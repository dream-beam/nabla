import time
import numpy as np
import threading
from collections import OrderedDict

from ophyd import ADComponent
from ophyd import ImagePlugin
from ophyd import SingleTrigger
from ophyd import Component, DetectorBase, CamBase, EpicsSignal, EpicsSignalRO
from ophyd.areadetector.filestore_mixins import FileStoreTIFFIterativeWrite
from ophyd.areadetector.plugins import TIFFPlugin
import os

BASLER_FILES_ROOT = "/mnt/data531"
BASLER_TEST_IMAGE_DIR = "20251113_test/%Y/%m/%d"   # e.g. "20251113_test/%Y/%m/%d" if you want date subdirs



def test():
    print("this worked!")


from ophyd import (
    EpicsMotor, Device, Signal, PVPositioner, EpicsSignal, 
    EpicsSignalRO, Component as Cpt
)
from ophyd.pseudopos import (
    PseudoPositioner, PseudoSingle,
    pseudo_position_argument, real_position_argument
)
from ophyd.signal import AttributeSignal
from ophyd.status import DeviceStatus
import ophyd

from epics import caget, caput
# ============================================================================
# Physical Constants for Monochromator
# ============================================================================

H_M2KGPS = 6.6261e-34      # Planck constant (J·s)
C_MPS = 299792458           # Speed of light (m/s)
E_EV = 6.2415e18            # Elementary charge (1/eV to J conversion)

# Silicon crystal parameters
SI_M = 5.43e-10             # Si lattice constant (m)
A_SI111_M = SI_M / np.sqrt(3)  # Si(1,1,1) d-spacing (m)
# 19.2567degree at copper edge 8980.3eV
# H_M2KGPS * C_MPS * E_EV/(energies_kev*1000)/(2*A_SI111_M)
# Calibration
# 19.2525
# 19.223 Dec 4, 2025
DEFAULT_MONO_OFFSET_DEG = 19.16745 - np.arcsin(H_M2KGPS * C_MPS * E_EV/(8978.8)/(2*A_SI111_M)) * 180/np.pi  # Default calibration offset









class MonoEnergy(PseudoPositioner):
    """
    Monochromator energy pseudo positioner.
    
    Provides energy control (eV) by moving the mono_angle motor (degrees).
    Uses Bragg's law: E = h*c / (2*d*sin(θ)) for Si(111) crystal.
    
    Real axis:
        mono_angle: Physical monochromator angle in degrees
        
    Pseudo axis:
        energy_eV: Photon energy in eV
        
    Example:
        >>> mono.energy_eV.position  # Read current energy
        8930.0
        >>> mono.energy_eV.move(9000)  # Move to 9 keV
    """
    
    # Pseudo axis - what the user controls
    energy_eV = Cpt(PseudoSingle, limits=(2400, 12000), egu='eV', kind='hinted')
    
    # Real axis - the physical motor
    mono_angle = Cpt(EpicsMotor, 'bl531_xps1:mono_angle_deg', labels={'motors'}, kind='normal')
    
    # Calibration offset (can be changed at runtime)
    offset = Cpt(AttributeSignal, attr='_offset', name='offset')
    
    def __init__(self, *args, offset=DEFAULT_MONO_OFFSET_DEG, **kwargs):
        """
        Initialize monochromator energy positioner.
        
        Args:
            offset: Calibration offset in degrees (default: -18.1361915)
        """
        self._offset = offset
        super().__init__(*args, **kwargs)
    
    @property
    def _d_spacing(self):
        """Si(111) d-spacing in meters."""
        return A_SI111_M
    
    @property
    def _hc_factor(self):
        """Constant factor h*c*E_EV for energy calculation."""
        return H_M2KGPS * C_MPS * E_EV
    
    @pseudo_position_argument
    def forward(self, pseudo_pos):
        """
        Convert energy (eV) to mono_angle (degrees).
        
        Bragg's law: θ = arcsin(h*c/(2*d*E))
        
        Args:
            pseudo_pos: PseudoPosition with .energy_eV attribute (eV)
            
        Returns:
            RealPosition with .mono_angle attribute (degrees)
        """
        energy_ev = pseudo_pos.energy_eV
        
        # Calculate angle from energy using Bragg's law
        sin_theta = self._hc_factor / (2 * self._d_spacing * energy_ev)
        
        # Check if physically possible
        if abs(sin_theta) > 1:
            raise ValueError(
                f"Energy {energy_ev} eV is outside physical range. "
                f"sin(θ) = {sin_theta:.3f} (must be ≤ 1)"
            )
        
        theta_rad = np.arcsin(sin_theta)
        theta_deg = np.degrees(theta_rad)
        
        # Apply calibration offset
        mono_angle_deg = theta_deg + self._offset
        
        return self.RealPosition(mono_angle=mono_angle_deg)
    
    @real_position_argument
    def inverse(self, real_pos):
        """
        Convert mono_angle (degrees) to energy (eV).
        
        Bragg's law: E = h*c / (2*d*sin(θ))
        
        Handles invalid angles gracefully to prevent subscription errors.
        
        Args:
            real_pos: RealPosition with .mono_angle attribute (degrees)
            
        Returns:
            PseudoPosition with .energy_eV attribute (eV)
        """
        mono_angle_deg = real_pos.mono_angle
        
        # Remove calibration offset
        theta_deg = mono_angle_deg - self._offset
        theta_rad = np.radians(theta_deg)
        
        sin_theta = np.sin(theta_rad)
        
        # Handle invalid Bragg angles gracefully
        # Valid Bragg angles need 0 < sin(θ) ≤ 1
        if sin_theta <= 0 or sin_theta > 1:
            # Return fallback value to prevent subscription errors
            return self.PseudoPosition(energy_eV=self.energy_eV.limits[0])
        
        # Calculate energy from angle using Bragg's law
        energy_ev = self._hc_factor / (2 * self._d_spacing * sin_theta)
        
        # Clamp to valid energy range
        energy_ev = max(self.energy_eV.limits[0], 
                       min(energy_ev, self.energy_eV.limits[1]))
        
        return self.PseudoPosition(energy_eV=energy_ev)





############### Basler TIFF Plugin ################

class BaslerTIFFPlugin(FileStoreTIFFIterativeWrite, TIFFPlugin):

    def describe(self):
        ret = super().describe()
        key = self.parent._image_name

        color_mode = self.parent.cam.color_mode.get(as_string=True)
        num_images = self.parent.cam.num_images.get()
        height = self.array_size.height.get()
        width  = self.array_size.width.get()

        if color_mode == 'Mono':
            ret[key]['shape'] = [num_images, height, width]
        elif color_mode in ['RGB1', 'Bayer']:
            ret[key]['shape'] = [num_images, height, width, 3]
        else:
            raise RuntimeError(f"Unexpected color_mode: {color_mode!r}")

        # dtype mapping
        cam_dtype = self.data_type.get(as_string=True)
        type_map = {
            'UInt8':   '|u1',
            'UInt16':  '<u2',
            'Float32': '<f4',
            'Float64': '<f8',
            'Int32':   '<i4',
        }
        if cam_dtype in type_map:
            ret[key].setdefault('dtype_str', type_map[cam_dtype])

        return ret


############### Basler Cam ################

class BaslerCam(CamBase):
    """
    Basler camera CAM plugin.
    Extends CamBase with Basler-specific PVs.
    CamBase already provides: acquire, acquire_time, acquire_period,
    num_images, image_mode, trigger_mode, color_mode, data_type, etc.
    Only add PVs that CamBase does NOT already define.
    """

    # Basler-specific PVs not in CamBase
    pixel_format  = Component(EpicsSignal,   'PixelFormat',  string=True)
    gain          = Component(EpicsSignal,   'Gain')
    exposure_auto = Component(EpicsSignal,   'ExposureAuto', string=True)
    gain_auto     = Component(EpicsSignal,   'GainAuto',     string=True)

    # Readback for acquire so SingleTrigger can poll it
    acquire_rbv   = Component(EpicsSignalRO, 'Acquire_RBV')


############### Basler Detector ################

class BaslerDetector(SingleTrigger, DetectorBase):
    """
    Complete Basler area-detector device.
    Uses SingleTrigger mixin so bp.count() / bp.scan() work out of the box.
    """

    cam   = ADComponent(BaslerCam,          'cam1:')
    image = ADComponent(ImagePlugin,        'image1:')
    tiff  = ADComponent(
        BaslerTIFFPlugin,
        'TIFF1:',
        write_path_template=os.path.join(BASLER_FILES_ROOT, BASLER_TEST_IMAGE_DIR),
        read_path_template=os.path.join(BASLER_FILES_ROOT, BASLER_TEST_IMAGE_DIR),
    )




import time
import threading
import re
import numpy as np
from ophyd import Device, Component
from ophyd.status import DeviceStatus


# ============================================================
# Helper: parse LabView float responses safely
# ============================================================

def parse_labview_float(s):
    """Parse a LabView TCP response like '10.000000' or '0.000000!0' to float."""
    s = str(s).strip()
    m = re.match(r'^[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', s)
    if m:
        return float(m.group(0))
    raise ValueError(f"Cannot parse float from {s!r}")


# ============================================================
# BCS Motor ophyd Device — Tiled/bluesky compatible
# ============================================================

class BCSMotor(Device):
    """
    Ophyd Device wrapping a single BCS LabView TCP motor axis.
    Fully compatible with bluesky RunEngine and Tiled.

    Parameters
    ----------
    bcs : connection
        Your existing LabView TCP connection object.
    motor_id : str
        The motor identifier string used by BCS (e.g. 'Galil x.20 A').
    settle_time : float
        Seconds to wait after reaching target before marking done. Default 0.5s.
    tolerance : float
        Position tolerance in engineering units. Default 0.01.
    units : str
        Engineering units string for Tiled metadata. Default 'mm'.
    name : str
        Ophyd device name (required).
    """

    def __init__(
        self,
        bcs,
        motor_id,
        settle_time=0.5,
        tolerance=0.01,
        units='mm',
        *args,
        **kwargs,
    ):
        super().__init__(prefix='', *args, **kwargs)
        self._bcs       = bcs
        self._motor_id  = motor_id
        self._settle_time = settle_time
        self._tolerance   = tolerance
        self._units       = units
        self._target      = None   # last commanded target
        self._set_status  = None   # current move DeviceStatus

    # ----------------------------------------------------------
    # Core read/describe — what bluesky records per event
    # ----------------------------------------------------------

    def get(self):
        """Return current motor position as float."""
        raw = self._bcs.GetMotorPos(self._motor_id)
        return parse_labview_float(raw)

    def read(self):
        """
        Called by bluesky at every event point.
        Must return dict of {signal_name: {'value': ..., 'timestamp': ...}}
        """
        pos = self.get()
        ts  = time.time()
        return {
            f'{self.name}':         {'value': pos,           'timestamp': ts},
            f'{self.name}_setpoint': {'value': self._target if self._target is not None else pos,
                                      'timestamp': ts},
        }

    def describe(self):
        """
        Called by bluesky at run start.
        Must return dict matching the keys in read().
        """
        return {
            f'{self.name}': {
                'source':   f'BCS:{self._motor_id}',
                'dtype':    'number',
                'shape':    [],
                'units':    self._units,
                'lower_ctrl_limit': 0.0,
                'upper_ctrl_limit': 0.0,
            },
            f'{self.name}_setpoint': {
                'source':   f'BCS:{self._motor_id}:setpoint',
                'dtype':    'number',
                'shape':    [],
                'units':    self._units,
                'lower_ctrl_limit': 0.0,
                'upper_ctrl_limit': 0.0,
            },
        }

    # ----------------------------------------------------------
    # Configuration read/describe — static metadata for Tiled
    # ----------------------------------------------------------

    def read_configuration(self):
        """
        Called by bluesky at run start/stop.
        Returns slow/static configuration values.
        """
        return {
            f'{self.name}_motor_id':    {'value': self._motor_id,    'timestamp': time.time()},
            f'{self.name}_settle_time': {'value': self._settle_time, 'timestamp': time.time()},
            f'{self.name}_tolerance':   {'value': self._tolerance,   'timestamp': time.time()},
            f'{self.name}_units':       {'value': self._units,       'timestamp': time.time()},
        }

    def describe_configuration(self):
        """
        Called by bluesky at run start.
        Must return dict matching the keys in read_configuration().
        """
        return {
            f'{self.name}_motor_id': {
                'source': f'BCS:{self._motor_id}:config',
                'dtype':  'string',
                'shape':  [],
                'units':  None,
            },
            f'{self.name}_settle_time': {
                'source': f'BCS:{self._motor_id}:config',
                'dtype':  'number',
                'shape':  [],
                'units':  's',
            },
            f'{self.name}_tolerance': {
                'source': f'BCS:{self._motor_id}:config',
                'dtype':  'number',
                'shape':  [],
                'units':  self._units,
            },
            f'{self.name}_units': {
                'source': f'BCS:{self._motor_id}:config',
                'dtype':  'string',
                'shape':  [],
                'units':  None,
            },
        }

    # ----------------------------------------------------------
    # Hints — tells Tiled/bluesky which field is the primary one
    # ----------------------------------------------------------

    @property
    def hints(self):
        """
        Tells bluesky/Tiled which field to use as the primary
        axis in LivePlot, BestEffortCallback, and Tiled plots.
        """
        return {'fields': [self.name]}

    # ----------------------------------------------------------
    # Move / set
    # ----------------------------------------------------------

    def set(self, target, timeout=120):
        """
        Move the motor to target position.
        Returns a DeviceStatus that completes when the motor
        reaches the target (within tolerance) plus settle time.
        The RunEngine will block until this status is marked done.
        """
        self._target = target
        st = DeviceStatus(self, timeout=timeout)
        self._set_status = st

        def _move():
            try:
                # Issue the move command
                print(f"[BCSMotor] {self.name}: moving to {target:.4f} {self._units}")
                self._bcs.MoveMotor(self._motor_id, target)

                # Wait before first readback so the motor controller
                # has time to update its position
                time.sleep(2)

                # Step 1: poll until motor is within tolerance of target
                deadline = time.time() + timeout
                pos = self.get()
                while time.time() < deadline:
                    try:
                        pos = self.get()
                    except Exception as e:
                        print(f"[BCSMotor] position read error: {e}")
                        time.sleep(0.1)
                        continue

                    remaining = abs(pos - target)
                    # print(f"[BCSMotor] {self.name}: pos={pos:.4f}, "
                    #     f"target={target:.4f}, remaining={remaining:.4f}")

                    if remaining <= self._tolerance:
                        break

                    time.sleep(0.05)
                else:
                    # Loop exhausted without reaching target
                    st.set_exception(
                        TimeoutError(
                            f"[BCSMotor] {self.name}: timed out after {timeout}s. "
                            f"Last pos={pos:.4f}, target={target:.4f}"
                        )
                    )
                    return

                # Step 2: settle time
                # print(f"[BCSMotor] {self.name}: within tolerance, "
                #     f"settling for {self._settle_time}s...")
                time.sleep(self._settle_time)

                # Step 3: final position readback
                final = self.get()
                print(f"[BCSMotor] {self.name}: done. "
                    f"Final position = {final:.4f} {self._units}")

                # Mark done — RunEngine unblocks here
                st.set_finished()

            except Exception as e:
                st.set_exception(e)

        t = threading.Thread(target=_move, daemon=True)
        t.start()
        return st

    # ----------------------------------------------------------
    # Stage / unstage (called by bluesky at scan start/end)
    # ----------------------------------------------------------

    def stage(self):
        """Called by bluesky before a scan starts."""
        return [self]

    def unstage(self):
        """Called by bluesky after a scan ends."""
        return [self]

    # ----------------------------------------------------------
    # Stop
    # ----------------------------------------------------------

    def stop(self, *, success=False):
        """Stop the motor immediately (called on RE.abort())."""
        try:
            self._bcs.StopMotor(self._motor_id)
        except Exception as e:
            print(f"[BCSMotor] stop error: {e}")

    # ----------------------------------------------------------
    # Convenience
    # ----------------------------------------------------------

    @property
    def position(self):
        """Current motor position as float."""
        return self.get()

    def __repr__(self):
        try:
            pos = self.get()
            return (f"BCSMotor(name={self.name!r}, "
                    f"motor_id={self._motor_id!r}, "
                    f"position={pos:.4f} {self._units})")
        except Exception:
            return (f"BCSMotor(name={self.name!r}, "
                    f"motor_id={self._motor_id!r})")
        
