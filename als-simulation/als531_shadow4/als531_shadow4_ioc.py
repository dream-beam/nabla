#!/usr/bin/env python
"""
Standalone caproto IOC exposing the ALS 5.3.1 Shadow4 "digital twin"
beamline as EPICS process variables.

This IOC has two families of PVs, all under the prefix ``ALS531S4:``:

1. **Degree-of-freedom (DOF) PVs** -- ``ALS531S4:DOF:<NAME>`` -- ten
   writable ``ao`` (analog output) PVs corresponding to the beamline's
   degrees of freedom (mirror pitch, mono height/angle, hexapod
   offsets, etc). Writing to any of these PVs immediately triggers a
   full re-trace of the beamline (see :mod:`als531_shadow4_sim`) in a
   background thread, so the Channel Access server itself is never
   blocked by a (potentially slow) ray-tracing calculation.

2. **AreaDetector-style camera PVs** -- mimicking the minimal
   ``cam1:``/``image1:`` NDArray interface already used elsewhere in
   this repository to talk to the *real* ALS 5.3.1 cameras (see e.g.
   ``blop-als531-test/20251125khc_blop-als531-test.ipynb`` or
   ``wfs-als531/20260309aiw_wfs_scan_labview.ipynb``):

   * ``ALS531S4:cam1:Acquire``       -- write 1 to manually trigger a
                                        re-trace using the current DOF
                                        values.
   * ``ALS531S4:cam1:Acquire_RBV``   -- 1 while a trace is running, 0
                                        when idle/done.
   * ``ALS531S4:cam1:Done``          -- reset to 0 as soon as a trace is
                                        (re-)triggered (by writing
                                        ``cam1:Acquire`` or any DOF PV),
                                        and set to 1 once that trace has
                                        finished. Unlike ``Acquire_RBV``
                                        (which pulses 0 -> 1 -> 0 around
                                        the trace and can therefore race
                                        with a client that subscribes
                                        after a fast trace has already
                                        completed), ``Done`` is safe to
                                        poll or subscribe to *after*
                                        writing ``cam1:Acquire``: simply
                                        wait for it to read back 1.
   * ``ALS531S4:cam1:AcquireTime``   -- simulated exposure time, in
                                        seconds (does not affect the
                                        trace; provided for API
                                        compatibility with real ALS
                                        5.3.1 camera IOCs).
   * ``ALS531S4:cam1:ArrayCounter_RBV`` -- increments on each
                                        completed frame.
   * ``ALS531S4:cam1:ArraySizeX_RBV``/``ArraySizeY_RBV`` -- static
                                        camera image dimensions, in
                                        pixels.
   * ``ALS531S4:image1:ArrayData``  -- the actual camera image,
                                        flattened row-major (C order),
                                        matching the convention used
                                        by real areaDetector IOCs (and
                                        by the existing notebooks that
                                        do
                                        ``img.reshape((ny, nx))``).
   * ``ALS531S4:SIM:INTENSITY_TOTAL`` -- convenience scalar PV with the
                                        total (summed) ray intensity of
                                        the last completed trace.

Run this script directly to start the IOC::

    python als531_shadow4_ioc.py --list-pvs

and connect to it from the companion notebook (``als531_shadow4.ipynb``)
or any other EPICS client using ``ophyd``/``caproto``/``pyepics``.
"""


#export EPICS_CAS_AUTO_BEACON_ADDR_LIST=no
#export EPICS_CAS_BEACON_ADDR_LIST="127.0.0.1"

#with a mac in a terminal you need to have
# caproto-repeater


from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from textwrap import dedent
import platform
import os

# Disable caproto beacon on macOS to prevent CaprotoNetworkError (broadcast restricted)
if platform.system() == "Darwin":
    os.environ["EPICS_CAS_AUTO_BEACON_ADDR_LIST"] = "no"

import numpy as np

import caproto.server.common
from caproto.server import PVGroup, SubGroup, ioc_arg_parser, pvproperty, run

# Robustly disable caproto beacon on macOS to prevent CaprotoNetworkError (broadcast restricted)
if platform.system() == "Darwin":
    async def _disabled_beacon(*args, **kwargs):
        pass
    caproto.server.common.broadcast_beacon_loop = _disabled_beacon

from als531_shadow4_sim import (
    CAMERA_NX,
    CAMERA_NY,
    DEFAULT_DOF_VALUES,
    DOF_NAMES,
    build_and_trace,
)

logger = logging.getLogger("als531_shadow4_ioc")

#: Map from DOF name (as used in als531_shadow4_sim) to the PV attribute
#: name used below. EPICS PV names are upper-cased versions of the DOF
#: name, e.g. ``m1_rel_pitch_deg`` -> ``ALS531S4:DOF:M1_REL_PITCH_DEG``.
_DOF_PV_SUFFIX = {name: name.upper() for name in DOF_NAMES}


class DOFGroup(PVGroup):
    """The ten degree-of-freedom PVs for the ALS 5.3.1 Shadow4 model.

    Each is a plain analog-output PV. Writing a new value triggers a
    re-trace of the beamline via the parent IOC's ``request_trace``
    method.
    """

    m1_rel_pitch_deg = pvproperty(
        value=DEFAULT_DOF_VALUES["m1_rel_pitch_deg"],
        name="M1_REL_PITCH_DEG",
        doc="M1 mirror pitch offset relative to nominal, in degrees",
    )
    m1_bending_m = pvproperty(
        value=DEFAULT_DOF_VALUES["m1_bending_m"],
        name="M1_BENDING_M",
        doc="M1 mirror tangential (meridional) bending radius, in metres",
    )
    mono_height_m = pvproperty(
        value=DEFAULT_DOF_VALUES["mono_height_m"],
        name="MONO_HEIGHT_M",
        doc="DCM1 crystal height offset, in metres",
    )
    mono_angle_rel_angle_deg = pvproperty(
        value=DEFAULT_DOF_VALUES["mono_angle_rel_angle_deg"],
        name="MONO_ANGLE_REL_ANGLE_DEG",
        doc="DCM1 crystal angle offset relative to Bragg angle, in degrees",
    )
    hexapod_rel_x_m = pvproperty(
        value=DEFAULT_DOF_VALUES["hexapod_rel_x_m"],
        name="HEXAPOD_REL_X_M",
        doc="Hexapod mirror X offset, in metres",
    )
    hexapod_rel_y_m = pvproperty(
        value=DEFAULT_DOF_VALUES["hexapod_rel_y_m"],
        name="HEXAPOD_REL_Y_M",
        doc="Hexapod mirror Y offset, in metres",
    )
    hexapod_rel_z_m = pvproperty(
        value=DEFAULT_DOF_VALUES["hexapod_rel_z_m"],
        name="HEXAPOD_REL_Z_M",
        doc="Hexapod mirror Z offset, in metres",
    )
    hexapod_rel_Rx_deg = pvproperty(
        value=DEFAULT_DOF_VALUES["hexapod_rel_Rx_deg"],
        name="HEXAPOD_REL_RX_DEG",
        doc="Hexapod mirror rotation about X, in degrees",
    )
    hexapod_rel_Ry_deg = pvproperty(
        value=DEFAULT_DOF_VALUES["hexapod_rel_Ry_deg"],
        name="HEXAPOD_REL_RY_DEG",
        doc="Hexapod mirror rotation about Y, in degrees",
    )
    hexapod_rel_Rz_deg = pvproperty(
        value=DEFAULT_DOF_VALUES["hexapod_rel_Rz_deg"],
        name="HEXAPOD_REL_RZ_DEG",
        doc="Hexapod mirror rotation about Z, in degrees",
    )

    def _pvproperties(self):
        """Yield (dof_name, pvproperty instance) pairs for all 10 DOFs."""
        for dof_name in DOF_NAMES:
            yield dof_name, getattr(self, dof_name)


def _make_dof_putter(dof_name):
    """Create a putter callback for the given DOF name that requests a
    beamline re-trace after the new value has been accepted."""

    async def putter(self, instance, value):
        # `self` here is the DOFGroup instance; its parent is the IOC.
        ioc = self.parent
        ioc.request_trace_threadsafe(**{dof_name: value})
        return value

    return putter


# Attach putter callbacks to each DOF pvproperty after the class body,
# since we need the DOF name captured in the closure above.
for _dof_name in DOF_NAMES:
    _prop = getattr(DOFGroup, _dof_name)
    _prop.putter(_make_dof_putter(_dof_name))


class CameraGroup(PVGroup):
    """Minimal areaDetector-style ``cam1:``/``image1:`` interface backed
    by the Shadow4 simulation, so existing analysis code written against
    the real ALS 5.3.1 cameras works unmodified against this simulated
    beamline.
    """

    acquire = pvproperty(
        value=0,
        name="cam1:Acquire",
        doc="Write 1 to trigger a re-trace using the current DOF values",
    )
    acquire_rbv = pvproperty(
        value=0,
        name="cam1:Acquire_RBV",
        doc="1 while a trace is running, 0 when idle/done",
        read_only=True,
    )
    done = pvproperty(
        value=1,
        name="cam1:Done",
        doc=(
            "0 as soon as a trace is (re-)triggered, 1 once that trace "
            "has finished. Safe to poll/subscribe to immediately after "
            "triggering, unlike Acquire_RBV which can race with very "
            "fast traces."
        ),
        read_only=True,
    )
    acquire_time = pvproperty(
        value=1.0,
        name="cam1:AcquireTime",
        doc=(
            "Simulated exposure time, in seconds. Does not affect the "
            "trace itself; provided purely for API compatibility with "
            "real ALS 5.3.1 camera IOCs, which expose a real "
            "cam1:AcquireTime PV."
        ),
    )
    array_counter_rbv = pvproperty(
        value=0,
        name="cam1:ArrayCounter_RBV",
        doc="Increments on each completed frame",
        read_only=True,
    )
    array_size_x_rbv = pvproperty(
        value=CAMERA_NX,
        name="cam1:ArraySizeX_RBV",
        doc="Camera image width, in pixels",
        read_only=True,
    )
    array_size_y_rbv = pvproperty(
        value=CAMERA_NY,
        name="cam1:ArraySizeY_RBV",
        doc="Camera image height, in pixels",
        read_only=True,
    )
    array_data = pvproperty(
        value=[0.0] * (CAMERA_NX * CAMERA_NY),
        name="image1:ArrayData",
        doc="Flattened (row-major) camera image intensity array",
        read_only=True,
        max_length=CAMERA_NX * CAMERA_NY,
    )

    @acquire.putter
    async def acquire(self, instance, value):
        if int(value):
            ioc = self.parent
            ioc.request_trace_threadsafe()
        return value


class SimGroup(PVGroup):
    """Convenience scalar PVs summarizing the last completed trace."""

    intensity_total = pvproperty(
        value=0.0,
        name="INTENSITY_TOTAL",
        doc="Sum of ray intensities on the camera from the last completed trace",
        read_only=True,
    )


class ALS531Shadow4IOC(PVGroup):
    """Top-level IOC grouping the DOF, camera, and summary PVs together,
    and coordinating background beamline re-traces.
    """

    dof = SubGroup(DOFGroup, prefix="DOF:")
    cam1 = SubGroup(CameraGroup, prefix="")
    sim = SubGroup(SimGroup, prefix="SIM:")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._current_dof_values = dict(DEFAULT_DOF_VALUES)

        # -- Trace serialization state ---------------------------------
        # Exactly one _run_trace() coroutine may be "active" (i.e. have
        # driven done: 1 -> 0 -> ... -> 1) at a time. If a new trace is
        # requested while one is already running, we do NOT start a
        # second, overlapping _run_trace() -- doing so would let two
        # coroutines race to write cam1:done / cam1:acquire_rbv /
        # image1:ArrayData, so a client (e.g. bluesky/ophyd) watching
        # for a single 0 -> 1 transition on `done` could observe the
        # *other* trace's completion and read back a stale or
        # about-to-be-overwritten image (exactly the symptom reported
        # when driving this IOC from a bluesky RunEngine: a motor set()
        # and a detector trigger() each request a trace, and without
        # serialization those two requests can race).
        #
        # Instead, an in-flight trace is left to run to completion, and
        # any additional request(s) that arrive in the meantime are
        # coalesced into a single *pending* flag. Once the current trace
        # finishes, if a request came in while it was running, exactly
        # one more trace is started (using the latest DOF values) before
        # `done` is allowed to settle at 1.
        self._trace_lock = asyncio.Lock()
        self._trace_pending = False

    # -- Trace orchestration -------------------------------------------------

    def request_trace_threadsafe(self, **dof_overrides):
        """Request a beamline re-trace on the IOC's async loop.

        This may be called from caproto putter coroutines (already on
        the event loop) or, in principle, from another thread; caproto
        putters run on the IOC's asyncio loop, so we simply create a
        task here.

        Multiple calls that arrive while a trace is already running are
        coalesced: at most one additional trace will run after the
        current one finishes, using whatever DOF values are current at
        that time. This guarantees traces never overlap.
        """
        self._current_dof_values.update(dof_overrides)

        try:
            asyncio.get_running_loop().create_task(self._request_trace_async())
        except RuntimeError:
            # No running loop (shouldn't normally happen inside a
            # putter) -- run synchronously as a last resort.
            asyncio.run(self._request_trace_async())

    async def _request_trace_async(self):
        if self._trace_lock.locked():
            # A trace is already running (or another coalesced request
            # is already queued behind it) -- just flag that another
            # trace should run once the current one completes, and
            # return immediately without starting a second, overlapping
            # _run_trace().
            self._trace_pending = True
            return

        async with self._trace_lock:
            # Run the requested trace, then keep re-running (using the
            # latest DOF values each time) for as long as further
            # requests keep arriving while we work -- this collapses any
            # burst of overlapping requests (e.g. a motor set() followed
            # immediately by a detector trigger()) into the minimum
            # number of actual traces, while guaranteeing `done` only
            # ever reflects one unambiguous trace's start/finish at a
            # time.
            while True:
                self._trace_pending = False
                await self._run_trace()
                if not self._trace_pending:
                    break

    async def _run_trace(self):
        # `done` goes to 0 the instant a trace starts (safe to poll or
        # subscribe to right after writing cam1:Acquire -- no race with
        # fast traces, unlike acquire_rbv's 0 -> 1 -> 0 pulse) and back
        # to 1 only once the trace has fully completed (including the
        # image/intensity/counter PV writes below). Because this method
        # is only ever invoked from within `_trace_lock` (see
        # `_request_trace_async` above), at most one `_run_trace()` call
        # is ever in flight, so these PV writes cannot race with another
        # trace's.
        await self.cam1.done.write(0)
        await self.cam1.acquire_rbv.write(1)
        try:
            loop = asyncio.get_event_loop()
            dof_values = dict(self._current_dof_values)
            result = await loop.run_in_executor(
                self._executor, build_and_trace, dof_values
            )
            flat_image = np.asarray(result["camera_image"], dtype=float).reshape(-1)
            await self.cam1.array_data.write(flat_image.tolist())
            await self.sim.intensity_total.write(result["intensity_total"])
            new_counter = int(self.cam1.array_counter_rbv.value) + 1
            await self.cam1.array_counter_rbv.write(new_counter)
        except Exception:
            logger.exception("Beamline trace failed")
        finally:
            await self.cam1.acquire_rbv.write(0)
            await self.cam1.done.write(1)


def main():
    ioc_options, run_options = ioc_arg_parser(
        default_prefix="ALS531S4:",
        desc=dedent(main.__doc__ or ALS531Shadow4IOC.__doc__ or ""),
    )
    ioc = ALS531Shadow4IOC(**ioc_options)
    run(ioc.pvdb, **run_options)


main.__doc__ = """
ALS 5.3.1 Shadow4 digital-twin caproto IOC.

Exposes the beamline's degrees of freedom as writable PVs under
``<prefix>DOF:`` and the resulting simulated camera image as an
areaDetector-style ``<prefix>cam1:``/``<prefix>image1:`` interface.
"""


if __name__ == "__main__":
    main()
