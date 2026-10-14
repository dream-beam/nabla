import os
import ctypes
from ctypes import c_int, c_double, c_char_p, POINTER, create_string_buffer

class PiezoError(Exception):
    """Exception raised for errors in the Piezo controller."""
    pass

class PiezoController:
    DLL_PATH_BASE = r"C:\Program Files (x86)\Thorlabs\MDT69XB\Sample\Thorlabs_MDT69XB_PythonSDK"

    def __init__(self, serial_number=None):
        self.serial_number = serial_number
        self.handle = -1
        self._load_dll()

    def _load_dll(self):
        # Try x64 first, then win32
        dll_files = ["MDT_COMMAND_LIB_x64.dll", "MDT_COMMAND_LIB_win32.dll"]
        self.lib = None
        for dll_file in dll_files:
            full_path = os.path.join(self.DLL_PATH_BASE, dll_file)
            if os.path.exists(full_path):
                try:
                    self.lib = ctypes.cdll.LoadLibrary(full_path)
                    # print(f"Successfully loaded {dll_file}")
                    break
                except Exception as e:
                    print(f"Failed to load {dll_file}: {e}")
        
        if self.lib is None:
            raise PiezoError(f"Could not load MDT DLL from {self.DLL_PATH_BASE}. Please ensure the SDK is installed.")

        # Define function prototypes
        self._setup_prototypes()

    def _setup_prototypes(self):
        # Open
        self.lib.Open.restype = c_int
        self.lib.Open.argtypes = [c_char_p, c_int, c_int]
        
        # IsOpen
        self.lib.IsOpen.restype = c_int
        self.lib.IsOpen.argtypes = [c_char_p]
        
        # Close
        self.lib.Close.restype = c_int
        self.lib.Close.argtypes = [c_int]
        
        # List
        self.lib.List.restype = c_int
        self.lib.List.argtypes = [c_char_p, c_int]
        
        # GetId
        self.lib.GetId.restype = c_int
        self.lib.GetId.argtypes = [c_int, c_char_p]
        
        # GetLimitVoltage
        self.lib.GetLimitVoltage.restype = c_int
        self.lib.GetLimitVoltage.argtypes = [c_int, POINTER(c_double)]
        
        # SetAllVoltage
        self.lib.SetAllVoltage.restype = c_int
        self.lib.SetAllVoltage.argtypes = [c_int, c_double]
        
        # Master Scan
        self.lib.GetMasterScanEnable.restype = c_int
        self.lib.GetMasterScanEnable.argtypes = [c_int, POINTER(c_int)]
        self.lib.SetMasterScanEnable.restype = c_int
        self.lib.SetMasterScanEnable.argtypes = [c_int, c_int]
        self.lib.GetMasterScanVoltage.restype = c_int
        self.lib.GetMasterScanVoltage.argtypes = [c_int, POINTER(c_double)]
        self.lib.SetMasterScanVoltage.restype = c_int
        self.lib.SetMasterScanVoltage.argtypes = [c_int, c_double]
        
        # X Axis
        self.lib.GetXAxisVoltage.restype = c_int
        self.lib.GetXAxisVoltage.argtypes = [c_int, POINTER(c_double)]
        self.lib.SetXAxisVoltage.restype = c_int
        self.lib.SetXAxisVoltage.argtypes = [c_int, c_double]
        self.lib.GetXAxisMinVoltage.restype = c_int
        self.lib.GetXAxisMinVoltage.argtypes = [c_int, POINTER(c_double)]
        self.lib.SetXAxisMinVoltage.restype = c_int
        self.lib.SetXAxisMinVoltage.argtypes = [c_int, c_double]
        self.lib.GetXAxisMaxVoltage.restype = c_int
        self.lib.GetXAxisMaxVoltage.argtypes = [c_int, POINTER(c_double)]
        self.lib.SetXAxisMaxVoltage.restype = c_int
        self.lib.SetXAxisMaxVoltage.argtypes = [c_int, c_double]
        
        # Y Axis
        self.lib.GetYAxisVoltage.restype = c_int
        self.lib.GetYAxisVoltage.argtypes = [c_int, POINTER(c_double)]
        self.lib.SetYAxisVoltage.restype = c_int
        self.lib.SetYAxisVoltage.argtypes = [c_int, c_double]
        self.lib.GetYAxisMinVoltage.restype = c_int
        self.lib.GetYAxisMinVoltage.argtypes = [c_int, POINTER(c_double)]
        self.lib.SetYAxisMinVoltage.restype = c_int
        self.lib.SetYAxisMinVoltage.argtypes = [c_int, c_double]
        self.lib.GetYAxisMaxVoltage.restype = c_int
        self.lib.GetYAxisMaxVoltage.argtypes = [c_int, POINTER(c_double)]
        self.lib.SetYAxisMaxVoltage.restype = c_int
        self.lib.SetYAxisMaxVoltage.argtypes = [c_int, c_double]
        
        # Z Axis
        self.lib.GetZAxisVoltage.restype = c_int
        self.lib.GetZAxisVoltage.argtypes = [c_int, POINTER(c_double)]
        self.lib.SetZAxisVoltage.restype = c_int
        self.lib.SetZAxisVoltage.argtypes = [c_int, c_double]
        self.lib.GetZAxisMinVoltage.restype = c_int
        self.lib.GetZAxisMinVoltage.argtypes = [c_int, POINTER(c_double)]
        self.lib.SetZAxisMinVoltage.restype = c_int
        self.lib.SetZAxisMinVoltage.argtypes = [c_int, c_double]
        self.lib.GetZAxisMaxVoltage.restype = c_int
        self.lib.GetZAxisMaxVoltage.argtypes = [c_int, POINTER(c_double)]
        self.lib.SetZAxisMaxVoltage.restype = c_int
        self.lib.SetZAxisMaxVoltage.argtypes = [c_int, c_double]
        
        # General/XYZ
        self.lib.GetVoltageAdjustmentResolution.restype = c_int
        self.lib.GetVoltageAdjustmentResolution.argtypes = [c_int, POINTER(c_int)]
        self.lib.SetVoltageAdjustmentResolution.restype = c_int
        self.lib.SetVoltageAdjustmentResolution.argtypes = [c_int, c_int]
        self.lib.GetXYZAxisVoltage.restype = c_int
        self.lib.GetXYZAxisVoltage.argtypes = [c_int, POINTER(c_double), POINTER(c_double), POINTER(c_double)]
        self.lib.SetXYZAxisVoltage.restype = c_int
        self.lib.SetXYZAxisVoltage.argtypes = [c_int, c_double, c_double, c_double]

    @staticmethod
    def list_devices():
        """List all connected MDT devices."""
        buf = create_string_buffer(10240, '\0')
        # We need a temporary DLL handle or a way to call List. 
        # Since list_devices is static, we load a temporary lib.
        temp_controller = PiezoController()
        result = temp_controller.lib.List(buf, 10240)
        if result < 0:
            return []
        
        devices_str = buf.raw.decode("utf-8").rstrip('\x00').split(',')
        devices = []
        # The SDK format is [serial, type, serial, type...]
        for i in range(0, len(devices_str) - 1, 2):
            sn = devices_str[i]
            dtype = devices_str[i+1]
            if sn:
                devices.append((sn, dtype))
        return devices

    def open(self, serial_number=None, baud=115200, timeout=3):
        """Open the MDT device."""
        sn = serial_number or self.serial_number
        if not sn:
            raise PiezoError("Serial number is required to open the device.")
        
        self.serial_number = sn
        res = self.lib.Open(sn.encode('utf-8'), baud, timeout)
        if res < 0:
            raise PiezoError(f"Failed to open device {sn}. Error code: {res}")
        self.handle = res
        return self.handle

    def close(self):
        """Close the MDT device."""
        if self.handle == -1:
            return
        res = self.lib.Close(self.handle)
        if res < 0:
            raise PiezoError(f"Failed to close device. Error code: {res}")
        self.handle = -1

    def is_open(self):
        """Check if the device is open."""
        if not self.serial_number:
            return False
        return self.lib.IsOpen(self.serial_number.encode('utf-8')) != 0

    def get_id(self):
        """Get the product header and firmware version."""
        buf = create_string_buffer(1024, '\0')
        res = self.lib.GetId(self.handle, buf)
        if res < 0:
            raise PiezoError(f"Failed to get ID. Error code: {res}")
        return buf.raw.decode("utf-8").rstrip('\x00')

    def get_limit_voltage(self):
        """Get output voltage limit setting."""
        vol = c_double(0)
        res = self.lib.GetLimitVoltage(self.handle, vol)
        if res < 0:
            raise PiezoError(f"Failed to get limit voltage. Error code: {res}")
        return vol.value

    def set_voltage(self, axis, voltage):
        """Set output voltage for a specific axis ('X', 'Y', or 'Z')."""
        axis = axis.upper()
        if axis == 'X':
            res = self.lib.SetXAxisVoltage(self.handle, float(voltage))
        elif axis == 'Y':
            res = self.lib.SetYAxisVoltage(self.handle, float(voltage))
        elif axis == 'Z':
            res = self.lib.SetZAxisVoltage(self.handle, float(voltage))
        else:
            raise ValueError("Axis must be 'X', 'Y', or 'Z'")
        
        if res < 0:
            raise PiezoError(f"Failed to set voltage for axis {axis}. Error code: {res}")

    def get_voltage(self, axis):
        """Get output voltage for a specific axis ('X', 'Y', or 'Z')."""
        axis = axis.upper()
        vol = c_double(0)
        if axis == 'X':
            res = self.lib.GetXAxisVoltage(self.handle, vol)
        elif axis == 'Y':
            res = self.lib.GetYAxisVoltage(self.handle, vol)
        elif axis == 'Z':
            res = self.lib.GetZAxisVoltage(self.handle, vol)
        else:
            raise ValueError("Axis must be 'X', 'Y', or 'Z'")
        
        if res < 0:
            raise PiezoError(f"Failed to get voltage for axis {axis}. Error code: {res}")
        return vol.value

    def set_all_voltage(self, voltage):
        """Set all outputs to the desired voltage."""
        res = self.lib.SetAllVoltage(self.handle, float(voltage))
        if res < 0:
            raise PiezoError(f"Failed to set all voltage. Error code: {res}")

    def set_master_scan_enable(self, enable: bool):
        """Enable or disable Master Scan mode."""
        state = 1 if enable else 0
        res = self.lib.SetMasterScanEnable(self.handle, state)
        if res < 0:
            raise PiezoError(f"Failed to set master scan enable. Error code: {res}")

    def set_master_scan_voltage(self, voltage):
        """Set a master scan voltage that adds to the x, y, and z axis voltages."""
        res = self.lib.SetMasterScanVoltage(self.handle, float(voltage))
        if res < 0:
            raise PiezoError(f"Failed to set master scan voltage. Error code: {res}")

    def get_xyz_voltages(self):
        """Get voltages for X, Y, and Z axes as a tuple."""
        volX, volY, volZ = c_double(0), c_double(0), c_double(0)
        res = self.lib.GetXYZAxisVoltage(
            self.handle,
            volX,
            volY,
            volZ
        )
        if res < 0:
            raise PiezoError(f"Failed to get XYZ voltages. Error code: {res}")
        return (volX.value, volY.value, volZ.value)

    def set_xyz_voltages(self, x, y, z):
        """Set voltages for X, Y, and Z axes."""
        res = self.lib.SetXYZAxisVoltage(self.handle, float(x), float(y), float(z))
        if res < 0:
            raise PiezoError(f"Failed to set XYZ voltages. Error code: {res}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
