"""Ophyd adapters for the MDT piezo, Kinesis linear stage, and Basler camera.

Instantiate the drivers explicitly; importing this module never opens hardware.
The piezo axes share one PiezoController and should share one threading lock.
Positions exposed by the KCube adapter are in millimeters, not pylablib's meters.
"""

import math
import threading
import time

import numpy as np
from ophyd import Device
from ophyd.status import DeviceStatus


def _finished(device):
    status = DeviceStatus(device)
    status.set_finished()
    return status


class PiezoAxis(Device):
    """One MDT69XB voltage axis, with independent Bluesky read/set semantics.

    The controller must already be open; X and Y should use the *same* lock.
    Voltage readback is the controller output setting, not a position sensor.
    """

    def __init__(self, controller, axis, *, name, lock=None, **kwargs):
        if axis.upper() not in ("X", "Y"):
            raise ValueError("PiezoAxis supports X and Y only")
        super().__init__(name=name, **kwargs)
        self.controller = controller
        self.axis = axis.upper()
        self._lock = lock if lock is not None else threading.RLock()

    def read(self):
        with self._lock:
            voltage = self.controller.get_voltage(self.axis)
        return {self.name: {"value": voltage, "timestamp": time.time()}}

    def describe(self):
        return {self.name: {"source": f"MDT69XB:{self.controller.serial_number}:{self.axis}",
                            "dtype": "number", "shape": [], "units": "V"}}

    @property
    def hints(self):
        return {"fields": [self.name]}

    def set(self, value, **kwargs):
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("Voltage must be finite")
        with self._lock:
            maximum = self.controller.get_limit_voltage()
            if not 0 <= value <= maximum:
                raise ValueError(f"Requested {value} V is outside 0..{maximum} V")
            self.controller.set_voltage(self.axis, value)
        return _finished(self)


class KCubeAxis(Device):
    """One translational pylablib KinesisMotor, reporting and accepting mm.

    Pass a motor opened with scale='stage' and checked travel limits in mm.
    The limit tuple must be provided explicitly from the attached stage's specs.
    No homing or automatic return to a prior position is performed.
    """

    def __init__(self, motor, *, name, limits_mm, timeout=30, **kwargs):
        super().__init__(name=name, **kwargs)
        if not motor.get_stage() or motor.get_scale_units() != "m":
            raise ValueError("KCube requires a detected linear stage in meters (scale='stage')")
        if (limits_mm is None or len(limits_mm) != 2
                or not all(math.isfinite(x) for x in limits_mm)
                or limits_mm[0] >= limits_mm[1]):
            raise ValueError("Provide the verified (minimum, maximum) stage travel in mm")
        self.motor = motor
        self.limits_mm = tuple(limits_mm)
        self.timeout = timeout
        self._lock = threading.RLock()
        self._moving = False

    def read(self):
        with self._lock:
            position_mm = self.motor.get_position() * 1000
        return {self.name: {"value": position_mm, "timestamp": time.time()}}

    def describe(self):
        return {self.name: {"source": f"Kinesis:{self.motor.get_stage()}",
                            "dtype": "number", "shape": [], "units": "mm",
                            "lower_ctrl_limit": self.limits_mm[0],
                            "upper_ctrl_limit": self.limits_mm[1]}}

    @property
    def hints(self):
        return {"fields": [self.name]}

    def set(self, value, **kwargs):
        target_mm = float(value)
        if not math.isfinite(target_mm) or not self.limits_mm[0] <= target_mm <= self.limits_mm[1]:
            raise ValueError(f"Target {target_mm} mm outside configured limits {self.limits_mm}")
        with self._lock:
            if self._moving:
                raise RuntimeError("A KCube move is already in progress")
            self._moving = True
        status = DeviceStatus(self, timeout=self.timeout + 5)

        def move():
            error = None
            try:
                with self._lock:
                    self.motor.move_to(target_mm / 1000)
                    self.motor.wait_move(timeout=self.timeout)
                    reached_mm = self.motor.get_position() * 1000
                if abs(reached_mm - target_mm) > 0.05:
                    raise RuntimeError(f"KCube stopped at {reached_mm} mm instead of {target_mm} mm")
            except Exception as exc:
                error = exc
            finally:
                with self._lock:
                    self._moving = False
            if error is None:
                status.set_finished()
            else:
                status.set_exception(error)

        threading.Thread(target=move, daemon=True).start()
        return status


class BaslerCamera(Device):
    """Single-shot pypylon detector; each Bluesky trigger caches a Mono8 image.

    The camera is selected by IP, rather than relying on enumeration order.
    This embeds each image array in the event document; for large scans use
    an external asset writer instead of collecting full frames in memory.
    """

    def __init__(self, ip_address, *, name, grab_timeout_ms=5000, **kwargs):
        super().__init__(name=name, **kwargs)
        from pypylon import pylon

        factory = pylon.TlFactory.GetInstance()
        info = next((dev for dev in factory.EnumerateDevices()
                     if hasattr(dev, "GetIpAddress") and dev.GetIpAddress() == ip_address), None)
        if info is None:
            raise RuntimeError(f"No accessible Basler camera at {ip_address}")
        self._camera = pylon.InstantCamera(factory.CreateDevice(info))
        self._camera.Open()
        self._converter = pylon.ImageFormatConverter()
        self._converter.OutputPixelFormat = pylon.PixelType_Mono8
        self._shape = [int(self._camera.Height.GetValue()), int(self._camera.Width.GetValue())]
        self._timeout_ms = grab_timeout_ms
        self._ip_address = ip_address
        self._image = None
        self._timestamp = None
        self._lock = threading.RLock()

    def trigger(self):
        with self._lock:
            result = self._camera.GrabOne(self._timeout_ms)
            try:
                if not result.GrabSucceeded():
                    raise RuntimeError(f"Basler grab failed: {result.GetErrorCode()} "
                                       f"{result.GetErrorDescription()}")
                image = self._converter.Convert(result).GetArray().copy()
            finally:
                result.Release()
            if list(image.shape) != self._shape:
                raise RuntimeError(f"Basler image shape changed: {image.shape} != {self._shape}")
            self._image = image
            self._timestamp = time.time()
        return _finished(self)

    def read(self):
        with self._lock:
            if self._image is None:
                raise RuntimeError("Trigger camera before reading an image")
            return {self.name: {"value": self._image, "timestamp": self._timestamp}}

    def describe(self):
        # Bluesky's event-model schema requires NumPy's dtype.str notation,
        # not the human-readable dtype name "uint8".
        return {self.name: {"source": f"Basler:{self._ip_address}", "dtype": "array",
                            "shape": self._shape.copy(), "dtype_numpy": "|u1"}}

    def close(self):
        with self._lock:
            self._camera.Close()


def beam_statistics(image, relative_threshold=0.1):
    """Compute intensity and thresholded intensity-weighted beam moments.

    Pixel coordinates start at (x=0, y=0) in the upper-left corner.
    ``total_intensity`` sums all nonnegative pixels (before thresholding);
    centroid and RMS sizes use only pixels >= relative_threshold * peak.
    Sizes are standard deviations in pixels; ``size_quadrature_px`` is the
    square root of the sum of their squares. Negative values are clipped to
    zero; nonfinite pixels are rejected rather than silently biasing moments.
    """
    pixels = np.asarray(image, dtype=np.float64)
    if pixels.ndim != 2 or pixels.size == 0 or not np.isfinite(pixels).all():
        raise ValueError("Image must be a nonempty, finite 2-D array")
    if not np.isfinite(relative_threshold) or not 0 <= relative_threshold < 1:
        raise ValueError("relative_threshold must be between 0 (inclusive) and 1 (exclusive)")

    pixels = np.maximum(pixels, 0)
    peak = float(pixels.max())
    if peak <= 0:
        raise ValueError("Image has no positive beam signal")
    total = float(pixels.sum())
    threshold = relative_threshold * peak
    weights = np.where(pixels >= threshold, pixels, 0.0)
    selected_total = float(weights.sum())
    if selected_total <= 0:
        raise ValueError("No signal above threshold")

    x_profile = weights.sum(axis=0)
    y_profile = weights.sum(axis=1)
    x = np.arange(pixels.shape[1], dtype=np.float64)
    y = np.arange(pixels.shape[0], dtype=np.float64)
    center_x = float(np.dot(x, x_profile) / selected_total)
    center_y = float(np.dot(y, y_profile) / selected_total)
    size_x = math.sqrt(float(np.dot((x - center_x) ** 2, x_profile) / selected_total))
    size_y = math.sqrt(float(np.dot((y - center_y) ** 2, y_profile) / selected_total))
    return {
        "peak_intensity": peak,
        "total_intensity": total,
        "thresholded_intensity": selected_total,
        "center_x_px": center_x,
        "center_y_px": center_y,
        "size_x_px": size_x,
        "size_y_px": size_y,
        "size_quadrature_px": math.hypot(size_x, size_y),
    }
