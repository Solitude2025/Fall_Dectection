"""Camera configuration and bounded, same-frame preview hand-off."""

from collections import deque
import math
import threading
import time

import cv2
import numpy as np


def prepare_detection_frame(frame, detector):
    """Shared live/replay preprocessing; normalized poses map to this frame."""
    target_width = min(frame.shape[1], detector.input_size[0])
    scale = target_width / frame.shape[1]
    reduced = cv2.resize(
        frame, (target_width, max(2, round(frame.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
    ) if scale < 1 else frame
    return detector.preprocess_for_lowlight(reduced)


def _set_exposure(cap, config):
    """Apply exposure controls and return the values reported by the driver."""
    # A previous application can leave a persistent manual UVC exposure.
    # Explicitly restore auto mode even when no override was provided.
    auto_exposure = config.get('camera_auto_exposure', 0.75)
    auto_accepted = None
    if auto_exposure is not None:
        auto_accepted = bool(cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, float(auto_exposure)))
    exposure = config.get('camera_exposure')
    manual_exposure = (auto_exposure is not None and float(auto_exposure) <= 0.5)
    if exposure is not None and manual_exposure:
        cap.set(cv2.CAP_PROP_EXPOSURE, float(exposure))
    return {
        'camera_auto_exposure': cap.get(cv2.CAP_PROP_AUTO_EXPOSURE),
        'camera_exposure': cap.get(cv2.CAP_PROP_EXPOSURE),
        # Some DirectShow drivers accept SET but report -1 for GET. Keep the
        # request/acceptance separate from the unsupported readback.
        'camera_exposure_mode_requested': 'manual' if manual_exposure else 'auto' if auto_exposure is not None else 'driver',
        'camera_auto_exposure_accepted': auto_accepted,
    }


def configure_capture(cap, config):
    """Apply format LAST: setting width/height can reset UVC to raw YUY2.

    This camera advertises 720p30 with MJPEG but only 720p8 with YUY2.
    An accepted set(FPS, 30) alone does not confirm the negotiated format.
    """
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.get('camera_width', 1280))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.get('camera_height', 720))
    cap.set(cv2.CAP_PROP_FPS, config.get('camera_fps', 30))
    # Request MJPEG before exposure controls. Some DirectShow drivers reset
    # auto-exposure to a manual default when the format is changed; applying
    # exposure after the final format request avoids that dark fallback.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    # On DirectShow, 0.75 means auto exposure and 0.25 means manual
    # exposure.  Setting a manual exposure after enabling auto exposure can
    # silently put some laptop cameras back into a dark fixed-exposure mode,
    # so only apply the exposure value when the configuration explicitly asks
    # for manual control.
    exposure_values = _set_exposure(cap, config)
    code = int(cap.get(cv2.CAP_PROP_FOURCC))
    return {
        'camera_width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'camera_height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'camera_format': ''.join(chr((code >> (8 * n)) & 255) for n in range(4)).strip('\x00'),
        'requested_fps': config.get('camera_fps', 30),
        **exposure_values,
    }


class RollingFPS:
    """Recent completed frames, excluding model/driver startup time."""
    def __init__(self, window=3.0):
        self.window = window
        self.samples = deque()

    def tick(self, now=None):
        now = time.perf_counter() if now is None else now
        self.samples.append(now)
        while len(self.samples) > 2 and self.samples[0] < now - self.window:
            self.samples.popleft()
        elapsed = self.samples[-1] - self.samples[0]
        return round((len(self.samples) - 1) / elapsed, 2) if elapsed > 0 else 0.0


class LatestPreview:
    """One input frame + its OWN result, dropped together if encoding lags."""
    def __init__(self):
        self._condition = threading.Condition()
        self._item = None
        self._closed = False
        self.dropped = 0

    def put(self, item):
        with self._condition:
            if self._closed:
                return
            if self._item is not None:
                self.dropped += 1
            self._item = item
            self._condition.notify()

    def get(self):
        with self._condition:
            self._condition.wait_for(lambda: self._item is not None or self._closed)
            item, self._item = self._item, None
            return item

    def close(self):
        with self._condition:
            self._closed = True
            self._condition.notify_all()


class PreviewBrightness:
    """Mild, black-preserving display lift after real camera auto exposure.

    A severe underexposure cannot be repaired by raising every dark pixel:
    gamma .4 turns sensor noise/black pedestal into a grey veil. Limit lift
    to gamma .8 and taper it near black; normal/bright frames stay untouched.
    """
    def __init__(self, target=100.0):
        self.target = float(target)
        self.gamma = 1.0
        self._last_gamma = None
        self._table = None
        self.metrics = {}

    def apply(self, frame):
        small = frame[::8, ::8]
        luma = float(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).mean())
        desired = 1.0
        if 4 < luma < self.target:
            desired = max(.8, min(1.0, math.log(self.target / 255) / math.log(luma / 255)))
        if self._last_gamma is None:
            self.gamma = desired
        else:
            self.gamma += .08 * (desired - self.gamma)
        gamma = round(self.gamma, 2)
        # Auto exposure should never be overridden by an old display gain
        # when the scene becomes bright. Resume gently on the next dark frame.
        if luma >= self.target or luma <= 4 or gamma >= .99:
            self.metrics = {'raw_brightness': round(luma, 2), 'preview_brightness': round(luma, 2), 'preview_gamma': 1.0}
            return frame
        if gamma != self._last_gamma:
            values = np.arange(256, dtype=np.float32)
            lifted = (values / 255.0) ** gamma * 255
            # Keep black at zero and retain shadow contrast/noise floor.
            toe = np.minimum(1.0, values / 32.0)
            self._table = np.clip(values + (lifted - values) * toe, 0, 255).astype(np.uint8)
            self._last_gamma = gamma
        output = cv2.LUT(frame, self._table)
        self.metrics = {
            'raw_brightness': round(luma, 2),
            'preview_brightness': round(float(cv2.cvtColor(output[::8, ::8], cv2.COLOR_BGR2GRAY).mean()), 2),
            'preview_gamma': gamma,
        }
        return output
