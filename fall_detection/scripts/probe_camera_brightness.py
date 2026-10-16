"""Compare real UVC exposure modes; run with live detection stopped."""
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from services.realtime_camera import configure_capture, PreviewBrightness


def main():
    output = Path('app_data/diagnostics/camera_brightness')
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, auto, exposure in [('manual_fast', .25, -5), ('auto_fraction', .75, None),
                                  ('auto_native', 1, None), ('manual_balanced', .25, -3)]:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        try:
            if not cap.isOpened():
                raise RuntimeError('Stop live camera detection before probing')
            config = dict(camera_auto_exposure=auto, camera_exposure=exposure)
            mode = configure_capture(cap, config)
            started = time.perf_counter()
            while time.perf_counter() - started < 3:
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError('Camera read failed')
            frames = 0
            levels = []
            started = time.perf_counter()
            while time.perf_counter() - started < 2:
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError('Camera read failed')
                frames += 1
                levels.append(float(cv2.cvtColor(frame[::8, ::8], cv2.COLOR_BGR2GRAY).mean()))
            fps = frames / (time.perf_counter() - started)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            row = dict(name=name, **mode, fps=round(fps, 2),
                       raw_luma=round(float(np.mean(levels)), 2),
                       p05_p50_p95=np.percentile(gray, [5, 50, 95]).round(2).tolist(),
                       gain=cap.get(cv2.CAP_PROP_GAIN), brightness=cap.get(cv2.CAP_PROP_BRIGHTNESS))
            shown = PreviewBrightness().apply(frame)
            cv2.imwrite(str(output / f'{name}.jpg'), np.hstack((frame, shown)))
            rows.append(row)
            print(json.dumps(row), flush=True)
        finally:
            cap.release()
    (output / 'probe.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
