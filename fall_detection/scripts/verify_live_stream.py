"""Measure unique multipart JPEG frames delivered through Spring Boot."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
import requests


def measure(url, seconds):
    received = unique = 0
    previous = None
    first_at = None
    last_at = None
    buffer = b''
    latest = None
    with requests.get(url, stream=True, timeout=(5, 5)) as response:
        response.raise_for_status()
        started = time.perf_counter()
        for chunk in response.iter_content(8192):
            buffer += chunk
            while True:
                start = buffer.find(b'\xff\xd8')
                if start < 0:
                    break
                end = buffer.find(b'\xff\xd9', start + 2)
                if end < 0:
                    break
                jpeg = buffer[start:end + 2]
                buffer = buffer[end + 2:]
                received += 1
                now = time.perf_counter()
                first_at = now if first_at is None else first_at
                last_at = now
                digest = hashlib.sha256(jpeg).digest()
                unique += digest != previous
                previous = digest
                latest = jpeg
            if time.perf_counter() - started >= seconds:
                break
    elapsed = last_at - first_at if received > 1 else 0
    status = requests.get('http://127.0.0.1:8001/api/status', timeout=5).json()
    image = cv2.imdecode(np.frombuffer(latest, np.uint8), cv2.IMREAD_COLOR)
    return {
        'url': url,
        'interval_seconds': round(elapsed, 3),
        'received_frames': received,
        'consecutive_unique_jpegs': unique,
        'delivered_fps': round((received - 1) / elapsed, 2) if elapsed else 0,
        'resolution': [image.shape[1], image.shape[0]],
        'preview_brightness_mean': round(float(image.mean()), 2),
        'status': status,
    }, image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seconds', type=int, default=15)
    parser.add_argument('--url', default='http://127.0.0.1:8080/api/camera/stream')
    parser.add_argument('--output', default='eval_report/realtime_stream_verification.json')
    args = parser.parse_args()
    result, image = measure(args.url, args.seconds)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    cv2.imwrite(str(path.with_suffix('.jpg')), image)
    print(json.dumps(result, ensure_ascii=False, indent=2))
