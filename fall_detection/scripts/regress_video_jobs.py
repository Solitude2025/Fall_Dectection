"""Run the production upload worker on every local regression video.

Reports and replays are isolated from the user's upload history by default.
The expected label is used only to report a mismatch, never for inference.
"""
import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import torch
from services.inference_service import InferenceService
from services.video_analysis import VideoAnalysisJobs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--negative', action='store_true')
    parser.add_argument('--config', default='configs/config.yaml')
    parser.add_argument('--only', nargs='*')
    args = parser.parse_args()
    cv2.setNumThreads(2)
    torch.set_num_threads(2)
    service = InferenceService(config_path=args.config)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config_path = Path(args.config).resolve()
    manifest = {'config': str(config_path),
                'config_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest(),
                'source': str(Path(args.source).resolve()),
                'expected_fall': not args.negative,
                'label_used_for_inference': False}
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf8')
    jobs = VideoAnalysisJobs(service, output / 'jobs')
    rows = []
    paths = sorted(p for p in Path(args.source).iterdir() if p.suffix.lower() == '.mp4')
    if args.only:
        paths = [p for p in paths if p.name in args.only]
    for path in paths:
        print('START', path.name, flush=True)
        job = jobs.reserve(path.name)
        shutil.copyfile(path, jobs.storage / job['job_id'] / 'input.video')
        started = time.perf_counter()
        jobs._run(job['job_id'])
        report = jobs.get(job['job_id'])
        row = {key: report.get(key) for key in (
            'filename', 'job_id', 'status', 'analysis_version', 'event_count',
            'events', 'quality', 'warnings', 'error', 'processing_seconds')}
        row['expected_fall'] = not args.negative
        row['passed'] = (report['status'] == 'completed' and
                         bool(report.get('event_count')) == (not args.negative))
        row['wall_seconds'] = round(time.perf_counter() - started, 2)
        rows.append(row)
        (output / 'summary.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(row, ensure_ascii=False), flush=True)
    print('COMPLETE', sum(row['passed'] for row in rows), '/', len(rows), flush=True)
    if not rows or not all(row['passed'] for row in rows):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
