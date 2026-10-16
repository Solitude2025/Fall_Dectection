"""Compare live decision policies on the same newly inferred observations.

Expectations only score the report. They never reach model/decision code.
Camera frames use exactly the live resize/enhancement; files use source time.
"""
import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import torch
from pipeline.inference import FallDetectionPipeline
from services.inference_service import InferenceService
from services.realtime_camera import prepare_detection_frame


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True,nargs='+')
    parser.add_argument('--output',required=True)
    parser.add_argument('--negative',action='store_true')
    parser.add_argument('--sample-fps',type=float,default=30)
    args=parser.parse_args()
    cv2.setNumThreads(2)
    torch.set_num_threads(2)
    fixed=InferenceService()._build_pipeline(realtime=True)
    baseline=FallDetectionPipeline(
        detector=fixed.detector,fall_detector=fixed.fall_detector,
        sequence_length=fixed.sequence_length,detection_interval=1,auto_lowlight=False,
        min_kp_completeness=fixed.min_kp_completeness,
        rule_config=fixed.rule_config,require_observed_transition=True,cpu_threads=2)
    paths=[]
    for value in args.source:
        source=Path(value)
        paths.extend(sorted(source.rglob('*.mp4')) if source.is_dir() else [source])
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    summary=[]
    for number,path in enumerate(paths):
        print('START',path.name,flush=True)
        for pipeline in (fixed,baseline):
            pipeline.reset()
            pipeline.alarm_state.use_wall_clock=True
        cap=cv2.VideoCapture(str(path))
        fps=cap.get(cv2.CAP_PROP_FPS) or 30
        for pipeline in (fixed,baseline):
            pipeline.fps=fps
            pipeline.alarm_state.set_fps(fps)
        base=datetime(2026,10,16,tzinfo=timezone.utc)
        index=0
        last_sample=-1e3
        rows=[]
        timings=[]
        started=time.perf_counter()
        while True:
            ok,frame=cap.read()
            if not ok: break
            seconds=index/fps
            if seconds-last_sample >= 1/args.sample_fps-1e-6:
                t0=time.perf_counter()
                prepared=prepare_detection_frame(frame,fixed.detector)
                now=base+timedelta(seconds=seconds)
                result=fixed.process_frame(prepared,index,timestamp=now)
                timings.append((time.perf_counter()-t0)*1000)
                old=baseline.process_frame(prepared,index,timestamp=now,detections_override=result.raw_detections)
                rows.append(dict(frame=index,seconds=round(seconds,4),persons=len(result.persons),
                                 fixed_candidate=result.model_alarms,fixed_alarm=any(result.alarms.values()),
                                 baseline_alarm=any(old.alarms.values()),details=result.decision_details))
                last_sample=seconds
            index+=1
        cap.release()
        row=dict(filename=path.name,path=str(path.resolve()),source_frames=index,observations=len(rows),
                 expected_fall=not args.negative,baseline_event_count=len(baseline._fall_events),
                 event_count=len(fixed._fall_events),passed=bool(fixed._fall_events)==(not args.negative),
                 alarm_times=[round(e.trigger_frame/fps,3) for e in fixed._fall_events],
                 baseline_alarm_times=[round(e.trigger_frame/fps,3) for e in baseline._fall_events],
                 mean_processing_ms=round(float(np.mean(timings[8:])),2) if len(timings)>8 else None,
                 wall_seconds=round(time.perf_counter()-started,2))
        summary.append(row)
        (out/f'{number:03}_timeline.json').write_text(json.dumps(rows,ensure_ascii=False),encoding='utf8')
        (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
        print(json.dumps(row,ensure_ascii=False),flush=True)
    print('COMPLETE',sum(r['passed'] for r in summary),'/',len(summary),flush=True)
    if not summary or not all(r['passed'] for r in summary):
        raise SystemExit(1)


if __name__=='__main__': main()
