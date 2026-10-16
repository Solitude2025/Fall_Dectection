"""Application-level alarm state for the fall detection pipeline.

The model produces a per-frame *candidate* signal.  This module turns that
signal into the product behaviour used by the web application:

* a candidate must remain true for at least ``trigger_seconds`` before an
  alarm event is created;
* an active alarm stays active until a person clears it or ``hold_seconds``
  elapse;
* clearing an alarm while the model is still signalling a fall does not
  immediately create duplicate events.  The model must first return to a
  non-fall state.

Keeping this state machine outside the neural model makes the timing policy
easy to test and change without retraining the model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import uuid
from typing import Dict, List, Optional


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: Optional[datetime]) -> Optional[str]:
    return value.isoformat().replace("+00:00", "Z") if value else None


@dataclass
class AlarmEventRecord:
    """A persistent, application-facing alarm event.

    The first eight fields are the stable JSON contract shared with the
    Spring Boot service.  Frame fields are retained for model diagnostics and
    offline evaluation, while the web UI can safely ignore them.
    """

    event_id: str
    camera_id: str
    person_id: int
    event_type: str
    confidence: float
    start_time: str
    alarm_level: str
    status: str
    start_frame: int
    trigger_frame: int
    fps: float
    end_frame: Optional[int] = None
    duration_frames: int = 0
    cleared_at: Optional[str] = None
    clear_reason: Optional[str] = None
    expires_at: Optional[str] = None
    created_at: str = field(default_factory=lambda: _iso(_utc_now()) or "")

    @property
    def max_probability(self) -> float:
        """Backward-compatible name used by the existing evaluation script."""

        return self.confidence

    @property
    def duration_seconds(self) -> float:
        return self.duration_frames / max(self.fps, 1e-6)

    @property
    def timestamp(self) -> float:
        """Start time in seconds for legacy offline evaluation code."""

        return self.start_frame / max(self.fps, 1e-6)

    def update_duration(self, frame_idx: int) -> None:
        self.end_frame = frame_idx
        self.duration_frames = max(1, frame_idx - self.start_frame + 1)

    def to_dict(self) -> dict:
        """Return the snake_case JSON contract used by Python and Java."""

        return {
            "event_id": self.event_id,
            "camera_id": self.camera_id,
            "person_id": self.person_id,
            "event_type": self.event_type,
            "confidence": round(float(self.confidence), 6),
            "start_time": self.start_time,
            "alarm_level": self.alarm_level,
            "status": self.status,
            "start_frame": self.start_frame,
            "trigger_frame": self.trigger_frame,
            "end_frame": self.end_frame,
            "duration_frames": self.duration_frames,
            "duration_seconds": round(self.duration_seconds, 3),
            "cleared_at": self.cleared_at,
            "clear_reason": self.clear_reason,
            "expires_at": self.expires_at,
            "created_at": self.created_at,
            "timestamp": round(self.timestamp, 3),
            # Kept for compatibility with older output files.
            "max_probability": round(float(self.confidence), 6),
        }


@dataclass
class _TrackAlarmState:
    candidate_start_frame: Optional[int] = None
    candidate_started_at: Optional[datetime] = None
    candidate_frames: int = 0
    active_event_id: Optional[str] = None
    rearm_required: bool = False


class AlarmStateManager:
    """Convert frame-level fall signals into debounced alarm events."""

    def __init__(
        self,
        camera_id: str = "laptop_camera",
        fps: float = 30.0,
        trigger_seconds: float = 0.5,
        hold_seconds: float = 30.0,
        use_wall_clock: bool = False,
    ):
        self.camera_id = camera_id
        self.fps = max(float(fps), 1.0)
        self.trigger_seconds = max(float(trigger_seconds), 0.0)
        self.hold_seconds = max(float(hold_seconds), 0.0)
        self.use_wall_clock = use_wall_clock
        self.trigger_frames = max(1, int(round(self.fps * self.trigger_seconds)))
        self.hold_frames = max(1, int(round(self.fps * self.hold_seconds)))
        self._tracks: Dict[int, _TrackAlarmState] = {}
        self._events: List[AlarmEventRecord] = []
        self._event_by_id: Dict[str, AlarmEventRecord] = {}
        self._pending_updates: List[dict] = []

    def set_fps(self, fps: float) -> None:
        """Update timing for a newly opened video or camera stream."""

        self.fps = max(float(fps), 1.0)
        self.trigger_frames = max(1, int(round(self.fps * self.trigger_seconds)))
        self.hold_frames = max(1, int(round(self.fps * self.hold_seconds)))

    def reset(self) -> None:
        self._tracks.clear()
        self._events.clear()
        self._event_by_id.clear()
        self._pending_updates.clear()

    @property
    def events(self) -> List[AlarmEventRecord]:
        return self._events

    @property
    def active_track_ids(self) -> List[int]:
        return [
            track_id
            for track_id, state in self._tracks.items()
            if state.active_event_id is not None
            and self._event_by_id.get(state.active_event_id, None) is not None
            and self._event_by_id[state.active_event_id].status == "active"
        ]

    @property
    def track_ids(self) -> List[int]:
        return list(self._tracks.keys())

    def mark_missing(self, track_id: int) -> None:
        """Break a pending continuous candidate when a track has no frame.

        An active alarm is intentionally left alone: the product rule keeps
        an already-triggered alarm visible through short detector gaps.
        """

        state = self._tracks.get(track_id)
        if state is None or state.active_event_id:
            return
        state.candidate_start_frame = None
        state.candidate_started_at = None
        state.candidate_frames = 0

    def update(
        self,
        track_id: int,
        fall_signal: bool,
        confidence: float,
        frame_idx: int,
        now: Optional[datetime] = None,
    ) -> bool:
        """Advance one person's state and return whether the alarm is active."""

        now = now or _utc_now()
        state = self._tracks.setdefault(track_id, _TrackAlarmState())

        # Existing alarms are deliberately independent from the model signal.
        # They remain visible until manual or timed clearing.
        if state.active_event_id:
            event = self._event_by_id.get(state.active_event_id)
            if event and event.status == "active":
                event.update_duration(frame_idx)
                event.confidence = max(event.confidence, float(confidence))
                if self._expired(event, frame_idx, now):
                    self._clear(event, frame_idx, now, "timeout_30s")
                    state.active_event_id = None
                    state.rearm_required = bool(fall_signal)
                    return False
                return True
            state.active_event_id = None

        # After a manual/automatic clear, wait for a clean frame before
        # allowing a new event.  This avoids event spam while somebody remains
        # on the floor.
        if not fall_signal:
            state.candidate_start_frame = None
            state.candidate_started_at = None
            state.candidate_frames = 0
            state.rearm_required = False
            return False
        if state.rearm_required:
            return False

        if state.candidate_start_frame is None:
            state.candidate_start_frame = frame_idx
            state.candidate_started_at = now
            state.candidate_frames = 0
        state.candidate_frames = frame_idx - state.candidate_start_frame + 1

        elapsed = (now - state.candidate_started_at).total_seconds()
        qualified = elapsed >= self.trigger_seconds if self.use_wall_clock else state.candidate_frames >= self.trigger_frames
        if not qualified:
            return False

        started_at = state.candidate_started_at or now
        # ``start_time`` describes the beginning of the qualifying behaviour,
        # while ``created_at`` is the moment the alarm event is emitted.
        event_id = "evt_{}_{}".format(
            now.strftime("%Y%m%dT%H%M%S%fZ"), uuid.uuid4().hex[:8]
        )
        event = AlarmEventRecord(
            event_id=event_id,
            camera_id=self.camera_id,
            person_id=int(track_id),
            event_type="fall_suspected",
            confidence=float(confidence),
            start_time=_iso(started_at) or "",
            alarm_level="high",
            status="active",
            start_frame=state.candidate_start_frame,
            trigger_frame=frame_idx,
            fps=self.fps,
            created_at=_iso(now) or "",
            expires_at=_iso(now + timedelta(seconds=self.hold_seconds)),
        )
        event.update_duration(frame_idx)
        self._events.append(event)
        self._event_by_id[event_id] = event
        self._pending_updates.append(event.to_dict())
        state.active_event_id = event_id
        state.candidate_start_frame = None
        state.candidate_started_at = None
        state.candidate_frames = 0
        return True

    def tick(self, frame_idx: int, now: Optional[datetime] = None) -> None:
        """Advance timeout handling for tracks missing from the current frame."""

        now = now or _utc_now()
        for state in self._tracks.values():
            if not state.active_event_id:
                continue
            event = self._event_by_id.get(state.active_event_id)
            if event is None or event.status != "active":
                continue
            event.update_duration(frame_idx)
            if self._expired(event, frame_idx, now):
                self._clear(event, frame_idx, now, "timeout_30s")
                state.active_event_id = None
                state.rearm_required = True

    def _expired(self, event: AlarmEventRecord, frame_idx: int, now: datetime) -> bool:
        if self.use_wall_clock:
            return now >= datetime.fromisoformat(event.expires_at.replace("Z", "+00:00"))
        return frame_idx - event.trigger_frame >= self.hold_frames

    def clear_event(
        self,
        event_id: str,
        reason: str = "manual",
        frame_idx: Optional[int] = None,
    ) -> Optional[dict]:
        """Clear an active event and return its updated JSON record."""

        event = self._event_by_id.get(event_id)
        if event is None:
            return None
        if event.status != "active":
            return event.to_dict()
        track_id = event.person_id
        state = self._tracks.get(track_id)
        if frame_idx is not None:
            event.update_duration(frame_idx)
        self._clear(event, frame_idx, _utc_now(), reason)
        if state is not None:
            state.active_event_id = None
            state.rearm_required = True
            state.candidate_start_frame = None
            state.candidate_started_at = None
            state.candidate_frames = 0
        return event.to_dict()

    def _clear(
        self,
        event: AlarmEventRecord,
        frame_idx: Optional[int],
        now: datetime,
        reason: str,
    ) -> None:
        if frame_idx is not None:
            event.update_duration(frame_idx)
        event.status = "cleared"
        event.cleared_at = _iso(now)
        event.clear_reason = reason
        self._pending_updates.append(event.to_dict())

    def consume_updates(self) -> List[dict]:
        """Return event create/update messages once, then clear the queue."""

        updates = list(self._pending_updates)
        self._pending_updates.clear()
        return updates

