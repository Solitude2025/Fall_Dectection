from datetime import datetime, timedelta, timezone

from pipeline.alarm_state import AlarmStateManager


def test_alarm_requires_half_second_of_continuous_signal():
    manager = AlarmStateManager(fps=10, trigger_seconds=0.5, hold_seconds=30)
    now = datetime(2026, 10, 14, tzinfo=timezone.utc)

    states = [manager.update(1, True, 0.8, frame, now) for frame in range(4)]
    assert states == [False, False, False, False]
    assert manager.events == []

    assert manager.update(1, True, 0.85, 4, now) is True
    assert len(manager.events) == 1
    event = manager.events[0]
    assert event.duration_seconds == 0.5
    assert event.status == "active"


def test_gap_breaks_candidate_and_manual_clear_requires_rearm():
    manager = AlarmStateManager(fps=10, trigger_seconds=0.5, hold_seconds=30)
    now = datetime(2026, 10, 14, tzinfo=timezone.utc)

    for frame in range(3):
        assert manager.update(1, True, 0.8, frame, now) is False
    assert manager.update(1, False, 0.0, 3, now) is False
    for frame in range(4, 8):
        assert manager.update(1, True, 0.8, frame, now) is False
    assert manager.events == []

    assert manager.update(1, True, 0.8, 8, now) is True
    event_id = manager.events[0].event_id
    cleared = manager.clear_event(event_id)
    assert cleared["status"] == "cleared"
    assert manager.update(1, True, 0.9, 9, now) is False
    assert manager.update(1, False, 0.0, 10, now) is False


def test_timeout_clears_event_and_emits_update():
    manager = AlarmStateManager(fps=10, trigger_seconds=0.1, hold_seconds=1)
    now = datetime(2026, 10, 14, tzinfo=timezone.utc)
    assert manager.update(1, True, 0.8, 0, now) is True
    assert manager.update(1, True, 0.8, 10, now) is False
    assert manager.events[0].status == "cleared"
    assert manager.events[0].clear_reason == "timeout_30s"


def test_realtime_uses_elapsed_seconds_when_inference_slows_down():
    manager = AlarmStateManager(fps=30, use_wall_clock=True)
    now = datetime(2026, 10, 15, tzinfo=timezone.utc)
    assert not manager.update(1, True, .7, 0, now)
    assert not manager.update(1, True, .7, 1, now + timedelta(seconds=.3))
    assert manager.update(1, True, .7, 2, now + timedelta(seconds=.6))
    manager.tick(3, now + timedelta(seconds=30.59))
    assert manager.events[0].status == 'active'
    manager.tick(4, now + timedelta(seconds=30.6))
    assert manager.events[0].status == 'cleared'

