package com.falldetection.event;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import jakarta.persistence.Table;

import java.time.Instant;
import java.util.Locale;

@Entity
@Table(name = "fall_events")
public class FallEvent {
    @Id
    @Column(name = "event_id", length = 80, nullable = false)
    private String eventId;

    @Column(name = "camera_id", length = 80, nullable = false)
    private String cameraId;

    @Column(name = "person_id", nullable = false)
    private Integer personId;

    @Column(name = "event_type", length = 40, nullable = false)
    private String eventType;

    @Column(nullable = false)
    private Double confidence;

    @Column(name = "start_time", length = 50, nullable = false)
    private String startTime;

    @Column(name = "alarm_level", length = 20, nullable = false)
    private String alarmLevel;

    @Column(length = 20, nullable = false)
    private String status;

    @Column(name = "start_frame")
    private Integer startFrame;

    @Column(name = "trigger_frame")
    private Integer triggerFrame;

    @Column(name = "end_frame")
    private Integer endFrame;

    @Column(name = "duration_frames")
    private Integer durationFrames;

    @Column(name = "duration_seconds")
    private Double durationSeconds;

    @Column(name = "cleared_at", length = 50)
    private String clearedAt;

    @Column(name = "clear_reason", length = 40)
    private String clearReason;

    @Column(name = "expires_at", length = 50)
    private String expiresAt;

    @Column(name = "created_at", nullable = false)
    private Instant createdAt;

    @Column(name = "updated_at", nullable = false)
    private Instant updatedAt;

    protected FallEvent() {
    }

    public FallEvent(FallEventPayload payload) {
        apply(payload);
        this.createdAt = Instant.now();
        this.updatedAt = Instant.now();
    }

    public void apply(FallEventPayload payload) {
        this.eventId = payload.eventId();
        this.cameraId = payload.cameraId();
        this.personId = payload.personId();
        this.eventType = payload.eventType();
        this.confidence = payload.confidence();
        this.startTime = payload.startTime();
        this.alarmLevel = payload.alarmLevel();
        this.status = payload.status() == null ? "active" : payload.status().toLowerCase(Locale.ROOT);
        this.startFrame = payload.startFrame();
        this.triggerFrame = payload.triggerFrame();
        this.endFrame = payload.endFrame();
        this.durationFrames = payload.durationFrames();
        this.durationSeconds = payload.durationSeconds();
        this.clearedAt = payload.clearedAt();
        this.clearReason = payload.clearReason();
        this.expiresAt = payload.expiresAt();
        this.updatedAt = Instant.now();
    }

    public void clear(String reason) {
        this.status = "cleared";
        this.clearReason = reason;
        this.clearedAt = Instant.now().toString();
        this.updatedAt = Instant.now();
    }

    public String getEventId() { return eventId; }
    public String getCameraId() { return cameraId; }
    public Integer getPersonId() { return personId; }
    public String getEventType() { return eventType; }
    public Double getConfidence() { return confidence; }
    public String getStartTime() { return startTime; }
    public String getAlarmLevel() { return alarmLevel; }
    public String getStatus() { return status; }
    public Integer getStartFrame() { return startFrame; }
    public Integer getTriggerFrame() { return triggerFrame; }
    public Integer getEndFrame() { return endFrame; }
    public Integer getDurationFrames() { return durationFrames; }
    public Double getDurationSeconds() { return durationSeconds; }
    public String getClearedAt() { return clearedAt; }
    public String getClearReason() { return clearReason; }
    public String getExpiresAt() { return expiresAt; }
    public Instant getCreatedAt() { return createdAt; }
    public Instant getUpdatedAt() { return updatedAt; }
}
