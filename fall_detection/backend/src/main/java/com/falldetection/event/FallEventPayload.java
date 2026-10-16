package com.falldetection.event;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;

/** JSON contract shared by the Python inference service and Spring Boot. */
@JsonIgnoreProperties(ignoreUnknown = true)
public record FallEventPayload(
        @JsonProperty("event_id") @NotBlank String eventId,
        @JsonProperty("camera_id") @NotBlank String cameraId,
        @JsonProperty("person_id") @NotNull Integer personId,
        @JsonProperty("event_type") @NotBlank String eventType,
        @JsonProperty("confidence") @NotNull Double confidence,
        @JsonProperty("start_time") @NotBlank String startTime,
        @JsonProperty("alarm_level") @NotBlank String alarmLevel,
        @JsonProperty("status") @NotBlank String status,
        @JsonProperty("start_frame") Integer startFrame,
        @JsonProperty("trigger_frame") Integer triggerFrame,
        @JsonProperty("end_frame") Integer endFrame,
        @JsonProperty("duration_frames") Integer durationFrames,
        @JsonProperty("duration_seconds") Double durationSeconds,
        @JsonProperty("cleared_at") String clearedAt,
        @JsonProperty("clear_reason") String clearReason,
        @JsonProperty("expires_at") String expiresAt,
        @JsonProperty("created_at") String createdAt
) {
    public static FallEventPayload from(FallEvent event) {
        return new FallEventPayload(
                event.getEventId(), event.getCameraId(), event.getPersonId(),
                event.getEventType(), event.getConfidence(), event.getStartTime(),
                event.getAlarmLevel(), event.getStatus(), event.getStartFrame(),
                event.getTriggerFrame(), event.getEndFrame(), event.getDurationFrames(),
                event.getDurationSeconds(), event.getClearedAt(), event.getClearReason(),
                event.getExpiresAt(), event.getCreatedAt() == null ? null : event.getCreatedAt().toString()
        );
    }
}
