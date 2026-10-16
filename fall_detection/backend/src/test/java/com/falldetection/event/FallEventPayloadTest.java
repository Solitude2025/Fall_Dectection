package com.falldetection.event;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class FallEventPayloadTest {
    @Test
    void jsonContractUsesSnakeCaseFields() {
        FallEventPayload payload = new FallEventPayload(
                "evt-1", "laptop_camera", 0, "fall_suspected", 0.91,
                "2026-10-14T10:20:11Z", "high", "active", 1, 15, 15,
                15, 0.5, null, null, "2026-10-14T10:20:41Z", null
        );
        assertEquals("evt-1", payload.eventId());
        assertEquals("laptop_camera", payload.cameraId());
        assertEquals("active", payload.status());
    }
}
