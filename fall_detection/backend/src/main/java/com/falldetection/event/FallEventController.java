package com.falldetection.event;

import jakarta.validation.Valid;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api")
public class FallEventController {
    private final FallEventService service;

    public FallEventController(FallEventService service) {
        this.service = service;
    }

    @GetMapping("/health")
    public Map<String, String> health() {
        return Map.of("status", "UP", "service", "fall-detection-backend");
    }

    /** Called by the Python camera process when an event is created or cleared. */
    @PostMapping("/internal/events")
    public FallEventPayload ingest(@Valid @RequestBody FallEventPayload payload) {
        return FallEventPayload.from(service.saveFromInference(payload));
    }

    @GetMapping("/events")
    public List<FallEventPayload> list(
            @RequestParam(required = false) String status,
            @RequestParam(defaultValue = "50") int limit) {
        return service.list(status, limit);
    }

    @GetMapping("/events/{eventId}")
    public FallEventPayload get(@PathVariable String eventId) {
        return service.get(eventId);
    }

    @PostMapping("/events/{eventId}/clear")
    public FallEventPayload clear(
            @PathVariable String eventId,
            @RequestParam(defaultValue = "manual") String reason) {
        return service.clear(eventId, reason);
    }

    @GetMapping("/events/stats")
    public Map<String, Object> stats(@RequestParam(defaultValue = "7") int days) {
        return service.statistics(days);
    }
}
