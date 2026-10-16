package com.falldetection.event;

import com.falldetection.camera.CameraService;
import org.springframework.data.domain.PageRequest;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.Instant;
import java.time.temporal.ChronoUnit;
import java.time.LocalDate;
import java.time.ZoneOffset;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

@Service
public class FallEventService {
    private final FallEventRepository repository;
    private final CameraService cameraService;

    public FallEventService(FallEventRepository repository, CameraService cameraService) {
        this.repository = repository;
        this.cameraService = cameraService;
    }

    @Transactional
    public FallEvent saveFromInference(FallEventPayload payload) {
        FallEvent event = repository.findById(payload.eventId()).orElseGet(() -> new FallEvent(payload));
        event.apply(payload);
        return repository.save(event);
    }

    public List<FallEventPayload> list(String status, int limit) {
        PageRequest page = PageRequest.of(0, Math.min(Math.max(limit, 1), 200));
        List<FallEvent> events = status == null || status.isBlank()
                ? repository.findAllByOrderByCreatedAtDesc(page)
                : repository.findByStatusOrderByCreatedAtDesc(status, page);
        return events.stream().map(FallEventPayload::from).toList();
    }

    public List<FallEventPayload> listAllForStats() {
        return repository.findAll().stream().map(FallEventPayload::from).toList();
    }

    public FallEventPayload get(String eventId) {
        return repository.findById(eventId).map(FallEventPayload::from)
                .orElseThrow(() -> new EventNotFoundException(eventId));
    }

    @Transactional
    public FallEventPayload clear(String eventId, String reason) {
        FallEvent event = repository.findById(eventId)
                .orElseThrow(() -> new EventNotFoundException(eventId));
        cameraService.clearPythonEvent(eventId, reason == null || reason.isBlank() ? "manual" : reason);
        event.clear(reason);
        return FallEventPayload.from(repository.save(event));
    }

    public Map<String, Object> statistics(int days) {
        int safeDays = Math.min(Math.max(days, 1), 90);
        Instant since = Instant.now().minus(safeDays, ChronoUnit.DAYS);
        Map<String, Object> response = new LinkedHashMap<>();
        response.put("days", safeDays);
        response.put("total", repository.countByCreatedAtAfter(since));
        response.put("active", repository.countByStatus("active"));
        response.put("cleared", repository.countByStatus("cleared"));
        List<FallEventPayload> recent = listAllForStats();
        Map<LocalDate, Long> byDate = new TreeMap<>();
        for (FallEventPayload event : recent) {
            if (event.createdAt() == null || event.createdAt().isBlank()) {
                continue;
            }
            try {
                LocalDate date = Instant.parse(event.createdAt()).atZone(ZoneOffset.UTC).toLocalDate();
                if (!date.isBefore(since.atZone(ZoneOffset.UTC).toLocalDate())) {
                    byDate.merge(date, 1L, Long::sum);
                }
            } catch (RuntimeException ignored) {
                // A malformed client timestamp should not break the dashboard.
            }
        }
        List<Map<String, Object>> trend = new ArrayList<>();
        byDate.forEach((date, count) -> trend.add(Map.of("date", date.toString(), "count", count)));
        response.put("trend", trend);
        return response;
    }

    public static class EventNotFoundException extends RuntimeException {
        public EventNotFoundException(String eventId) {
            super("event not found: " + eventId);
        }
    }
}
