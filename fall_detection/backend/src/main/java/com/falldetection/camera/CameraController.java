package com.falldetection.camera;

import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.servlet.mvc.method.annotation.StreamingResponseBody;

import java.util.Map;

@RestController
@RequestMapping("/api/camera")
public class CameraController {
    private final CameraService service;

    public CameraController(CameraService service) {
        this.service = service;
    }

    @GetMapping("/status")
    public Map<String, Object> status() {
        return service.status();
    }

    @PostMapping("/start")
    public Map<String, Object> start() {
        return service.start();
    }

    @PostMapping("/stop")
    public Map<String, Object> stop() {
        return service.stop();
    }

    @GetMapping("/stream-url")
    public Map<String, String> streamUrl() {
        return Map.of("url", "/api/camera/stream");
    }

    @GetMapping("/stream")
    public ResponseEntity<StreamingResponseBody> stream() {
        StreamingResponseBody body = outputStream -> {
            try {
                service.copyStream(outputStream);
            } catch (Exception ignored) {
                // The status endpoint reports the actual camera error. The
                // stream simply ends when Python is stopped.
            }
        };
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.parseMediaType("multipart/x-mixed-replace; boundary=frame"));
        headers.setCacheControl("no-cache, no-store, must-revalidate");
        headers.setPragma("no-cache");
        return ResponseEntity.ok().headers(headers).body(body);
    }
}
