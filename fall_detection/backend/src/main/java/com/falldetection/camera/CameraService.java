package com.falldetection.camera;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestClientException;
import org.springframework.web.client.RestTemplate;

import java.util.LinkedHashMap;
import java.util.Map;
import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;
import java.io.InputStream;
import java.io.IOException;
import java.net.URL;

/** Passes camera lifecycle commands to the local Python inference process. */
@Service
public class CameraService {
    private final RestTemplate restTemplate = new RestTemplate();
    private final String inferenceUrl;

    public CameraService(@Value("${fall-detection.inference-url:http://127.0.0.1:8001}") String inferenceUrl) {
        this.inferenceUrl = inferenceUrl.replaceAll("/$", "");
    }

    public Map<String, Object> status() {
        try {
            ResponseEntity<Map> response = restTemplate.getForEntity(inferenceUrl + "/api/status", Map.class);
            return response.getBody() == null ? Map.of("running", false) : response.getBody();
        } catch (RestClientException ex) {
            Map<String, Object> error = new LinkedHashMap<>();
            error.put("running", false);
            error.put("available", false);
            error.put("error", ex.getMessage() == null ? "inference service unavailable" : ex.getMessage());
            return error;
        }
    }

    public Map<String, Object> start() {
        return postStatus("/api/camera/start");
    }

    public Map<String, Object> stop() {
        return postStatus("/api/camera/stop");
    }

    public String streamUrl() {
        return inferenceUrl + "/api/stream.mjpeg";
    }

    public void copyStream(java.io.OutputStream outputStream) throws IOException {
        URL url = new URL(streamUrl());
        try (InputStream inputStream = url.openStream()) {
            inputStream.transferTo(outputStream);
        }
    }

    public void clearPythonEvent(String eventId, String reason) {
        try {
            restTemplate.postForEntity(
                    inferenceUrl + "/api/internal/events/" + URLEncoder.encode(eventId, StandardCharsets.UTF_8)
                            + "/clear?reason=" + URLEncoder.encode(reason == null ? "manual" : reason, StandardCharsets.UTF_8),
                    null, Map.class);
        } catch (RestClientException ignored) {
            // The database event is still cleared if the Python process has
            // already stopped. The next start begins with a clean state.
        }
    }

    private Map<String, Object> postStatus(String path) {
        try {
            ResponseEntity<Map> response = restTemplate.postForEntity(inferenceUrl + path, null, Map.class);
            return response.getBody() == null ? Map.of("running", false) : response.getBody();
        } catch (RestClientException ex) {
            Map<String, Object> error = new LinkedHashMap<>();
            error.put("running", false);
            error.put("available", false);
            error.put("error", ex.getMessage());
            return error;
        }
    }
}
