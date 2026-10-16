package com.falldetection.video;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.server.ResponseStatusException;
import org.springframework.web.servlet.mvc.method.annotation.StreamingResponseBody;

import java.io.InputStream;
import java.io.UncheckedIOException;
import java.net.URI;
import java.net.URLEncoder;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.Locale;
import java.util.Set;

@RestController
@RequestMapping("/api/videos")
public class VideoController {
    static final long MAX_BYTES = 100L * 1024 * 1024;
    private final HttpClient client = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(5)).build();
    private final ObjectMapper mapper;
    private final String inferenceUrl;

    public VideoController(@Value("${fall-detection.inference-url:http://127.0.0.1:8001}") String url,
                           ObjectMapper mapper) {
        this.inferenceUrl = url.replaceAll("/$", "");
        this.mapper = mapper;
    }

    @PostMapping
    public ResponseEntity<JsonNode> upload(@RequestParam("file") MultipartFile file) {
        String name = file.getOriginalFilename() == null ? "" : file.getOriginalFilename();
        int dot = name.lastIndexOf('.');
        String suffix = dot < 0 ? "" : name.substring(dot).toLowerCase(Locale.ROOT);
        if (file.isEmpty()) throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "视频文件为空");
        if (file.getSize() > MAX_BYTES) throw new ResponseStatusException(HttpStatus.PAYLOAD_TOO_LARGE, "视频不能超过 100 MB");
        if (!Set.of(".mp4", ".avi", ".mov", ".mkv", ".webm").contains(suffix)) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "请选择支持的视频格式");
        }
        HttpRequest request = HttpRequest.newBuilder(URI.create(inferenceUrl + "/api/videos?filename="
                        + URLEncoder.encode(name, StandardCharsets.UTF_8)))
                .timeout(Duration.ofMinutes(3)).header("Content-Type", "application/octet-stream")
                .POST(HttpRequest.BodyPublishers.ofInputStream(() -> {
                    try { return file.getInputStream(); }
                    catch (java.io.IOException ex) { throw new UncheckedIOException(ex); }
                })).build();
        return ResponseEntity.accepted().body(json(request));
    }

    @GetMapping
    public JsonNode list() {
        return json(HttpRequest.newBuilder(URI.create(inferenceUrl + "/api/videos"))
                .timeout(Duration.ofSeconds(10)).GET().build());
    }

    @DeleteMapping
    public JsonNode deleteAll() {
        return json(HttpRequest.newBuilder(URI.create(inferenceUrl + "/api/videos"))
                .timeout(Duration.ofSeconds(20)).DELETE().build());
    }

    @DeleteMapping("/{jobId}")
    public JsonNode delete(@PathVariable String jobId) {
        validateId(jobId);
        return json(HttpRequest.newBuilder(URI.create(inferenceUrl + "/api/videos/" + jobId))
                .timeout(Duration.ofSeconds(20)).DELETE().build());
    }

    @GetMapping("/{jobId}")
    public JsonNode status(@PathVariable String jobId) {
        validateId(jobId);
        return json(HttpRequest.newBuilder(URI.create(inferenceUrl + "/api/videos/" + jobId))
                .timeout(Duration.ofSeconds(10)).GET().build());
    }

    @GetMapping("/{jobId}/result")
    public ResponseEntity<StreamingResponseBody> result(@PathVariable String jobId,
            @RequestHeader(value = "Range", required = false) String range) {
        validateId(jobId);
        try {
            HttpRequest.Builder request = HttpRequest.newBuilder(URI.create(inferenceUrl + "/api/videos/" + jobId + "/result"))
                    .timeout(Duration.ofSeconds(30)).GET();
            if (range != null) request.header("Range", range);
            HttpResponse<InputStream> response = client.send(request.build(), HttpResponse.BodyHandlers.ofInputStream());
            if (response.statusCode() >= 400) {
                try (InputStream input = response.body()) {
                    throw new ResponseStatusException(org.springframework.http.HttpStatusCode.valueOf(response.statusCode()),
                            new String(input.readAllBytes(), StandardCharsets.UTF_8));
                }
            }
            HttpHeaders headers = new HttpHeaders();
            for (String name : new String[]{"Content-Type", "Content-Length", "Content-Range", "Accept-Ranges"}) {
                response.headers().firstValue(name).ifPresent(value -> headers.set(name, value));
            }
            headers.setCacheControl("no-store");
            StreamingResponseBody body = output -> {
                try (InputStream input = response.body()) { input.transferTo(output); output.flush(); }
            };
            return ResponseEntity.status(response.statusCode()).headers(headers).body(body);
        } catch (ResponseStatusException ex) { throw ex; }
        catch (Exception ex) { throw unavailable(ex); }
    }

    private JsonNode json(HttpRequest request) {
        try {
            HttpResponse<String> response = client.send(request, HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8));
            JsonNode payload = mapper.readTree(response.body());
            if (response.statusCode() >= 400) {
                throw new ResponseStatusException(org.springframework.http.HttpStatusCode.valueOf(response.statusCode()),
                        payload.path("detail").asText("视频服务请求失败"));
            }
            return payload;
        } catch (ResponseStatusException ex) { throw ex; }
        catch (Exception ex) { throw unavailable(ex); }
    }

    private ResponseStatusException unavailable(Exception ex) {
        if (ex instanceof InterruptedException) Thread.currentThread().interrupt();
        return new ResponseStatusException(HttpStatus.SERVICE_UNAVAILABLE, "Python 推理服务未连接，请先启动服务", ex);
    }

    private void validateId(String id) {
        if (!id.matches("[a-f0-9]{32}")) throw new ResponseStatusException(HttpStatus.NOT_FOUND, "任务不存在");
    }

}
