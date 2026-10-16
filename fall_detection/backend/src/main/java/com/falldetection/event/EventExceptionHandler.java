package com.falldetection.event;

import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.ResponseStatus;
import org.springframework.web.bind.annotation.RestControllerAdvice;
import org.springframework.web.server.ResponseStatusException;
import org.springframework.web.multipart.MaxUploadSizeExceededException;

import java.util.Map;

@RestControllerAdvice
public class EventExceptionHandler {
    @ExceptionHandler(MaxUploadSizeExceededException.class)
    @ResponseStatus(HttpStatus.PAYLOAD_TOO_LARGE)
    public Map<String, String> tooLarge() {
        return Map.of("error", "视频不能超过 100 MB");
    }

    @ExceptionHandler(ResponseStatusException.class)
    public ResponseEntity<Map<String, String>> serviceError(ResponseStatusException ex) {
        return ResponseEntity.status(ex.getStatusCode()).body(Map.of("error",
                ex.getReason() == null ? "请求失败" : ex.getReason()));
    }

    @ExceptionHandler(FallEventService.EventNotFoundException.class)
    @ResponseStatus(HttpStatus.NOT_FOUND)
    public Map<String, String> notFound(FallEventService.EventNotFoundException ex) {
        return Map.of("error", ex.getMessage());
    }
}
