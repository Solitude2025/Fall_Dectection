package com.falldetection.video;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.falldetection.event.EventExceptionHandler;
import com.sun.net.httpserver.HttpServer;
import org.junit.jupiter.api.Test;
import org.springframework.mock.web.MockMultipartFile;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.MvcResult;
import org.springframework.test.web.servlet.setup.MockMvcBuilders;

import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.*;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;

class VideoControllerTest {
    @Test
    void forwardsFileAndPreservesVideoByteRanges() throws Exception {
        HttpServer python = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        AtomicReference<byte[]> uploaded = new AtomicReference<>();
        AtomicReference<String> requestedRange = new AtomicReference<>();
        String id = "0123456789abcdef0123456789abcdef";
        python.createContext("/api/videos", exchange -> {
            byte[] body;
            int code;
            if (exchange.getRequestURI().getPath().endsWith("/result")) {
                requestedRange.set(exchange.getRequestHeaders().getFirst("Range"));
                body = "2345".getBytes(StandardCharsets.US_ASCII);
                exchange.getResponseHeaders().set("Content-Type", "video/mp4");
                exchange.getResponseHeaders().set("Content-Range", "bytes 2-5/10");
                exchange.getResponseHeaders().set("Accept-Ranges", "bytes");
                code = 206;
            } else {
                uploaded.set(exchange.getRequestBody().readAllBytes());
                body = ("{\"job_id\":\"" + id + "\",\"status\":\"queued\"}").getBytes(StandardCharsets.UTF_8);
                exchange.getResponseHeaders().set("Content-Type", "application/json");
                code = 202;
            }
            exchange.sendResponseHeaders(code, body.length);
            exchange.getResponseBody().write(body);
            exchange.close();
        });
        python.start();
        try {
            MockMvc mvc = MockMvcBuilders.standaloneSetup(new VideoController(
                    "http://127.0.0.1:" + python.getAddress().getPort(), new ObjectMapper())).build();
            byte[] input = new byte[]{0, 1, 2, 3};
            mvc.perform(multipart("/api/videos").file(new MockMultipartFile("file", "sample.mp4", "video/mp4", input)))
                    .andExpect(status().isAccepted()).andExpect(jsonPath("$.job_id").value(id));
            assertArrayEquals(input, uploaded.get());

            MvcResult stream = mvc.perform(get("/api/videos/" + id + "/result").header("Range", "bytes=2-5"))
                    .andExpect(request().asyncStarted()).andReturn();
            mvc.perform(asyncDispatch(stream)).andExpect(status().isPartialContent())
                    .andExpect(header().string("Content-Range", "bytes 2-5/10"))
                    .andExpect(content().contentType("video/mp4"))
                    .andExpect(content().string("2345"));
            assertEquals("bytes=2-5", requestedRange.get());
        } finally {
            python.stop(0);
        }
    }

    @Test
    void rejectsEmptyOrNonVideoUploadsBeforeCallingPython() throws Exception {
        MockMvc mvc = MockMvcBuilders.standaloneSetup(new VideoController("http://127.0.0.1:1", new ObjectMapper()))
                .setControllerAdvice(new EventExceptionHandler()).build();
        mvc.perform(multipart("/api/videos").file(new MockMultipartFile("file", "empty.mp4", "video/mp4", new byte[0])))
                .andExpect(status().isBadRequest()).andExpect(jsonPath("$.error").value("视频文件为空"));
        mvc.perform(multipart("/api/videos").file(new MockMultipartFile("file", "notes.txt", "text/plain", new byte[]{1})))
                .andExpect(status().isBadRequest());
    }
}
