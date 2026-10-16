package com.falldetection.camera;

import org.junit.jupiter.api.Test;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.MvcResult;
import org.springframework.test.web.servlet.setup.MockMvcBuilders;

import java.io.OutputStream;
import java.nio.charset.StandardCharsets;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.asyncDispatch;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.content;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.header;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.request;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

class CameraStreamTest {
    @Test
    void declaresMjpegBoundaryBeforeStreamingAndPreservesFrameBytes() throws Exception {
        // The proxy must identify the multipart response before the browser
        // reads any frames. A produces annotation alone did not send it.
        byte[] frame = "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: 4\r\n\r\n\u00ff\u00d8\u00ff\u00d9\r\n"
                .getBytes(StandardCharsets.ISO_8859_1);
        CameraService service = mock(CameraService.class);
        doAnswer(invocation -> {
            invocation.<OutputStream>getArgument(0).write(frame);
            return null;
        }).when(service).copyStream(any(OutputStream.class));
        MockMvc mvc = MockMvcBuilders.standaloneSetup(new CameraController(service)).build();

        MvcResult result = mvc.perform(get("/api/camera/stream"))
                .andExpect(request().asyncStarted())
                .andExpect(header().string("Content-Type", "multipart/x-mixed-replace;boundary=frame"))
                .andReturn();

        mvc.perform(asyncDispatch(result))
                .andExpect(status().isOk())
                .andExpect(content().bytes(frame));
    }
}
