package com.vijay.faceswap.controller;

import com.vijay.faceswap.service.FaceSwapClientService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;

@CrossOrigin(origins = "*")
@RestController
@RequestMapping("/api/v1")
public class FaceSwapController {

    @Autowired
    private FaceSwapClientService faceSwapService;

    @PostMapping(value = "/swap", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public ResponseEntity<byte[]> processFaceSwap(
            @RequestParam("source") MultipartFile sourceFile,
            @RequestParam("target") MultipartFile targetFile) {

        try {
            // Pass the uploaded files to our gRPC bridge
            byte[] resultImage = faceSwapService.swapFaces(sourceFile.getBytes(), targetFile.getBytes());

            // Set the headers so the browser knows it's receiving an image back
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.IMAGE_JPEG);

            return new ResponseEntity<>(resultImage, headers, HttpStatus.OK);

        } catch (IOException e) {
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).build();
        } catch (RuntimeException e) {
            return ResponseEntity.status(HttpStatus.BAD_REQUEST).body(e.getMessage().getBytes());
        }
    }
}
