package com.vijay.faceswap.service;

import com.google.protobuf.ByteString;
import com.vijay.faceswap.grpc.SwapRequest;
import com.vijay.faceswap.grpc.SwapResponse;
import com.vijay.faceswap.grpc.FaceSwapServiceGrpc;
import net.devh.boot.grpc.client.inject.GrpcClient;
import org.springframework.stereotype.Service;

@Service
public class FaceSwapClientService {

    // This annotation automatically wires this stub to the properties we just set
    @GrpcClient("faceswap-python-service")
    private FaceSwapServiceGrpc.FaceSwapServiceBlockingStub faceSwapStub;

    public byte[] swapFaces(byte[] sourceImageBytes, byte[] targetImageBytes) {

        System.out.println("Sending image bytes to Python GPU Server...");

        // Build the protobuf request
        SwapRequest request = SwapRequest.newBuilder()
                .setSourceImage(ByteString.copyFrom(sourceImageBytes))
                .setTargetImage(ByteString.copyFrom(targetImageBytes))
                .build();

        // Make the gRPC call (this blocks until Python returns the result)
        SwapResponse response = faceSwapStub.swapFace(request);

        if (!response.getSuccess()) {
            throw new RuntimeException("AI processing failed: " + response.getMessage());
        }

        System.out.println("Received swapped image from GPU!");

        // Return the raw swapped bytes
        return response.getSwappedImage().toByteArray();
    }
}
