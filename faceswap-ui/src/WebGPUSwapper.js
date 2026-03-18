import * as ort from 'onnxruntime-web/webgpu';

let session = null;

export const isWebGPULoaded = () => session !== null;

export const initWebGPU = async (onProgress) => {
    if (session) return;
    try {
        console.log("Fetching model into RAM...");
        const response = await fetch('/hyperswap_1c_256.onnx');
        
        if (!response.ok) throw new Error("Failed to fetch model.");
        
        const contentLength = response.headers.get('content-length');
        const total = contentLength ? parseInt(contentLength, 10) : 0;
        let loaded = 0;
        
        if (onProgress && total) {
            onProgress(0, total);
        }
        
        const reader = response.body.getReader();
        const chunks = [];
        
        while(true) {
            const {done, value} = await reader.read();
            if (done) break;
            chunks.push(value);
            loaded += value.length;
            if (onProgress && total) {
                onProgress(loaded, total);
            }
        }
        
        console.log("Model downloaded to RAM, stitching chunks...");
        const modelBuffer = new Uint8Array(loaded);
        let position = 0;
        for(let chunk of chunks) {
            modelBuffer.set(chunk, position);
            position += chunk.length;
        }

        console.log("Initializing ONNX Inference Session...");
        ort.env.wasm.numThreads = Math.max(1, navigator.hardwareConcurrency - 1 || 1);
        
        // Critical: Added 'wasm' (WebAssembly CPU) fallback.
        // If a device blocks WebGPU (like strict MacOS Safari settings or insecure LANs), it will automatically fallback to CPU!
        session = await ort.InferenceSession.create(modelBuffer.buffer, { executionProviders: ['webgpu', 'wasm'] });
        console.log("Session initialized successfully!");
    } catch (e) {
        console.error("Initialization failed:", e);
        throw new Error("Failed to initialize AI session.");
    }
};

export const runWebGPUSwap = async (cropBase64, embeddingNormArray) => {
    if (!session) {
        throw new Error("Model is not loaded into RAM yet.");
    }

    // 1. Convert base64 cropped face image into uncompressed Image
    const img = new Image();
    img.src = `data:image/png;base64,${cropBase64}`;
    await new Promise((resolve, reject) => { 
        img.onload = resolve; 
        img.onerror = reject;
    });

    // 2. Extact raw pixel data to shape [256, 256, 4] using HTML canvas
    const canvas = document.createElement('canvas');
    canvas.width = 256;
    canvas.height = 256;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    ctx.drawImage(img, 0, 0, 256, 256);
    const imageData = ctx.getImageData(0, 0, 256, 256);

    // 3. Mathematical Normalization (Pre-processing for AI Model)
    // Convert from [H, W, C=RGBA] to [C=RGB, H, W]
    // Hyperswap format: Normalized RGB Float32 ((val/255) - 0.5) / 0.5
    const float32Data = new Float32Array(3 * 256 * 256);
    for (let i = 0; i < 256 * 256; i++) {
        const r = imageData.data[i * 4];
        const g = imageData.data[i * 4 + 1];
        const b = imageData.data[i * 4 + 2];
        
        float32Data[i]                   = (r / 255.0 - 0.5) / 0.5; // Red channel
        float32Data[256 * 256 + i]       = (g / 255.0 - 0.5) / 0.5; // Green channel
        float32Data[2 * 256 * 256 + i]   = (b / 255.0 - 0.5) / 0.5; // Blue channel
    }
    const targetTensor = new ort.Tensor('float32', float32Data, [1, 3, 256, 256]);

    // 4. Source Embedding (Vectorized Face Print)
    // Hyperswap specifically expects the 512d *normalized* embedding
    const sourceTensor = new ort.Tensor('float32', new Float32Array(embeddingNormArray), [1, 512]);

    console.log("Running massive Face Swap AI directly on local Graphics Card via WebGPU...");
    const feeds = {
        'source': sourceTensor,
        'target': targetTensor
    };
    
    // 5. Execute Generative Model Wait (Synchronous blocker for GPU queue)
    const results = await session.run(feeds);
    
    const outputName = session.outputNames[0];
    const outputTensor = results[outputName];
    const outData = outputTensor.data;
    
    // 6. Mathematical Denormalization (Post-processing to pixels)
    // Inverse translation: (val * 0.5) + 0.5
    const finalImageData = new ImageData(256, 256);
    for (let i = 0; i < 256 * 256; i++) {
        let r = outData[i] * 0.5 + 0.5;
        let g = outData[256 * 256 + i] * 0.5 + 0.5;
        let b = outData[2 * 256 * 256 + i] * 0.5 + 0.5;
        
        finalImageData.data[i * 4]     = Math.max(0, Math.min(1, r)) * 255;
        finalImageData.data[i * 4 + 1] = Math.max(0, Math.min(1, g)) * 255;
        finalImageData.data[i * 4 + 2] = Math.max(0, Math.min(1, b)) * 255;
        finalImageData.data[i * 4 + 3] = 255; // Alpha channel solid
    }
    
    // 7. Render back to Image URL
    ctx.putImageData(finalImageData, 0, 0);
    return canvas.toDataURL('image/png');
};
