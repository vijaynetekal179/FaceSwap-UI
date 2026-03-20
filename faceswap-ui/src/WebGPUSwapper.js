import * as ort from 'onnxruntime-web/webgpu';

let session = null;

export const isWebGPULoaded = () => session !== null;

export const initWebGPU = async (onProgress) => {
    if (session) return;
    try {
        const url = '/hyperswap_1c_256.onnx';
        const cacheName = 'facemorph-ai-models-v1';
        
        // 1. Check if the model is ALREADY perfectly saved on the user's hard drive!
        const cache = await caches.open(cacheName);
        let cachedResponse = await cache.match(url);
        
        let modelBuffer;

        if (cachedResponse) {
            console.log("CACHE HIT: Model found in persistent local storage! Loading instantly...");
            
            // It's already fully downloaded, so we instantly complete the progress bar UI
            const total = parseInt(cachedResponse.headers.get('content-length') || 0, 10);
            if (onProgress && total) onProgress(total, total);
            
            // Read from the persistent hard drive cache into RAM
            modelBuffer = await cachedResponse.arrayBuffer();
        } else {
            console.log("CACHE MISS: Fetching model from server for the first time...");
            const response = await fetch(url);
            
            if (!response.ok) throw new Error("Failed to fetch model from server.");
            
            const contentLength = response.headers.get('content-length');
            const total = contentLength ? parseInt(contentLength, 10) : 0;
            
            if (!total) {
                if (onProgress) onProgress(0, 0);
                ort.env.wasm.numThreads = Math.max(1, navigator.hardwareConcurrency - 1 || 1);
                session = await ort.InferenceSession.create(url, { executionProviders: ['webgpu', 'wasm'] });
                if (onProgress) onProgress(1, 1);
                return;
            }

            // Memory-safe direct streaming
            const bufferArray = new Uint8Array(total);
            let loaded = 0;
            
            if (onProgress) onProgress(0, total);
            
            const reader = response.body.getReader();
            while(true) {
                const {done, value} = await reader.read();
                if (done) break;
                
                bufferArray.set(value, loaded); 
                loaded += value.length;
                
                if (onProgress) onProgress(loaded, total);
            }
            
            // 2. SAVING THE MODEL PERMANENTLY:
            // We successfully downloaded the clean Array. Before we boot it up,
            // we save this exact array into the browser's persistent hard drive cache!
            console.log("Saving 400MB model permanently to Offline Storage Cache...");
            const offlineResponse = new Response(bufferArray, {
                headers: { 'content-length': total.toString(), 'content-type': 'application/octet-stream' }
            });
            await cache.put(url, offlineResponse);
            
            modelBuffer = bufferArray.buffer;
        }

        console.log("Initializing ONNX Inference Session from ArrayBuffer...");
        ort.env.wasm.numThreads = Math.max(1, navigator.hardwareConcurrency - 1 || 1);
        session = await ort.InferenceSession.create(modelBuffer, { executionProviders: ['webgpu', 'wasm'] });
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
