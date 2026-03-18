export const cropTargetCanvas = async (imgFile, affineMatrixArray) => {
    return new Promise((resolve, reject) => {
        const url = URL.createObjectURL(imgFile);
        const img = new Image();
        img.onload = () => {
            const canvas = document.createElement('canvas');
            canvas.width = 256;
            canvas.height = 256;
            const ctx = canvas.getContext('2d');
            
            const [m00, m01, m02] = affineMatrixArray[0];
            const [m10, m11, m12] = affineMatrixArray[1];
            
            // Map original image coordinates to crop coordinates
            ctx.setTransform(m00, m10, m01, m11, m02, m12);
            ctx.drawImage(img, 0, 0);
            
            URL.revokeObjectURL(url);
            
            // Output cropped base64 string
            const dataUrl = canvas.toDataURL('image/png');
            // Remove 'data:image/png;base64,' prefix for uniformity with python endpoint
            resolve(dataUrl.split(',')[1]);
        };
        img.onerror = (e) => {
            URL.revokeObjectURL(url);
            reject(e);
        };
        img.src = url;
    });
};

export const pasteBackCanvas = async (originalImgFile, swappedCropBase64, affineMatrixArray) => {
    return new Promise((resolve, reject) => {
        const url = URL.createObjectURL(originalImgFile);
        const original = new Image();
        original.onload = () => {
            const swapped = new Image();
            swapped.onload = () => {
                const canvas = document.createElement('canvas');
                canvas.width = original.width;
                canvas.height = original.height;
                const ctx = canvas.getContext('2d');
                
                // 1. Draw the high-res background
                ctx.drawImage(original, 0, 0);
                
                // 2. Prepare the Inverse Matrix using native DOMMatrix API
                const [m00, m01, m02] = affineMatrixArray[0];
                const [m10, m11, m12] = affineMatrixArray[1];
                const forwardMatrix = new DOMMatrix([m00, m10, m01, m11, m02, m12]);
                const invMatrix = forwardMatrix.inverse();
                
                // 3. Create a soft-blended mask of the 256x256 WebGPU AI crop
                const maskCanvas = document.createElement('canvas');
                maskCanvas.width = 256;
                maskCanvas.height = 256;
                const mCtx = maskCanvas.getContext('2d');
                
                mCtx.drawImage(swapped, 0, 0);
                
                // Apply feathering to the border of the face so skin blends naturally
                mCtx.globalCompositeOperation = 'destination-in';
                const gradient = mCtx.createRadialGradient(128, 128, 80, 128, 128, 128);
                gradient.addColorStop(0, 'rgba(0,0,0,1)');
                gradient.addColorStop(0.7, 'rgba(0,0,0,1)');
                gradient.addColorStop(1, 'rgba(0,0,0,0)'); // Transparent edges
                mCtx.fillStyle = gradient;
                mCtx.fillRect(0, 0, 256, 256);
                
                // 4. Transform the Canvas Context backwards and paste the feathered frame
                ctx.setTransform(invMatrix);
                ctx.drawImage(maskCanvas, 0, 0);
                
                URL.revokeObjectURL(url);
                resolve(canvas.toDataURL('image/jpeg', 0.9));
            };
            swapped.onerror = reject;
            swapped.src = swappedCropBase64;
        };
        original.onerror = reject;
        original.src = url;
    });
};
