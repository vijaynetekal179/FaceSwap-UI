import os
import sys
import tempfile
import traceback
import time
import asyncio
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import cv2

# Adjust this path if your facefusion folder is located elsewhere
FACEFUSION_DIR = r"C:\Users\devadmin\facefusion"
FACEFUSION_SCRIPT = os.path.join(FACEFUSION_DIR, "facefusion.py")

# Emulate command line so FaceFusion thinks we are running via CLI
if FACEFUSION_DIR not in sys.path:
    sys.path.append(FACEFUSION_DIR)

from facefusion import state_manager, core, face_analyser, face_recognizer, face_selector
from facefusion.args import apply_args
from facefusion.workflows import image_to_image
from facefusion.program import create_program
from facefusion.processors.core import get_processors_modules
from facefusion.face_analyser import get_many_faces, get_average_face, scale_face
from facefusion.types import Face
from facefusion.vision import read_static_image, write_image, extract_vision_mask, conditional_merge_vision_mask
from facefusion.face_selector import select_faces
import numpy as np
import json
import csv
import base64
import hashlib

def init_facefusion():
    print("Initializing FaceFusion AI models in GPU. This will take ~6 seconds...", flush=True)
    
    # Very important: FaceFusion relies on relative path resolution for finding its processors
    original_cwd = os.getcwd()
    os.chdir(FACEFUSION_DIR)
    
    # FaceFusion argument parser requires arguments to actually exist on the hard drive
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as dummy_file:
        dummy_path = dummy_file.name
        dummy_file.write(b"")

    try:
        program = create_program()
        args = vars(program.parse_args([
            'headless-run',
            '-s', dummy_path,
            '-t', dummy_path,
            '-o', dummy_path,
            
            # --- Processors ---
            '--processors', 'face_swapper', 'face_enhancer',

            # --- Face Swapper ---
            '--face-swapper-model', 'hyperswap_1c_256',

            # --- Face Enhancer: gfpgan_1.4 — fast and high quality ---
            '--face-enhancer-model', 'gfpgan_1.4',
            '--face-enhancer-blend', '100',

            # --- Execution: Nvidia GPU ---
            '--execution-providers', 'cuda',
            # 4 threads is often better for laptop GPUs to avoid overheating bottlenecks
            '--execution-thread-count', '4',
            '--video-memory-strategy', 'tolerant'
        ]))
        apply_args(args, state_manager.init_item)
        
        # Pre-check and load models into VRAM permanently
        if core.common_pre_check() and core.processors_pre_check():
            print("FaceFusion Models successfully loaded into GPU memory!", flush=True)
            return True
        return False
    finally:
        # Restore directory and cleanup
        os.chdir(original_cwd)
        if os.path.exists(dummy_path):
            os.remove(dummy_path)


# Pre-initialize FaceFusion before spinning up the web server
if not init_facefusion():
    print("CRITICAL: Failed to initialize FaceFusion models during boot. Exiting.", flush=True)
    sys.exit(1)

app = FastAPI(title="FaceMorph Native API")

# Allow requests from React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the massive FaceFusion AI models directly to the browser
models_dir = os.path.join(FACEFUSION_DIR, ".assets", "models")
if os.path.exists(models_dir):
    app.mount("/models", StaticFiles(directory=models_dir), name="models")

# A critically important lock to ensure FaceFusion only processes 1 image at a time natively.
# This completely prevents VRAM overflow crashes when multiple users hit the API at once.
# They will be neatly queued instead of crashing the server!
swap_queue_lock = asyncio.Lock()

@app.post("/api/v1/swap")
async def process_face_swap(source: UploadFile = File(...), target: UploadFile = File(...)):
    print(f"Incoming swap request queued! Source={source.filename}, Target={target.filename}", flush=True)
    
    t_queued = time.time()

    # Acquring the lock means waiting our turn if someone else is currently running
    async with swap_queue_lock:
        queue_wait_time = time.time() - t_queued
        print(f"Request acquired GPU lock after waiting {queue_wait_time:.2f}s in queue.", flush=True)
        
        src_path = None
        tgt_path = None
        out_path = None

        try:
            # Write source image to a temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as src_file:
                src_path = src_file.name
                src_file.write(await source.read())

            # Write target image to a temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tgt_file:
                tgt_path = tgt_file.name
                tgt_file.write(await target.read())

            # Output path must NOT exist yet — FaceFusion creates it itself
            out_path = tempfile.mktemp(suffix=".jpg")

            start_time = time.time()
            
            # Update the global state manager with the new image paths for this specific request
            state_manager.set_item('source_paths', [src_path])
            state_manager.set_item('target_path', tgt_path)
            state_manager.set_item('output_path', out_path)
            
            # Run FaceFusion natively (Blocking, but single-threaded by the queue lock)
            error_code = image_to_image.process(start_time)
            
            if error_code != 0:
                raise HTTPException(status_code=500, detail=f"FaceFusion processing failed with error code: {error_code}")

            if not os.path.isfile(out_path):
                raise HTTPException(status_code=500, detail="FaceFusion returned exit 0 but produced no output file.")

            # Read the swapped image bytes
            with open(out_path, "rb") as image_file:
                swapped_bytes = image_file.read()

            processing_time = time.time() - start_time
            print(f"Swap successful! Queue Wait: {queue_wait_time:.2f}s | Processing: {processing_time:.2f}s", flush=True)
            
            return Response(content=swapped_bytes, media_type="image/jpeg")

        except HTTPException as e:
            raise e
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Server exception: {type(e).__name__}: {e}")
            
        finally:
            # Clean up temp files immediately to save storage
            for path in [src_path, tgt_path, out_path]:
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                    except Exception:
                        pass

@app.post("/api/v1/extract-embedding")
async def extract_face_embedding(source: UploadFile = File(...)):
    """
    Extracts face embedding from an uploaded image and deletes it immediately.
    """
    print(f"Extracting embedding for {source.filename}...", flush=True)
    src_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as src_file:
            src_path = src_file.name
            src_file.write(await source.read())
        
        from facefusion.vision import read_static_image
        vision_frame = read_static_image(src_path)
        faces = get_many_faces([vision_frame])
        source_face = get_average_face(faces)
        
        if not source_face:
            raise HTTPException(status_code=400, detail="No face detected in the source image.")
            
        # Convert numpy arrays to lists for JSON serialization
        embedding_data = {
            "embedding": source_face.embedding.tolist(),
            "embedding_norm": source_face.embedding_norm.tolist()
        }
        
        return embedding_data
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if src_path and os.path.exists(src_path):
            os.remove(src_path)

CSV_FILE_PATH = "embeddings.csv"
CSV_TARGETS_PATH = "targets.csv"

def get_target_face_cache(target_hash: str):
    if not os.path.isfile(CSV_TARGETS_PATH):
        return None
    try:
        with open(CSV_TARGETS_PATH, mode='r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['target_hash'] == target_hash:
                    bbox = np.array(json.loads(row['bounding_box']), dtype=np.float32)
                    lmark = {k: np.array(v, dtype=np.float32) for k, v in json.loads(row['landmark_set']).items()}
                    return Face(
                        bounding_box=bbox, score_set=None, landmark_set=lmark,
                        angle=None, embedding=None, embedding_norm=None, gender=None, age=None, race=None
                    )
    except Exception as e:
        print(f"Error reading target cache: {e}")
    return None

def save_target_face_cache(target_hash: str, target_face: Face):
    try:
        bbox_json = json.dumps(target_face.bounding_box.tolist())
        lmark_json = json.dumps({k: v.tolist() for k, v in target_face.landmark_set.items()})
        file_exists = os.path.isfile(CSV_TARGETS_PATH)
        with open(CSV_TARGETS_PATH, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['target_hash', 'bounding_box', 'landmark_set'])
            writer.writerow([target_hash, bbox_json, lmark_json])
    except Exception as e:
        print(f"Error saving target cache: {e}")

@app.post("/api/v1/save-embedding")
async def save_face_embedding(name: str = Form(...), source: UploadFile = File(...)):
    """
    Extracts face embedding from an uploaded image, saves it to a CSV, and deletes the image.
    """
    print(f"Extracting and saving embedding for {name} from {source.filename}...", flush=True)
    src_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as src_file:
            src_path = src_file.name
            src_file.write(await source.read())
        
        from facefusion.vision import read_static_image
        vision_frame = read_static_image(src_path)
        faces = get_many_faces([vision_frame])
        source_face = get_average_face(faces)
        
        if not source_face:
            raise HTTPException(status_code=400, detail="No face detected in the source image.")
            
        embedding_list = source_face.embedding.tolist()
        norm_list = source_face.embedding_norm.tolist()
        
        file_exists = os.path.isfile(CSV_FILE_PATH)
        with open(CSV_FILE_PATH, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['name', 'embedding', 'embedding_norm'])
            writer.writerow([name, json.dumps(embedding_list), json.dumps(norm_list)])
            
        return {"message": f"Embedding for '{name}' saved successfully"}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if src_path and os.path.exists(src_path):
            os.remove(src_path)

@app.get("/api/v1/get-embedding/{name}")
async def get_face_embedding(name: str):
    """
    Fetches the 512d face embedding for a given name from the CSV file.
    """
    if not os.path.isfile(CSV_FILE_PATH):
        raise HTTPException(status_code=404, detail="No embeddings database found.")
    
    with open(CSV_FILE_PATH, mode='r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['name'] == name:
                return {
                    "embedding": json.loads(row['embedding']),
                    "embedding_norm": json.loads(row['embedding_norm'])
                }
                
    raise HTTPException(status_code=404, detail=f"Embedding for '{name}' not found.")

@app.post("/api/v1/prepare-crop")
async def prepare_crop(target: UploadFile = File(...)):
    """ Returns a perfectly aligned 256x256 crop of the largest face for WebGPU processing """
    print(f"Preparing crop for WebGPU...", flush=True)
    tgt_path = None
    try:
        target_bytes = await target.read()
        target_hash = hashlib.md5(target_bytes).hexdigest()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tgt_file:
            tgt_path = tgt_file.name
            tgt_file.write(target_bytes)
            
        from facefusion.vision import read_static_image
        target_vision_frame = read_static_image(tgt_path)
        
        target_face = get_target_face_cache(target_hash)
        if target_face:
            print(f"CACHE HIT [{target_hash}]: Bypassed Face Detection for Prepare Crop!", flush=True)
        else:
            print("CACHE MISS: Detecting face on target...", flush=True)
            faces = get_many_faces([target_vision_frame])
            if not faces:
                raise HTTPException(status_code=400, detail="No face detected in target image")
            from facefusion.face_selector import sort_faces_by_order
            faces = sort_faces_by_order(faces, 'large-small')
            target_face = faces[0]
            save_target_face_cache(target_hash, target_face)
        
        from facefusion.face_helper import warp_face_by_face_landmark_5
        crop_vision_frame, affine_matrix = warp_face_by_face_landmark_5(
            target_vision_frame, target_face.landmark_set.get('5/68'), 'arcface_128', (256, 256)
        )
        
        import cv2
        _, buffer = cv2.imencode('.png', crop_vision_frame)
        crop_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return {
            "crop_base64": crop_base64
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tgt_path and os.path.exists(tgt_path):
            os.remove(tgt_path)

@app.post("/api/v1/finalize-swap")
async def finalize_swap(target: UploadFile = File(...), swapped_crop_base64: str = Form(...)):
    """ Pastes the WebGPU swapped crop back and enhances it """
    print(f"Finalizing WebGPU swap with GFPGAN...", flush=True)
    
    tgt_path = None
    out_path = tempfile.mktemp(suffix=".jpg")
    try:
        target_bytes = await target.read()
        target_hash = hashlib.md5(target_bytes).hexdigest()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tgt_file:
            tgt_path = tgt_file.name
            tgt_file.write(target_bytes)

        import cv2
        import numpy as np
        
        img_data = base64.b64decode(swapped_crop_base64.split(",")[1] if "," in swapped_crop_base64 else swapped_crop_base64)
        nparr = np.frombuffer(img_data, np.uint8)
        swapped_crop_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        from facefusion.vision import read_static_image
        target_vision_frame = read_static_image(tgt_path, 'rgba')
        reference_vision_frame = read_static_image(tgt_path)
        temp_vision_frame = target_vision_frame.copy()
        temp_vision_mask = extract_vision_mask(temp_vision_frame)

        target_face = get_target_face_cache(target_hash)
        if target_face:
            print(f"CACHE HIT [{target_hash}]: Bypassed Face Detection for Finalizing Swap!", flush=True)
        else:
            faces = get_many_faces([reference_vision_frame])
            if not faces:
                raise HTTPException(status_code=400, detail="No face detected during finalization")
            from facefusion.face_selector import sort_faces_by_order
            faces = sort_faces_by_order(faces, 'large-small')
            target_face = faces[0]
            save_target_face_cache(target_hash, target_face)

        from facefusion.face_helper import warp_face_by_face_landmark_5, paste_back
        from facefusion.face_masker import create_box_mask, create_occlusion_mask, create_area_mask, create_region_mask
        
        _, affine_matrix = warp_face_by_face_landmark_5(
            temp_vision_frame[:, :, :3], target_face.landmark_set.get('5/68'), 'arcface_128', (256, 256)
        )

        crop_masks = []
        box_mask = create_box_mask(swapped_crop_frame, 0.3, (0, 0, 0, 0))
        crop_masks.append(box_mask)
        occlusion_mask = create_occlusion_mask(swapped_crop_frame)
        crop_masks.append(occlusion_mask)
        
        crop_mask = np.minimum.reduce(crop_masks).clip(0, 1)

        temp_vision_frame[:, :, :3] = paste_back(temp_vision_frame[:, :, :3], swapped_crop_frame, crop_mask, affine_matrix)

        processors = state_manager.get_item('processors')
        for processor_module in get_processors_modules(processors):
            processor_name = processor_module.__name__.split('.')[-2]
            if processor_name == 'face_enhancer':
                from facefusion.processors.modules.face_enhancer.core import enhance_face
                scaled_target_face = scale_face(target_face, target_vision_frame[:, :, :3], temp_vision_frame[:, :, :3])
                temp_vision_frame[:, :, :3] = enhance_face(scaled_target_face, temp_vision_frame[:, :, :3])

        temp_vision_frame = conditional_merge_vision_mask(temp_vision_frame, temp_vision_mask)
        write_image(out_path, temp_vision_frame)

        with open(out_path, "rb") as image_file:
            swapped_bytes = image_file.read()

        from fastapi.responses import Response
        return Response(content=swapped_bytes, media_type="image/jpeg")

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        for p in [tgt_path, out_path]:
            if p and os.path.exists(p):
                os.remove(p)

@app.post("/api/v1/extract-source")
async def extract_source(name: str = Form(...), source: UploadFile = File(...)):
    """ Extracts and returns the Source Face math exclusively for the Client LocalStorage """
    print(f"Extracting source embedding for {name} to hand to LocalStorage...", flush=True)
    src_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as src_file:
            src_path = src_file.name
            src_file.write(await source.read())
            
        from facefusion.vision import read_static_image
        source_vision_frame = read_static_image(src_path)
        faces = get_many_faces([source_vision_frame])
        
        if not faces:
            raise HTTPException(status_code=400, detail="No face detected in source image")
            
        from facefusion.face_selector import sort_faces_by_order
        faces = sort_faces_by_order(faces, 'large-small')
        
        return {
            "name": name,
            "embedding_norm": faces[0].embedding_norm.tolist()
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if src_path and os.path.exists(src_path):
            os.remove(src_path)

def extract_skin_tone(vision_frame, face):
    x1, y1, x2, y2 = map(int, face.bounding_box)
    h, w = vision_frame.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    
    crop = vision_frame[y1:y2, x1:x2]
    if crop.size == 0:
        return [0.0] * 6
        
    # FIX: Robust skin isolation. HSV hardcoded limits completely fail on extreme dark complexions (low V/Brightness).
    # Instead, we mathematically sample pure facial skin by grabbing the central 30% of the bounding box (nose, inner cheeks), completely avoiding hair/backgrounds.
    bh, bw = crop.shape[:2]
    cy, cx = bh // 2, bw // 2
    qy, qx = int(bh * 0.15), int(bw * 0.15)
    
    center_crop = crop[cy - qy: cy + qy, cx - qx: cx + qx]
    if center_crop.size == 0:
        center_crop = crop
        
    lab = cv2.cvtColor(center_crop, cv2.COLOR_BGR2LAB)
    skin_pixels = lab.reshape(-1, 3)
        
    L_mean, a_mean, b_mean = np.mean(skin_pixels, axis=0)
    L_std, a_std, b_std = np.std(skin_pixels, axis=0)
    
    return [float(L_mean), float(a_mean), float(b_mean), float(L_std), float(a_std), float(b_std)]

@app.post("/api/v1/extract-user-tone")
async def extract_user_tone(source: UploadFile = File(...)):
    """ Extracts the 6D LAB skin tone mathematically for matching """
    print(f"Extracting user skin tone for {source.filename}...", flush=True)
    src_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as src_file:
            src_path = src_file.name
            src_file.write(await source.read())
            
        from facefusion.vision import read_static_image
        source_vision_frame = read_static_image(src_path)
        faces = get_many_faces([source_vision_frame])
        
        if not faces:
            raise HTTPException(status_code=400, detail="No face detected in source image")
            
        from facefusion.face_selector import sort_faces_by_order
        faces = sort_faces_by_order(faces, 'large-small')
        
        skin_tone = extract_skin_tone(source_vision_frame, faces[0])
        return {"skin_tone": skin_tone}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if src_path and os.path.exists(src_path):
            os.remove(src_path)

class StockPaths(BaseModel):
    paths: list[str]

@app.post("/api/v1/extract-stock-tones")
async def extract_stock_tones(payload: StockPaths):
    """ Batch extracts 6D LAB skin tone encodings internally for a list of local UI stock images """
    print(f"Extracting skin tones for {len(payload.paths)} stock images...", flush=True)
    results = {}
    
    # Path navigation: from FaceAPI to faceswap-ui/public
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "faceswap-ui", "public"))
    
    try:
        from facefusion.vision import read_static_image
        from facefusion.face_selector import sort_faces_by_order
        
        for path in payload.paths:
            clean_path = path.lstrip('/')
            full_path = os.path.join(base_dir, clean_path)
            if not os.path.exists(full_path):
                print(f"Warning: Stock image not found at {full_path}", flush=True)
                continue
                
            vision_frame = read_static_image(full_path)
            if vision_frame is None:
                continue
                
            faces = get_many_faces([vision_frame])
            if not faces:
                continue
                
            faces = sort_faces_by_order(faces, 'large-small')
            tone = extract_skin_tone(vision_frame, faces[0])
            results[path] = tone
            
        return {"stock_tones": results}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/extract-target")
async def extract_target(target: UploadFile = File(...)):
    """ Extracts and returns the Target Face Affine Matrix for Canvas LocalStorage processing """
    print("Extracting target Affine Matrix to hand to LocalStorage...", flush=True)
    tgt_path = None
    try:
        target_bytes = await target.read()
        target_hash = hashlib.md5(target_bytes).hexdigest()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tgt_file:
            tgt_path = tgt_file.name
            tgt_file.write(target_bytes)
            
        from facefusion.vision import read_static_image
        target_vision_frame = read_static_image(tgt_path)
        
        faces = get_many_faces([target_vision_frame])
        if not faces:
            raise HTTPException(status_code=400, detail="No face detected in target image")
            
        from facefusion.face_selector import sort_faces_by_order
        faces = sort_faces_by_order(faces, 'large-small')
        target_face = faces[0]
        
        from facefusion.face_helper import warp_face_by_face_landmark_5
        _, affine_matrix = warp_face_by_face_landmark_5(
            target_vision_frame, target_face.landmark_set.get('5/68'), 'arcface_128', (256, 256)
        )
        
        return {
            "target_hash": target_hash,
            "affine_matrix": affine_matrix.tolist()
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tgt_path and os.path.exists(tgt_path):
            os.remove(tgt_path)

@app.post("/api/v1/swap-embedding")
async def process_swap_with_embedding(target: UploadFile = File(...), embedding_data: str = File(...)):
    """
    Swaps face using a pre-calculated embedding.
    """
    data = json.loads(embedding_data)
    # ONNX Runtime usually expects float32
    source_embedding = np.array(data["embedding"], dtype=np.float32)
    source_embedding_norm = np.array(data["embedding_norm"], dtype=np.float32)
    
    # Create a mock Face object
    # We use None for things like bounding_box as they aren't used for source faces in swap
    mock_source_face = Face(
        bounding_box=None,
        score_set=None,
        landmark_set=None, 
        angle=None,
        embedding=source_embedding,
        embedding_norm=source_embedding_norm,
        gender=None,
        age=None,
        race=None
    )

    t_queued = time.time()
    async with swap_queue_lock:
        queue_wait_time = time.time() - t_queued
        tgt_path = None
        out_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tgt_file:
                tgt_path = tgt_file.name
                tgt_file.write(await target.read())

            out_path = tempfile.mktemp(suffix=".jpg")
            start_time = time.time()

            target_vision_frame = read_static_image(tgt_path, 'rgba')
            reference_vision_frame = read_static_image(tgt_path) # Simplified reference
            temp_vision_frame = target_vision_frame.copy()
            temp_vision_mask = extract_vision_mask(temp_vision_frame)
            
            processors = state_manager.get_item('processors')
            for processor_module in get_processors_modules(processors):
                processor_name = processor_module.__name__.split('.')[-2]
                print(f"Running processor: {processor_name}", flush=True)

                if processor_name == 'face_swapper':
                    from facefusion.processors.modules.face_swapper.core import swap_face
                    target_faces = select_faces(reference_vision_frame, target_vision_frame[:, :, :3])
                    for target_face in target_faces:
                        scaled_target_face = scale_face(target_face, target_vision_frame[:, :, :3], temp_vision_frame[:, :, :3])
                        temp_vision_frame_swap = swap_face(mock_source_face, scaled_target_face, temp_vision_frame[:, :, :3])
                        temp_vision_frame[:, :, :3] = temp_vision_frame_swap
                
                elif processor_name == 'face_enhancer':
                    from facefusion.processors.modules.face_enhancer.core import enhance_face
                    target_faces = select_faces(reference_vision_frame, target_vision_frame[:, :, :3])
                    for target_face in target_faces:
                        scaled_target_face = scale_face(target_face, target_vision_frame[:, :, :3], temp_vision_frame[:, :, :3])
                        temp_vision_frame_enhance = enhance_face(scaled_target_face, temp_vision_frame[:, :, :3])
                        temp_vision_frame[:, :, :3] = temp_vision_frame_enhance

            temp_vision_frame = conditional_merge_vision_mask(temp_vision_frame, temp_vision_mask)
            write_image(out_path, temp_vision_frame)

            with open(out_path, "rb") as image_file:
                swapped_bytes = image_file.read()

            return Response(content=swapped_bytes, media_type="image/jpeg")

        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            for path in [tgt_path, out_path]:
                if path and os.path.exists(path):
                    os.remove(path)

if __name__ == '__main__':
    # Running native Python API directly on port 8000
    # You no longer need Java or gRPC!
    uvicorn.run("server:app", host="0.0.0.0", port=8000, log_level="info")