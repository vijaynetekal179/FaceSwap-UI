"""
Skin Tone Matching Test Script v3
===================================
  1. Extracts skin tone from user and all stock images using HSV masking
  2. Labels stock models using per-folder relative ranking (highest L = FAIR,
     middle = LIGHT, lowest = DUSKY) — no global thresholds needed
  3. Classifies user into FAIR / LIGHT / DUSKY by comparing against
     the average L of each category across all folders
  4. Selects the matching-category model from each folder
  5. Full distance matrix for debugging

Usage:
    python test_skin_tone.py
"""

import os
import sys
import numpy as np
import cv2

# ─── Setup ─────────────────────────────────────────────────────────
FACEFUSION_DIR = r"C:\faceSwap\facefusion"
if FACEFUSION_DIR not in sys.path:
    sys.path.append(FACEFUSION_DIR)

venv_scripts = os.path.abspath(os.path.join(os.path.dirname(__file__), "venv", "Scripts"))
if venv_scripts not in os.environ.get("PATH", ""):
    os.environ["PATH"] += os.pathsep + venv_scripts

PUBLIC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "faceswap-ui", "public"))
USER_DIR = os.path.join(PUBLIC_DIR, "user")
STOCKS_DIR = os.path.join(PUBLIC_DIR, "stocks")

# ─── Skin tone category order (highest L to lowest L) ─────────────
CATEGORIES = ["FAIR", "LIGHT", "DUSKY"]


def extract_skin_tone(image_path, face_bbox=None):
    img = cv2.imread(image_path)
    if img is None:
        return None

    if face_bbox is not None:
        x1, y1, x2, y2 = face_bbox
        h, w = img.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        face_crop = img[y1:y2, x1:x2]
    else:
        h, w = img.shape[:2]
        margin_y, margin_x = int(h * 0.2), int(w * 0.2)
        face_crop = img[margin_y:h - margin_y, margin_x:w - margin_x]

    if face_crop.size == 0:
        return None

    hsv = cv2.cvtColor(face_crop, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, np.array([0, 20, 30]), np.array([50, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([160, 20, 30]), np.array([180, 255, 255]))
    skin_mask = mask1 | mask2

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)

    skin_pixels = face_crop[skin_mask > 0]

    if len(skin_pixels) < 50:
        h2, w2 = face_crop.shape[:2]
        forehead = face_crop[int(h2*0.15):int(h2*0.35), int(w2*0.3):int(w2*0.7)]
        if forehead.size == 0:
            return None
        lab = cv2.cvtColor(forehead, cv2.COLOR_BGR2LAB)
        pixels = lab.reshape(-1, 3).astype(np.float64)
        return np.mean(pixels, axis=0)

    skin_lab = cv2.cvtColor(skin_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB)
    skin_lab = skin_lab.reshape(-1, 3).astype(np.float64)
    return np.mean(skin_lab, axis=0)


def detect_face_bbox(image_path):
    try:
        from facefusion.face_analyser import get_many_faces
        from facefusion.vision import read_static_image
        vision_frame = read_static_image(image_path)
        if vision_frame is None:
            return None
        faces = get_many_faces([vision_frame])
        if not faces:
            return None
        return tuple(map(int, faces[0].bounding_box))
    except:
        return None


def euclidean_distance(a, b):
    return np.linalg.norm(a - b)


def assign_folder_labels(folder_images):
    """Assign FAIR / LIGHT / DUSKY labels by ranking images within a folder
    by their L (lightness) value.  Highest L → FAIR, middle → LIGHT,
    lowest → DUSKY.  Works regardless of how many images the folder has."""

    # Sort by L value descending (lightest first)
    sorted_imgs = sorted(folder_images.items(), key=lambda x: x[1]["tone"][0], reverse=True)

    if len(sorted_imgs) >= 3:
        labels = CATEGORIES  # FAIR, LIGHT, DUSKY
    elif len(sorted_imgs) == 2:
        labels = ["FAIR", "DUSKY"]
    else:
        labels = ["FAIR"]

    for i, (img_file, data) in enumerate(sorted_imgs):
        label = labels[i] if i < len(labels) else labels[-1]
        folder_images[img_file]["label"] = label


def classify_user(user_L, category_avg_L):
    """Classify user skin tone by finding which category's average L
    is closest to the user's L value."""
    best_cat = "FAIR"
    best_diff = float("inf")
    for cat, avg_L in category_avg_L.items():
        diff = abs(user_L - avg_L)
        if diff < best_diff:
            best_diff = diff
            best_cat = cat
    return best_cat


def main():
    use_facefusion = False

    try:
        os.chdir(FACEFUSION_DIR)
        from facefusion import state_manager, core
        from facefusion.args import apply_args
        from facefusion.program import create_program
        import tempfile

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as f:
            dummy = f.name
            f.write(b"")

        program = create_program()
        args = vars(program.parse_args([
            'headless-run', '-s', dummy, '-t', dummy, '-o', dummy,
            '--processors', 'face_swapper',
            '--face-swapper-model', 'hyperswap_1c_256',
            '--execution-providers', 'cpu',
            '--execution-thread-count', '4',
            '--video-memory-strategy', 'tolerant'
        ]))
        apply_args(args, state_manager.init_item)
        if core.common_pre_check():
            use_facefusion = True
            print("✓ FaceFusion loaded for face detection.\n")
        os.remove(dummy)
    except Exception as e:
        print(f"⚠ FaceFusion not available ({e}). Using center-crop.\n")

    # ── Step 1: User Skin Tone ─────────────────────────────────────
    print("=" * 70)
    print("STEP 1: EXTRACTING USER SKIN TONE (HSV Skin Masking)")
    print("=" * 70)

    user_files = [f for f in os.listdir(USER_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not user_files:
        print("ERROR: No user image found in", USER_DIR)
        return

    user_path = os.path.join(USER_DIR, user_files[0])
    print(f"  User image: {user_files[0]}")

    bbox = detect_face_bbox(user_path) if use_facefusion else None
    if bbox:
        print(f"  Face detected at bbox: {bbox}")

    user_tone = extract_skin_tone(user_path, bbox)
    if user_tone is None:
        print("ERROR: Could not extract skin tone.")
        return

    print(f"  User LAB Skin Tone: L={user_tone[0]:.1f}  a={user_tone[1]:.1f}  b={user_tone[2]:.1f}")

    # ── Step 2: Extract All Stock Tones ────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 2: EXTRACTING STOCK IMAGE SKIN TONES (HSV Skin Masking)")
    print("=" * 70)

    folder_data = {}

    for folder in sorted(os.listdir(STOCKS_DIR)):
        folder_path = os.path.join(STOCKS_DIR, folder)
        if not os.path.isdir(folder_path):
            continue

        print(f"\n  📁 {folder}/")
        folder_data[folder] = {}

        for img_file in sorted(os.listdir(folder_path)):
            if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            img_path = os.path.join(folder_path, img_file)
            bbox = detect_face_bbox(img_path) if use_facefusion else None
            tone = extract_skin_tone(img_path, bbox)

            if tone is None:
                print(f"    {img_file:<30s}  SKIPPED")
                continue

            dist = euclidean_distance(user_tone, tone)
            folder_data[folder][img_file] = {"tone": tone, "label": "", "distance": dist}

    # ── Step 2b: Assign labels by per-folder relative ranking ─────
    print("\n" + "=" * 70)
    print("STEP 2b: ASSIGNING LABELS (per-folder L-value ranking)")
    print("=" * 70)
    print("  (Highest L in folder → FAIR, middle → LIGHT, lowest → DUSKY)\n")

    for folder, images in folder_data.items():
        assign_folder_labels(images)
        print(f"  📁 {folder}/")
        for img_file, data in sorted(images.items(), key=lambda x: x[1]["tone"][0], reverse=True):
            print(f"    {img_file:<30s}  L={data['tone'][0]:6.1f}  → {data['label']}")

    # ── Step 2c: Compute category averages & classify user ────────
    category_L_values = {cat: [] for cat in CATEGORIES}
    for folder, images in folder_data.items():
        for img_file, data in images.items():
            category_L_values[data["label"]].append(data["tone"][0])

    category_avg_L = {}
    print(f"\n  Category average L values (across all folders):")
    for cat in CATEGORIES:
        vals = category_L_values[cat]
        if vals:
            avg = np.mean(vals)
            category_avg_L[cat] = avg
            print(f"    {cat:8s}  avg L = {avg:.1f}  (from {len(vals)} images)")

    user_label = classify_user(user_tone[0], category_avg_L)
    print(f"\n  ✓ User L={user_tone[0]:.1f} → classified as: {user_label}")

    # ── Step 3: Results Table ──────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 3: MATCHING RESULTS — Label-based matching")
    print("=" * 70)
    print(f"  Strategy: Pick the {user_label} model from each folder")
    print(f"            (fallback to closest distance if no exact label match)\n")

    print(f"  {'Jewellery Folder':<20s}  {'Best Match':<30s}  {'Label':<8s}  {'Distance':>10s}")
    print(f"  {'─' * 20}  {'─' * 30}  {'─' * 8}  {'─' * 10}")

    for folder, images in folder_data.items():
        if not images:
            continue

        # Primary: pick model with matching label
        same_label = {f: d for f, d in images.items() if d["label"] == user_label}
        if same_label:
            best_file = min(same_label, key=lambda f: same_label[f]["distance"])
        else:
            # Fallback: closest distance overall
            best_file = min(images, key=lambda f: images[f]["distance"])

        best = images[best_file]
        print(f"  {folder:<20s}  {best_file:<30s}  {best['label']:<8s}  {best['distance']:>10.2f}")

    # ── Step 4: Full Distance Matrix ───────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 4: FULL DISTANCE MATRIX (for debugging)")
    print("=" * 70)

    for folder, images in folder_data.items():
        print(f"\n  📁 {folder}/")
        sorted_images = sorted(images.items(), key=lambda x: x[1]["distance"])
        for rank, (img_file, data) in enumerate(sorted_images, 1):
            selected = " ← SELECTED" if data["label"] == user_label else ""
            print(f"    #{rank}  dist={data['distance']:8.2f}  [{data['label']:<5s}]  {img_file}{selected}")


if __name__ == "__main__":
    main()
