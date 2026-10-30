#!/usr/bin/env python3
"""
extract_video_to_staging.py
Mengekstrak patch 224x224 dari video ke folder staging:
vid/drive_pdp/staging_patches/<video_stem>/<class_name>/
"""

import sys
import cv2
import numpy as np
from pathlib import Path

BASE_DIR = Path("/home/fawwazfa/Program/Harvestmoon")
sys.path.append(str(BASE_DIR / "Pigeon_Harvest/training"))
from build_dataset_from_drive import classify_patch_agronomy

STAGING_DIR = BASE_DIR / "vid/drive_pdp/staging_patches"

def process_video(video_path: Path, step_seconds: float = 1.0, patch_size: int = 224):
    if not video_path.exists():
        print(f"File {video_path} tidak ditemukan!")
        return

    vid_staging = STAGING_DIR / video_path.stem
    vid_staging.mkdir(parents=True, exist_ok=True)
    for c in ["0_zona_normal", "1_ketidakmerataan_irigasi", "2_defisit_air_kanopi", "3_defisit_air_tanah"]:
        (vid_staging / c).mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Gagal membuka video {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step_frames = max(1, int(step_seconds * fps))

    print(f"🎬 Memproses {video_path.name}: {total_frames} frames @ {fps:.1f} fps (step {step_frames} frames)...")

    counts = {"0_zona_normal": 0, "1_ketidakmerataan_irigasi": 0, "2_defisit_air_kanopi": 0, "3_defisit_air_tanah": 0}
    saved_count = 0

    for f_idx in range(0, total_frames, step_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        h, w = frame.shape[:2]
        # Grid sampling 3x4 = 12 patch per frame
        ys = [80, int((h - patch_size) * 0.5), h - patch_size - 80]
        xs = [100, int((w - patch_size) * 0.33), int((w - patch_size) * 0.66), w - patch_size - 100]

        for y in ys:
            for x in xs:
                patch = frame[y:y+patch_size, x:x+patch_size]
                if patch.shape[:2] == (patch_size, patch_size):
                    cls = classify_patch_agronomy(patch)
                    p_path = vid_staging / cls / f"{video_path.stem}_{f_idx:06d}_{y}_{x}.jpg"
                    cv2.imwrite(str(p_path), patch, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    counts[cls] += 1
                    saved_count += 1

    cap.release()
    print(f"✅ Selesai mengekstrak {video_path.name}: {saved_count} total patch tersimpan di {vid_staging}")
    for c, cnt in counts.items():
        print(f"   - {c}: {cnt}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        vpath = Path(sys.argv[1])
    else:
        vpath = BASE_DIR / "vid/drive_pdp/playback/flight 1.mp4"
    process_video(vpath, step_seconds=1.0)
