#!/usr/bin/env python3
"""
build_dataset_from_drive.py
===========================
Memproses video sawah yang diunduh dari Google Drive (flight 1, flight 2, flight 3),
mengekstrak patch 224x224 resolusi spasial tinggi,
dan mengklasifikasikan patch ke dalam 4 Kelas Target PDP:
  - 0_zona_normal: Kanopi hijau lebat, vegetasi sehat, kadar air cukup (macak-macak).
  - 1_ketidakmerataan_irigasi: Anomali saluran/padi muda/pola tumbuh tidak merata.
  - 2_defisit_air_kanopi: Kanopi stres klorosis, daun menguning/pucat.
  - 3_defisit_air_tanah: Lahan kering, retakan tanah, bare soil / jerami kering.

Dilengkapi deduplikasi hash, balancing, dan split Train (1000/kelas), Val (200/kelas), Test (75/kelas).
"""

import os
import cv2
import shutil
import hashlib
import random
import numpy as np
from pathlib import Path
from collections import defaultdict

random.seed(42)
np.random.seed(42)

BASE_DIR = Path("/home/fawwazfa/Program/Harvestmoon")
TRAINING_DIR = BASE_DIR / "Pigeon_Harvest/training"
DATASET_DIR = TRAINING_DIR / "datasets/moonharvest-uav"
OUT_DIR = DATASET_DIR / "dataset_pdp_drive"
DRIVE_VID_DIR = BASE_DIR / "vid/drive_pdp"

CLASSES = [
    "0_zona_normal",
    "1_ketidakmerataan_irigasi",
    "2_defisit_air_kanopi",
    "3_defisit_air_tanah"
]

def get_hash(img_bytes: bytes) -> str:
    return hashlib.md5(img_bytes).hexdigest()

def compute_exg(bgr):
    """Excess Green Index = 2*G - R - B."""
    b, g, r = cv2.split(bgr.astype(np.float32))
    tot = b + g + r + 1e-6
    return (2 * g - r - b) / tot

def classify_patch_agronomy(patch_bgr):
    """
    Menganalisis patch kanopi/lahan padi berdasarkan indeks agronomi ExG & HSV:
    Mengembalikan nama kelas PDP yang paling sesuai:
    0_zona_normal, 1_ketidakmerataan_irigasi, 2_defisit_air_kanopi, atau 3_defisit_air_tanah.
    """
    hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)
    H = hsv[:, :, 0]
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]
    exg = compute_exg(patch_bgr)

    # Masker tanah terbuka / retakan tanah / jerami kering
    # S rendah (abu-abu/tanah) atau Hue coklat/kering (H < 22) dengan ExG negatif / sangat rendah
    soil_mask = ((S <= 55) & (V >= 60) & (exg < 0.010)) | ((H <= 20) & (V >= 80) & (exg < 0.005))
    soil_ratio = np.mean(soil_mask)

    # Masker kanopi hijau subur (Lush Green)
    lush_mask = (H >= 32) & (H <= 88) & (S >= 40) & (V >= 45) & (exg >= 0.032)
    lush_ratio = np.mean(lush_mask)

    # Masker kanopi stres / klorosis / daun menguning (Defisit Air Kanopi)
    # Hue kuning-keemasan (20 <= H < 32), atau ExG rendah tapi masih vegetasi (0.005 <= exg < 0.025)
    stress_mask = ((H >= 18) & (H < 32) & (S >= 35) & (V >= 50)) | ((H >= 32) & (H <= 85) & (exg >= 0.008) & (exg < 0.025))
    stress_ratio = np.mean(stress_mask)

    # Anomali / Ketidakmerataan (campuran air/lumpur/padi anakan jarang, variansi spasial tinggi)
    # Sering ditandai variasi ExG tinggi di dalam patch (std dev tinggi) atau air/lumpur dengan vegetasi sporadis
    water_mud_mask = (V < 50) | ((S < 40) & (V < 75) & (exg < 0.015))
    water_mud_ratio = np.mean(water_mud_mask)
    exg_std = float(np.std(exg))

    if soil_ratio > 0.45 or (soil_ratio > 0.30 and lush_ratio < 0.20):
        return "3_defisit_air_tanah"
    elif stress_ratio > 0.40 or (stress_ratio > 0.25 and lush_ratio < 0.35):
        return "2_defisit_air_kanopi"
    elif lush_ratio > 0.55 and exg_std < 0.06:
        return "0_zona_normal"
    elif (lush_ratio > 0.20 and (water_mud_ratio > 0.25 or soil_ratio > 0.20)) or exg_std >= 0.05:
        return "1_ketidakmerataan_irigasi"
    elif lush_ratio >= 0.40:
        return "0_zona_normal"
    else:
        return "1_ketidakmerataan_irigasi"

def extract_patches_from_video(video_path: Path, step_frames: int = 60, patch_size: int = 224):
    """
    Ekstrak grid patch dari frame video resolusi Full HD (1920x1080).
    Menghasilkan patch berukuran 224x224 px dengan sampling cepat (1 frame per detik).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Gagal membuka video: {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Mengekstrak dari {video_path.name}: {total_frames} frames @ {fps:.1f} fps (step {step_frames} frames)...")

    patches = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        if frame_idx % step_frames == 0:
            h, w = frame.shape[:2]
            # Grid crop 3x4 = 12 patch per frame
            ys = [80, int((h - patch_size) * 0.5), h - patch_size - 80]
            xs = [100, int((w - patch_size) * 0.33), int((w - patch_size) * 0.66), w - patch_size - 100]
            for y in ys:
                for x in xs:
                    patch = frame[y:y+patch_size, x:x+patch_size]
                    if patch.shape[0] == patch_size and patch.shape[1] == patch_size:
                        patches.append(patch)

        frame_idx += 1

    cap.release()
    print(f"  -> Total patch diekstrak dari {video_path.name}: {len(patches)}")
    return patches

def main():
    print("=" * 70)
    print("🌾 MEMPROSES DATASET BARU DARI VIDEO SAWAH GOOGLE DRIVE (PDP)")
    print("=" * 70)

    # 1. Cari video di vid/drive_pdp/
    video_files = list(DRIVE_VID_DIR.rglob("*.mp4")) + list(DRIVE_VID_DIR.rglob("*.MOV")) + list(DRIVE_VID_DIR.rglob("*.mkv"))
    if not video_files:
        print(f"❌ Tidak ada file video di {DRIVE_VID_DIR}")
        return

    print(f"Ditemukan {len(video_files)} video:")
    for v in video_files:
        print(f"  - {v.name} ({v.stat().st_size / (1024*1024):.1f} MB)")

    raw_pool = {c: [] for c in CLASSES}

    # Muat patch yang sudah ada di staging (jika ada)
    staging_dir = DRIVE_VID_DIR / "staging_patches"
    if staging_dir.exists():
        print("Memuat patch tersimpan dari staging...")
        for c in CLASSES:
            for f in staging_dir.rglob(f"{c}/*.jpg"):
                img = cv2.imread(str(f))
                if img is not None and img.shape[:2] == (224, 224):
                    raw_pool[c].append(img)
        for c in CLASSES:
            print(f"  Staging {c}: {len(raw_pool[c])} patch")

    for vpath in sorted(video_files):
        # Hindari ekstrak ulang flight 1 jika sudah ada di staging dengan jumlah banyak
        if "flight 1" in vpath.stem and len(raw_pool["3_defisit_air_tanah"]) >= 2000:
            print(f"Melewati ekstraksi ulang {vpath.name} (sudah dimuat dari staging).")
            continue
        patches = extract_patches_from_video(vpath, step_frames=60, patch_size=224)
        for p in patches:
            cls_name = classify_patch_agronomy(p)
            raw_pool[cls_name].append(p)

    print("\nHasil Klasifikasi Gabungan Seluruh Video Sawah:")
    for c in CLASSES:
        print(f"  {c}: {len(raw_pool[c])} patch")

    # Deduplikasi hash
    print("\nMelakukan deduplikasi patch...")
    clean_pool = {c: [] for c in CLASSES}
    seen_hashes = set()

    for c in CLASSES:
        for patch in raw_pool[c]:
            _, buf = cv2.imencode(".jpg", patch, [cv2.IMWRITE_JPEG_QUALITY, 95])
            h = get_hash(buf.tobytes())
            if h not in seen_hashes:
                seen_hashes.add(h)
                clean_pool[c].append(patch)

    for c in CLASSES:
        print(f"  {c} unik: {len(clean_pool[c])} patch")

    # Target kuota sesuai PDP: 1000 train, 200 val, 75 test per kelas
    target_train = 1000
    target_val = 200
    target_test = 75
    target_total = target_train + target_val + target_test

    balanced_pool = {}
    for c in CLASSES:
        imgs = clean_pool[c]
        random.shuffle(imgs)
        if len(imgs) < target_total:
            print(f"Menyeimbangkan {c} ({len(imgs)} < {target_total}) dengan augmentasi rotasi/flip...")
            augmented = list(imgs)
            while len(augmented) < target_total:
                base = random.choice(imgs)
                aug = base.copy()
                if random.random() > 0.5:
                    aug = cv2.flip(aug, 1)
                if random.random() > 0.5:
                    aug = cv2.flip(aug, 0)
                k = random.choice([0, 1, 2, 3])
                if k > 0:
                    aug = np.rot90(aug, k)
                augmented.append(aug)
            balanced_pool[c] = augmented[:target_total]
        else:
            balanced_pool[c] = imgs[:target_total]

    # Simpan ke folder dataset baru
    print(f"\nMenyimpan dataset baru ke: {OUT_DIR}")
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val", "test"]:
        for c in CLASSES:
            (OUT_DIR / split / c).mkdir(parents=True, exist_ok=True)

    for c in CLASSES:
        imgs = balanced_pool[c]
        train_imgs = imgs[:target_train]
        val_imgs = imgs[target_train:target_train+target_val]
        test_imgs = imgs[target_train+target_val:target_total]

        for i, img in enumerate(train_imgs):
            cv2.imwrite(str(OUT_DIR / "train" / c / f"{c}_{i:05d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        for i, img in enumerate(val_imgs):
            cv2.imwrite(str(OUT_DIR / "val" / c / f"{c}_{i:05d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        for i, img in enumerate(test_imgs):
            cv2.imwrite(str(OUT_DIR / "test" / c / f"{c}_{i:05d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])

    # data.yaml
    yaml_content = f"""path: {OUT_DIR.absolute()}
train: train
val: val
test: test

nc: 4
names:
  0: 0_zona_normal
  1: 1_ketidakmerataan_irigasi
  2: 2_defisit_air_kanopi
  3: 3_defisit_air_tanah
"""
    with open(OUT_DIR / "data.yaml", "w") as f:
        f.write(yaml_content)

    print("\n✅ Dataset Selesai Dibuat!")
    for split in ["train", "val", "test"]:
        print(f"Split {split}:")
        for c in CLASSES:
            count = len(list((OUT_DIR / split / c).glob("*.jpg")))
            print(f"  - {c}: {count} gambar")

if __name__ == "__main__":
    main()
