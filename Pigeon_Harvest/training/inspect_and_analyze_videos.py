#!/usr/bin/env python3
"""
inspect_and_analyze_videos.py
Memeriksa file video hasil download dari Google Drive,
mengambil metadata (FPS, durasi, resolusi), dan menghasilkan cuplikan visual.
"""

import sys
from pathlib import Path
import cv2

VIDEO_DIR = Path("/home/fawwazfa/Program/Harvestmoon/vid/drive_pdp")

def inspect_videos():
    if not VIDEO_DIR.exists():
        print(f"Direktori {VIDEO_DIR} belum ada.")
        return

    videos = list(VIDEO_DIR.rglob("*.mp4")) + list(VIDEO_DIR.rglob("*.mov")) + list(VIDEO_DIR.rglob("*.mkv"))
    print(f"Ditemukan {len(videos)} video di {VIDEO_DIR}:")
    
    for v in sorted(videos):
        cap = cv2.VideoCapture(str(v))
        if not cap.isOpened():
            print(f"  ❌ Gagal membuka {v.name}")
            continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        file_size_mb = v.stat().st_size / (1024 * 1024)
        print(f"  🎬 {v.name}: {w}x{h} @ {fps:.1f} fps, {total_frames} frames ({duration:.1f}s), {file_size_mb:.1f} MB")
        cap.release()

if __name__ == "__main__":
    inspect_videos()
