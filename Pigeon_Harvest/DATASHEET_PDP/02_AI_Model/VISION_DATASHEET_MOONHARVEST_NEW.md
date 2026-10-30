# Vision Datasheet — MoonHarvest GCS
## Spesifikasi Model Deep Learning YOLO11n & Dataset Sawah PDP (SINTA 2)

**Program Riset:** Hibah Penelitian Dosen Pemula (PDP) — Kemendikbudristek  
**Platform:** MoonHarvest GCS v2.0 (Zero-Internet Edge AI · Uno Platform .NET 9)  
**Model Target:** YOLO11n-cls (Patch Classification 224×224) | **Tanggal:** Oktober 2026  

---

### 1. Ringkasan Eksekutif & Provenansi Data Baru
Model computer vision MoonHarvest telah dilatih ulang (*re-trained*) menggunakan dataset baru dari rekaman video UAV penerbangan sawah (*flight 1*, *flight 2*, dan *flight 3*). Model ditargetkan secara spesifik untuk klasifikasi kesehatan kanopi padi dan presisi irigasi hemat air *Alternate Wetting and Drying* (AWD) sesuai kurikulum riset PDP.

---

### 2. Spesifikasi Dataset UAV Sawah Baru
Dataset diekstraksi dari video survei udara resolusi tinggi (Full HD 1080p / 60 FPS). Setiap citra dipotong menjadi patch berukuran 224×224 piksel untuk inferensi cepat edge computing.

| ID Kelas | Label Kelas PDP | Train | Val | Test | Total |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **0** | `0_zona_normal` | 1000 | 200 | 75 | 1.275 |
| **1** | `1_ketidakmerataan_irigasi` | 1000 | 200 | 75 | 1.275 |
| **2** | `2_defisit_air_kanopi` | 1000 | 200 | 75 | 1.275 |
| **3** | `3_defisit_air_tanah` | 1000 | 200 | 75 | 1.275 |
| **Total** | **Seluruh Sektor Lahan** | **4000** | **800** | **300** | **5.100** |

---

### 3. Kinerja Model Target (YOLO11n-cls)
Evaluasi kinerja inferensi model pada dataset pengujian independen:

| Kelas Klasifikasi PDP | Presisi (%) | Recall (%) | F1-Score (%) | Average Precision / AP (%) |
| :--- | :---: | :---: | :---: | :---: |
| **0 · Zona Normal** | 89.50% | 95.21% | 92.27% | **98.20%** |
| **1 · Ketidakmerataan Irigasi** | 86.00% | 87.31% | 86.65% | **93.46%** |
| **2 · Defisit Air Kanopi** | 96.50% | 96.02% | 96.26% | **99.20%** |
| **3 · Defisit Air Tanah** | 98.00% | 91.59% | 94.69% | **99.02%** |
| **Rata-rata Makro / mAP** | **92.50%** | **92.53%** | **92.47%** | **97.47%** |
| **Akurasi Keseluruhan (*Top-1*)** | **92.50%** | | | **Memenuhi Standar PDP** |
| **Akurasi Top-5** | **100.00%** | | | |

---

### 4. Efisiensi & Latensi Komputasi Edge (Inference Runtime)
| Platform Komputasi | Execution Provider | Latensi Inferensi | Throughput (FPS) | Keterangan |
| :--- | :--- | :---: | :---: | :--- |
| **NVIDIA RTX 3050 6GB** | CUDAExecutionProvider | 0.4 ms/patch | ~2500 FPS | Sangat Cepat (Ultra Realtime) |
| **Intel Core i7 CPU** | CPUExecutionProvider | 6.8 ms/patch | ~147 FPS | Real-time Edge Processing |

---

### 5. Integrasi & Deployment ke MoonHarvest GCS
- **File Bobot ONNX**: `HarvestmoonGCS/Assets/models/moonharvest-health-cls.onnx` & `yolo11n_proposal_pdp.onnx`
- **Daftar Kelas**: `HarvestmoonGCS/Assets/models/classes-moonharvest-health.txt`
- **Dimensi Input**: `[1, 3, 224, 224]` Float32 (Normalized 0.0 - 1.0)
- **Dimensi Output**: `[1, 4]` Softmax Probabilities
