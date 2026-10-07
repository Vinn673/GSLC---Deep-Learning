# Tugas Take-Home Sesi 05 — Advanced CNNs & Transfer Learning

Studi komparatif strategi transfer learning untuk klasifikasi citra dengan data terbatas (small-data regime), menggunakan subset 10 kelas dari dataset **Oxford-IIIT Pet** dan arsitektur utama **ResNet-18**.

## Ringkasan

Eksperimen membandingkan tiga paradigma pelatihan plus uji kontrol dan arsitektur pembanding:

| Kode | Kondisi | Deskripsi |
|------|---------|-----------|
| B1 | Baseline from Scratch | ResNet-18 `weights=None` (inisialisasi acak) |
| B2 | Feature Extraction | Backbone dibekukan, hanya `fc` dilatih |
| B3 | Progressive Fine-Tuning | Unfreeze `layer4` + `fc`, differential LR (1e-4 / 1e-3) |
| B4 | Uji Kontrol Augmentasi | Fine-tuning dengan vs tanpa data augmentation |
| B5 | Arsitektur Pembanding (bonus) | EfficientNet-B0 & MobileNetV3-Small |

## Hasil Utama

Macro F1 pada data uji (sumber: `logs/metrics.csv`):

| Kode | Strategi | Trainable Params | Macro F1 (test) |
|------|----------|-----------------:|----------------:|
| B1 | From Scratch | 11.181.642 | 0.3992 |
| B2 | Feature Extraction | 5.130 | 0.9664 |
| **B3** | **Progressive Fine-Tuning** | **8.398.858** | **0.9733** |
| B4a | Fine-Tuning tanpa Augmentasi | 8.398.858 | 0.9698 |
| B4b | Fine-Tuning dengan Augmentasi | 8.398.858 | 0.9700 |
| B5-EffB0 | EfficientNet-B0 | 424.970 | 0.9664 |
| B5-MobV3 | MobileNetV3-Small | 657.546 | 0.9391 |

Temuan utama: transfer learning jauh mengungguli pelatihan dari nol (selisih ~57 poin persen). Progressive fine-tuning (B3) memberi performa terbaik sekaligus konvergen paling cepat. Model terbaik hanya keliru pada 8 dari 296 citra uji.

## Struktur Repo

```
.
├── Tugas05_TransferLearning.ipynb   # Notebook eksperimen Bagian B + visualisasi Bagian C
├── Tugas05_BagianA.docx             # Jawaban Bagian A (analisis konseptual)
├── Laporan_BagianC.docx             # Laporan analisis Bagian C (3-5 halaman)
├── requirements.txt                 # Dependensi
├── README.md
├── logs/                            # Log metrik mentah (artefak verifikasi)
│   ├── metrics.csv                  # Ringkasan metrik seluruh kondisi
│   ├── metrics.json                 # Metrik + histori kurva loss per epoch
│   └── class_distribution.csv       # Distribusi kelas per split
└── figures/                         # Visualisasi hasil
    ├── loss_curves_b1_b2_b3.png
    ├── macro_f1_comparison.png
    ├── failure_analysis.png
    ├── confusion_matrix.png
    └── gradcam_b2_vs_b3.png
```

> Catatan: folder `data/` (dataset Oxford-IIIT Pet, ±800 MB) tidak disertakan di repo karena diunduh otomatis oleh `torchvision` saat notebook dijalankan.

## Cara Menjalankan

### Google Colab (disarankan — GPU T4 gratis sudah cukup)
1. Upload `Tugas05_TransferLearning.ipynb` ke Colab.
2. Pilih runtime GPU: `Runtime > Change runtime type > T4 GPU`.
3. `Runtime > Run all`. Dataset Oxford-IIIT Pet terunduh otomatis via `torchvision`.

### Lokal
```bash
pip install -r requirements.txt
jupyter notebook Tugas05_TransferLearning.ipynb
```

## Protokol Eksperimen (Terstandar)

- **Dataset:** Oxford-IIIT Pet, subset 10 kelas (5 ras kucing + 5 ras anjing), 1.962 citra
- **Split:** stratified 70% train / 15% validation / 15% test (1.387 / 287 / 296)
- **Input:** 224×224, normalisasi ImageNet (`mean=[0.485,0.456,0.406]`, `std=[0.229,0.224,0.225]`)
- **Hyperparameter:** batch size 32, optimizer AdamW, `CrossEntropyLoss`, weight decay 1e-2
- **Early stopping:** memonitor validation loss, patience 4
- **Reproduksibilitas:** `seed = 42` (PyTorch, NumPy, Python), cuDNN deterministic

## Catatan Integritas Data

Seluruh klaim performa pada laporan bersumber dari `logs/metrics.csv` dan `logs/metrics.json` yang dihasilkan otomatis oleh notebook. Jangan menyunting angka secara manual — jalankan ulang notebook untuk memperbarui.
