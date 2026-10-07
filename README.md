# Tugas Take-Home Sesi 05 — Advanced CNNs & Transfer Learning

Studi komparatif strategi transfer learning untuk klasifikasi citra dengan data terbatas (small-data regime), menggunakan subset 10 kelas dari dataset **Oxford-IIIT Pet** dan arsitektur utama **ResNet-18**.

## Ringkasan

Eksperimen membandingkan tiga paradigma pelatihan plus uji kontrol dan arsitektur pembanding:

| Kode | Kondisi | Deskripsi |
|------|---------|-----------|
| B1 | Baseline from Scratch | ResNet-18 `weights=None` (inisialisasi acak) |
| B2 | Feature Extraction | Backbone dibekukan, hanya `fc` dilatih |
| B3 | Progressive Fine-Tuning | Unfreeze `layer4` + `fc`, differential LR (1e-4 / 1e-3) |
| B4 | Uji Kontrol Augmentasi | B3 dengan vs tanpa data augmentation |
| B5 | Arsitektur Pembanding (bonus) | EfficientNet-B0 & MobileNetV3-Small |

## Struktur Repo

```
.
├── Tugas05_TransferLearning.ipynb   # Notebook eksperimen Bagian B + visualisasi Bagian C
├── Laporan_BagianC.md               # Draft laporan analisis (3-5 halaman)
├── requirements.txt                 # Dependensi
├── README.md
├── logs/                            # Dibuat otomatis saat run
│   ├── metrics.csv                  # Log metrik mentah (artefak verifikasi)
│   ├── metrics.json                 # Log metrik + histori kurva loss
│   └── class_distribution.csv       # Distribusi kelas per split
└── figures/                         # Dibuat otomatis saat run
    ├── loss_curves_b1_b2_b3.png
    ├── macro_f1_comparison.png
    ├── failure_analysis.png
    ├── confusion_matrix.png
    └── gradcam_b2_vs_b3.png
```

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

- **Dataset:** Oxford-IIIT Pet, subset 10 kelas (5 ras kucing + 5 ras anjing), ±2.000 citra
- **Split:** stratified 70% train / 15% validation / 15% test
- **Input:** 224×224, normalisasi ImageNet (`mean=[0.485,0.456,0.406]`, `std=[0.229,0.224,0.225]`)
- **Hyperparameter:** batch size 32, optimizer AdamW, `CrossEntropyLoss`, weight decay 1e-2
- **Early stopping:** memonitor validation loss, patience 4
- **Reproduksibilitas:** `seed = 42` (PyTorch, NumPy, Python), cuDNN deterministic

## Catatan Integritas Data

Seluruh klaim performa pada laporan bersumber dari `logs/metrics.csv` dan `logs/metrics.json` yang dihasilkan otomatis oleh notebook. Jangan menyunting angka secara manual — jalankan ulang notebook untuk memperbarui.
