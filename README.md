# A-eye — Detecting AI-Generated Videos with Frequency-Domain Features

**IE 643: Deep Learning — Theory and Practice | IIT Bombay | Course Project (Team A-eye)**

A-eye classifies a video as **real** or **AI-generated**. Instead of feeding raw frames to a large network, it extracts a compact set of **frequency-domain, wavelet, chroma and temporal features** from each video and classifies them with a small **multi-layer perceptron (MLP)**. The idea is that video generators leave characteristic traces in the spatial spectrum, in the high-frequency residual, and in how the spectrum changes from frame to frame, even when the frames look realistic to the eye.

The whole pipeline (feature extraction → training → evaluation → interpretability → external testing → single-video prediction) is contained in one Google Colab notebook.

---

## Table of Contents
1. [Repository Structure](#repository-structure)
2. [Dataset (Google Drive)](#dataset-google-drive)
3. [Quick Start (Google Colab)](#quick-start-google-colab)
4. [Method](#method)
5. [Results](#results)
6. [Interpretability](#interpretability)
7. [Classifying Your Own Video](#classifying-your-own-video)
8. [Configuration Reference](#configuration-reference)
9. [Outputs Written to Drive](#outputs-written-to-drive)
10. [Limitations](#limitations)
11. [Team](#team)

---

## Repository Structure

```
A-eye/
├── ie643_optimized_data_sep.ipynb   # Complete pipeline (Colab notebook)
└── README.md                        # This file
```

The videos, cached features, trained model and result figures are **not stored in this repository** because of their size. They are shared through Google Drive (next section).

---

## Dataset (Google Drive)

📁 **Data folder:** [`Data_1` on Google Drive](https://drive.google.com/drive/folders/1lkY4ZFNJIdxDjl99WU7Dr3xzH2BoqQbB?usp=drive_link)

### Folder layout



```
Data_1/
├── Real/                     # Real videos, one sub-folder per source
│   ├── Camera/
│   ├── Compress/
│   └── net/
├── Fake/                     # AI-generated videos, one sub-folder per generator
│   ├── crafter/    ├── gen2/        ├── hotshot/
│   ├── lavie/      ├── modelscope/  ├── moonvalley/
│   ├── morph/      ├── reduced_videos/
│   ├── self/       ├── sora/        └── veo3/
├── Test/                     # Independent external test set
│   ├── Real/
│   └── Fake/
├── Test_compress/            # External test set with compressed videos
│   ├── Real/
│   └── Fake/
├── feature_cache_v4/         # Pre-computed feature files (.npz) + saved split
├── models_v4/                # Trained checkpoint (mlp_detector_v4.pt)
└── results_v4/               # Metrics, plots, prediction CSVs, interpretability figures
```

Supported video formats: `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`. Folder names `Real`/`Fake` are matched case-insensitively, and nested sub-folders are allowed.

### Training data summary

| Class | Videos |
|---|---|
| Real | 1187 |
| Fake | 1193 |
| **Total** | **2380** |

Per-source counts in the source-aware split:

| Source | Train | Val | Test | Total |
|---|---|---|---|---|
| Fake/crafter | 71 | 15 | 15 | 101 |
| Fake/gen2 | 87 | 19 | 19 | 125 |
| Fake/hotshot | 69 | 15 | 15 | 99 |
| Fake/lavie | 78 | 17 | 17 | 112 |
| Fake/modelscope | 78 | 16 | 16 | 110 |
| Fake/moonvalley | 118 | 25 | 25 | 168 |
| Fake/morph | 85 | 18 | 18 | 121 |
| Fake/reduced_videos | 62 | 14 | 14 | 90 |
| Fake/self | 45 | 9 | 9 | 63 |
| Fake/sora | 71 | 15 | 15 | 101 |
| Fake/veo3 | 73 | 15 | 15 | 103 |
| Real/Camera | 40 | 8 | 8 | 56 |
| Real/Compress | 82 | 17 | 17 | 116 |
| Real/net | 711 | 152 | 152 | 1015 |
| **All** | **1670** | **355** | **355** | **2380** |

External test sets: `Test` (201 videos: 97 real, 104 fake) and `Test_compress` (107 videos: 52 real, 55 fake).

### Getting the data into your Colab session

Colab can only see folders that are inside **your own** "My Drive". The notebook also **writes** caches, models and results into `Data_1`, so it needs write access. The simplest option is:

1. Open the Drive link above.
2. Download the `Data_1` folder, or copy it into your own Drive, so that it sits at:
   ```
   My Drive/Data_1
   ```
3. If you place it anywhere else, change `DRIVE_ROOT` in Section 3 of the notebook:
   ```python
   DRIVE_ROOT = Path('/content/drive/MyDrive/<your-path>/Data_1')
   ```

> **Tip:** If you only want to reproduce the results, you do **not** need the raw videos. The folders `feature_cache_v4/` and `models_v4/` are enough, because the notebook loads the cached features by default and never decodes a video. The raw videos are only needed to re-extract features (`REBUILD_FEATURES = True`), for the 2-D spectrum figure (Section 16D), and for the misclassified-video paths.

---

## Quick Start (Google Colab)

1. Open `ie643_optimized_data_sep.ipynb` in Google Colab (*File → Upload notebook*, or open it directly from GitHub).
2. Make sure `Data_1` is in your Drive as described above.
3. Run the cells from top to bottom:
   - **Section 1** mounts Google Drive (authorise when prompted).
   - **Section 2** installs `PyWavelets` and imports the libraries.
   - **Section 3** sets all paths and settings.
4. With the default settings (`REBUILD_FEATURES = False`, `REBUILD_EXTERNAL_FEATURES = False`, `REBUILD_SPLIT = False`), the notebook:
   - loads the cached features and the saved train/val/test split,
   - trains the MLP (takes about a minute; a GPU is optional),
   - evaluates on the internal test set and on both external test sets,
   - produces all interpretability figures.

### Dependencies

Colab already provides everything except `PyWavelets`, which the notebook installs itself. To run locally:

```bash
pip install torch numpy pandas matplotlib scikit-learn opencv-python PyWavelets tqdm
```

To run outside Colab, also remove the `google.colab` imports (Sections 1 and 18) and point `DRIVE_ROOT` to a local copy of `Data_1`.

---

## Method

### 1. Temporal sampling and spatial crops
Each video is split into **4 evenly spaced temporal segments** of **8 consecutive frames**, so the detector does not depend only on the start of a clip. From each segment, up to **5 crops of 256 × 256** pixels (centre + four offset crops) are taken. Frames smaller than 256 pixels are upscaled first.

### 2. Per-crop features (767 values)

| Feature group | Description | Size |
|---|---|---|
| Radial FFT profile | Hann-windowed 2-D FFT log-power, averaged in 64 radial bins: mean, std and frame-to-frame drift across frames | 192 |
| Frame-difference FFT | Radial spectrum of the difference between consecutive frames (temporal change) | 64 |
| Non-radial 2-D FFT statistics | Low/mid/high band energies, spectral centroid, entropy, horizontal–vertical anisotropy, quadrant energies, HF/LF ratio (mean, std, drift) | 33 |
| High-frequency residual | Frame minus Gaussian-blurred frame (σ = 1): radial spectrum (mean, std, drift) + RMS/std/abs-mean statistics | 198 |
| Wavelet energies | `db2` wavelet, 2 levels: H/V/D detail-band energies (mean, std, drift) | 18 |
| Chroma spectra | Radial spectra of the Cr and Cb channels (YCrCb): mean and std | 256 |
| Temporal scalars | Spectral change between frames (mean, std, max) and HF−LF energy statistics | 6 |

### 3. Aggregation → 3068-D video descriptor
- Across the crops of a segment: **mean and std** (767 × 2 = 1534).
- Across the 4 segments: **mean and std** (1534 × 2 = **3068**).

Keeping the standard deviations retains the spatial and temporal variability, not just the average spectrum.

### 4. Source-aware split
Videos are split into **70% train / 15% validation / 15% test inside each source/generator folder**. Every real source and every generator is therefore represented in all three splits. The split is saved and reused as long as the list of videos is unchanged. Feature normalisation (z-score) uses **training-set statistics only**.

### 5. Classifier — `FreqMLP`

```
3068 → [Linear 256 → BatchNorm → ReLU → Dropout 0.3]
     → [Linear 128 → BatchNorm → ReLU → Dropout 0.3]
     → [Linear  64 → BatchNorm → ReLU → Dropout 0.3]
     → Linear 2   (real / fake)
```
Trainable parameters: **827,842**.

| Training setting | Value |
|---|---|
| Loss | Class-weighted cross-entropy |
| Optimiser | AdamW (lr = 2 × 10⁻⁴, weight decay = 10⁻⁴) |
| Scheduler | ReduceLROnPlateau (factor 0.5, patience 5) |
| Batch size | 32 |
| Max epochs / early stopping | 100 / patience 15 on validation loss (best weights restored) |
| Random seed | 42 |

### 6. Decision threshold
The threshold on P(fake) is chosen to **maximise F1 on the validation set only** (search over 0.10–0.90). The selected value is **0.63**. The internal test set and the external folders are never used for this choice.

---



## Configuration Reference

All settings are in **Section 3** of the notebook.

| Setting | Default | Meaning |
|---|---|---|
| `DRIVE_ROOT` | `/content/drive/MyDrive/Data_1` | Root data folder |
| `REBUILD_FEATURES` | `False` | `True` re-extracts features from all training videos and overwrites the cache (slow) |
| `REBUILD_EXTERNAL_FEATURES` | `False` | Same, for the `Test` / `Test_compress` folders |
| `REBUILD_SPLIT` | `False` | `True` forces a new train/val/test split |
| `NUM_SEGMENTS`, `FRAMES_PER_SEGMENT`, `FRAME_STRIDE` | 4, 8, 1 | Temporal sampling |
| `CROP_SIZE`, `N_CROPS` | 256, 5 | Spatial crops |
| `NUM_BINS` | 64 | Radial frequency bins |
| `HIDDEN`, `DROPOUT` | (256, 128, 64), 0.30 | MLP architecture |
| `LR`, `WEIGHT_DECAY`, `EPOCHS`, `PATIENCE` | 2e-4, 1e-4, 100, 15 | Training |
| `SEED`, `TEST_SIZE`, `VAL_SIZE`, `BATCH_SIZE` | 42, 0.15, 0.15, 32 | Split and batching |

> ⚠️ If you change any feature-extraction setting (segments, crops, bins), the cached features no longer match. Set `REBUILD_FEATURES = True` and `REBUILD_EXTERNAL_FEATURES = True` to regenerate them. The checkpoint stores the feature version and is rejected if it does not match.

---

## Outputs Written to Drive

| File | Location |
|---|---|
| Training features + metadata | `feature_cache_v4/features_v4.npz`, `features_v4_metadata.json` |
| Saved split | `feature_cache_v4/train_val_test_split_v4.npz` |
| External-test features | `feature_cache_v4/external_test_features_v1.npz`, `external_test_compress_features_v4.npz` |
| Trained model (weights + normalisation + threshold + config) | `models_v4/mlp_detector_v4.pt` |
| Test metrics | `results_v4/metrics_v4.json` |
| Training curves, confusion matrix, ROC curve | `results_v4/training_curves_v4.png`, `confusion_matrix_v4.png`, `roc_curve_v4.png` |
| Misclassified test videos | `results_v4/misclassified_test_v4.csv` |
| External-test predictions and confusion matrices | `results_v4/external_folder_*_predictions.csv`, `*_confusion_matrix.png` |
| Summary of all evaluations | `results_v4/evaluation_summary_v4.csv` |
| Interpretability figures and tables | `results_v4/interpretability_v4_top6/` |



*This project was developed for academic purposes as part of the IE 643 course at IIT Bombay.*
