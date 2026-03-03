# EnteroVision: Virtual Endoscopy via CT-based Intestinal Straightening

**[한국어](README_KR.md) | English**

> **Paper:** *EnteroVision: Virtual Endoscopy via CT-based Intestinal Straightening*
>
> A deep learning framework for early diagnosis of intestinal stenosis, adhesion, and obstruction via CT-based 3D segmentation and automated intestinal straightening (Curved Planar Reformation).

---

## Overview

![EnteroVision Conceptual Overview](imgs/fig1.png)

*Fig. 1 — EnteroVision conceptual overview: (Left) Conventional CT shows overlapping intestinal loops. (Center) The AI engine straightens the intestinal tract. (Right) Outputs include a 3D unfolded intestinal map and a localization view for precise surgical site identification.*

---

![EnteroVision Pipeline](imgs/fig2.png)

*Fig. 2 — Full EnteroVision pipeline: (A) Diffusion-augmented fine-tuning using MAISI to address data scarcity. (B) Automated inference pipeline with TotalSegmentator, centerline extraction, and CPR for intestinal straightening.*

---

## Background & Motivation

Diagnosing intestinal stenosis, adhesion, and obstruction currently requires colonoscopy — an invasive, uncomfortable procedure. While abdominal CT is widely used as a supplementary tool, it is difficult to:

- Precisely identify lesion locations in the tangled 3D intestinal structure
- Determine exact surgical sites from conventional CT views alone

**EnteroVision** addresses this by treating CT as the primary diagnostic modality, using deep learning to:
1. Generate scarce CT data (MAISI latent diffusion)
2. Segment the 3D intestinal structure (fine-tuned TotalSegmentator)
3. Straighten the intestine into a linear "virtual endoscopy" view (CPR)
4. Localize lesions and surgical sites with distance-based coordinates

---

## Code Architecture

```
EnteroVision_v002/
├── app_small_bowel.py          # Streamlit app: Small bowel analysis
├── app_colon_analysis.py       # Streamlit app: CT colonography (CPR)
├── src/
│   ├── totalsegmentator_wrapper.py   # TotalSegmentator v2 integration
│   ├── colon_cpr_visualizer.py       # Curved Planar Reformation engine
│   ├── volume_renderer.py            # 3D volume rendering (Plotly)
│   └── ui_logger.py                  # Real-time UI logging
├── imgs/                       # Paper figures
├── requirements.txt
└── datasets/                   # CT data (not tracked by git)
    ├── ct_images/              # Input CT scans (.nii.gz)
    └── ct_labels/              # TotalSegmentator output (auto-generated)
```

### Module Breakdown

#### `src/totalsegmentator_wrapper.py`
- Wraps TotalSegmentator v2 CLI to segment 104 anatomical structures from CT
- Maintains full label mapping (small bowel = 48, colon = 50, etc.)
- Auto-discovers all organs present in a segmentation file
- Caches results in `datasets/ct_labels/` to avoid redundant computation

#### `src/colon_cpr_visualizer.py`
- Implements full **Curved Planar Reformation (CPR)** pipeline:
  1. Mask preprocessing (noise removal, hole filling, erosion)
  2. 3D skeletonization → centerline extraction
  3. Graph-based longest-path ordering (`networkx`)
  4. Spline smoothing (`splprep`/`splev`)
  5. Trilinear interpolation along perpendicular cross-sections
- Outputs interactive Plotly 3D + 2D CPR heatmap

#### `src/volume_renderer.py`
- Marching Cubes surface reconstruction for 3D organ rendering
- Organ-by-organ color mapping and opacity control
- CT slice viewer with organ overlay (axial/sagittal/coronal)
- `create_straightened_view()`: projects intestine to a 2D linear map

#### `app_small_bowel.py`
- End-to-end Streamlit UI for small bowel analysis
- Tabs: 3D Visualization, All Organs 3D, CT Slices, Straightened View, Analysis
- Organ group filtering (digestive, urogenital, respiratory, vascular, skeletal)
- Real-time processing log with sidebar display

#### `app_colon_analysis.py`
- Specialized CT colonography application
- Tabs: 3D Colon View, CPR Analysis, CT Slices, Centerline Analysis, Report
- Curvature statistics, HU value analysis, downloadable report

---

## Pipeline

```
CT Input (.nii.gz)
    │
    ▼
TotalSegmentator v2  ──[MAISI fine-tune for rare lesions]──►  Fine-tuned Model
    │
    ▼
3D Segmentation (small_bowel, colon, + 102 structures)
    │
    ▼
Centerline Extraction (3D skeletonize → graph-based ordering → spline smooth)
    │
    ▼
Curved Planar Reformation (CPR) — trilinear interpolation along centerline
    │
    ▼
Straightened Intestinal Map + Lesion Localization (linear distance coordinate)
```

---

## Installation

```bash
# Python 3.9+ recommended
pip install -r requirements.txt

# GPU acceleration (optional but recommended)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Data Setup

Place CT scans in:
```
EnteroVision_v002/datasets/ct_images/<case>_image.nii.gz
```

---

## Running

```bash
# Small bowel analysis
streamlit run app_small_bowel.py --server.port 8501

# CT colonography (CPR)
streamlit run app_colon_analysis.py --server.port 8502
```

Access at `http://localhost:8501` and `http://localhost:8502`.

### Remote Server (SSH Tunneling)
```bash
ssh -L 18501:localhost:8501 introai16@147.46.121.39
ssh -L 18502:localhost:8502 introai16@147.46.121.39
```
Then access `http://localhost:18501` and `http://localhost:18502`.

---

## Experiments

Experiments are organized under `experiments/` with timestamp-based folders:
```
experiments/
└── YYYYMMDD_HHMMSS_<experiment_name>/
    ├── config.yaml
    ├── logs/
    └── results/
```

---

## Tech Stack

| Component | Technology |
|---|---|
| AI Segmentation | TotalSegmentator v2 (104 structures) |
| Data Augmentation | MAISI (Latent Diffusion) |
| Intestinal Straightening | CPR + 3D Skeletonization |
| 3D Visualization | Plotly + Marching Cubes |
| UI Framework | Streamlit |
| Image Processing | SimpleITK, scikit-image |
| Numerical | NumPy, SciPy |

---

## Limitations & Notes

- **Research use only** — not for clinical diagnosis
- Small bowel segmentation accuracy with TotalSegmentator is limited; MAISI fine-tuning addresses this
- CPR quality depends on centerline extraction quality
- GPU strongly recommended for TotalSegmentator inference

---

## License

Research and educational use. Contact for commercial licensing.
