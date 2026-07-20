# MVB Spatial Tracker: Automated Multivesicular Body Tracking & Spatial Dynamics Pipeline

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![Cellpose](https://img.shields.io/badge/segmentation-Cellpose--SAM-green.svg)](https://github.com/MouseLand/cellpose)
[![Trackpy](https://img.shields.io/badge/tracking-Trackpy%20LAP-orange.svg)](https://soft-matter.github.io/trackpy/)

This repository contains the official Python implementation for automated segmentation, trajectory linking, and spatial dynamics analysis of Multivesicular Bodies (MVBs) in 5D live-cell confocal hyperstacks.

---

## Key Features

- **Dual-Pass Cellpose-SAM Segmentation**: High-precision segmentation of cellular boundaries and punctate vesicular structures.
- **Single-Object Morphological Filtering**: Filters organelle candidates based on cell body bounding rules, cross-sectional area, and circularity.
- **Linear Assignment Problem (LAP) Tracking**: Frame-to-frame particle linking using `trackpy` with disappearance memory tolerance.
- **Dynamic Geometric Analysis**: Dynamic fitting of cell major/minor axes to measure vesicle distances to the cell center and longitudinal poles across time.
- **Kinetic Motion Classification**: Categorizes trajectories into **Processive** vs. **Brownian/Confined** transport profiles based on path straightness and calculates frame-by-frame centrifugal vs. centripetal velocities.
- **Publication-Ready Figures**: Automated generation of high-resolution single-track spaghetti plots, directional charts, and longitudinal kymographs.

---

## Pipeline Overview

```text
5D Hyperstack (T, Z, C, Y, X)
       │
       ▼
[ mvb_segmentation.py ] ──► Dual-pass Cellpose-SAM & Morphological Filters
       │
       ▼
Segmented Stack (*_GEOMETRIC_NUC.tif)
       │
       ▼
[ mvb_tracker.py ]      ──► LAP Linking (trackpy) & Spatial Coordinate Analysis
       │
       ▼
Final Output (*_FINAL_ANALYSIS.csv & High-Res PNG Figures)
