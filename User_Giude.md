## Usage Guide

### Step 1: Segmentation & Mask Extraction
Run `mvb_segmentation.py` to convert raw 5D confocal hyperstacks into segmented multi-channel mask stacks.

```bash
python mvb_segmentation.py --input /path/to/hyperstack.tif --cell_dia 180 --mvb_dia 15
```
Note: If --input is omitted, an interactive file dialog will pop up automatically.

Outputs:
*_GEOMETRIC_NUC.tif: Multi-channel TCYX hyperstack containing:

Channel 0: Outer Cell Boundary Mask

Channel 1: Filtered MVB / Lysosome Mask

Channel 2: Nuclear Mask Region

### Step 2: Trajectory Linking & Spatial Analysis
Run mvb_tracker.py on the output from Step 1 to perform tracking and compute spatial metrics.
```bash
python mvb_tracker.py --input /path/to/hyperstack_GEOMETRIC_NUC.tif --pixel_size 0.0962 --frame_rate 30.0
```
##Experimental Methods ContextThe parameters provided in this repository default to standard neuroblastoma live-cell imaging setups:
Cell Line: Human neuroblastoma SH-SY5Y cells.
Image Acquisition: 63× objective (2.5× optical zoom), field of view $98.41 \times 98.41\ \mu\text{m}$, pixel size $96.20 \times 96.20\text{ nm}$.
Time-Lapse Settings: 30-minute total acquisition duration with frames acquired every 30 seconds (60 frames total).
