#!/usr/bin/env python3
"""
MVB Segmentation Pipeline (Cellpose-SAM)
=======================================
Processes 5D live-cell confocal hyperstacks (T, Z, C, Y, X) to segment
cellular boundaries and Multivesicular Bodies (MVBs) / Lysosomes.

Usage:
    python mvb_segmentation.py --input /path/to/hyperstack.tif
"""

import argparse
import os
import sys
import tkinter as tk
from tkinter import filedialog
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import filters, measure, morphology
import tifffile

try:
    from cellpose import models
except ImportError:
    sys.exit("Error: 'cellpose' is required. Install via 'pip install cellpose'.")


def parse_args():
    parser = argparse.ArgumentParser(description="Segment MVBs and Cell Boundaries from Hyperstack.")
    parser.add_argument("-i", "--input", type=str, help="Path to input 5D TIFF hyperstack")
    parser.add_argument("--cell_dia", type=float, default=180.0, help="Estimated cell diameter in pixels (default: 180)")
    parser.add_argument("--mvb_dia", type=float, default=15.0, help="Estimated MVB diameter in pixels (default: 15)")
    parser.add_argument("--gpu", action="store_true", default=True, help="Use GPU acceleration for Cellpose")
    return parser.parse_args()


def select_file_via_gui():
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    file_path = filedialog.askopenfilename(
        title="Select 5D Confocal Hyperstack",
        filetypes=[("TIFF files", "*.tif *.tiff")]
    )
    return file_path


def preprocess_hyperstack(stack):
    """Extracts middle Z-slices, performs max projection, median filtering, and contrast normalization."""
    # Assuming stack layout: (T, Z, C, Y, X)
    if stack.ndim != 5:
        raise ValueError(f"Expected 5D array (T, Z, C, Y, X), got shape {stack.shape}")

    # Extract middle slices (slices 1 to 3, i.e., indices 1:4) for vesicle channel
    middle_slices = stack[:, 1:4, :, :, :]
    lyso_raw = np.max(middle_slices, axis=1)[:, 0, :, :]

    lyso_enhanced = []
    cell_blurred = []

    for frame in lyso_raw:
        denoised = filters.median(frame, footprint=np.ones((2, 2)))
        p2, p99 = np.percentile(denoised, (2, 99.99))
        
        # Percentile clipping & normalization
        res_lyso = np.clip((denoised - p2) / (p99 - p2 + 1e-8), 0, 1)
        lyso_enhanced.append(res_lyso)
        
        # Blur cytoplasm to generate clean cellular boundary masks
        cell_blurred.append(filters.gaussian(res_lyso, sigma=8))

    return np.array(lyso_enhanced), np.array(cell_blurred)


def run_segmentation(lyso_enhanced, cell_blurred, cell_dia, mvb_dia, use_gpu=True):
    """Executes Cellpose-SAM segmentation and morphological post-processing."""
    print("\n[INFO] Initializing Cellpose-SAM Model ('cpsam')...")
    model = models.CellposeModel(gpu=use_gpu, model_type="cpsam")

    final_cell_masks = []
    final_lyso_masks = []
    final_nuc_masks = []

    num_frames = lyso_enhanced.shape[0]

    for t in range(num_frames):
        # 1. Primary Cell Body Segmentation
        c_masks_raw, _, _ = model.eval(
            cell_blurred[t],
            diameter=cell_dia,
            cellprob_threshold=1.0,
            flow_threshold=0.4
        )

        solid_cell = np.zeros_like(c_masks_raw, dtype=bool)
        nucleus_mask = np.zeros_like(c_masks_raw, dtype=bool)

        if np.max(c_masks_raw) > 0:
            binary_c = c_masks_raw > 0
            struct = morphology.disk(6)
            
            # Morphological closing & hole filling
            dilated = morphology.binary_dilation(binary_c, struct)
            solid_filled = ndi.binary_fill_holes(dilated)
            solid_cell = morphology.binary_erosion(solid_filled, struct)

            # Derive geometric nuclear mask region
            nuc_raw = (solid_cell ^ binary_c) & solid_cell
            nucleus_mask = morphology.remove_small_objects(nuc_raw, min_size=300)

        # 2. MVB / Organelle Segmentation
        l_masks, _, _ = model.eval(
            lyso_enhanced[t],
            diameter=mvb_dia,
            cellprob_threshold=-4.0,
            flow_threshold=0.0
        )

        # 3. Morphological Filtering
        cleaned_lyso = np.zeros_like(l_masks, dtype=np.uint16)
        if np.max(solid_cell) > 0:
            l_props = measure.regionprops(l_masks.astype(int))
            for prop in l_props:
                y, x = map(int, prop.centroid)
                # Keep puncta inside cell body and outside central nuclear region
                if solid_cell[y, x] and not nucleus_mask[y, x]:
                    circ = (4 * np.pi * prop.area) / (prop.perimeter ** 2) if prop.perimeter > 0 else 0
                    if (10 <= prop.area <= 350) and (0.8 <= circ <= 1.0):
                        cleaned_lyso[l_masks == prop.label] = 1

        final_cell_masks.append(solid_cell.astype(np.uint16))
        final_lyso_masks.append(cleaned_lyso)
        final_nuc_masks.append(nucleus_mask.astype(np.uint16))

        print(f" -> Processed Frame [{t+1:02d}/{num_frames:02d}]")

    output_stack = np.stack([
        np.array(final_cell_masks),
        np.array(final_lyso_masks),
        np.array(final_nuc_masks)
    ], axis=1).astype(np.uint16)

    return output_stack


def main():
    args = parse_args()
    file_path = args.input

    if not file_path:
        file_path = select_file_via_gui()

    if not file_path or not os.path.exists(file_path):
        sys.exit("No valid input file provided. Exiting.")

    print(f"[INFO] Loading hyperstack: {file_path}")
    stack = tifffile.imread(file_path)

    lyso_enhanced, cell_blurred = preprocess_hyperstack(stack)
    segmented_stack = run_segmentation(
        lyso_enhanced, cell_blurred, 
        cell_dia=args.cell_dia, 
        mvb_dia=args.mvb_dia, 
        use_gpu=args.gpu
    )

    save_path = os.path.splitext(file_path)[0] + "_GEOMETRIC_NUC.tif"
    tifffile.imwrite(save_path, segmented_stack, imagej=True, metadata={'axes': 'TCYX'})
    print(f"\n[SUCCESS] Processed hyperstack saved to: {save_path}")


if __name__ == "__main__":
    main()