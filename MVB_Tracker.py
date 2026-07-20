#!/usr/bin/env python3
"""
MVB Trajectory Linking & Spatial Coordinates Analysis Pipeline
==============================================================
Links segmented MVB centroids across frames, computes radial dynamics relative
to cell center and poles, and classifies trajectory kinetics.

Usage:
    python mvb_tracker.py --input /path/to/segmented_stack.tif
"""

import argparse
import os
import sys
import tkinter as tk
from tkinter import filedialog

import numpy as np
import pandas as pd
import tifffile
import trackpy as tp
from skimage import measure

import matplotlib.pyplot as plt
import seaborn as sns

# Configurable defaults matching experimental setup
DEFAULT_PIXEL_SIZE = 0.0962  # µm per pixel (96.20 nm)
DEFAULT_FRAME_RATE = 30.0    # Seconds per frame
DEFAULT_SEARCH_RANGE = 20    # Pixels
DEFAULT_MEMORY = 3           # Frames
DEFAULT_MIN_FRAMES = 3       # Minimum trajectory duration


def setup_publication_theme():
    """Sets publication-standard global typography and styling."""
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Liberation Sans', 'DejaVu Sans']
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12


def parse_args():
    parser = argparse.ArgumentParser(description="MVB Trajectory Tracking & Spatial Analysis.")
    parser.add_argument("-i", "--input", type=str, help="Path to segmented TIFF stack")
    parser.add_argument("--pixel_size", type=float, default=DEFAULT_PIXEL_SIZE, help="Microns per pixel")
    parser.add_argument("--frame_rate", type=float, default=DEFAULT_FRAME_RATE, help="Time interval between frames (s)")
    parser.add_argument("--search_range", type=int, default=DEFAULT_SEARCH_RANGE, help="Max particle displacement (pixels)")
    parser.add_argument("--memory", type=int, default=DEFAULT_MEMORY, help="Disappearance frame tolerance")
    parser.add_argument("--min_frames", type=int, default=DEFAULT_MIN_FRAMES, help="Min track length filter")
    return parser.parse_args()


def select_file_via_gui():
    root = tk.Tk()
    root.withdraw()
    return filedialog.askopenfilename(
        title="Select Segmented TIF Stack", 
        filetypes=[("TIFF Files", "*.tif *.tiff")]
    )


def extract_features(mvb_masks, pixel_size):
    """Extracts vesicle centroids and bounding geometries frame-by-frame."""
    frames_data = []
    for t in range(mvb_masks.shape[0]):
        labeled = measure.label(mvb_masks[t] > 0)
        props = measure.regionprops_table(
            labeled, 
            properties=('centroid', 'area', 'major_axis_length', 'minor_axis_length')
        )
        df = pd.DataFrame(props)
        if df.empty:
            continue
        df['frame'] = t
        df['major_axis_um'] = df['major_axis_length'] * pixel_size
        df['minor_axis_um'] = df['minor_axis_length'] * pixel_size
        df = df.rename(columns={'centroid-0': 'y', 'centroid-1': 'x'})
        frames_data.append(df)
    
    return pd.concat(frames_data, ignore_index=True) if frames_data else pd.DataFrame()


def compute_spatial_dynamics(tracks, cell_masks, pixel_size, frame_rate):
    """Calculates distances to cell center and longitudinal poles across time."""
    dist_center, dist_pole_a, dist_pole_b = [], [], []

    for t in range(cell_masks.shape[0]):
        cell_label = measure.label(cell_masks[t] > 0)
        props = measure.regionprops(cell_label)
        if not props:
            continue
        main_cell = max(props, key=lambda x: x.area)

        cy, cx = main_cell.centroid
        orientation = main_cell.orientation
        halflength = main_cell.major_axis_length / 2.0

        p1_y = cy - halflength * np.cos(orientation)
        p1_x = cx + halflength * np.sin(orientation)
        p2_y = cy + halflength * np.cos(orientation)
        p2_x = cx - halflength * np.sin(orientation)

        f_tracks = tracks[tracks['frame'] == t]
        for _, row in f_tracks.iterrows():
            my, mx = row['y'], row['x']
            d_center = np.sqrt((mx - cx) ** 2 + (my - cy) ** 2) * pixel_size
            d_p1 = np.sqrt((mx - p1_x) ** 2 + (my - p1_y) ** 2) * pixel_size
            d_p2 = np.sqrt((mx - p2_x) ** 2 + (my - p2_y) ** 2) * pixel_size

            dist_center.append(d_center)
            dist_pole_a.append(d_p1)
            dist_pole_b.append(d_p2)

    tracks['dist_to_center'] = dist_center
    tracks['dist_to_pole_1'] = dist_pole_a
    tracks['dist_to_pole_2'] = dist_pole_b

    # Radial velocity calculation (dd_center / dt)
    tracks['radial_velocity'] = tracks.groupby('particle')['dist_to_center'].diff() / frame_rate
    tracks['motion_type'] = tracks['radial_velocity'].apply(
        lambda v: "Stationary" if pd.isna(v) else ("Centrifugal" if v > 0 else "Centripetal")
    )
    return tracks


def classify_trajectories(tracks):
    """Calculates trajectory straightness index and classifies movement profiles."""
    classification_results = []
    for p_id, group in tracks.groupby('particle'):
        if len(group) < 2:
            continue
        path_length = group['velocity'].sum()
        start_pos = np.array([group['x'].iloc[0], group['y'].iloc[0]])
        end_pos = np.array([group['x'].iloc[-1], group['y'].iloc[-1]])
        net_disp = np.linalg.norm(end_pos - start_pos)
        
        straightness = (net_disp / path_length) if path_length > 0 else 0.0
        motion_class = "Processive" if straightness > 0.5 else "Brownian/Confined"

        classification_results.append({
            'particle': p_id,
            'straightness': straightness,
            'classification': motion_class
        })

    class_df = pd.DataFrame(classification_results)
    return tracks.merge(class_df, on='particle', how='left'), class_df


def plot_spaghetti_distance(data, base_path):
    """Plots distance to cell center across time."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.set_style("ticks")

    sns.lineplot(
        data=data, x='frame', y='dist_to_center', hue='particle', 
        palette='turbo', legend=None, alpha=0.5, lw=2.0, ax=ax
    )
    sns.lineplot(data=data, x='frame', y='dist_to_center', color='white', lw=5, errorbar=None, zorder=9, ax=ax)
    sns.lineplot(data=data, x='frame', y='dist_to_center', color='#111111', lw=2.5, label='Population Mean', errorbar=None, zorder=10, ax=ax)

    ax.set_title("Distance to Cell Center Over Time", fontweight='bold')
    ax.set_xlabel("Time (Frames)")
    ax.set_ylabel("Distance to Center (μm)")
    sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(base_path + "_Center_Distance.png", dpi=300)
    plt.close()


def generate_kymograph(tracks, cell_masks, pixel_size, base_path):
    """Generates longitudinal kymograph projected along major cell axis."""
    kymograph_data = []
    for t in range(cell_masks.shape[0]):
        cell_label = measure.label(cell_masks[t] > 0)
        props = measure.regionprops(cell_label)
        if not props:
            continue
        main_cell = max(props, key=lambda x: x.area)

        cy, cx = main_cell.centroid
        orientation = main_cell.orientation
        v_axis = np.array([np.sin(orientation), -np.cos(orientation)])

        f_tracks = tracks[tracks['frame'] == t]
        for _, row in f_tracks.iterrows():
            v_mvb = np.array([row['x'] - cx, row['y'] - cy])
            proj_dist = np.dot(v_mvb, v_axis) * pixel_size
            kymograph_data.append({'frame': t, 'long_pos': proj_dist, 'particle': row['particle']})

    df_kymo = pd.DataFrame(kymograph_data)
    if df_kymo.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.set_style("ticks")

    unique_particles = df_kymo['particle'].unique()
    base_cmap = plt.cm.get_cmap('tab20', len(unique_particles))

    for i, (p_id, group) in enumerate(df_kymo.groupby('particle')):
        group = group.sort_values('frame')
        color = base_cmap(i % 20)
        ax.plot(group['long_pos'], group['frame'], color=color, lw=2.5, alpha=0.85)

    ax.axvline(0, color='#FF3333', linestyle='--', lw=2.0, label='Cell Center')
    ax.gca().invert_yaxis()
    ax.set_xlabel("Position along Major Axis (μm)", fontweight='bold')
    ax.set_ylabel("Time (Frames)", fontweight='bold')
    ax.set_title("Longitudinal Kymograph", fontweight='bold')
    sns.despine(ax=ax)
    plt.legend(loc='upper left', frameon=False)
    plt.tight_layout()
    plt.savefig(base_path + "_Longitudinal_Kymograph.png", dpi=300)
    plt.close()


def main():
    setup_publication_theme()
    args = parse_args()
    file_path = args.input or select_file_via_gui()

    if not file_path or not os.path.exists(file_path):
        sys.exit("No valid file provided. Exiting.")

    print(f"[INFO] Loading segmented file: {file_path}")
    stack = tifffile.imread(file_path)
    
    cell_masks = stack[:, 0]
    mvb_masks = stack[:, 1]

    print("[INFO] Extracting feature centroids...")
    all_features = extract_features(mvb_masks, args.pixel_size)
    if all_features.empty:
        sys.exit("No valid objects detected for tracking.")

    print("[INFO] Linking trajectories (trackpy)...")
    tracks = tp.link_df(
        all_features, search_range=args.search_range, 
        memory=args.memory, adaptive_stop=10
    )
    tracks = tp.filter_stubs(tracks, args.min_frames).reset_index(drop=True)
    tracks = tracks.sort_values(['particle', 'frame'])

    # Compute step velocity
    tracks['velocity'] = np.sqrt(
        tracks.groupby('particle')['x'].diff() ** 2 + 
        tracks.groupby('particle')['y'].diff() ** 2
    )

    print("[INFO] Calculating polar metrics and radial velocities...")
    tracks = compute_spatial_dynamics(tracks, cell_masks, args.pixel_size, args.frame_rate)

    print("[INFO] Classifying movement dynamics...")
    tracks, class_df = classify_trajectories(tracks)

    base_path = os.path.splitext(file_path)[0]
    out_csv = base_path + "_MVB_FINAL_ANALYSIS.csv"
    tracks.to_csv(out_csv, index=False)
    print(f"[SUCCESS] Exported trajectory statistics to: {out_csv}")

    print("[INFO] Plotting figures...")
    plot_spaghetti_distance(tracks, base_path)
    generate_kymograph(tracks, cell_masks, args.pixel_size, base_path)
    print("[SUCCESS] Pipeline completed successfully.")


if __name__ == "__main__":
    main()