#!/usr/bin/env python3
"""
01_process_input_functions.py

Process raw input function TSV files (aorta + plasma) into interpolated
input functions for FUR calculation.

Input:  RawData/InputFunctions/sub-XXX_ses-{Baseline|Followup}_desc-IF_tacs.tsv
Output: DerivedData/input_functions/sub-XXX_ses-{timepoint}_if_processed.csv

The FUR calculation requires AUC from time 0 to mid-scan (~45 min post-injection).
This script:
1. Concatenates aorta (early) and plasma (late) measurements
2. Interpolates linearly through the gap between them
3. Saves processed input function with timing metadata
4. Creates QC plots showing the input function curve
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# Add Scripts directory to path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

from config import (
    RAWDATA_DIR, DERIVED_DIR, QC_DIR, LOGNOTES_DIR,
    ensure_directories
)
from utils.io_utils import load_blinding_key, find_pet_file

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
INPUT_FUNC_RAW_DIR = RAWDATA_DIR / "InputFunctions"
INPUT_FUNC_OUT_DIR = DERIVED_DIR / "input_functions"
PLASMA_QC_DIR = QC_DIR / "plasma"
UPDATED_JSON_DIR = RAWDATA_DIR / "json_side_cars_updated"  # Corrected JSON sidecars


def load_pet_json(subject_id: str, timepoint: str, session_id: str = None) -> dict:
    """Load PET JSON sidecar for scan timing.

    Priority:
    1. Updated JSON from json_side_cars_updated/ (preferred, has correct timing)
    2. Fallback to JSON alongside PET NIfTI file (may have incorrect ScanStart)
    """
    # Try updated JSON first (uses timepoint in filename)
    updated_json = UPDATED_JSON_DIR / f"{subject_id}_ses-{timepoint}_trc-18FFDG_rec-StaticMoCo_chunk-1_pet.json"
    if updated_json.exists():
        logger.info(f"    Using updated JSON: {updated_json.name}")
        with open(updated_json) as f:
            return json.load(f)

    # Fallback to original JSON alongside PET file
    if session_id:
        session_dir = RAWDATA_DIR / subject_id / session_id
        pet_file = find_pet_file(session_dir)
        if pet_file:
            json_file = pet_file.with_suffix('.json')
            if json_file.exists():
                logger.warning(f"    Using original JSON (may have incorrect timing): {json_file.name}")
                with open(json_file) as f:
                    return json.load(f)

    logger.warning(f"    No JSON sidecar found for {subject_id} {timepoint}")
    return {}


def get_scan_timing(pet_json: dict) -> dict:
    """Extract scan timing parameters from PET JSON.

    Returns:
        dict with scan_start_s, scan_end_s, tissue_time_s (mid-scan)

    Note: Some PET JSONs have ScanStart=0 which is incorrect (scan starts ~30 min
    post-injection). We use 1800s as default when ScanStart is 0 or missing.
    """
    scan_start = pet_json.get('ScanStart', 1800)  # Default 30 min

    # Handle incorrect ScanStart=0 (should be ~1800s for 30-min post-injection start)
    if scan_start == 0:
        logger.warning(f"    ScanStart=0 in JSON, using default 1800s (30 min)")
        scan_start = 1800

    frame_duration_ms = pet_json.get('FrameDuration', [1800000])
    if isinstance(frame_duration_ms, list):
        frame_duration_ms = frame_duration_ms[0]
    frame_duration_s = frame_duration_ms / 1000

    scan_end = scan_start + frame_duration_s
    tissue_time = (scan_start + scan_end) / 2  # Mid-scan for FUR AUC cutoff

    return {
        'scan_start_s': scan_start,
        'scan_end_s': scan_end,
        'tissue_time_s': tissue_time,
        'frame_duration_s': frame_duration_s,
    }


def process_input_function(subject_id: str, session_id: str, timepoint: str,
                           force: bool = False) -> bool:
    """Process raw input function TSV into interpolated CSV.

    Args:
        subject_id: Subject identifier (e.g., 'sub-101')
        session_id: Session identifier (e.g., 'ses-qbimm')
        timepoint: 'Baseline' or 'Followup'
        force: Overwrite existing output

    Returns:
        True if successful, False otherwise
    """
    # Find raw input function file
    raw_file = INPUT_FUNC_RAW_DIR / f"{subject_id}_ses-{timepoint}_desc-IF_tacs.tsv"
    if not raw_file.exists():
        logger.warning(f"  Raw IF not found: {raw_file.name}")
        return False

    # Output file
    out_file = INPUT_FUNC_OUT_DIR / f"{subject_id}_ses-{timepoint}_if_processed.csv"
    if out_file.exists() and not force:
        logger.info(f"  Already exists: {out_file.name}")
        return True

    # Load raw TSV
    raw_df = pd.read_csv(raw_file, sep='\t')

    # Filter to aorta and plasma only (exclude wbl)
    aorta = raw_df[raw_df['ROI'] == 'aorta'][['Time(s)', 'Radioactivity(Bq/mL)']].copy()
    plasma = raw_df[raw_df['ROI'] == 'plasma'][['Time(s)', 'Radioactivity(Bq/mL)']].copy()

    if len(aorta) == 0:
        logger.warning(f"  No aorta data found")
        return False
    if len(plasma) == 0:
        logger.warning(f"  No plasma data found")
        return False

    # Rename columns for consistency
    aorta.columns = ['time_s', 'activity_Bq_mL']
    plasma.columns = ['time_s', 'activity_Bq_mL']

    # Concatenate (if overlap, keep plasma values)
    combined = pd.concat([aorta, plasma]).drop_duplicates(subset='time_s', keep='last')
    combined = combined.sort_values('time_s').reset_index(drop=True)

    # Remove negative times (pre-injection baseline)
    combined = combined[combined['time_s'] >= 0].reset_index(drop=True)

    time_raw = combined['time_s'].values
    activity_raw = combined['activity_Bq_mL'].values

    # Get scan timing from PET JSON (prefer updated JSONs)
    pet_json = load_pet_json(subject_id, timepoint, session_id)
    timing = get_scan_timing(pet_json)

    # Interpolate to 1-second grid
    # Use linear interpolation (handles gap between aorta and plasma)
    interp_func = interp1d(time_raw, activity_raw, kind='linear',
                           bounds_error=False, fill_value=(0, activity_raw[-1]))

    # Create regular time grid from 0 to max available time
    max_time = min(time_raw.max(), timing['scan_end_s'] + 600)  # Include some buffer
    time_interp = np.arange(0, max_time + 1, 1.0)
    activity_interp = interp_func(time_interp)

    # Ensure non-negative values
    activity_interp = np.maximum(activity_interp, 0)

    # Create output dataframe
    out_df = pd.DataFrame({
        'time_s': time_interp,
        'plasma_Bq_mL': activity_interp
    })

    # Save processed input function
    INPUT_FUNC_OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_file, index=False)

    # Compute AUC to mid-scan for QC
    mask = time_interp <= timing['tissue_time_s']
    auc_to_mid = np.trapz(activity_interp[mask], time_interp[mask])

    logger.info(f"  Saved: {out_file.name}")
    logger.info(f"    Aorta points: {len(aorta)}, Plasma points: {len(plasma)}")
    logger.info(f"    Time range: 0 to {time_interp.max():.0f}s")
    logger.info(f"    Mid-scan (AUC cutoff): {timing['tissue_time_s']:.0f}s ({timing['tissue_time_s']/60:.1f} min)")
    logger.info(f"    AUC(0→mid): {auc_to_mid:.2e} Bq·s/mL")

    # Create QC plot
    create_qc_plot(subject_id, session_id, timepoint,
                   time_raw, activity_raw, time_interp, activity_interp,
                   timing, auc_to_mid)

    return True


def create_qc_plot(subject_id: str, session_id: str, timepoint: str,
                   time_raw: np.ndarray, activity_raw: np.ndarray,
                   time_interp: np.ndarray, activity_interp: np.ndarray,
                   timing: dict, auc: float):
    """Create QC plot for input function processing."""
    PLASMA_QC_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Convert to minutes for plotting
    time_raw_min = time_raw / 60
    time_interp_min = time_interp / 60
    scan_start_min = timing['scan_start_s'] / 60
    tissue_time_min = timing['tissue_time_s'] / 60
    scan_end_min = timing['scan_end_s'] / 60

    # Plot raw data points
    ax.scatter(time_raw_min, activity_raw, c='red', s=30, zorder=5,
               label='Raw data (aorta + plasma)', alpha=0.7)

    # Plot interpolated curve
    ax.plot(time_interp_min, activity_interp, 'b-', linewidth=1.5,
            label='Interpolated', alpha=0.8)

    # Shade AUC region (0 to mid-scan)
    mask = time_interp <= timing['tissue_time_s']
    ax.fill_between(time_interp_min[mask], 0, activity_interp[mask],
                    alpha=0.3, color='green', label=f'AUC (0→{tissue_time_min:.1f} min)')

    # Mark key timepoints
    ax.axvline(scan_start_min, color='orange', linestyle='--', linewidth=2,
               label=f'Scan start ({scan_start_min:.1f} min)')
    ax.axvline(tissue_time_min, color='green', linestyle='-', linewidth=2,
               label=f'Mid-scan ({tissue_time_min:.1f} min)')
    ax.axvline(scan_end_min, color='purple', linestyle='--', linewidth=2,
               label=f'Scan end ({scan_end_min:.1f} min)')

    # Labels and formatting
    ax.set_xlabel('Time post-injection (minutes)', fontsize=12)
    ax.set_ylabel('Activity (Bq/mL)', fontsize=12)
    ax.set_title(f'{subject_id} / {session_id} ({timepoint})\n'
                 f'AUC(0→{tissue_time_min:.1f}min) = {auc:.2e} Bq·s/mL', fontsize=12)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_xlim(0, max(70, scan_end_min + 5))
    ax.set_ylim(0, None)
    ax.grid(True, alpha=0.3)

    # Add timing info text box
    textstr = (f'ScanStart: {timing["scan_start_s"]:.0f}s ({scan_start_min:.1f} min)\n'
               f'Duration: {timing["frame_duration_s"]:.0f}s ({timing["frame_duration_s"]/60:.1f} min)\n'
               f'Mid-scan: {timing["tissue_time_s"]:.0f}s ({tissue_time_min:.1f} min)\n'
               f'Scan end: {timing["scan_end_s"]:.0f}s ({scan_end_min:.1f} min)')
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)

    # Save
    qc_file = PLASMA_QC_DIR / f"{subject_id}_{session_id}_input_function.png"
    fig.savefig(qc_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"    QC plot: {qc_file.name}")


def discover_sessions():
    """Discover all available input function files."""
    sessions = []
    blinding_map = load_blinding_key()

    for if_file in sorted(INPUT_FUNC_RAW_DIR.glob("sub-*_ses-*_desc-IF_tacs.tsv")):
        # Parse filename: sub-XXX_ses-{Baseline|Followup}_desc-IF_tacs.tsv
        name = if_file.stem
        parts = name.split('_')
        subject_id = parts[0]
        timepoint = parts[1].replace('ses-', '')  # 'Baseline' or 'Followup'

        # Find session_id from blinding key
        session_id = None
        for (subj, sess), tp in blinding_map.items():
            if subj == subject_id and tp == timepoint:
                session_id = sess
                break

        if session_id:
            sessions.append({
                'subject_id': subject_id,
                'session_id': session_id,
                'timepoint': timepoint,
            })

    return sessions


def main():
    parser = argparse.ArgumentParser(
        description='Process raw input functions for FUR calculation'
    )
    parser.add_argument('--subjects', nargs='+', type=str, default=None,
                        help='Process specific subjects only (e.g., sub-101 sub-102)')
    parser.add_argument('--force', action='store_true',
                        help='Overwrite existing processed files')
    args = parser.parse_args()

    ensure_directories()

    # Setup logging to file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = LOGNOTES_DIR / f"input_function_processing_{timestamp}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    logger.info("=" * 70)
    logger.info("INPUT FUNCTION PROCESSING")
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 70)

    # Discover sessions
    sessions = discover_sessions()

    if args.subjects:
        sessions = [s for s in sessions if s['subject_id'] in args.subjects]

    logger.info(f"Sessions to process: {len(sessions)}")

    success_count = 0
    fail_count = 0

    for sess in sessions:
        subject_id = sess['subject_id']
        session_id = sess['session_id']
        timepoint = sess['timepoint']

        logger.info(f"\n--- {subject_id} / {session_id} ({timepoint}) ---")

        success = process_input_function(subject_id, session_id, timepoint,
                                          force=args.force)
        if success:
            success_count += 1
        else:
            fail_count += 1

    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Total sessions: {len(sessions)}")
    logger.info(f"  Success: {success_count}")
    logger.info(f"  Failed:  {fail_count}")
    logger.info(f"Output: {INPUT_FUNC_OUT_DIR}")
    logger.info(f"QC plots: {PLASMA_QC_DIR}")
    logger.info(f"Log: {log_file}")


if __name__ == "__main__":
    main()
