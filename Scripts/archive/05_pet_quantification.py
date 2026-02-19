#!/usr/bin/env python3
"""
05_pet_quantification.py - PET Metric Extraction Script

This script extracts FDG uptake metrics from peridental ROIs for all sessions.

Metrics extracted:
- SUV (mean, 90th percentile)
- TPR (mean, 90th percentile)
- FUR (mean, 90th percentile)
- Raw intensity (mean, 90th percentile)

Usage:
    cd Periodontal_Analysis/Scripts
    python 05_pet_quantification.py

    # Or with options:
    python 05_pet_quantification.py --subject sub-101
    python 05_pet_quantification.py --force

Prerequisites:
    - 01_segmentation.py
    - 02_roi_generation.py
    - 04_plasma_processing.py

Output:
    Outputs/tooth_level_metrics.csv
    Outputs/quadrant_level_metrics.csv
"""

import argparse
import sys
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

# Add Scripts directory to path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

from config import (
    RAWDATA_DIR, ROI_DIR, OUTPUTS_DIR, DERIVED_DIR, LOGNOTES_DIR,
    QUADRANTS, ALL_UPPER_TEETH, TOOTH_METRICS_FILE, QUADRANT_METRICS_FILE,
    ensure_directories
)
from utils.io_utils import (
    load_blinding_key, load_ecrf_data, load_clinical_ratings,
    get_suv_parameters, discover_subjects, discover_sessions,
    find_pet_file, find_pet_json, load_nifti, get_voxel_volume_ml
)
from utils.pet_utils import (
    validate_pet_units, extract_pet_metrics, extract_metrics_for_tooth,
    load_pet_json as load_pet_timing, resample_roi_to_pet, check_dimensions_match
)
from utils.plasma_utils import load_input_function, process_input_function


def setup_logging(log_dir: Path) -> logging.Logger:
    """Setup logging to file and console."""
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"pet_quantification_log_{timestamp}.txt"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Log file: {log_file}")
    return logger


def load_session_info(subject_id: str, timepoint: str) -> dict:
    """Load processed session info from plasma processing step."""
    session_file = DERIVED_DIR / "session_info.csv"
    if not session_file.exists():
        return {}

    df = pd.read_csv(session_file)
    match = df[(df['subject_id'] == subject_id) & (df['timepoint'] == timepoint)]

    if len(match) == 0:
        return {}

    return match.iloc[0].to_dict()


def process_session(subject_id: str, session_id: str, timepoint: str,
                    suv_params: dict, clinical_df: pd.DataFrame,
                    logger: logging.Logger, force: bool = False) -> dict:
    """
    Extract PET metrics for a single session.
    """
    result = {
        'status': 'failed',
        'subject_id': subject_id,
        'session_id': session_id,
        'timepoint': timepoint,
        'tooth_metrics': [],
        'quadrant_metrics': [],
        'warnings': []
    }

    # Find ROI directory
    roi_dir = ROI_DIR / f"{subject_id}_{session_id}"
    peridental_file = roi_dir / "peridental_soft_tissue_4mm.nii.gz"

    if not peridental_file.exists():
        logger.warning(f"  {session_id}: No ROI found, skipping")
        result['warnings'].append("ROI files not found")
        return result

    # Find PET file
    session_dir = RAWDATA_DIR / subject_id / session_id
    pet_file = find_pet_file(session_dir)

    if pet_file is None:
        logger.warning(f"  {session_id}: No PET file found, skipping")
        result['warnings'].append("PET file not found")
        return result

    logger.info(f"  {session_id} ({timepoint}): Loading PET and ROIs...")

    # Load PET data
    pet_data, pet_img = load_nifti(pet_file)
    voxel_vol_ml = get_voxel_volume_ml(pet_img)

    # Validate PET units
    unit_info = validate_pet_units(pet_data)
    logger.info(f"    PET units: {unit_info['inferred_units']} (max={unit_info['max_value']:.0f})")

    # Get SUV parameters
    weight_kg = suv_params.get('weight_kg')
    dose_mbq = suv_params.get('injected_mbq')

    if weight_kg is None or dose_mbq is None:
        logger.warning(f"    Missing SUV parameters")
        result['warnings'].append("Missing weight or dose")

    # Get plasma data
    session_info = load_session_info(subject_id, timepoint)
    plasma_mean = session_info.get('plasma_mean_Bq_mL', np.nan)
    plasma_auc = session_info.get('plasma_auc_0_to_T_Bq_s_mL', np.nan)

    if np.isnan(plasma_mean):
        logger.warning(f"    No plasma data available")
        result['warnings'].append("No plasma data")

    # Load main peridental ROI
    peridental_roi_data, peridental_roi_img = load_nifti(peridental_file)

    # Check if dimensions match - if not, resample ROI to PET space
    if not check_dimensions_match(pet_data, peridental_roi_data):
        logger.info(f"    ROI shape {peridental_roi_data.shape} != PET shape {pet_data.shape}")
        logger.info(f"    Resampling ROI to PET space...")
        import nibabel as nib
        peridental_roi_nii = nib.Nifti1Image(peridental_roi_data.astype(np.float32),
                                               peridental_roi_img.affine,
                                               peridental_roi_img.header)
        peridental_roi = resample_roi_to_pet(peridental_roi_nii, pet_img)
        peridental_roi = peridental_roi > 0
        # Use PET voxel volume for ROI volume calculation when resampled
        voxel_vol_ml = get_voxel_volume_ml(pet_img)
    else:
        peridental_roi = peridental_roi_data > 0

    # Extract metrics for main ROI
    logger.info(f"    Extracting metrics from peridental ROI...")
    main_metrics = extract_pet_metrics(
        pet_data, peridental_roi, weight_kg, dose_mbq,
        plasma_mean, plasma_auc, unit_info
    )
    main_metrics['roi_volume_mL'] = float(np.sum(peridental_roi) * voxel_vol_ml)

    logger.info(f"    SUV_mean: {main_metrics['SUV_mean']:.3f}, "
                f"SUV_90th: {main_metrics['SUV_90th']:.3f}")
    logger.info(f"    TPR_mean: {main_metrics['TPR_mean']:.3f}, "
                f"FUR_mean: {main_metrics['FUR_mean_per_min']:.4f}")

    if main_metrics['warnings']:
        for w in main_metrics['warnings']:
            logger.warning(f"    {w}")
        result['warnings'].extend(main_metrics['warnings'])

    # Load per-tooth ROIs from combined labeled NIfTI (new format) or individual files (legacy)
    tooth_metrics_list = []
    tooth_labeled_file = roi_dir / "tooth_shells_labeled.nii.gz"
    tooth_dir = roi_dir / "tooth_shells"  # Legacy format

    if tooth_labeled_file.exists():
        # New format: single labeled NIfTI
        import json
        tooth_labeled_data, tooth_labeled_img = load_nifti(tooth_labeled_file)

        # Load lookup to get tooth IDs
        lookup_file = roi_dir / "tooth_shells_lookup.json"
        if lookup_file.exists():
            with open(lookup_file) as f:
                lookup = json.load(f)
            tooth_ids = lookup.get('tooth_ids', [])
        else:
            tooth_ids = [int(x) for x in np.unique(tooth_labeled_data) if x > 0]

        # Filter for valid FDI tooth IDs
        valid_tooth_ids = [tid for tid in tooth_ids if tid in ALL_UPPER_TEETH]

        if valid_tooth_ids:
            logger.info(f"    Processing {len(valid_tooth_ids)} valid tooth ROIs (FDI 11-28)...")

            # Resample labeled volume to PET space if needed
            if not check_dimensions_match(pet_data, tooth_labeled_data):
                logger.info(f"    Resampling tooth labels to PET space...")
                import nibabel as nib
                tooth_labeled_nii = nib.Nifti1Image(tooth_labeled_data.astype(np.float32),
                                                      tooth_labeled_img.affine,
                                                      tooth_labeled_img.header)
                tooth_labeled_resampled = resample_roi_to_pet(tooth_labeled_nii, pet_img)
            else:
                tooth_labeled_resampled = tooth_labeled_data

            for tooth_id in valid_tooth_ids:
                tooth_roi = (tooth_labeled_resampled == tooth_id)

                if np.sum(tooth_roi) == 0:
                    continue

                # Extract metrics
                tooth_metrics = extract_metrics_for_tooth(
                    pet_data, tooth_roi, weight_kg, dose_mbq,
                    plasma_mean, plasma_auc, tooth_id, pet_img
                )

                # Add session info
                tooth_metrics['subject_id'] = subject_id
                tooth_metrics['session_id'] = session_id
                tooth_metrics['timepoint'] = timepoint

                # Get clinical rating if available
                clinical_match = clinical_df[
                    (clinical_df['subject_id'] == subject_id) &
                    (clinical_df['session_id'] == session_id) &
                    (clinical_df['tooth_id'] == tooth_id)
                ]
                if len(clinical_match) > 0:
                    tooth_metrics['clinical_rating'] = clinical_match.iloc[0]['rating']
                    tooth_metrics['clinical_category'] = clinical_match.iloc[0]['rating_category']
                else:
                    tooth_metrics['clinical_rating'] = np.nan
                    tooth_metrics['clinical_category'] = 'unknown'

                tooth_metrics_list.append(tooth_metrics)
        else:
            logger.info(f"    Found {len(tooth_ids)} tooth shells (connected components, not FDI notation)")
            logger.info(f"    Skipping per-tooth analysis - no instance segmentation available")

        logger.info(f"    Extracted metrics for {len(tooth_metrics_list)} teeth")

    elif tooth_dir.exists():
        # Legacy format: individual files
        tooth_files = list(tooth_dir.glob("tooth_*.nii.gz"))
        logger.info(f"    Found {len(tooth_files)} tooth shells (legacy format, connected components)")
        logger.info(f"    Skipping per-tooth analysis - no FDI instance segmentation available")

    result['tooth_metrics'] = tooth_metrics_list

    # Calculate quadrant-level metrics
    quadrant_metrics_list = []

    for quadrant_name, tooth_ids in QUADRANTS.items():
        # Find teeth belonging to this quadrant
        quadrant_teeth = [t for t in tooth_metrics_list if t['tooth_id'] in tooth_ids]

        if len(quadrant_teeth) == 0:
            continue

        # Aggregate metrics (mean of per-tooth means)
        quadrant_metrics = {
            'subject_id': subject_id,
            'session_id': session_id,
            'timepoint': timepoint,
            'quadrant': quadrant_name,
            'n_teeth': len(quadrant_teeth),
            'SUV_mean': np.mean([t['SUV_mean'] for t in quadrant_teeth if not np.isnan(t['SUV_mean'])]),
            'SUV_90th': np.mean([t['SUV_90th'] for t in quadrant_teeth if not np.isnan(t['SUV_90th'])]),
            'TPR_mean': np.mean([t['TPR_mean'] for t in quadrant_teeth if not np.isnan(t['TPR_mean'])]),
            'FUR_mean_per_min': np.mean([t['FUR_mean_per_min'] for t in quadrant_teeth if not np.isnan(t['FUR_mean_per_min'])]),
        }

        quadrant_metrics_list.append(quadrant_metrics)

    # Add whole upper jaw
    if tooth_metrics_list:
        whole_jaw_metrics = {
            'subject_id': subject_id,
            'session_id': session_id,
            'timepoint': timepoint,
            'quadrant': 'whole_upper_jaw',
            'n_teeth': len(tooth_metrics_list),
            'SUV_mean': np.mean([t['SUV_mean'] for t in tooth_metrics_list if not np.isnan(t['SUV_mean'])]),
            'SUV_90th': np.mean([t['SUV_90th'] for t in tooth_metrics_list if not np.isnan(t['SUV_90th'])]),
            'TPR_mean': np.mean([t['TPR_mean'] for t in tooth_metrics_list if not np.isnan(t['TPR_mean'])]),
            'FUR_mean_per_min': np.mean([t['FUR_mean_per_min'] for t in tooth_metrics_list if not np.isnan(t['FUR_mean_per_min'])]),
        }
        quadrant_metrics_list.append(whole_jaw_metrics)

    result['quadrant_metrics'] = quadrant_metrics_list
    result['main_metrics'] = main_metrics
    result['status'] = 'success'

    return result


def main():
    """Main PET quantification pipeline."""
    parser = argparse.ArgumentParser(description='PET Metric Extraction Pipeline')
    parser.add_argument('--subject', type=str, help='Process only this subject')
    parser.add_argument('--force', action='store_true', help='Re-run even if output exists')
    args = parser.parse_args()

    # Setup
    ensure_directories()
    logger = setup_logging(LOGNOTES_DIR)

    logger.info("=" * 60)
    logger.info("PERIODONTAL ANALYSIS - PET QUANTIFICATION")
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    # Load blinding key
    try:
        blinding_map = load_blinding_key()
        logger.info(f"Loaded blinding key with {len(blinding_map)} entries")
    except FileNotFoundError as e:
        logger.error(f"ERROR: {e}")
        sys.exit(1)

    # Load eCRF data
    try:
        ecrf_df = load_ecrf_data()
        logger.info(f"Loaded eCRF data")
    except FileNotFoundError as e:
        logger.error(f"ERROR: {e}")
        sys.exit(1)

    # Load clinical ratings
    try:
        clinical_df = load_clinical_ratings()
        logger.info(f"Loaded clinical ratings with {len(clinical_df)} entries")
    except FileNotFoundError as e:
        logger.warning(f"Clinical ratings not found: {e}")
        clinical_df = pd.DataFrame(columns=['subject_id', 'session_id', 'tooth_id', 'rating', 'rating_category'])

    # Discover subjects
    subjects = discover_subjects()
    if args.subject:
        if args.subject in subjects:
            subjects = [args.subject]
        else:
            logger.error(f"Subject {args.subject} not found")
            sys.exit(1)

    logger.info(f"Processing {len(subjects)} subjects")

    # Collect all metrics
    all_tooth_metrics = []
    all_quadrant_metrics = []
    all_session_metrics = []  # Main peridental ROI metrics per session
    all_results = []

    # Process each subject
    for subject_id in subjects:
        logger.info(f"\n{'='*40}")
        logger.info(f"SUBJECT: {subject_id}")
        logger.info(f"{'='*40}")

        subject_dir = RAWDATA_DIR / subject_id
        sessions = discover_sessions(subject_dir)

        for session_id in sessions:
            # Get timepoint
            key = (subject_id, session_id)
            timepoint = blinding_map.get(key, "Unknown")
            if timepoint == "Unknown":
                logger.warning(f"  {session_id}: Not in blinding key, skipping")
                continue

            # Get SUV parameters
            suv_params = get_suv_parameters(ecrf_df, subject_id, timepoint)

            # Process session
            result = process_session(
                subject_id, session_id, timepoint, suv_params,
                clinical_df, logger, force=args.force
            )

            all_results.append(result)

            if result['status'] == 'success':
                all_tooth_metrics.extend(result['tooth_metrics'])
                all_quadrant_metrics.extend(result['quadrant_metrics'])
                # Add session-level metrics from main peridental ROI
                session_metrics = {
                    'subject_id': result['subject_id'],
                    'session_id': result['session_id'],
                    'timepoint': result['timepoint'],
                    'roi_type': 'peridental_4mm',
                    **{k: v for k, v in result['main_metrics'].items() if k != 'warnings'}
                }
                all_session_metrics.append(session_metrics)

    # Save results
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    if all_tooth_metrics:
        tooth_df = pd.DataFrame(all_tooth_metrics)
        tooth_df = tooth_df.sort_values(['subject_id', 'timepoint', 'tooth_id'])
        tooth_file = OUTPUTS_DIR / TOOTH_METRICS_FILE
        # Merge with existing data to avoid overwriting when run per-subject
        if tooth_file.exists():
            existing = pd.read_csv(tooth_file)
            processed_subjects = tooth_df['subject_id'].unique()
            existing = existing[~existing['subject_id'].isin(processed_subjects)]
            tooth_df = pd.concat([existing, tooth_df], ignore_index=True)
            tooth_df = tooth_df.sort_values(['subject_id', 'timepoint', 'tooth_id'])
        tooth_df.to_csv(tooth_file, index=False)
        logger.info(f"\nTooth-level metrics saved to: {tooth_file}")
        logger.info(f"  Total rows: {len(tooth_df)}")

    if all_quadrant_metrics:
        quadrant_df = pd.DataFrame(all_quadrant_metrics)
        quadrant_df = quadrant_df.sort_values(['subject_id', 'timepoint', 'quadrant'])
        quadrant_file = OUTPUTS_DIR / QUADRANT_METRICS_FILE
        # Merge with existing data to avoid overwriting when run per-subject
        if quadrant_file.exists():
            existing = pd.read_csv(quadrant_file)
            processed_subjects = quadrant_df['subject_id'].unique()
            existing = existing[~existing['subject_id'].isin(processed_subjects)]
            quadrant_df = pd.concat([existing, quadrant_df], ignore_index=True)
            quadrant_df = quadrant_df.sort_values(['subject_id', 'timepoint', 'quadrant'])
        quadrant_df.to_csv(quadrant_file, index=False)
        logger.info(f"Quadrant-level metrics saved to: {quadrant_file}")
        logger.info(f"  Total rows: {len(quadrant_df)}")

    # Always save session-level metrics (main peridental ROI)
    if all_session_metrics:
        session_df = pd.DataFrame(all_session_metrics)
        session_df = session_df.sort_values(['subject_id', 'timepoint'])
        session_file = OUTPUTS_DIR / "periodontal_metrics.csv"
        # Merge with existing data to avoid overwriting when run per-subject
        if session_file.exists():
            existing = pd.read_csv(session_file)
            processed_subjects = session_df['subject_id'].unique()
            existing = existing[~existing['subject_id'].isin(processed_subjects)]
            session_df = pd.concat([existing, session_df], ignore_index=True)
            session_df = session_df.sort_values(['subject_id', 'timepoint'])
        session_df.to_csv(session_file, index=False)
        logger.info(f"\nSession-level metrics saved to: {session_file}")
        logger.info(f"  Total rows: {len(session_df)}")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    success_count = sum(1 for r in all_results if r['status'] == 'success')
    fail_count = sum(1 for r in all_results if r['status'] == 'failed')

    logger.info(f"Sessions processed successfully: {success_count}")
    logger.info(f"Sessions failed: {fail_count}")

    if all_tooth_metrics:
        tooth_df = pd.DataFrame(all_tooth_metrics)
        logger.info(f"\nMetric ranges (tooth-level):")
        logger.info(f"  SUV_mean: {tooth_df['SUV_mean'].min():.3f} - {tooth_df['SUV_mean'].max():.3f}")
        logger.info(f"  TPR_mean: {tooth_df['TPR_mean'].min():.3f} - {tooth_df['TPR_mean'].max():.3f}")
        logger.info(f"  FUR_mean: {tooth_df['FUR_mean_per_min'].min():.5f} - {tooth_df['FUR_mean_per_min'].max():.5f}")

    logger.info("\n" + "=" * 60)
    logger.info(f"PET quantification completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
