#!/usr/bin/env python3
"""
04_batch_quantify.py - Batch PET Quantification with Cross-Session Harmonization

Runs PET quantification for all sessions with:
1. Per-tooth metrics (original + tongue-trimmed variants)
2. Jaw-level metrics with cross-session tooth harmonization
3. Support for both 3mm and 8mm tongue trimming

Cross-session harmonization:
    For paired Baseline/Followup analysis, jaw-level ROIs include only
    teeth present in BOTH sessions. This ensures comparable metrics.

Output:
    Outputs/tooth_level_metrics.csv          - Per-tooth metrics (all variants)
    Outputs/jaw_level_metrics.csv            - Jaw-level metrics (harmonized + unharmonized)

Usage:
    python 04_batch_quantify.py                    # All sessions
    python 04_batch_quantify.py --subjects sub-101 # Specific subjects
"""

import argparse
import json
import sys
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import nibabel as nib
import pandas as pd
from scipy.interpolate import interp1d

# Add Scripts directory to path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

from config import (
    RAWDATA_DIR, TOTALSEG_ROI_DIR, OUTPUTS_DIR, LOGNOTES_DIR, DERIVED_DIR,
    CROSS_SECTIONAL_OUTPUT_DIR, ensure_directories, get_tooth_type
)
from utils.io_utils import (
    load_nifti, get_voxel_volume_ml, load_blinding_key, find_pet_file
)

logger = logging.getLogger(__name__)


# =============================================================================
# BLINDING / SESSION HELPERS
# =============================================================================

def get_subject_sessions(blinding_map):
    """Return dict {subject_id: {'Baseline': session_id, 'Followup': session_id}}."""
    subjects = {}
    for (subj, sess), tp in blinding_map.items():
        if subj not in subjects:
            subjects[subj] = {}
        subjects[subj][tp] = sess
    return subjects


def get_shared_teeth(subject_id, sessions_dict, roi_base_dir):
    """Get set of teeth present in both Baseline and Followup sessions."""
    baseline_sess = sessions_dict.get('Baseline')
    followup_sess = sessions_dict.get('Followup')

    if not baseline_sess or not followup_sess:
        return None  # Single session, no harmonization needed

    baseline_teeth = set()
    followup_teeth = set()

    for sess, teeth_set in [(baseline_sess, baseline_teeth), (followup_sess, followup_teeth)]:
        lookup_file = roi_base_dir / f"{subject_id}_{sess}" / "tooth_shells_lookup.json"
        if lookup_file.exists():
            with open(lookup_file) as f:
                lookup = json.load(f)
            # Handle both old and new lookup formats
            if 'tooth_ids' in lookup:
                teeth_set.update(lookup['tooth_ids'])
            elif 'tooth_volumes_ml' in lookup:
                teeth_set.update(int(k) for k in lookup['tooth_volumes_ml'].keys())

    return baseline_teeth & followup_teeth


# =============================================================================
# eCRF / INPUT FUNCTION LOADING
# =============================================================================

def load_ecrf_data():
    """Load eCRF data for weight and dose."""
    # Try eCRF_data subdirectory first, then root RawData
    ecrf_dir = RAWDATA_DIR / "eCRF_data"
    ecrf_files = list(ecrf_dir.glob("K8ERAPKIH22001_DATA_*.csv")) if ecrf_dir.exists() else []
    if not ecrf_files:
        ecrf_files = list(RAWDATA_DIR.glob("K8ERAPKIH22001_DATA_*.csv"))
    if not ecrf_files:
        logger.warning("No eCRF file found")
        return {}

    ecrf_file = sorted(ecrf_files)[-1]  # Most recent
    df = pd.read_csv(ecrf_file)

    # Map to subject/timepoint
    data = {}
    for _, row in df.iterrows():
        subj_num = row.get('record_id')
        if pd.isna(subj_num):
            continue

        # eCRF uses record_id 1-14, pipeline uses sub-101 to sub-114
        subject_id = f"sub-{int(subj_num) + 100:03d}"

        # Map timepoint to column prefixes (actual eCRF column names)
        for tp, weight_col, dose_col in [
            ('Baseline', 'weight_kg_pet_1', 'injected_mbq_pet_1'),
            ('Followup', 'weight_kg_pet_2', 'injected_mbq_pet_2')
        ]:
            weight = row.get(weight_col)
            dose = row.get(dose_col)

            if pd.notna(weight) and pd.notna(dose):
                # Handle European decimal format (comma as decimal separator)
                weight_str = str(weight).replace(',', '.')
                dose_str = str(dose).replace(',', '.')
                try:
                    key = (subject_id, tp)
                    data[key] = {
                        'weight_kg': float(weight_str),
                        'dose_mbq': float(dose_str),
                    }
                except ValueError:
                    logger.warning(f"Could not parse weight/dose for {subject_id} {tp}: {weight}, {dose}")

    return data


def load_input_function(subject_id, timepoint):
    """Load input function data for TPR/FUR calculation."""
    if_dir = DERIVED_DIR / "input_functions"

    # Use timepoint directly in filename (new naming convention)
    if_file = if_dir / f"{subject_id}_ses-{timepoint}_if_processed.csv"
    if not if_file.exists():
        # Try legacy naming convention with actual session ID
        blinding_map = load_blinding_key()
        session_id = None
        for (subj, sess), tp in blinding_map.items():
            if subj == subject_id and tp == timepoint:
                session_id = sess
                break
        if session_id:
            if_file = if_dir / f"{subject_id}_{session_id}_desc-IF_tacs.tsv"
        if not if_file.exists():
            return None

    # Read CSV (comma-separated)
    df = pd.read_csv(if_file)

    # Rename column if using old naming convention
    if 'activity_Bq_mL' in df.columns and 'plasma_Bq_mL' not in df.columns:
        df = df.rename(columns={'activity_Bq_mL': 'plasma_Bq_mL'})

    return df


def compute_plasma_denominators(if_data, scan_start_s, scan_end_s, tissue_time_s):
    """Compute plasma denominators for TPR and FUR."""
    time_s = if_data['time_s'].values
    plasma = if_data['plasma_Bq_mL'].values

    # Interpolate to get plasma at tissue time
    interp_func = interp1d(time_s, plasma, kind='linear', fill_value='extrapolate')

    # Mean plasma during scan window
    scan_times = time_s[(time_s >= scan_start_s) & (time_s <= scan_end_s)]
    if len(scan_times) < 2:
        plasma_mean = float(interp_func(tissue_time_s))
    else:
        plasma_values = plasma[(time_s >= scan_start_s) & (time_s <= scan_end_s)]
        plasma_mean = float(np.mean(plasma_values))

    # AUC from 0 to tissue_time
    mask = time_s <= tissue_time_s
    auc = float(np.trapz(plasma[mask], time_s[mask]))

    return {
        'plasma_mean_Bq_mL': plasma_mean,
        'plasma_auc_0_to_T_Bq_s_mL': auc,
    }


def load_pet_json(subject_id, timepoint):
    """Load PET JSON sidecar for scan timing.

    Priority:
    1. Updated JSON from json_side_cars_updated/ (preferred, has correct timing)
    2. Fallback to JSON alongside PET NIfTI file (may have incorrect ScanStart)
    """
    # Try updated JSON first (uses timepoint in filename)
    updated_json_dir = RAWDATA_DIR / "json_side_cars_updated"
    updated_json = updated_json_dir / f"{subject_id}_ses-{timepoint}_trc-18FFDG_rec-StaticMoCo_chunk-1_pet.json"
    if updated_json.exists():
        with open(updated_json) as f:
            return json.load(f)

    # Fallback to original JSON alongside PET file
    blinding_map = load_blinding_key()
    session_id = None
    for (subj, sess), tp in blinding_map.items():
        if subj == subject_id and tp == timepoint:
            session_id = sess
            break

    if session_id is None:
        return {}

    session_dir = RAWDATA_DIR / subject_id / session_id
    pet_file = find_pet_file(session_dir)
    if pet_file is None:
        return {}

    json_file = pet_file.with_suffix('.json')
    if json_file.exists():
        logger.warning(f"  Using original JSON (may have incorrect timing) for {subject_id} {timepoint}")
        with open(json_file) as f:
            return json.load(f)
    return {}


# =============================================================================
# METRIC EXTRACTION
# =============================================================================

def extract_metrics_from_mask(pet_data, mask_data, voxel_vol_ml,
                               suv_scaler, plasma_mean, auc_0_to_T,
                               weight_threshold=0.05):
    """Extract metrics from a single continuous mask."""
    mask = mask_data > weight_threshold
    if not np.any(mask):
        return None

    pet_vals = pet_data[mask]
    w = mask_data[mask]

    valid_idx = np.isfinite(pet_vals) & (pet_vals > 0)
    if not np.any(valid_idx):
        return None

    pet_valid = pet_vals[valid_idx]
    w_valid = w[valid_idx]

    intensity_wmean = float(np.average(pet_valid, weights=w_valid))
    intensity_median = float(np.median(pet_valid))
    intensity_p90 = float(np.percentile(pet_valid, 90))
    intensity_max = float(np.max(pet_valid))
    effective_vol = float(np.sum(w_valid)) * voxel_vol_ml

    suv_mean = intensity_wmean * suv_scaler if np.isfinite(suv_scaler) else np.nan
    suv_p90 = intensity_p90 * suv_scaler if np.isfinite(suv_scaler) else np.nan

    if np.isfinite(plasma_mean) and plasma_mean > 0:
        tpr_mean = intensity_wmean / plasma_mean
        tpr_p90 = intensity_p90 / plasma_mean
    else:
        tpr_mean = tpr_p90 = np.nan

    if np.isfinite(auc_0_to_T) and auc_0_to_T > 0:
        fur_mean = (intensity_wmean / auc_0_to_T) * 60
        fur_p90 = (intensity_p90 / auc_0_to_T) * 60
    else:
        fur_mean = fur_p90 = np.nan

    return {
        'n_voxels': int(np.sum(valid_idx)),
        'roi_volume_ml': effective_vol,
        'intensity_mean_Bq_mL': intensity_wmean,
        'intensity_median_Bq_mL': intensity_median,
        'intensity_p90_Bq_mL': intensity_p90,
        'intensity_max_Bq_mL': intensity_max,
        'SUV_mean': suv_mean,
        'SUV_p90': suv_p90,
        'TPR_mean': tpr_mean,
        'TPR_p90': tpr_p90,
        'FUR_mean_per_min': fur_mean,
        'FUR_p90_per_min': fur_p90,
    }


def quantify_session(subject_id, session_id, timepoint, ecrf_data, shared_teeth=None):
    """Quantify a single session, returning list of records."""

    roi_dir = TOTALSEG_ROI_DIR / f"{subject_id}_{session_id}"
    cont_dir = roi_dir / "continuous_masks_PETspace"

    if not cont_dir.exists():
        logger.warning(f"  No continuous masks directory")
        return []

    # Load PET
    session_dir = RAWDATA_DIR / subject_id / session_id
    pet_file = find_pet_file(session_dir)
    if pet_file is None:
        logger.error(f"  No PET file found")
        return []

    pet_data, pet_img = load_nifti(pet_file)
    voxel_vol_ml = get_voxel_volume_ml(pet_img)

    # Get SUV params
    ecrf_key = (subject_id, timepoint)
    suv_params = ecrf_data.get(ecrf_key, {})
    weight_kg = suv_params.get('weight_kg', np.nan)
    dose_mbq = suv_params.get('dose_mbq', np.nan)

    if np.isfinite(weight_kg) and np.isfinite(dose_mbq) and dose_mbq > 0:
        suv_scaler = weight_kg / (dose_mbq * 1000)
    else:
        suv_scaler = np.nan

    # Get plasma params
    pet_json = load_pet_json(subject_id, timepoint)
    scan_start_s = pet_json.get('ScanStart', 1800)
    frame_duration_ms = pet_json.get('FrameDuration', [1800000])
    if isinstance(frame_duration_ms, list):
        frame_duration_ms = frame_duration_ms[0]
    scan_end_s = scan_start_s + frame_duration_ms / 1000
    tissue_time_s = (scan_start_s + scan_end_s) / 2

    if_data = load_input_function(subject_id, timepoint)
    plasma_mean = np.nan
    auc_0_to_T = np.nan
    if if_data is not None:
        plasma_denom = compute_plasma_denominators(if_data, scan_start_s, scan_end_s, tissue_time_s)
        plasma_mean = plasma_denom.get('plasma_mean_Bq_mL', np.nan)
        auc_0_to_T = plasma_denom.get('plasma_auc_0_to_T_Bq_s_mL', np.nan)

    records = []

    # --- Per-tooth metrics ---
    # Find all mask variants (original: tooth_XX_continuous.nii.gz, trimmed: tooth_XX_trimmed_Xmm.nii.gz)
    for mask_file in sorted(cont_dir.glob("tooth_*.nii.gz")):
        if "exclusion" in mask_file.name:
            continue

        mask_data, _ = load_nifti(mask_file)

        # Parse FDI and trimming variant
        parts = mask_file.stem.split('_')
        fdi = int(parts[1])

        if 'trimmed' in mask_file.stem:
            trim_idx = parts.index('trimmed')
            trimming = parts[trim_idx + 1].replace('.nii', '')  # "3mm" or "8mm"
        elif 'continuous' in mask_file.stem:
            trimming = "none"
        else:
            continue  # Skip any other variants

        metrics = extract_metrics_from_mask(
            pet_data, mask_data, voxel_vol_ml,
            suv_scaler, plasma_mean, auc_0_to_T
        )

        if metrics:
            jaw = 'upper' if 11 <= fdi <= 28 else 'lower'
            metrics.update({
                'subject_id': subject_id,
                'session_id': session_id,
                'timepoint': timepoint,
                'fdi_tooth': fdi,
                'jaw': jaw,
                'tooth_type': get_tooth_type(fdi),
                'trimming': trimming,
            })
            records.append(metrics)

    # --- Jaw-level metrics (unharmonized) ---
    for jaw_name in ['upper_jaw', 'lower_jaw']:
        for trimming_suffix in ['', '_trimmed_0mm', '_trimmed_3mm', '_trimmed_5mm', '_trimmed_8mm', '_trimmed_10mm']:
            mask_file = cont_dir / f"peridental_{jaw_name}{trimming_suffix}.nii.gz"
            if trimming_suffix == '':
                mask_file = cont_dir / f"peridental_{jaw_name}_continuous.nii.gz"

            if not mask_file.exists():
                continue

            mask_data, _ = load_nifti(mask_file)

            if trimming_suffix:
                trimming = trimming_suffix.replace('_trimmed_', '')
            else:
                trimming = "none"

            metrics = extract_metrics_from_mask(
                pet_data, mask_data, voxel_vol_ml,
                suv_scaler, plasma_mean, auc_0_to_T
            )

            if metrics:
                metrics.update({
                    'subject_id': subject_id,
                    'session_id': session_id,
                    'timepoint': timepoint,
                    'fdi_tooth': jaw_name,
                    'jaw': jaw_name.replace('_jaw', ''),
                    'trimming': trimming,
                    'harmonized': False,
                })
                records.append(metrics)

    # --- Jaw-level metrics (harmonized) ---
    if shared_teeth is not None:
        for jaw_name, fdi_range in [('upper_jaw', range(11, 29)), ('lower_jaw', range(31, 49))]:
            for trimming in ['none', '0mm', '3mm', '5mm', '8mm', '10mm']:
                # Build composite mask from shared teeth only
                composite_mask = None

                for fdi in shared_teeth:
                    if fdi not in fdi_range:
                        continue

                    if trimming == 'none':
                        mask_file = cont_dir / f"tooth_{fdi:02d}_continuous.nii.gz"
                    else:
                        mask_file = cont_dir / f"tooth_{fdi:02d}_trimmed_{trimming}.nii.gz"

                    if not mask_file.exists():
                        continue

                    tooth_data, _ = load_nifti(mask_file)
                    if composite_mask is None:
                        composite_mask = tooth_data.copy()
                    else:
                        composite_mask = np.maximum(composite_mask, tooth_data)

                if composite_mask is not None:
                    metrics = extract_metrics_from_mask(
                        pet_data, composite_mask, voxel_vol_ml,
                        suv_scaler, plasma_mean, auc_0_to_T
                    )

                    if metrics:
                        n_shared = len([t for t in shared_teeth if t in fdi_range])
                        metrics.update({
                            'subject_id': subject_id,
                            'session_id': session_id,
                            'timepoint': timepoint,
                            'fdi_tooth': f'{jaw_name}_harmonized',
                            'jaw': jaw_name.replace('_jaw', ''),
                            'trimming': trimming,
                            'harmonized': True,
                            'n_shared_teeth': n_shared,
                        })
                        records.append(metrics)

    return records


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Batch PET Quantification')
    parser.add_argument('--subjects', nargs='+', type=str, help='Specific subjects')
    parser.add_argument('--force', action='store_true', help='Overwrite existing output')
    args = parser.parse_args()

    ensure_directories()

    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOGNOTES_DIR / f"batch_quantify_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    logger.info("=" * 70)
    logger.info("BATCH PET QUANTIFICATION")
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 70)

    # Load data
    blinding_map = load_blinding_key()
    ecrf_data = load_ecrf_data()
    subject_sessions = get_subject_sessions(blinding_map)

    # Filter subjects
    if args.subjects:
        subject_sessions = {k: v for k, v in subject_sessions.items() if k in args.subjects}

    logger.info(f"Subjects to process: {len(subject_sessions)}")

    all_records = []

    for subject_id, sessions_dict in sorted(subject_sessions.items()):
        logger.info(f"\n--- {subject_id} ---")

        # Get shared teeth for harmonization
        shared_teeth = get_shared_teeth(subject_id, sessions_dict, TOTALSEG_ROI_DIR)
        if shared_teeth:
            logger.info(f"  Shared teeth: {len(shared_teeth)} ({sorted(shared_teeth)[:5]}...)")
        else:
            logger.info(f"  No paired sessions for harmonization")

        for timepoint, session_id in sessions_dict.items():
            logger.info(f"  {session_id} ({timepoint})...")

            try:
                records = quantify_session(
                    subject_id, session_id, timepoint, ecrf_data, shared_teeth
                )
                all_records.extend(records)
                logger.info(f"    Extracted {len(records)} records")
            except Exception as e:
                logger.error(f"    FAILED: {e}")
                import traceback
                traceback.print_exc()

    # Save results
    df = pd.DataFrame(all_records)

    if len(df) == 0:
        logger.warning("No records extracted!")
        return

    # Separate tooth-level and jaw-level
    tooth_mask = df['fdi_tooth'].apply(lambda x: isinstance(x, (int, np.integer)))
    jaw_mask = ~tooth_mask

    tooth_df = df[tooth_mask].copy()
    jaw_df = df[jaw_mask].copy()

    # Define column order: metadata first, then metrics
    metadata_cols = ['subject_id', 'session_id', 'timepoint', 'fdi_tooth', 'jaw', 'tooth_type', 'trimming', 'harmonized', 'n_shared_teeth']
    metric_cols = ['n_voxels', 'roi_volume_ml', 'intensity_mean_Bq_mL', 'intensity_median_Bq_mL',
                   'intensity_p90_Bq_mL', 'intensity_max_Bq_mL', 'SUV_mean', 'SUV_p90',
                   'TPR_mean', 'TPR_p90', 'FUR_mean_per_min', 'FUR_p90_per_min']

    # Order tooth-level columns (no harmonized/n_shared_teeth)
    tooth_col_order = [c for c in metadata_cols if c in tooth_df.columns] + \
                      [c for c in metric_cols if c in tooth_df.columns]
    tooth_df = tooth_df[tooth_col_order]

    # Order jaw-level columns (includes harmonized/n_shared_teeth)
    jaw_col_order = [c for c in metadata_cols if c in jaw_df.columns] + \
                    [c for c in metric_cols if c in jaw_df.columns]
    jaw_df = jaw_df[jaw_col_order]

    # Save tooth-level metrics
    CROSS_SECTIONAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tooth_file = CROSS_SECTIONAL_OUTPUT_DIR / "tooth_level_metrics.csv"
    tooth_df.to_csv(tooth_file, index=False)
    logger.info(f"Saved: {tooth_file} ({len(tooth_df)} rows)")

    # Save jaw-level metrics
    jaw_file = CROSS_SECTIONAL_OUTPUT_DIR / "jaw_level_metrics.csv"
    jaw_df.to_csv(jaw_file, index=False)
    logger.info(f"Saved: {jaw_file} ({len(jaw_df)} rows)")

    # Summary
    logger.info(f"\n{'='*70}")
    logger.info("SUMMARY")
    logger.info(f"{'='*70}")
    logger.info(f"Total tooth records: {len(tooth_df)}")
    logger.info(f"Total jaw records: {len(jaw_df)}")

    # Group by trimming
    if 'trimming' in tooth_df.columns:
        for trim in tooth_df['trimming'].unique():
            n = len(tooth_df[tooth_df['trimming'] == trim])
            logger.info(f"  Trimming={trim}: {n} tooth records")

    if 'harmonized' in jaw_df.columns:
        n_harm = len(jaw_df[jaw_df['harmonized'] == True])
        n_unharm = len(jaw_df[jaw_df['harmonized'] == False])
        logger.info(f"  Jaw metrics: {n_unharm} unharmonized, {n_harm} harmonized")


if __name__ == "__main__":
    main()
