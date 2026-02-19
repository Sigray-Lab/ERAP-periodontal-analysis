#!/usr/bin/env python3
"""
06_longitudinal_coregistration.py

Longitudinal voxel-wise FDG-PET analysis using bias-free midpoint space.

This script:
1. Co-registers Baseline and Followup CT images using rigid registration
2. Computes a midpoint space using matrix logarithm (T^0.5)
3. Chains transforms to bring both PET images into common space
4. Creates intersection masks and computes delta images
5. Applies Gaussian smoothing and statistical analysis

Usage:
    python 06_longitudinal_coregistration.py [--subject SUB] [--force] [--trimming Xmm]
"""

import argparse
import gc
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import nibabel as nib
import pandas as pd
from scipy import ndimage
from scipy import stats
import matplotlib.pyplot as plt

# Add Scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    RAWDATA_DIR, TOTALSEG_ROI_DIR, OUTPUTS_DIR, LOGNOTES_DIR, DERIVED_DIR,
    TRANSFORM_DIR, LONGITUDINAL_DIR, LONGITUDINAL_QC_DIR, BLINDING_KEY_FILE,
    LONGITUDINAL_SMOOTHING_FWHM_MM, LONGITUDINAL_DEFAULT_TRIMMING,
    INPUT_FUNC_DIR, ensure_directories,
)

from utils.io_utils import load_nifti, find_ct_file, find_pet_file
from utils.registration_utils import (
    register_ct_to_ct, save_transform,
    compute_halfway_transform, create_midpoint_reference_grid, resample_to_midpoint,
)

# Setup logging
logger = logging.getLogger(__name__)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_blinding_key():
    """Load blinding key to map sessions to Baseline/Followup."""
    df = pd.read_csv(BLINDING_KEY_FILE)
    mapping = {}
    for _, row in df.iterrows():
        subject_id = row['participant_id']
        session = row['Session']
        blind_code = row['Blind.code']
        session_id = f"ses-{blind_code}"
        if subject_id not in mapping:
            mapping[subject_id] = {}
        mapping[subject_id][session] = session_id
    return mapping


def load_ecrf_data():
    """Load eCRF data for metabolic calculations."""
    ecrf_pattern = "K8ERAPKIH22001_DATA_*.csv"
    ecrf_files = list(RAWDATA_DIR.glob(ecrf_pattern))
    if not ecrf_files:
        logger.warning("No eCRF files found")
        return {}

    ecrf_file = sorted(ecrf_files)[-1]  # Most recent
    df = pd.read_csv(ecrf_file)

    data = {}
    for _, row in df.iterrows():
        try:
            subject_id = f"sub-{int(row.get('subj_no', 0)):03d}"
            timepoint = row.get('redcap_event_name', '')
            if 'baseline' in timepoint.lower():
                tp = 'Baseline'
            elif 'followup' in timepoint.lower() or 'follow_up' in timepoint.lower():
                tp = 'Followup'
            else:
                continue

            weight = row.get('pet_weight', np.nan)
            dose = row.get('pet_dose', np.nan)
            if pd.notna(weight) and pd.notna(dose):
                data[(subject_id, tp)] = {
                    'weight_kg': float(weight),
                    'dose_mbq': float(dose),
                }
        except (ValueError, KeyError):
            continue

    return data


def load_input_function(subject_id, timepoint):
    """Load input function data for a subject/timepoint."""
    if_file = INPUT_FUNC_DIR / f"{subject_id}_{timepoint.lower()}_input_function.csv"
    if not if_file.exists():
        return None
    return pd.read_csv(if_file)


def load_pet_json_timing(subject_id, timepoint):
    """Load PET timing from JSON sidecar."""
    blinding = load_blinding_key()
    session = blinding.get(subject_id, {}).get(timepoint)
    if not session:
        return 1800, 3600, 2700  # Defaults

    pet_dir = RAWDATA_DIR / subject_id / session / "pet"
    json_files = list(pet_dir.glob("*.json"))
    if not json_files:
        return 1800, 3600, 2700

    with open(json_files[0]) as f:
        meta = json.load(f)

    scan_start = meta.get('FrameTimesStart', [1800])[0]
    duration = meta.get('FrameDuration', [1800])[0]
    scan_end = scan_start + duration
    tissue_time = (scan_start + scan_end) / 2

    return scan_start, scan_end, tissue_time


def compute_plasma_params(if_data, scan_start, scan_end, tissue_time):
    """Compute plasma parameters from input function."""
    if if_data is None:
        return {}

    time = if_data['time_s'].values
    plasma = if_data['plasma_Bq_mL'].values

    # Mean plasma during scan
    mask = (time >= scan_start) & (time <= scan_end)
    if mask.sum() > 0:
        plasma_mean = plasma[mask].mean()
    else:
        plasma_mean = np.nan

    # AUC from 0 to tissue_time
    mask_auc = time <= tissue_time
    if mask_auc.sum() > 1:
        plasma_auc = np.trapz(plasma[mask_auc], time[mask_auc])
    else:
        plasma_auc = np.nan

    return {
        'plasma_mean_Bq_mL': plasma_mean,
        'plasma_auc_Bq_s_mL': plasma_auc,
    }


# =============================================================================
# METRIC COMPUTATION
# =============================================================================

def compute_suv_image(pet_data, weight_kg, dose_mbq):
    """Compute SUV image."""
    if np.isnan(weight_kg) or np.isnan(dose_mbq) or dose_mbq == 0:
        return np.full_like(pet_data, np.nan)
    return pet_data * weight_kg / (dose_mbq * 1e6)


def compute_tpr_image(pet_data, plasma_mean):
    """Compute tissue-to-plasma ratio image."""
    if np.isnan(plasma_mean) or plasma_mean == 0:
        return np.full_like(pet_data, np.nan)
    return pet_data / plasma_mean


def compute_fur_image(pet_data, plasma_auc):
    """Compute fractional uptake rate image (min^-1)."""
    if np.isnan(plasma_auc) or plasma_auc == 0:
        return np.full_like(pet_data, np.nan)
    return (pet_data / plasma_auc) * 60  # Convert from s^-1 to min^-1


def create_intersection_mask(mask_bl, mask_fu, threshold=0.5):
    """Create intersection of two masks."""
    bl_binary = mask_bl > threshold
    fu_binary = mask_fu > threshold

    intersection = bl_binary & fu_binary

    # Compute statistics
    n_bl = bl_binary.sum()
    n_fu = fu_binary.sum()
    n_int = intersection.sum()

    dice = 2 * n_int / (n_bl + n_fu) if (n_bl + n_fu) > 0 else 0
    jaccard = n_int / (n_bl + n_fu - n_int) if (n_bl + n_fu - n_int) > 0 else 0

    stats = {
        'n_voxels_baseline': int(n_bl),
        'n_voxels_followup': int(n_fu),
        'n_voxels_intersection': int(n_int),
        'dice_coefficient': float(dice),
        'jaccard_index': float(jaccard),
    }

    return intersection.astype(np.uint8), stats


def gaussian_smooth_3d(image, fwhm_mm, voxel_dims):
    """Apply 3D Gaussian smoothing."""
    sigma_mm = fwhm_mm / 2.355
    sigma_vox = [sigma_mm / abs(d) for d in voxel_dims]
    return ndimage.gaussian_filter(image, sigma=sigma_vox)


# =============================================================================
# QC VISUALIZATION
# =============================================================================

def create_ct_alignment_qc(ct_bl, ct_fu, output_path, subject_id):
    """Create checkerboard visualization of CT alignment."""
    z_mid = ct_bl.shape[2] // 2

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Axial slice
    for ax, sl_idx, title in [
        (axes[0], z_mid, 'Axial'),
        (axes[1], z_mid - 20, 'Axial -20'),
        (axes[2], z_mid + 20, 'Axial +20'),
    ]:
        if 0 <= sl_idx < ct_bl.shape[2]:
            bl_slice = ct_bl[:, :, sl_idx].T
            fu_slice = ct_fu[:, :, sl_idx].T

            # Checkerboard
            checker = np.zeros_like(bl_slice)
            block_size = 32
            for i in range(0, bl_slice.shape[0], block_size * 2):
                for j in range(0, bl_slice.shape[1], block_size * 2):
                    checker[i:i+block_size, j:j+block_size] = 1
                    checker[i+block_size:i+2*block_size, j+block_size:j+2*block_size] = 1

            combined = np.where(checker, bl_slice, fu_slice)
            ax.imshow(combined, cmap='gray', vmin=-200, vmax=400)
            ax.set_title(f'{title} (z={sl_idx})')
            ax.axis('off')

    plt.suptitle(f'{subject_id} - CT Alignment (Checkerboard)', fontsize=14)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_mask_intersection_qc(mask_bl, mask_fu, intersection, output_path, subject_id, stats):
    """Create visualization of mask intersection."""
    z_mid = mask_bl.shape[2] // 2

    bl_slice = mask_bl[:, :, z_mid].T
    fu_slice = mask_fu[:, :, z_mid].T
    int_slice = intersection[:, :, z_mid].T

    # RGB overlay
    rgb = np.zeros((*bl_slice.shape, 3))
    rgb[bl_slice > 0.5, 0] = 1  # Red: Baseline only
    rgb[fu_slice > 0.5, 2] = 1  # Blue: Followup only
    rgb[int_slice > 0, 1] = 1  # Green: Intersection
    rgb[int_slice > 0, 0] = 0
    rgb[int_slice > 0, 2] = 0

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(rgb)
    axes[0].set_title('Mask Overlap\n(Red=BL only, Blue=FU only, Green=Both)')
    axes[0].axis('off')

    text = f"Dice: {stats['dice_coefficient']:.3f}\n"
    text += f"Jaccard: {stats['jaccard_index']:.3f}\n"
    text += f"Intersection: {stats['n_voxels_intersection']} vox\n"
    text += f"Baseline: {stats['n_voxels_baseline']} vox\n"
    text += f"Followup: {stats['n_voxels_followup']} vox"
    axes[1].text(0.1, 0.5, text, fontsize=14, family='monospace', va='center')
    axes[1].axis('off')
    axes[1].set_title('Mask Statistics')

    plt.suptitle(f'{subject_id} - Mask Intersection QC', fontsize=14)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_delta_maps_qc(delta_suv, delta_tpr, delta_fur, mask, output_path, subject_id):
    """Create visualization of delta maps."""
    z_mid = mask.shape[2] // 2

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, data, title in [
        (axes[0], delta_suv, 'Delta SUV'),
        (axes[1], delta_tpr, 'Delta TPR'),
        (axes[2], delta_fur, 'Delta FUR (min⁻¹)'),
    ]:
        slice_data = data[:, :, z_mid].T.copy()
        slice_mask = mask[:, :, z_mid].T

        slice_data[slice_mask == 0] = np.nan

        valid = slice_data[np.isfinite(slice_data)]
        vmax = np.percentile(np.abs(valid), 95) if len(valid) > 0 else 1

        im = ax.imshow(slice_data, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax.set_title(title)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle(f'{subject_id} - Delta Maps (Followup - Baseline)', fontsize=14)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# MAIN PROCESSING
# =============================================================================

def process_subject(subject_id, baseline_session, followup_session,
                    ecrf_data, trimming='0mm', force=False):
    """Process one subject through the full longitudinal pipeline."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing {subject_id}")
    logger.info(f"  Baseline: {baseline_session}, Followup: {followup_session}")
    logger.info(f"{'='*60}")

    # Setup directories
    subj_out_dir = LONGITUDINAL_DIR / subject_id
    subj_out_dir.mkdir(parents=True, exist_ok=True)
    transform_dir = TRANSFORM_DIR / f"{subject_id}_longitudinal"
    transform_dir.mkdir(parents=True, exist_ok=True)

    # Check if already done
    delta_suv_file = subj_out_dir / "delta_SUV_smoothed_3mm.nii.gz"
    if delta_suv_file.exists() and not force:
        logger.info(f"  Already processed (use --force to re-run)")
        return {'subject_id': subject_id, 'status': 'skipped_exists'}

    # =========================================================================
    # Step 1: Load CT and PET images
    # =========================================================================
    logger.info(f"\n[Step 1] Loading CT and PET images...")

    bl_session_dir = RAWDATA_DIR / subject_id / baseline_session
    fu_session_dir = RAWDATA_DIR / subject_id / followup_session

    bl_ct_file = find_ct_file(bl_session_dir, prefer_bone=False)
    fu_ct_file = find_ct_file(fu_session_dir, prefer_bone=False)

    if bl_ct_file is None or fu_ct_file is None:
        logger.error(f"  Missing CT file(s)")
        return {'subject_id': subject_id, 'status': 'missing_ct'}

    bl_ct_data, bl_ct_img = load_nifti(bl_ct_file)
    fu_ct_data, fu_ct_img = load_nifti(fu_ct_file)

    logger.info(f"  Baseline CT: {bl_ct_file.name}, shape={bl_ct_img.shape}")
    logger.info(f"  Followup CT: {fu_ct_file.name}, shape={fu_ct_img.shape}")

    bl_pet_file = find_pet_file(bl_session_dir)
    fu_pet_file = find_pet_file(fu_session_dir)

    if bl_pet_file is None or fu_pet_file is None:
        logger.error(f"  Missing PET file(s)")
        return {'subject_id': subject_id, 'status': 'missing_pet'}

    bl_pet_data, bl_pet_img = load_nifti(bl_pet_file)
    fu_pet_data, fu_pet_img = load_nifti(fu_pet_file)

    logger.info(f"  Baseline PET: {bl_pet_file.name}, shape={bl_pet_img.shape}")
    logger.info(f"  Followup PET: {fu_pet_file.name}, shape={fu_pet_img.shape}")

    # =========================================================================
    # Step 2: CT-CT Registration (cropped to PET FOV)
    # =========================================================================
    logger.info(f"\n[Step 2] CT-to-CT registration (cropped to PET FOV)...")

    ct_fwd_file = transform_dir / "ct_baseline_to_followup_fwd_0.mat"

    if ct_fwd_file.exists() and not force:
        logger.info(f"  Using existing CT-CT transform")
    else:
        reg_result, reg_meta = register_ct_to_ct(
            bl_ct_img, fu_ct_img,
            pet_baseline_nib=bl_pet_img,
            pet_followup_nib=fu_pet_img,
            transform_type='Rigid'
        )
        save_transform(reg_result, transform_dir, prefix="ct_baseline_to_followup")

        with open(transform_dir / "registration_meta.json", 'w') as f:
            json.dump(reg_meta, f, indent=2)

    # =========================================================================
    # Step 3: Compute halfway transforms
    # =========================================================================
    logger.info(f"\n[Step 3] Computing midpoint transforms...")

    halfway_files = compute_halfway_transform(
        str(ct_fwd_file),
        transform_dir,
        prefix='halfway'
    )

    # =========================================================================
    # Step 4: Create midpoint reference grid
    # =========================================================================
    logger.info(f"\n[Step 4] Creating midpoint reference grid...")

    midpoint_ref = create_midpoint_reference_grid(
        bl_ct_img, fu_ct_img,
        halfway_files['baseline_to_midpoint'],
        halfway_files['followup_to_midpoint']
    )

    nib.save(midpoint_ref, str(subj_out_dir / "midpoint_reference.nii.gz"))

    # =========================================================================
    # Step 5: Resample CTs to midpoint space
    # =========================================================================
    logger.info(f"\n[Step 5] Resampling CTs to midpoint space...")

    ct_bl_mid = resample_to_midpoint(
        bl_ct_img,
        [halfway_files['baseline_to_midpoint']],
        midpoint_ref,
        interpolator='linear'
    )

    ct_fu_mid = resample_to_midpoint(
        fu_ct_img,
        [halfway_files['followup_to_midpoint']],
        midpoint_ref,
        interpolator='linear'
    )

    nib.save(ct_bl_mid, str(subj_out_dir / "ct_baseline_midpoint.nii.gz"))
    nib.save(ct_fu_mid, str(subj_out_dir / "ct_followup_midpoint.nii.gz"))

    create_ct_alignment_qc(
        ct_bl_mid.get_fdata(), ct_fu_mid.get_fdata(),
        LONGITUDINAL_QC_DIR / f"{subject_id}_ct_alignment.png",
        subject_id
    )

    del bl_ct_data, fu_ct_data
    gc.collect()

    # =========================================================================
    # Step 6: Chain PET transforms
    # =========================================================================
    logger.info(f"\n[Step 6] Setting up PET transform chains...")

    bl_pet_to_ct_dir = TRANSFORM_DIR / f"{subject_id}_{baseline_session}"
    fu_pet_to_ct_dir = TRANSFORM_DIR / f"{subject_id}_{followup_session}"

    bl_pet_to_ct_files = [str(bl_pet_to_ct_dir / "ct_to_pet_inv_0.mat")]
    fu_pet_to_ct_files = [str(fu_pet_to_ct_dir / "ct_to_pet_inv_0.mat")]

    if not Path(bl_pet_to_ct_files[0]).exists():
        bl_pet_to_ct_files = [str(bl_pet_to_ct_dir / "ct_to_pet_fwd_0.mat")]
        logger.warning(f"  Using forward transform for baseline")

    if not Path(fu_pet_to_ct_files[0]).exists():
        fu_pet_to_ct_files = [str(fu_pet_to_ct_dir / "ct_to_pet_fwd_0.mat")]
        logger.warning(f"  Using forward transform for followup")

    # Chain: PET → CT → Midpoint (ANTs applies in reverse order)
    bl_pet_to_mid_transforms = [halfway_files['baseline_to_midpoint']] + bl_pet_to_ct_files
    fu_pet_to_mid_transforms = [halfway_files['followup_to_midpoint']] + fu_pet_to_ct_files

    # =========================================================================
    # Step 7: Resample PET images to midpoint space
    # =========================================================================
    logger.info(f"\n[Step 7] Resampling PET images to midpoint space...")

    pet_bl_mid = resample_to_midpoint(
        bl_pet_img, bl_pet_to_mid_transforms, midpoint_ref, interpolator='linear'
    )
    pet_fu_mid = resample_to_midpoint(
        fu_pet_img, fu_pet_to_mid_transforms, midpoint_ref, interpolator='linear'
    )

    nib.save(pet_bl_mid, str(subj_out_dir / "pet_baseline_midpoint.nii.gz"))
    nib.save(pet_fu_mid, str(subj_out_dir / "pet_followup_midpoint.nii.gz"))

    del bl_pet_data, fu_pet_data
    gc.collect()

    # =========================================================================
    # Step 8: Load and resample masks to midpoint space
    # =========================================================================
    logger.info(f"\n[Step 8] Loading and resampling masks to midpoint space...")

    bl_roi_dir = TOTALSEG_ROI_DIR / f"{subject_id}_{baseline_session}" / "continuous_masks_PETspace"
    fu_roi_dir = TOTALSEG_ROI_DIR / f"{subject_id}_{followup_session}" / "continuous_masks_PETspace"

    bl_mask_file = bl_roi_dir / f"peridental_upper_jaw_trimmed_{trimming}.nii.gz"
    fu_mask_file = fu_roi_dir / f"peridental_upper_jaw_trimmed_{trimming}.nii.gz"

    if not bl_mask_file.exists() or not fu_mask_file.exists():
        logger.error(f"  Missing mask file(s) for {trimming} trimming")
        return {'subject_id': subject_id, 'status': 'missing_masks'}

    bl_mask_data, bl_mask_img = load_nifti(bl_mask_file)
    fu_mask_data, fu_mask_img = load_nifti(fu_mask_file)

    mask_bl_mid = resample_to_midpoint(
        bl_mask_img, bl_pet_to_mid_transforms, midpoint_ref, interpolator='linear'
    )
    mask_fu_mid = resample_to_midpoint(
        fu_mask_img, fu_pet_to_mid_transforms, midpoint_ref, interpolator='linear'
    )

    nib.save(mask_bl_mid, str(subj_out_dir / f"mask_baseline_upper_{trimming}_midpoint.nii.gz"))
    nib.save(mask_fu_mid, str(subj_out_dir / f"mask_followup_upper_{trimming}_midpoint.nii.gz"))

    intersection_mask, mask_stats = create_intersection_mask(
        mask_bl_mid.get_fdata(), mask_fu_mid.get_fdata(), threshold=0.5
    )

    nib.save(
        nib.Nifti1Image(intersection_mask, midpoint_ref.affine),
        str(subj_out_dir / f"mask_intersection_upper_{trimming}.nii.gz")
    )

    logger.info(f"  Intersection mask: {mask_stats['n_voxels_intersection']} voxels, "
               f"Dice={mask_stats['dice_coefficient']:.3f}")

    create_mask_intersection_qc(
        mask_bl_mid.get_fdata(), mask_fu_mid.get_fdata(), intersection_mask,
        LONGITUDINAL_QC_DIR / f"{subject_id}_mask_intersection.png",
        subject_id, mask_stats
    )

    # =========================================================================
    # Step 9: Compute metric images and deltas
    # =========================================================================
    logger.info(f"\n[Step 9] Computing delta images...")

    bl_ecrf = ecrf_data.get((subject_id, 'Baseline'), {})
    fu_ecrf = ecrf_data.get((subject_id, 'Followup'), {})

    bl_if = load_input_function(subject_id, 'Baseline')
    fu_if = load_input_function(subject_id, 'Followup')

    bl_scan_start, bl_scan_end, bl_tissue_time = load_pet_json_timing(subject_id, 'Baseline')
    fu_scan_start, fu_scan_end, fu_tissue_time = load_pet_json_timing(subject_id, 'Followup')

    bl_plasma = compute_plasma_params(bl_if, bl_scan_start, bl_scan_end, bl_tissue_time) if bl_if is not None else {}
    fu_plasma = compute_plasma_params(fu_if, fu_scan_start, fu_scan_end, fu_tissue_time) if fu_if is not None else {}

    pet_bl_mid_data = pet_bl_mid.get_fdata()
    pet_fu_mid_data = pet_fu_mid.get_fdata()

    suv_bl = compute_suv_image(pet_bl_mid_data, bl_ecrf.get('weight_kg', np.nan), bl_ecrf.get('dose_mbq', np.nan))
    suv_fu = compute_suv_image(pet_fu_mid_data, fu_ecrf.get('weight_kg', np.nan), fu_ecrf.get('dose_mbq', np.nan))

    tpr_bl = compute_tpr_image(pet_bl_mid_data, bl_plasma.get('plasma_mean_Bq_mL', np.nan))
    tpr_fu = compute_tpr_image(pet_fu_mid_data, fu_plasma.get('plasma_mean_Bq_mL', np.nan))

    fur_bl = compute_fur_image(pet_bl_mid_data, bl_plasma.get('plasma_auc_Bq_s_mL', np.nan))
    fur_fu = compute_fur_image(pet_fu_mid_data, fu_plasma.get('plasma_auc_Bq_s_mL', np.nan))

    delta_suv = np.full_like(suv_bl, np.nan)
    delta_tpr = np.full_like(tpr_bl, np.nan)
    delta_fur = np.full_like(fur_bl, np.nan)

    mask_bool = intersection_mask > 0
    delta_suv[mask_bool] = suv_fu[mask_bool] - suv_bl[mask_bool]
    delta_tpr[mask_bool] = tpr_fu[mask_bool] - tpr_bl[mask_bool]
    delta_fur[mask_bool] = fur_fu[mask_bool] - fur_bl[mask_bool]

    for name, data in [('delta_SUV', delta_suv), ('delta_TPR', delta_tpr), ('delta_FUR', delta_fur)]:
        nib.save(nib.Nifti1Image(data.astype(np.float32), midpoint_ref.affine),
                 str(subj_out_dir / f"{name}.nii.gz"))

    # =========================================================================
    # Step 10: Gaussian smoothing
    # =========================================================================
    logger.info(f"\n[Step 10] Applying {LONGITUDINAL_SMOOTHING_FWHM_MM}mm Gaussian smoothing...")

    voxel_dims = midpoint_ref.header.get_zooms()

    delta_suv_smooth = gaussian_smooth_3d(np.nan_to_num(delta_suv), LONGITUDINAL_SMOOTHING_FWHM_MM, voxel_dims)
    delta_tpr_smooth = gaussian_smooth_3d(np.nan_to_num(delta_tpr), LONGITUDINAL_SMOOTHING_FWHM_MM, voxel_dims)
    delta_fur_smooth = gaussian_smooth_3d(np.nan_to_num(delta_fur), LONGITUDINAL_SMOOTHING_FWHM_MM, voxel_dims)

    delta_suv_smooth[~mask_bool] = np.nan
    delta_tpr_smooth[~mask_bool] = np.nan
    delta_fur_smooth[~mask_bool] = np.nan

    for name, data in [('delta_SUV_smoothed_3mm', delta_suv_smooth),
                       ('delta_TPR_smoothed_3mm', delta_tpr_smooth),
                       ('delta_FUR_smoothed_3mm', delta_fur_smooth)]:
        nib.save(nib.Nifti1Image(data.astype(np.float32), midpoint_ref.affine),
                 str(subj_out_dir / f"{name}.nii.gz"))

    create_delta_maps_qc(
        delta_suv_smooth, delta_tpr_smooth, delta_fur_smooth, intersection_mask,
        LONGITUDINAL_QC_DIR / f"{subject_id}_delta_maps.png",
        subject_id
    )

    # =========================================================================
    # Step 11: Statistics
    # =========================================================================
    logger.info(f"\n[Step 11] Computing voxel-wise statistics...")

    results = {
        'subject_id': subject_id,
        'baseline_session': baseline_session,
        'followup_session': followup_session,
        'mask_stats': mask_stats,
    }

    for metric_name, delta_data in [('suv', delta_suv_smooth),
                                     ('tpr', delta_tpr_smooth),
                                     ('fur', delta_fur_smooth)]:
        valid = delta_data[mask_bool & np.isfinite(delta_data)]
        if len(valid) > 0:
            t_stat, t_pval = stats.ttest_1samp(valid, 0)
            w_stat, w_pval = stats.wilcoxon(valid)
            results[f'delta_{metric_name}_mean'] = float(np.mean(valid))
            results[f'delta_{metric_name}_std'] = float(np.std(valid))
            results[f'delta_{metric_name}_t_stat'] = float(t_stat)
            results[f'delta_{metric_name}_t_pval'] = float(t_pval)
            results[f'delta_{metric_name}_w_stat'] = float(w_stat)
            results[f'delta_{metric_name}_w_pval'] = float(w_pval)
            logger.info(f"  Delta {metric_name.upper()}: mean={np.mean(valid):.4f}, t={t_stat:.2f}, p={t_pval:.4f}")

    results['status'] = 'success'
    logger.info(f"\n  {subject_id} processing complete")
    return results


def main():
    parser = argparse.ArgumentParser(description='Longitudinal voxel-wise FDG-PET analysis')
    parser.add_argument('--subject', type=str, help='Process specific subject (e.g., sub-101)')
    parser.add_argument('--force', action='store_true', help='Force re-processing')
    parser.add_argument('--trimming', type=str, default=LONGITUDINAL_DEFAULT_TRIMMING,
                        help='Tongue trimming level (default: 0mm = original tongue mask)')
    args = parser.parse_args()

    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOGNOTES_DIR / f"longitudinal_coregistration_{timestamp}.log"
    ensure_directories()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    logger.info("=" * 70)
    logger.info("LONGITUDINAL VOXEL-WISE FDG-PET ANALYSIS")
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Trimming: {args.trimming}")
    logger.info(f"Smoothing FWHM: {LONGITUDINAL_SMOOTHING_FWHM_MM}mm")
    logger.info("=" * 70)

    # Load data
    blinding = load_blinding_key()
    ecrf_data = load_ecrf_data()

    # Determine subjects
    if args.subject:
        subjects = [args.subject]
    else:
        subjects = sorted([s for s in blinding.keys() if 'Baseline' in blinding[s] and 'Followup' in blinding[s]])

    logger.info(f"Subjects to process: {len(subjects)}")

    # Process
    results = []
    for subject_id in subjects:
        if subject_id not in blinding:
            logger.warning(f"Subject {subject_id} not in blinding key, skipping")
            continue

        baseline_session = blinding[subject_id].get('Baseline')
        followup_session = blinding[subject_id].get('Followup')

        if not baseline_session or not followup_session:
            logger.warning(f"Missing sessions for {subject_id}, skipping")
            continue

        try:
            result = process_subject(
                subject_id, baseline_session, followup_session,
                ecrf_data, trimming=args.trimming, force=args.force
            )
            results.append(result)
        except Exception as e:
            logger.error(f"Error processing {subject_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            results.append({'subject_id': subject_id, 'status': 'error', 'error': str(e)})

    # Save results
    logger.info("\n" + "=" * 70)
    logger.info("AGGREGATING GROUP STATISTICS")
    logger.info("=" * 70)

    df = pd.DataFrame([r for r in results if r.get('status') == 'success'])
    if not df.empty:
        output_file = OUTPUTS_DIR / "longitudinal_voxelwise_stats.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"Saved: {output_file} ({len(df)} rows)")

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    success = sum(1 for r in results if r.get('status') == 'success')
    failed = len(results) - success
    logger.info(f"Total subjects: {len(results)}")
    logger.info(f"  Success: {success}")
    logger.info(f"  Failed:  {failed}")
    logger.info(f"Log: {log_file}")
    logger.info(f"Completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
