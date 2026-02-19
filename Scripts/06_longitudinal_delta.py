#!/usr/bin/env python3
"""
06_longitudinal_delta.py - Longitudinal Delta Analysis

Computes voxel-wise Followup-Baseline difference images in unbiased midpoint space.

This script creates:
1. CT↔CT registration between Baseline and Followup sessions
2. Unbiased midpoint space (neither timepoint privileged)
3. PET images resampled to midpoint space via CT
4. Intersection mask (voxels present in both sessions)
5. Delta images (Followup - Baseline) within the intersection mask

Usage:
    python 06_longitudinal_delta.py                    # All subjects
    python 06_longitudinal_delta.py --subject sub-101  # Single subject
    python 06_longitudinal_delta.py --force            # Force re-run
    python 06_longitudinal_delta.py --mask-type jaw    # Use jaw-level masks (default)
    python 06_longitudinal_delta.py --mask-type tooth  # Use per-tooth masks

Output:
    DerivedData/longitudinal/sub-XXX/
        transforms/         - CT↔CT and midpoint transforms
        midpoint_space/     - All images in midpoint space
        delta/              - Delta images and statistics
    QC/longitudinal/sub-XXX/
        - Alignment and delta visualizations
"""

import argparse
import json
import logging
import sys
import gc
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, Optional, List

import numpy as np
import nibabel as nib
import ants
from scipy.linalg import sqrtm, inv
from scipy.spatial.transform import Rotation
from scipy.ndimage import gaussian_filter
from scipy.stats import ttest_1samp

# Add Scripts directory to path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

from config import (
    RAWDATA_DIR, DERIVED_DIR, QC_DIR, LOGNOTES_DIR, OUTPUTS_DIR,
    BLINDING_KEY_FILE, TRANSFORM_DIR, TOTALSEG_ROI_DIR,
    LONGITUDINAL_DIR, LONGITUDINAL_QC_DIR, INPUT_FUNC_DIR,
    LONGITUDINAL_SMOOTHING_FWHM_MM, LONGITUDINAL_OUTPUT_DIR,
    ensure_directories
)
from utils.io_utils import (
    load_nifti, discover_subjects, discover_sessions,
    load_blinding_key, find_pet_file, find_ct_file
)
from utils.registration_utils import crop_ct_to_pet_fov

import pandas as pd
from scipy.interpolate import interp1d


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def get_subject_sessions_map(blinding_map: dict) -> dict:
    """
    Create mapping from subject_id to {timepoint: session_id}.

    Args:
        blinding_map: Dict from (subject_id, session_id) to timepoint

    Returns:
        Dict mapping subject_id to {timepoint: session_id}
    """
    subject_sessions = {}
    for (subj, sess), tp in blinding_map.items():
        if subj not in subject_sessions:
            subject_sessions[subj] = {}
        subject_sessions[subj][tp] = sess
    return subject_sessions


def get_sessions_for_subject(subject_id: str, blinding_map: dict) -> list:
    """Get all session IDs for a given subject."""
    sessions = []
    for (subj, sess), _ in blinding_map.items():
        if subj == subject_id:
            sessions.append(sess)
    return sessions


def get_session_timepoint(subject_id: str, session_id: str, blinding_map: dict) -> Optional[str]:
    """Get the timepoint (Baseline/Followup) for a given session."""
    return blinding_map.get((subject_id, session_id))


# Registration parameters
CT_CT_REGISTRATION_TYPE = "Rigid"  # Rigid (6 DOF) for longitudinal

# Mask threshold for intersection
MASK_INTERSECTION_THRESHOLD = 0.5

# Smoothing parameter (3mm FWHM)
SMOOTHING_FWHM_MM = 3.0


def fwhm_to_sigma(fwhm_mm: float, voxel_size_mm: np.ndarray) -> np.ndarray:
    """
    Convert FWHM in mm to sigma in voxels for each dimension.

    FWHM = 2 * sqrt(2 * ln(2)) * sigma ≈ 2.355 * sigma

    Args:
        fwhm_mm: FWHM in millimeters
        voxel_size_mm: Voxel size in mm for each dimension (x, y, z)

    Returns:
        sigma in voxels for each dimension
    """
    sigma_mm = fwhm_mm / (2.0 * np.sqrt(2.0 * np.log(2.0)))  # ~2.355
    sigma_voxels = sigma_mm / voxel_size_mm
    return sigma_voxels


def smooth_pet_image(pet_data: np.ndarray, affine: np.ndarray,
                     fwhm_mm: float = SMOOTHING_FWHM_MM) -> np.ndarray:
    """
    Apply Gaussian smoothing to PET image.

    Args:
        pet_data: 3D PET data array
        affine: NIfTI affine matrix (used to extract voxel sizes)
        fwhm_mm: Full Width at Half Maximum in millimeters

    Returns:
        Smoothed PET data
    """
    # Extract voxel sizes from affine matrix
    voxel_sizes = np.abs(np.diag(affine[:3, :3]))

    # Convert FWHM to sigma in voxels
    sigma_voxels = fwhm_to_sigma(fwhm_mm, voxel_sizes)

    # Apply Gaussian filter
    smoothed = gaussian_filter(pet_data.astype(np.float64), sigma=sigma_voxels)

    return smoothed.astype(np.float32)


# ==============================================================================
# INPUT FUNCTION AND METRICS (matching 04_batch_quantify.py)
# ==============================================================================

def load_input_function(subject_id: str, timepoint: str) -> Optional[pd.DataFrame]:
    """
    Load input function data for TPR/FUR calculation.

    Args:
        subject_id: Subject ID (e.g., 'sub-101')
        timepoint: 'Baseline' or 'Followup'

    Returns:
        DataFrame with time_s and plasma_Bq_mL columns, or None if not found
    """
    if_file = INPUT_FUNC_DIR / f"{subject_id}_ses-{timepoint}_if_processed.csv"
    if not if_file.exists():
        return None

    df = pd.read_csv(if_file)

    # Ensure consistent column naming
    if 'activity_Bq_mL' in df.columns and 'plasma_Bq_mL' not in df.columns:
        df = df.rename(columns={'activity_Bq_mL': 'plasma_Bq_mL'})

    return df


def load_suv_scaling_factors() -> Dict:
    """Load SUV scaling factors from the curated CSV file.

    This file contains verified weight and dose data for all sessions.
    Format: subject_id;session_blinded;session_unblinded;injected_MBq;body_weight_kg;...
    """
    suv_file = RAWDATA_DIR / "SUV_info" / "session_scaling_factors.csv"
    if not suv_file.exists():
        return {}

    # CSV uses semicolon separator
    df = pd.read_csv(suv_file, sep=';')

    data = {}
    for _, row in df.iterrows():
        subject_id = row.get('subject_id')
        timepoint = row.get('session_unblinded')  # 'Baseline' or 'Followup'
        dose_mbq = row.get('injected_MBq')
        weight_kg = row.get('body_weight_kg')

        if pd.notna(subject_id) and pd.notna(timepoint) and pd.notna(dose_mbq) and pd.notna(weight_kg):
            data[(subject_id, timepoint)] = {
                'weight_kg': float(weight_kg),
                'dose_mbq': float(dose_mbq)
            }

    return data


def load_pet_json(subject_id: str, timepoint: str) -> Dict:
    """Load PET JSON sidecar for scan timing.

    Priority:
    1. Updated JSON from json_side_cars_updated/ (preferred, has correct timing)
    2. Returns empty dict if not found
    """
    updated_json_dir = RAWDATA_DIR / "json_side_cars_updated"
    updated_json = updated_json_dir / f"{subject_id}_ses-{timepoint}_trc-18FFDG_rec-StaticMoCo_chunk-1_pet.json"

    if updated_json.exists():
        with open(updated_json) as f:
            return json.load(f)

    return {}


def compute_plasma_denominators(if_data: pd.DataFrame,
                                 scan_start_s: float,
                                 scan_end_s: float,
                                 tissue_time_s: float) -> Dict:
    """
    Compute plasma denominators for TPR and FUR.

    Args:
        if_data: Input function DataFrame
        scan_start_s: Scan start time (default: 1800s = 30min)
        scan_end_s: Scan end time (default: 3600s = 60min)
        tissue_time_s: Tissue measurement time (default: 2700s = 45min)

    Returns:
        Dict with plasma_mean_Bq_mL and plasma_auc_0_to_T_Bq_s_mL
    """
    time_s = if_data['time_s'].values
    plasma = if_data['plasma_Bq_mL'].values

    # Mean plasma during scan window
    scan_mask = (time_s >= scan_start_s) & (time_s <= scan_end_s)
    if np.sum(scan_mask) >= 2:
        plasma_mean = float(np.mean(plasma[scan_mask]))
    else:
        # Interpolate if not enough points
        interp_func = interp1d(time_s, plasma, kind='linear', fill_value='extrapolate')
        plasma_mean = float(interp_func(tissue_time_s))

    # AUC from 0 to tissue_time for FUR
    auc_mask = time_s <= tissue_time_s
    auc = float(np.trapz(plasma[auc_mask], time_s[auc_mask]))

    return {
        'plasma_mean_Bq_mL': plasma_mean,
        'plasma_auc_0_to_T_Bq_s_mL': auc
    }


def compute_roi_metrics(pet_data: np.ndarray, mask: np.ndarray,
                        voxel_vol_ml: float, suv_scaler: float,
                        plasma_mean: float, auc_0_to_T: float) -> Dict:
    """
    Compute SUV, TPR, FUR metrics for a ROI.

    Args:
        pet_data: PET data in kBq/mL (typical convention for FDG-PET)
        mask: Binary mask
        voxel_vol_ml: Voxel volume in mL
        suv_scaler: SUV scaling factor (body_weight_kg / dose_kBq)
        plasma_mean: Mean plasma activity during scan (kBq/mL)
        auc_0_to_T: Plasma AUC from 0 to tissue time (kBq·s/mL)

    Returns:
        Dict with all metrics
    """
    if not np.any(mask):
        return None

    pet_vals = pet_data[mask]
    valid_idx = np.isfinite(pet_vals) & (pet_vals > 0)

    if not np.any(valid_idx):
        return None

    pet_valid = pet_vals[valid_idx]

    # Basic intensity metrics
    intensity_mean = float(np.mean(pet_valid))
    intensity_median = float(np.median(pet_valid))
    intensity_p90 = float(np.percentile(pet_valid, 90))
    n_voxels = int(np.sum(valid_idx))
    roi_volume_ml = float(n_voxels * voxel_vol_ml)

    # SUV
    suv_mean = intensity_mean * suv_scaler if np.isfinite(suv_scaler) else np.nan
    suv_p90 = intensity_p90 * suv_scaler if np.isfinite(suv_scaler) else np.nan

    # TPR (Tissue-to-Plasma Ratio)
    if np.isfinite(plasma_mean) and plasma_mean > 0:
        tpr_mean = intensity_mean / plasma_mean
        tpr_p90 = intensity_p90 / plasma_mean
    else:
        tpr_mean = tpr_p90 = np.nan

    # FUR (Fractional Uptake Rate) in min^-1
    if np.isfinite(auc_0_to_T) and auc_0_to_T > 0:
        fur_mean = (intensity_mean / auc_0_to_T) * 60  # Convert to per min
        fur_p90 = (intensity_p90 / auc_0_to_T) * 60
    else:
        fur_mean = fur_p90 = np.nan

    return {
        'n_voxels': n_voxels,
        'roi_volume_ml': roi_volume_ml,
        'intensity_mean_Bq_mL': intensity_mean,
        'intensity_p90_Bq_mL': intensity_p90,
        'SUV_mean': suv_mean,
        'SUV_p90': suv_p90,
        'TPR_mean': tpr_mean,
        'TPR_p90': tpr_p90,
        'FUR_mean_per_min': fur_mean,
        'FUR_p90_per_min': fur_p90
    }


# ==============================================================================
# LOGGING SETUP
# ==============================================================================

def setup_logging(log_dir: Path) -> logging.Logger:
    """Setup logging to file and console."""
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"longitudinal_delta_{timestamp}.log"

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


# ==============================================================================
# TRANSFORM UTILITIES
# ==============================================================================

def extract_rigid_params_from_ants(transform_file: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract rotation matrix and translation vector from ANTs .mat file.

    ANTs stores rigid transforms as:
    - 3x3 rotation matrix
    - 3x1 translation vector
    - 3x1 center of rotation

    Returns:
        rotation: 3x3 rotation matrix
        translation: 3x1 translation vector
    """
    # Read the ANTs transform
    tx = ants.read_transform(str(transform_file))
    params = tx.parameters
    fixed_params = tx.fixed_parameters

    # For ANTs rigid transforms:
    # params[0:9] = rotation matrix (row-major)
    # params[9:12] = translation
    # fixed_params[0:3] = center of rotation

    if len(params) == 12:
        # Affine/Rigid transform
        rotation = np.array(params[0:9]).reshape(3, 3)
        translation = np.array(params[9:12])
    elif len(params) == 6:
        # Euler angle representation (versor)
        # params[0:3] = versor (quaternion-like)
        # params[3:6] = translation
        versor = np.array(params[0:3])
        translation = np.array(params[3:6])
        # Convert versor to rotation matrix
        # ANTs versor: [vx, vy, vz] where angle = 2*acos(sqrt(1 - vx^2 - vy^2 - vz^2))
        v_norm_sq = np.sum(versor**2)
        if v_norm_sq < 1:
            w = np.sqrt(1 - v_norm_sq)
            quat = np.array([w, versor[0], versor[1], versor[2]])
            rotation = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
        else:
            rotation = np.eye(3)
    else:
        raise ValueError(f"Unexpected number of parameters: {len(params)}")

    return rotation, translation


def compute_half_rigid_transform(rotation: np.ndarray, translation: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the "half" of a rigid transform (square root).

    For a rigid transform T = [R, t], the half-transform is T^0.5 = [sqrt(R), t/2]

    The square root of a rotation matrix is computed via:
    1. Convert to axis-angle representation
    2. Halve the angle
    3. Convert back to rotation matrix

    Args:
        rotation: 3x3 rotation matrix
        translation: 3x1 translation vector

    Returns:
        half_rotation: 3x3 rotation matrix (square root)
        half_translation: 3x1 translation vector (halved)
    """
    # Convert rotation matrix to axis-angle
    r = Rotation.from_matrix(rotation)
    rotvec = r.as_rotvec()  # axis * angle

    # Halve the rotation
    half_rotvec = rotvec / 2.0
    half_rotation = Rotation.from_rotvec(half_rotvec).as_matrix()

    # Halve the translation
    half_translation = translation / 2.0

    return half_rotation, half_translation


def create_ants_rigid_transform(rotation: np.ndarray, translation: np.ndarray,
                                 center: np.ndarray = None) -> ants.ANTsTransform:
    """
    Create an ANTs rigid transform from rotation matrix and translation.

    Args:
        rotation: 3x3 rotation matrix
        translation: 3x1 translation vector
        center: 3x1 center of rotation (default: origin)

    Returns:
        ANTs transform object
    """
    if center is None:
        center = np.zeros(3)

    # Create affine matrix (4x4)
    affine = np.eye(4)
    affine[:3, :3] = rotation
    affine[:3, 3] = translation

    # Create ANTs transform
    tx = ants.create_ants_transform(
        transform_type='AffineTransform',
        precision='float',
        dimension=3
    )

    # Set parameters: rotation matrix (row-major) + translation
    params = list(rotation.flatten()) + list(translation)
    tx.set_parameters(params)
    tx.set_fixed_parameters(list(center))

    return tx


def save_ants_transform(transform: ants.ANTsTransform, output_path: Path):
    """Save ANTs transform to file."""
    ants.write_transform(transform, str(output_path))


# ==============================================================================
# REGISTRATION FUNCTIONS
# ==============================================================================

def register_ct_to_ct(ct_a_nib: nib.Nifti1Image, ct_b_nib: nib.Nifti1Image,
                       logger: logging.Logger) -> Dict:
    """
    Register CT_B to CT_A using rigid registration.

    Args:
        ct_a_nib: Baseline CT (fixed/reference)
        ct_b_nib: Followup CT (moving)
        logger: Logger instance

    Returns:
        Dictionary with registration result and transforms
    """
    logger.info("  Registering CT_B → CT_A (rigid)...")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Save as temp files for ANTs
        ct_a_file = tmpdir / "ct_a.nii.gz"
        ct_b_file = tmpdir / "ct_b.nii.gz"
        nib.save(ct_a_nib, str(ct_a_file))
        nib.save(ct_b_nib, str(ct_b_file))

        # Load as ANTs images
        ct_a_ants = ants.image_read(str(ct_a_file))
        ct_b_ants = ants.image_read(str(ct_b_file))

        # Register
        reg_result = ants.registration(
            fixed=ct_a_ants,
            moving=ct_b_ants,
            type_of_transform=CT_CT_REGISTRATION_TYPE,
            verbose=False
        )

        # Extract registration metrics
        warped_ct_b = reg_result['warpedmovout']

        # Compute translation magnitude from transform file
        fwd_transform = reg_result['fwdtransforms'][0]
        rotation, translation = extract_rigid_params_from_ants(Path(fwd_transform))

        translation_mag = np.linalg.norm(translation)
        rotation_angles = Rotation.from_matrix(rotation).as_euler('xyz', degrees=True)

        logger.info(f"    Translation: [{translation[0]:.2f}, {translation[1]:.2f}, {translation[2]:.2f}] mm")
        logger.info(f"    Translation magnitude: {translation_mag:.2f} mm")
        logger.info(f"    Rotation: [{rotation_angles[0]:.2f}, {rotation_angles[1]:.2f}, {rotation_angles[2]:.2f}] deg")

        return {
            'fwdtransforms': reg_result['fwdtransforms'],
            'invtransforms': reg_result['invtransforms'],
            'warped_moving': warped_ct_b,
            'rotation': rotation,
            'translation': translation,
            'translation_magnitude_mm': translation_mag,
            'rotation_angles_deg': rotation_angles.tolist()
        }


def compute_midpoint_transforms(ct_a_nib: nib.Nifti1Image, ct_b_nib: nib.Nifti1Image,
                                 reg_result: Dict, logger: logging.Logger) -> Dict:
    """
    Compute transforms to midpoint space from CT↔CT registration.

    Given T_B→A (transforms CT_B into CT_A space), compute:
    - T_A→mid: Transform CT_A to midpoint
    - T_B→mid: Transform CT_B to midpoint

    The midpoint is defined such that:
    - T_A→mid = inv(T_B→A^0.5) = T_B→A^(-0.5)
    - T_B→mid = T_B→A^0.5

    This ensures both timepoints are equally transformed.

    Args:
        ct_a_nib: Baseline CT
        ct_b_nib: Followup CT
        reg_result: Result from register_ct_to_ct()
        logger: Logger instance

    Returns:
        Dictionary with midpoint transforms and reference image
    """
    logger.info("  Computing midpoint transforms...")

    # Extract rotation and translation from B→A transform
    rotation_ba = reg_result['rotation']
    translation_ba = reg_result['translation']

    # Compute half transforms
    half_rotation, half_translation = compute_half_rigid_transform(rotation_ba, translation_ba)

    # T_B→mid = T_B→A^0.5 (half the transform from B toward A)
    rotation_b_to_mid = half_rotation
    translation_b_to_mid = half_translation

    # T_A→mid = inv(T_B→A^0.5) = T_B→A^(-0.5)
    # This is the inverse of the half-transform
    rotation_a_to_mid = half_rotation.T  # Transpose = inverse for rotation matrix
    translation_a_to_mid = -rotation_a_to_mid @ half_translation

    logger.info(f"    T_A→mid translation: [{translation_a_to_mid[0]:.3f}, {translation_a_to_mid[1]:.3f}, {translation_a_to_mid[2]:.3f}] mm")
    logger.info(f"    T_B→mid translation: [{translation_b_to_mid[0]:.3f}, {translation_b_to_mid[1]:.3f}, {translation_b_to_mid[2]:.3f}] mm")

    # Create ANTs transforms
    # Use center of CT_A as rotation center
    ct_a_center = np.array(ct_a_nib.shape[:3]) / 2.0
    ct_a_center_world = nib.affines.apply_affine(ct_a_nib.affine, ct_a_center)

    t_a_to_mid = create_ants_rigid_transform(rotation_a_to_mid, translation_a_to_mid, ct_a_center_world)
    t_b_to_mid = create_ants_rigid_transform(rotation_b_to_mid, translation_b_to_mid, ct_a_center_world)

    # Create midpoint reference image
    # Use CT_A's grid with averaged affine
    midpoint_affine = ct_a_nib.affine.copy()
    # Apply small shift to make it truly "midpoint" (optional refinement)
    # For now, use CT_A's grid as the midpoint grid

    midpoint_ref = nib.Nifti1Image(
        np.zeros(ct_a_nib.shape[:3], dtype=np.float32),
        midpoint_affine
    )

    return {
        't_a_to_mid': t_a_to_mid,
        't_b_to_mid': t_b_to_mid,
        'rotation_a_to_mid': rotation_a_to_mid,
        'translation_a_to_mid': translation_a_to_mid,
        'rotation_b_to_mid': rotation_b_to_mid,
        'translation_b_to_mid': translation_b_to_mid,
        'midpoint_ref': midpoint_ref,
        'center': ct_a_center_world
    }


# ==============================================================================
# RESAMPLING FUNCTIONS
# ==============================================================================

def resample_to_midpoint(source_nib: nib.Nifti1Image,
                          midpoint_ref: nib.Nifti1Image,
                          transform: ants.ANTsTransform,
                          interpolator: str = 'linear') -> nib.Nifti1Image:
    """
    Resample an image to midpoint space using a single transform.

    Args:
        source_nib: Source image (nibabel)
        midpoint_ref: Midpoint reference image defining output grid
        transform: ANTs transform from source to midpoint
        interpolator: 'linear', 'nearestNeighbor', or 'genericLabel'

    Returns:
        Resampled image in midpoint space
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Save images
        source_file = tmpdir / "source.nii.gz"
        ref_file = tmpdir / "ref.nii.gz"
        transform_file = tmpdir / "transform.mat"

        nib.save(source_nib, str(source_file))
        nib.save(midpoint_ref, str(ref_file))
        save_ants_transform(transform, transform_file)

        # Load as ANTs
        source_ants = ants.image_read(str(source_file))
        ref_ants = ants.image_read(str(ref_file))

        # Apply transform
        warped = ants.apply_transforms(
            fixed=ref_ants,
            moving=source_ants,
            transformlist=[str(transform_file)],
            interpolator=interpolator
        )

        # Convert back to nibabel
        warped_file = tmpdir / "warped.nii.gz"
        ants.image_write(warped, str(warped_file))
        warped_nib = nib.load(str(warped_file))

        return nib.Nifti1Image(warped_nib.get_fdata().copy(), warped_nib.affine)


def resample_pet_to_midpoint(pet_nib: nib.Nifti1Image,
                              ct_nib: nib.Nifti1Image,
                              midpoint_ref: nib.Nifti1Image,
                              ct_to_pet_inv_file: Path,
                              ct_to_mid_transform: ants.ANTsTransform,
                              logger: logging.Logger) -> nib.Nifti1Image:
    """
    Resample PET to midpoint space via CT space.

    Transform chain: PET → CT → Midpoint
    ANTs applies transforms in reverse order, so:
    transformlist = [ct_to_mid, pet_to_ct] applies pet_to_ct first, then ct_to_mid

    Args:
        pet_nib: Native PET image
        ct_nib: Cropped CT image (defines CT space for transforms)
        midpoint_ref: Midpoint reference image
        ct_to_pet_inv_file: Inverse CT→PET transform (i.e., PET→CT)
        ct_to_mid_transform: Transform from CT to midpoint
        logger: Logger instance

    Returns:
        PET resampled to midpoint space
    """
    logger.info("    Resampling PET to midpoint space...")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Save images
        pet_file = tmpdir / "pet.nii.gz"
        ref_file = tmpdir / "ref.nii.gz"
        ct_to_mid_file = tmpdir / "ct_to_mid.mat"

        nib.save(pet_nib, str(pet_file))
        nib.save(midpoint_ref, str(ref_file))
        save_ants_transform(ct_to_mid_transform, ct_to_mid_file)

        # Load as ANTs
        pet_ants = ants.image_read(str(pet_file))
        ref_ants = ants.image_read(str(ref_file))

        # Apply chained transforms: PET → CT → Midpoint
        # ANTs applies in reverse order, so list is [ct_to_mid, pet_to_ct]
        warped = ants.apply_transforms(
            fixed=ref_ants,
            moving=pet_ants,
            transformlist=[str(ct_to_mid_file), str(ct_to_pet_inv_file)],
            interpolator='linear'
        )

        # Convert back to nibabel
        warped_file = tmpdir / "warped_pet.nii.gz"
        ants.image_write(warped, str(warped_file))
        warped_nib = nib.load(str(warped_file))

        return nib.Nifti1Image(warped_nib.get_fdata().copy(), warped_nib.affine)


def resample_mask_to_midpoint(mask_nib: nib.Nifti1Image,
                               midpoint_ref: nib.Nifti1Image,
                               ct_to_mid_transform: ants.ANTsTransform,
                               logger: logging.Logger) -> nib.Nifti1Image:
    """
    Resample mask to midpoint space.

    Masks are already in CT space, so only need CT → Midpoint transform.
    Uses linear interpolation to produce continuous [0, 1] output.

    Args:
        mask_nib: Mask in CT space (binary or continuous)
        midpoint_ref: Midpoint reference image
        ct_to_mid_transform: Transform from CT to midpoint
        logger: Logger instance

    Returns:
        Mask resampled to midpoint space (continuous [0, 1])
    """
    # Ensure mask is float for linear interpolation
    mask_data = mask_nib.get_fdata().astype(np.float32)
    mask_float = nib.Nifti1Image(mask_data, mask_nib.affine)

    return resample_to_midpoint(
        mask_float,
        midpoint_ref,
        ct_to_mid_transform,
        interpolator='linear'
    )


# ==============================================================================
# DELTA COMPUTATION
# ==============================================================================

def create_intersection_mask(mask_a: np.ndarray, mask_b: np.ndarray,
                              threshold: float = MASK_INTERSECTION_THRESHOLD) -> np.ndarray:
    """
    Create binary intersection mask from two continuous masks.

    Args:
        mask_a: Mask A in midpoint space (continuous [0, 1])
        mask_b: Mask B in midpoint space (continuous [0, 1])
        threshold: Threshold for binarization (default: 0.5)

    Returns:
        Binary intersection mask (both masks > threshold)
    """
    return (mask_a > threshold) & (mask_b > threshold)


def compute_delta(pet_a: np.ndarray, pet_b: np.ndarray,
                   mask: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """
    Compute delta image (Followup - Baseline) within mask.

    Args:
        pet_a: Baseline PET in midpoint space
        pet_b: Followup PET in midpoint space
        mask: Binary intersection mask

    Returns:
        delta: Delta image (masked, NaN outside)
        stats: Dictionary of delta statistics
    """
    # Compute delta
    delta = pet_b - pet_a

    # Mask to NaN outside
    delta_masked = np.where(mask, delta, np.nan)

    # Compute statistics within mask
    delta_values = delta[mask]

    stats = {
        'mean_delta': float(np.mean(delta_values)),
        'std_delta': float(np.std(delta_values)),
        'median_delta': float(np.median(delta_values)),
        'p5_delta': float(np.percentile(delta_values, 5)),
        'p25_delta': float(np.percentile(delta_values, 25)),
        'p75_delta': float(np.percentile(delta_values, 75)),
        'p95_delta': float(np.percentile(delta_values, 95)),
        'min_delta': float(np.min(delta_values)),
        'max_delta': float(np.max(delta_values)),
        'n_voxels_total': int(np.sum(mask)),
        'n_voxels_increase': int(np.sum(delta_values > 0)),
        'n_voxels_decrease': int(np.sum(delta_values < 0)),
        'pct_increase': float(100.0 * np.sum(delta_values > 0) / len(delta_values))
    }

    return delta_masked, stats


# ==============================================================================
# QC VISUALIZATION
# ==============================================================================

def create_qc_visualizations(subject_id: str,
                              ct_a_mid: nib.Nifti1Image,
                              ct_b_mid: nib.Nifti1Image,
                              pet_a_mid: nib.Nifti1Image,
                              pet_b_mid: nib.Nifti1Image,
                              mask_intersection: np.ndarray,
                              delta: np.ndarray,
                              qc_dir: Path,
                              trimming: str,
                              logger: logging.Logger):
    """
    Generate QC visualizations for longitudinal analysis.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    logger.info("  Generating QC visualizations...")
    qc_dir.mkdir(parents=True, exist_ok=True)

    # Get data
    ct_a_data = ct_a_mid.get_fdata()
    ct_b_data = ct_b_mid.get_fdata()
    pet_a_data = pet_a_mid.get_fdata()
    pet_b_data = pet_b_mid.get_fdata()

    # Find best slices for each view based on mask content
    # Axial: sum over X,Y to find best Z
    axial_sums = np.sum(mask_intersection, axis=(0, 1))
    best_axial = np.argmax(axial_sums)

    # Coronal: sum over X,Z to find best Y
    coronal_sums = np.sum(mask_intersection, axis=(0, 2))
    best_coronal = np.argmax(coronal_sums)

    # Sagittal: sum over Y,Z to find best X
    sagittal_sums = np.sum(mask_intersection, axis=(1, 2))
    best_sagittal = np.argmax(sagittal_sums)

    logger.info(f"    QC slices: axial Z={best_axial}, coronal Y={best_coronal}, sagittal X={best_sagittal}")

    # 1. CT alignment overlay
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, (data_a, data_b, title) in zip(axes, [
        (ct_a_data[:, :, best_axial], ct_b_data[:, :, best_axial], 'Axial'),
        (ct_a_data[:, best_coronal, :], ct_b_data[:, best_coronal, :], 'Coronal'),
        (ct_a_data[best_sagittal, :, :], ct_b_data[best_sagittal, :, :], 'Sagittal')
    ]):
        # Checkerboard overlay
        checker = np.zeros_like(data_a)
        block_size = 20
        for i in range(0, checker.shape[0], block_size):
            for j in range(0, checker.shape[1], block_size):
                if ((i // block_size) + (j // block_size)) % 2 == 0:
                    checker[i:i+block_size, j:j+block_size] = data_a[i:i+block_size, j:j+block_size]
                else:
                    checker[i:i+block_size, j:j+block_size] = data_b[i:i+block_size, j:j+block_size]

        ax.imshow(checker.T, cmap='gray', origin='lower', vmin=-500, vmax=1500)
        ax.set_title(f'{title} - CT Checkerboard (A/B)')
        ax.axis('off')

    plt.suptitle(f'{subject_id} - CT Alignment in Baseline Reference Space')
    plt.tight_layout()
    plt.savefig(qc_dir / f'{subject_id}_ct_alignment.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 2. PET comparison
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    vmax_pet = np.percentile(np.concatenate([pet_a_data[mask_intersection], pet_b_data[mask_intersection]]), 95)

    # Row 1: PET A (Baseline, reference)
    axes[0, 0].imshow(pet_a_data[:, :, best_axial].T, cmap='hot', origin='lower', vmin=0, vmax=vmax_pet)
    axes[0, 0].set_title('Baseline PET [Reference] (Axial)')
    axes[0, 1].imshow(pet_a_data[:, best_coronal, :].T, cmap='hot', origin='lower', vmin=0, vmax=vmax_pet)
    axes[0, 1].set_title('Baseline PET [Reference] (Coronal)')
    axes[0, 2].imshow(pet_a_data[best_sagittal, :, :].T, cmap='hot', origin='lower', vmin=0, vmax=vmax_pet)
    axes[0, 2].set_title('Baseline PET [Reference] (Sagittal)')

    # Row 2: PET B (Followup, registered to Baseline)
    axes[1, 0].imshow(pet_b_data[:, :, best_axial].T, cmap='hot', origin='lower', vmin=0, vmax=vmax_pet)
    axes[1, 0].set_title('Followup PET [Registered] (Axial)')
    axes[1, 1].imshow(pet_b_data[:, best_coronal, :].T, cmap='hot', origin='lower', vmin=0, vmax=vmax_pet)
    axes[1, 1].set_title('Followup PET [Registered] (Coronal)')
    axes[1, 2].imshow(pet_b_data[best_sagittal, :, :].T, cmap='hot', origin='lower', vmin=0, vmax=vmax_pet)
    axes[1, 2].set_title('Followup PET [Registered] (Sagittal)')

    for ax in axes.flat:
        ax.axis('off')

    plt.suptitle(f'{subject_id} - PET in Baseline Reference Space (3mm smoothed)')
    plt.tight_layout()
    plt.savefig(qc_dir / f'{subject_id}_pet_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 3. Delta image
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Use diverging colormap centered at 0
    delta_masked = np.where(mask_intersection, delta, np.nan)
    vmax_delta = np.nanpercentile(np.abs(delta_masked), 95)
    norm = TwoSlopeNorm(vmin=-vmax_delta, vcenter=0, vmax=vmax_delta)

    im = axes[0].imshow(delta_masked[:, :, best_axial].T, cmap='RdBu_r', norm=norm, origin='lower')
    axes[0].set_title('Delta (Axial)')
    axes[1].imshow(delta_masked[:, best_coronal, :].T, cmap='RdBu_r', norm=norm, origin='lower')
    axes[1].set_title('Delta (Coronal)')
    axes[2].imshow(delta_masked[best_sagittal, :, :].T, cmap='RdBu_r', norm=norm, origin='lower')
    axes[2].set_title('Delta (Sagittal)')

    for ax in axes:
        ax.axis('off')

    plt.colorbar(im, ax=axes, label='Delta kBq/mL (Followup - Baseline)', shrink=0.8)
    plt.suptitle(f'{subject_id} - Delta Image (3mm smoothed, {trimming} tongue trim)\nBlue=Decrease, Red=Increase')
    plt.tight_layout()
    plt.savefig(qc_dir / f'{subject_id}_delta.png', dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"    QC images saved to: {qc_dir}")


# ==============================================================================
# MAIN PROCESSING
# ==============================================================================

def process_subject(subject_id: str, blinding_map: dict, force: bool = False,
                     mask_type: str = 'jaw', trimming: str = '0mm',
                     logger: logging.Logger = None) -> bool:
    """
    Process longitudinal delta analysis for one subject.

    Args:
        subject_id: Subject ID (e.g., 'sub-101')
        blinding_map: Dict from (subject_id, session_id) to timepoint
        force: Force re-run even if outputs exist
        mask_type: 'jaw' for jaw-level masks, 'tooth' for per-tooth
        trimming: Tongue trimming distance ('0mm', '3mm', '5mm', '8mm', '10mm')
        logger: Logger instance

    Returns:
        True if successful, False otherwise
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing: {subject_id}")
    logger.info(f"{'='*60}")

    # Setup output directories
    subj_output_dir = LONGITUDINAL_DIR / subject_id
    transform_dir = subj_output_dir / "transforms"
    midpoint_dir = subj_output_dir / "midpoint_space"
    delta_dir = subj_output_dir / "delta"
    qc_dir = LONGITUDINAL_QC_DIR / subject_id

    # Check if already done
    delta_file = delta_dir / "delta_pet_suv.nii.gz"
    if delta_file.exists() and not force:
        logger.info(f"  Delta image exists, skipping (use --force to re-run)")
        return True

    # Create directories
    for d in [transform_dir, midpoint_dir, delta_dir, qc_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Get sessions for this subject
    sessions = get_sessions_for_subject(subject_id, blinding_map)
    if len(sessions) != 2:
        logger.error(f"  Expected 2 sessions, found {len(sessions)}")
        return False

    # Identify Baseline and Followup
    session_timepoints = {}
    for ses_id in sessions:
        tp = get_session_timepoint(subject_id, ses_id, blinding_map)
        session_timepoints[tp] = ses_id

    if 'Baseline' not in session_timepoints or 'Followup' not in session_timepoints:
        logger.error(f"  Could not identify Baseline and Followup sessions")
        return False

    ses_a = session_timepoints['Baseline']
    ses_b = session_timepoints['Followup']
    logger.info(f"  Baseline: {ses_a}")
    logger.info(f"  Followup: {ses_b}")

    # -------------------------------------------------------------------------
    # Step 1: Load CT images (cropped)
    # -------------------------------------------------------------------------
    logger.info("  [Step 1] Loading CT images...")

    # Find CT files
    session_a_dir = RAWDATA_DIR / subject_id / ses_a
    session_b_dir = RAWDATA_DIR / subject_id / ses_b

    ct_a_file = find_ct_file(session_a_dir, prefer_bone=False)
    ct_b_file = find_ct_file(session_b_dir, prefer_bone=False)

    if ct_a_file is None or ct_b_file is None:
        logger.error(f"  Could not find CT files")
        return False

    ct_a_data, ct_a_nib = load_nifti(ct_a_file)
    ct_b_data, ct_b_nib = load_nifti(ct_b_file)

    # Find PET files (needed for cropping CT to PET FOV)
    pet_a_file = find_pet_file(session_a_dir)
    pet_b_file = find_pet_file(session_b_dir)

    if pet_a_file is None or pet_b_file is None:
        logger.error(f"  Could not find PET files")
        return False

    pet_a_data, pet_a_nib = load_nifti(pet_a_file)
    pet_b_data, pet_b_nib = load_nifti(pet_b_file)

    # Crop CTs to PET FOV (same as done for CT→PET registration)
    ct_a_cropped, crop_info_a = crop_ct_to_pet_fov(ct_a_nib, pet_a_nib)
    ct_b_cropped, crop_info_b = crop_ct_to_pet_fov(ct_b_nib, pet_b_nib)

    logger.info(f"    CT_A cropped: {ct_a_nib.shape} → {ct_a_cropped.shape}")
    logger.info(f"    CT_B cropped: {ct_b_nib.shape} → {ct_b_cropped.shape}")

    # -------------------------------------------------------------------------
    # Step 2: Register PET_B → PET_A directly (simpler approach)
    # -------------------------------------------------------------------------
    logger.info("  [Step 2] PET↔PET registration (direct approach)...")

    # Register PET_B to PET_A directly - this is simpler and more robust
    # than chaining CT→CT transforms through multiple coordinate systems
    import shutil

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        pet_a_file_tmp = tmpdir / "pet_a.nii.gz"
        pet_b_file_tmp = tmpdir / "pet_b.nii.gz"

        nib.save(pet_a_nib, str(pet_a_file_tmp))
        nib.save(pet_b_nib, str(pet_b_file_tmp))

        pet_a_ants = ants.image_read(str(pet_a_file_tmp))
        pet_b_ants = ants.image_read(str(pet_b_file_tmp))

        # Rigid registration of PET_B to PET_A
        reg_result = ants.registration(
            fixed=pet_a_ants,
            moving=pet_b_ants,
            type_of_transform='Rigid',
            verbose=False
        )

        # Extract registration parameters for logging
        tx_file = reg_result['fwdtransforms'][0]
        tx = ants.read_transform(tx_file)
        params = tx.parameters

        # Parse parameters (9 rotation params + 3 translation for AffineTransform)
        if len(params) == 12:
            translation = np.array(params[9:12])
        else:
            translation = np.array(params[-3:])

        translation_mag = np.linalg.norm(translation)
        logger.info(f"    Translation: [{translation[0]:.2f}, {translation[1]:.2f}, {translation[2]:.2f}] mm")
        logger.info(f"    Translation magnitude: {translation_mag:.2f} mm")

        # Save transforms
        for i, tf in enumerate(reg_result['fwdtransforms']):
            shutil.copy(tf, transform_dir / f"pet_b_to_pet_a_fwd_{i}.mat")
        for i, tf in enumerate(reg_result['invtransforms']):
            shutil.copy(tf, transform_dir / f"pet_b_to_pet_a_inv_{i}.mat")

        # Get warped PET_B in PET_A space
        pet_b_warped_ants = reg_result['warpedmovout']

    # -------------------------------------------------------------------------
    # Step 3: Use PET_A space as reference (skip complex midpoint computation)
    # -------------------------------------------------------------------------
    logger.info("  [Step 3] Using PET_A as reference space...")

    # Reference space is PET_A native space
    reference_nib = pet_a_nib

    # Save reference
    nib.save(reference_nib, str(midpoint_dir / "reference_space.nii.gz"))

    # Save metadata
    meta = {
        'subject_id': subject_id,
        'baseline_session': ses_a,
        'followup_session': ses_b,
        'registration': {
            'method': 'direct_pet_to_pet',
            'reference': 'PET_A (Baseline)',
            'translation_mm': translation.tolist(),
            'translation_magnitude_mm': float(translation_mag)
        },
        'crop_info_a': crop_info_a,
        'crop_info_b': crop_info_b
    }
    with open(transform_dir / "registration_meta.json", 'w') as f:
        json.dump(meta, f, indent=2)

    # -------------------------------------------------------------------------
    # Step 4: Save aligned PET images
    # -------------------------------------------------------------------------
    logger.info("  [Step 4] Saving aligned PET images...")

    # PET_A is already in reference space (itself)
    pet_a_aligned = pet_a_nib

    # PET_B warped to PET_A space
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        warped_file = tmpdir / "pet_b_warped.nii.gz"
        ants.image_write(pet_b_warped_ants, str(warped_file))
        pet_b_aligned = nib.load(str(warped_file))
        pet_b_aligned = nib.Nifti1Image(pet_b_aligned.get_fdata().copy(), pet_b_aligned.affine)

    nib.save(pet_a_aligned, str(midpoint_dir / "pet_a_aligned.nii.gz"))
    nib.save(pet_b_aligned, str(midpoint_dir / "pet_b_aligned.nii.gz"))

    # Also save as "in_midpoint" for compatibility with downstream code
    nib.save(pet_a_aligned, str(midpoint_dir / "pet_a_in_midpoint.nii.gz"))
    nib.save(pet_b_aligned, str(midpoint_dir / "pet_b_in_midpoint.nii.gz"))

    # Save CT images - cropped baseline CT for mask overlay
    # Also save to QC folder for easy access
    nib.save(ct_a_cropped, str(midpoint_dir / "ct_baseline_cropped.nii.gz"))
    nib.save(ct_a_cropped, str(qc_dir / f"{subject_id}_ct_baseline_cropped.nii.gz"))

    # For CT QC visualization - warp CT_B to match PET_A space
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        ct_b_file_tmp = tmpdir / "ct_b.nii.gz"
        ref_file_tmp = tmpdir / "ref.nii.gz"
        nib.save(ct_b_cropped, str(ct_b_file_tmp))
        nib.save(ct_a_cropped, str(ref_file_tmp))
        ct_b_ants = ants.image_read(str(ct_b_file_tmp))
        ref_ants = ants.image_read(str(ref_file_tmp))
        ct_b_warped = ants.apply_transforms(
            fixed=ref_ants,
            moving=ct_b_ants,
            transformlist=[str(transform_dir / "pet_b_to_pet_a_fwd_0.mat")],
            interpolator='linear'
        )
        warped_file = tmpdir / "ct_b_warped.nii.gz"
        ants.image_write(ct_b_warped, str(warped_file))
        ct_b_warped_nib = nib.load(str(warped_file))
        nib.save(nib.Nifti1Image(ct_b_warped_nib.get_fdata().copy(), ct_b_warped_nib.affine),
                 str(midpoint_dir / "ct_followup_registered.nii.gz"))

    # Legacy names for backward compatibility
    nib.save(ct_a_cropped, str(midpoint_dir / "ct_a_in_midpoint.nii.gz"))
    nib.save(nib.load(str(midpoint_dir / "ct_followup_registered.nii.gz")),
             str(midpoint_dir / "ct_b_in_midpoint.nii.gz"))

    # -------------------------------------------------------------------------
    # Step 5: Resample masks to reference space (PET_A space)
    # -------------------------------------------------------------------------
    logger.info("  [Step 5] Resampling masks to reference space...")

    # Load masks from ROI directory - use tongue-trimmed PET-space masks
    roi_a_dir = TOTALSEG_ROI_DIR / f"{subject_id}_{ses_a}"
    roi_b_dir = TOTALSEG_ROI_DIR / f"{subject_id}_{ses_b}"

    if mask_type == 'jaw':
        # Use tongue-trimmed jaw-level masks in PET space
        mask_a_file = roi_a_dir / "continuous_masks_PETspace" / f"peridental_upper_jaw_trimmed_{trimming}.nii.gz"
        mask_b_file = roi_b_dir / "continuous_masks_PETspace" / f"peridental_upper_jaw_trimmed_{trimming}.nii.gz"
    else:
        # Use combined tooth shells with trimming
        mask_a_file = roi_a_dir / "continuous_masks_PETspace" / f"tooth_shells_trimmed_{trimming}.nii.gz"
        mask_b_file = roi_b_dir / "continuous_masks_PETspace" / f"tooth_shells_trimmed_{trimming}.nii.gz"

    if not mask_a_file.exists() or not mask_b_file.exists():
        logger.error(f"  Missing mask files:")
        logger.error(f"    Mask A: {mask_a_file} (exists: {mask_a_file.exists()})")
        logger.error(f"    Mask B: {mask_b_file} (exists: {mask_b_file.exists()})")
        return False

    mask_a_data, mask_a_nib = load_nifti(mask_a_file)
    mask_b_data, mask_b_nib = load_nifti(mask_b_file)
    logger.info(f"    Mask A shape: {mask_a_nib.shape}, range: [{mask_a_data.min():.3f}, {mask_a_data.max():.3f}]")
    logger.info(f"    Mask B shape: {mask_b_nib.shape}, range: [{mask_b_data.min():.3f}, {mask_b_data.max():.3f}]")

    # Convert labeled to binary if needed
    if mask_type == 'tooth':
        mask_a_data = (mask_a_data > 0).astype(np.float32)
        mask_b_data = (mask_b_data > 0).astype(np.float32)
        mask_a_nib = nib.Nifti1Image(mask_a_data, mask_a_nib.affine)
        mask_b_nib = nib.Nifti1Image(mask_b_data, mask_b_nib.affine)

    # Mask A is already in PET_A space (reference space) - just use it directly
    mask_a_aligned = mask_a_nib
    logger.info("    Mask A already in reference space (PET_A)")

    # Mask B needs to be warped from PET_B space to PET_A space using the same transform as PET_B
    logger.info("    Warping Mask B (PET_B space) → reference space (PET_A)...")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        mask_b_file_tmp = tmpdir / "mask_b.nii.gz"
        ref_file_tmp = tmpdir / "ref.nii.gz"

        nib.save(mask_b_nib, str(mask_b_file_tmp))
        nib.save(reference_nib, str(ref_file_tmp))

        mask_b_ants = ants.image_read(str(mask_b_file_tmp))
        ref_ants = ants.image_read(str(ref_file_tmp))

        # Apply the PET_B → PET_A transform
        mask_b_warped = ants.apply_transforms(
            fixed=ref_ants,
            moving=mask_b_ants,
            transformlist=[str(transform_dir / "pet_b_to_pet_a_fwd_0.mat")],
            interpolator='linear'  # Linear for continuous masks
        )

        warped_file = tmpdir / "mask_b_warped.nii.gz"
        ants.image_write(mask_b_warped, str(warped_file))
        mask_b_aligned = nib.load(str(warped_file))
        mask_b_aligned = nib.Nifti1Image(mask_b_aligned.get_fdata().copy(), mask_b_aligned.affine)

    nib.save(mask_a_aligned, str(midpoint_dir / "mask_a_in_midpoint.nii.gz"))
    nib.save(mask_b_aligned, str(midpoint_dir / "mask_b_in_midpoint.nii.gz"))

    # -------------------------------------------------------------------------
    # Step 6: Create intersection mask
    # -------------------------------------------------------------------------
    logger.info("  [Step 6] Creating intersection mask...")

    mask_a_aligned_data = mask_a_aligned.get_fdata()
    mask_b_aligned_data = mask_b_aligned.get_fdata()

    intersection_mask = create_intersection_mask(mask_a_aligned_data, mask_b_aligned_data)

    # Compute overlap statistics
    vol_a = np.sum(mask_a_aligned_data > 0.5)
    vol_b = np.sum(mask_b_aligned_data > 0.5)
    vol_intersection = np.sum(intersection_mask)
    dice = 2 * vol_intersection / (vol_a + vol_b) if (vol_a + vol_b) > 0 else 0

    logger.info(f"    Mask A volume: {vol_a} voxels")
    logger.info(f"    Mask B volume: {vol_b} voxels")
    logger.info(f"    Intersection volume: {vol_intersection} voxels")
    logger.info(f"    Dice overlap: {dice:.3f}")

    intersection_nib = nib.Nifti1Image(
        intersection_mask.astype(np.float32),
        reference_nib.affine
    )
    nib.save(intersection_nib, str(midpoint_dir / "intersection_mask.nii.gz"))

    # -------------------------------------------------------------------------
    # Step 7: Apply smoothing and compute delta image
    # -------------------------------------------------------------------------
    logger.info("  [Step 7] Applying 3mm Gaussian smoothing and computing delta...")

    pet_a_aligned_data = pet_a_aligned.get_fdata()
    pet_b_aligned_data = pet_b_aligned.get_fdata()

    # Apply 3mm FWHM Gaussian smoothing to both PET images
    logger.info(f"    Smoothing with {SMOOTHING_FWHM_MM}mm FWHM Gaussian...")
    pet_a_smoothed = smooth_pet_image(pet_a_aligned_data, pet_a_aligned.affine, SMOOTHING_FWHM_MM)
    pet_b_smoothed = smooth_pet_image(pet_b_aligned_data, pet_b_aligned.affine, SMOOTHING_FWHM_MM)

    # Save smoothed PET images
    pet_a_smoothed_nib = nib.Nifti1Image(pet_a_smoothed, pet_a_aligned.affine)
    pet_b_smoothed_nib = nib.Nifti1Image(pet_b_smoothed, pet_b_aligned.affine)
    nib.save(pet_a_smoothed_nib, str(midpoint_dir / "pet_a_smoothed_3mm.nii.gz"))
    nib.save(pet_b_smoothed_nib, str(midpoint_dir / "pet_b_smoothed_3mm.nii.gz"))

    # Compute delta on smoothed images (full image, unmasked)
    delta_full = pet_b_smoothed - pet_a_smoothed

    # Save UNMASKED delta image (for visualization with mask overlays)
    delta_full_nib = nib.Nifti1Image(delta_full.astype(np.float32), reference_nib.affine)
    nib.save(delta_full_nib, str(delta_dir / "delta_pet_smoothed_3mm_unmasked.nii.gz"))

    # Compute masked delta and statistics
    delta_data, delta_stats = compute_delta(pet_a_smoothed, pet_b_smoothed, intersection_mask)

    logger.info(f"    Mean delta: {delta_stats['mean_delta']:.4f} kBq/mL")
    logger.info(f"    Std delta: {delta_stats['std_delta']:.4f}")
    logger.info(f"    Voxels increased: {delta_stats['n_voxels_increase']} ({delta_stats['pct_increase']:.1f}%)")

    # Save masked delta image
    delta_nib = nib.Nifti1Image(
        delta_data.astype(np.float32),
        reference_nib.affine
    )
    nib.save(delta_nib, str(delta_dir / "delta_pet_smoothed_3mm.nii.gz"))

    # -------------------------------------------------------------------------
    # Step 8: Compute SUV/TPR/FUR metrics using input functions
    # (matching 04_batch_quantify.py methodology)
    # -------------------------------------------------------------------------
    logger.info("  [Step 8] Computing SUV/TPR/FUR metrics...")

    # Load SUV scaling factors from curated CSV (preferred over eCRF)
    suv_data = load_suv_scaling_factors()

    # Compute voxel volume
    voxel_sizes = np.abs(np.diag(reference_nib.affine[:3, :3]))
    voxel_vol_ml = float(np.prod(voxel_sizes) / 1000.0)  # mm³ to mL

    # Process Baseline metrics
    # Note: PET is in kBq/mL, so SUV scaler uses dose_kBq = dose_MBq * 1000
    if_a = load_input_function(subject_id, 'Baseline')
    suv_a = suv_data.get((subject_id, 'Baseline'), {})
    pet_json_a = load_pet_json(subject_id, 'Baseline')

    if if_a is not None and suv_a:
        # Get scan timing from JSON sidecar
        scan_start_s_a = pet_json_a.get('ScanStart', 1800)
        frame_duration_ms_a = pet_json_a.get('FrameDuration', [1800000])
        if isinstance(frame_duration_ms_a, list):
            frame_duration_ms_a = frame_duration_ms_a[0]
        scan_end_s_a = scan_start_s_a + frame_duration_ms_a / 1000
        tissue_time_s_a = (scan_start_s_a + scan_end_s_a) / 2

        plasma_a = compute_plasma_denominators(if_a, scan_start_s_a, scan_end_s_a, tissue_time_s_a)
        suv_scaler_a = suv_a['weight_kg'] / (suv_a['dose_mbq'] * 1000)  # dose in kBq
        metrics_a = compute_roi_metrics(
            pet_a_smoothed, intersection_mask, voxel_vol_ml,
            suv_scaler_a, plasma_a['plasma_mean_Bq_mL'], plasma_a['plasma_auc_0_to_T_Bq_s_mL']
        )
        logger.info(f"    Baseline: SUV_mean={metrics_a['SUV_mean']:.3f}, TPR_mean={metrics_a['TPR_mean']:.3f}, FUR_mean={metrics_a['FUR_mean_per_min']:.4f}")
    else:
        metrics_a = None
        plasma_a = {'plasma_mean_Bq_mL': np.nan, 'plasma_auc_0_to_T_Bq_s_mL': np.nan}
        logger.warning("    Baseline: Missing input function or SUV data")

    # Process Followup metrics
    if_b = load_input_function(subject_id, 'Followup')
    suv_b = suv_data.get((subject_id, 'Followup'), {})
    pet_json_b = load_pet_json(subject_id, 'Followup')

    if if_b is not None and suv_b:
        # Get scan timing from JSON sidecar
        scan_start_s_b = pet_json_b.get('ScanStart', 1800)
        frame_duration_ms_b = pet_json_b.get('FrameDuration', [1800000])
        if isinstance(frame_duration_ms_b, list):
            frame_duration_ms_b = frame_duration_ms_b[0]
        scan_end_s_b = scan_start_s_b + frame_duration_ms_b / 1000
        tissue_time_s_b = (scan_start_s_b + scan_end_s_b) / 2

        plasma_b = compute_plasma_denominators(if_b, scan_start_s_b, scan_end_s_b, tissue_time_s_b)
        suv_scaler_b = suv_b['weight_kg'] / (suv_b['dose_mbq'] * 1000)  # dose in kBq
        metrics_b = compute_roi_metrics(
            pet_b_smoothed, intersection_mask, voxel_vol_ml,
            suv_scaler_b, plasma_b['plasma_mean_Bq_mL'], plasma_b['plasma_auc_0_to_T_Bq_s_mL']
        )
        logger.info(f"    Followup: SUV_mean={metrics_b['SUV_mean']:.3f}, TPR_mean={metrics_b['TPR_mean']:.3f}, FUR_mean={metrics_b['FUR_mean_per_min']:.4f}")
    else:
        metrics_b = None
        plasma_b = {'plasma_mean_Bq_mL': np.nan, 'plasma_auc_0_to_T_Bq_s_mL': np.nan}
        logger.warning("    Followup: Missing input function or SUV data")

    # Compute delta metrics (Followup - Baseline)
    if metrics_a is not None and metrics_b is not None:
        delta_suv_mean = metrics_b['SUV_mean'] - metrics_a['SUV_mean']
        delta_tpr_mean = metrics_b['TPR_mean'] - metrics_a['TPR_mean']
        delta_fur_mean = metrics_b['FUR_mean_per_min'] - metrics_a['FUR_mean_per_min']
        pct_change_suv = 100.0 * delta_suv_mean / metrics_a['SUV_mean'] if metrics_a['SUV_mean'] != 0 else np.nan
        pct_change_tpr = 100.0 * delta_tpr_mean / metrics_a['TPR_mean'] if metrics_a['TPR_mean'] != 0 else np.nan
        pct_change_fur = 100.0 * delta_fur_mean / metrics_a['FUR_mean_per_min'] if metrics_a['FUR_mean_per_min'] != 0 else np.nan

        logger.info(f"    Delta SUV_mean: {delta_suv_mean:+.3f} ({pct_change_suv:+.1f}%)")
        logger.info(f"    Delta TPR_mean: {delta_tpr_mean:+.3f} ({pct_change_tpr:+.1f}%)")
        logger.info(f"    Delta FUR_mean: {delta_fur_mean:+.4f} ({pct_change_fur:+.1f}%)")
    else:
        delta_suv_mean = delta_tpr_mean = delta_fur_mean = np.nan
        pct_change_suv = pct_change_tpr = pct_change_fur = np.nan

    # Save delta statistics
    # Parse trimming value (e.g., '0mm' -> 0, '5mm' -> 5)
    trimming_mm_value = int(trimming.replace('mm', ''))
    delta_summary = {
        'subject_id': subject_id,
        'sessions': {
            'baseline': ses_a,
            'followup': ses_b
        },
        'mask_type': mask_type,
        'smoothing_fwhm_mm': SMOOTHING_FWHM_MM,
        'tongue_trimming_mm': trimming_mm_value,
        'masks': {
            'mask_a_volume_voxels': int(vol_a),
            'mask_b_volume_voxels': int(vol_b),
            'intersection_volume_voxels': int(vol_intersection),
            'overlap_dice': float(dice)
        },
        'delta_stats_raw_Bq_mL': delta_stats,
        'metrics_baseline': metrics_a,
        'metrics_followup': metrics_b,
        'delta_metrics': {
            'delta_SUV_mean': float(delta_suv_mean) if np.isfinite(delta_suv_mean) else None,
            'delta_TPR_mean': float(delta_tpr_mean) if np.isfinite(delta_tpr_mean) else None,
            'delta_FUR_mean_per_min': float(delta_fur_mean) if np.isfinite(delta_fur_mean) else None,
            'pct_change_SUV': float(pct_change_suv) if np.isfinite(pct_change_suv) else None,
            'pct_change_TPR': float(pct_change_tpr) if np.isfinite(pct_change_tpr) else None,
            'pct_change_FUR': float(pct_change_fur) if np.isfinite(pct_change_fur) else None
        },
        'registration': {
            'method': 'direct_pet_to_pet',
            'translation_mm': translation.tolist(),
            'translation_magnitude_mm': float(translation_mag)
        }
    }
    with open(delta_dir / "delta_summary.json", 'w') as f:
        json.dump(delta_summary, f, indent=2)

    # -------------------------------------------------------------------------
    # Step 9: Generate QC visualizations and save key outputs
    # -------------------------------------------------------------------------
    # Load the saved CT images for QC (we saved them earlier)
    ct_a_for_qc = nib.load(str(midpoint_dir / "ct_a_in_midpoint.nii.gz"))
    ct_b_for_qc = nib.load(str(midpoint_dir / "ct_b_in_midpoint.nii.gz"))

    # Use smoothed PET for QC visualization
    create_qc_visualizations(
        subject_id, ct_a_for_qc, ct_b_for_qc,
        pet_a_smoothed_nib, pet_b_smoothed_nib,  # Use smoothed PET
        intersection_mask, delta_data, qc_dir, trimming, logger
    )

    # Save key NIfTI files to QC folder for easy access
    logger.info("  Saving key NIfTIs to QC folder...")
    nib.save(delta_full_nib, str(qc_dir / f"{subject_id}_delta_smoothed_3mm_unmasked.nii.gz"))
    nib.save(delta_nib, str(qc_dir / f"{subject_id}_delta_smoothed_3mm_masked.nii.gz"))
    nib.save(intersection_nib, str(qc_dir / f"{subject_id}_intersection_mask.nii.gz"))

    logger.info(f"  Completed: {subject_id}")
    return True


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Longitudinal Delta Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python 06_longitudinal_delta.py                    # All subjects
    python 06_longitudinal_delta.py --subject sub-101  # Single subject
    python 06_longitudinal_delta.py --force            # Force re-run
    python 06_longitudinal_delta.py --mask-type tooth  # Per-tooth masks
        """
    )

    parser.add_argument('--subject', type=str, help='Process only this subject')
    parser.add_argument('--force', action='store_true', help='Force re-run')
    parser.add_argument('--mask-type', choices=['jaw', 'tooth'], default='jaw',
                        help='Mask type: jaw (default) or tooth')
    parser.add_argument('--trimming', choices=['0mm', '3mm', '5mm', '8mm', '10mm'], default='0mm',
                        help='Tongue trimming distance (default: 0mm = original tongue mask, no dilation)')

    args = parser.parse_args()

    # Setup
    ensure_directories()
    LONGITUDINAL_DIR.mkdir(parents=True, exist_ok=True)
    LONGITUDINAL_QC_DIR.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(LOGNOTES_DIR)

    logger.info("="*70)
    logger.info("LONGITUDINAL DELTA ANALYSIS")
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Mask type: {args.mask_type}")
    logger.info(f"Tongue trimming: {args.trimming}")
    logger.info("="*70)

    # Load blinding key
    blinding_map = load_blinding_key()
    logger.info(f"Loaded blinding key with {len(blinding_map)} entries")

    # Get subjects
    if args.subject:
        subjects = [args.subject]
    else:
        subjects = discover_subjects()

    logger.info(f"Subjects to process: {len(subjects)}")

    # Process each subject
    results = {}
    for subject_id in subjects:
        try:
            success = process_subject(subject_id, blinding_map, args.force, args.mask_type, args.trimming, logger)
            results[subject_id] = 'SUCCESS' if success else 'FAILED'
        except Exception as e:
            logger.error(f"Error processing {subject_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            results[subject_id] = 'ERROR'

    # Summary
    logger.info("\n" + "="*70)
    logger.info("SUMMARY")
    logger.info("="*70)

    for subject_id, status in results.items():
        logger.info(f"  {subject_id}: {status}")

    n_success = sum(1 for s in results.values() if s == 'SUCCESS')
    n_failed = sum(1 for s in results.values() if s in ['FAILED', 'ERROR'])

    logger.info(f"\nCompleted: {n_success}/{len(subjects)}, Failed: {n_failed}")
    logger.info(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*70)

    # Create group summary CSV with metrics and t-test
    if n_success > 0:
        summary_rows = []
        for subject_id in subjects:
            summary_file = LONGITUDINAL_DIR / subject_id / "delta" / "delta_summary.json"
            if summary_file.exists():
                with open(summary_file) as f:
                    data = json.load(f)

                # Extract metrics if available
                delta_metrics = data.get('delta_metrics', {})
                metrics_a = data.get('metrics_baseline', {}) or {}
                metrics_b = data.get('metrics_followup', {}) or {}

                summary_rows.append({
                    'subject_id': subject_id,
                    'baseline_ses': data['sessions']['baseline'],
                    'followup_ses': data['sessions']['followup'],
                    'mask_type': data['mask_type'],
                    'smoothing_fwhm_mm': data.get('smoothing_fwhm_mm', 3.0),
                    'tongue_trimming_mm': data.get('tongue_trimming_mm', 0),
                    'overlap_dice': data['masks']['overlap_dice'],
                    'n_voxels': data.get('delta_stats_raw_Bq_mL', data.get('delta_stats', {})).get('n_voxels_total', 0),
                    'mean_delta_Bq_mL': data.get('delta_stats_raw_Bq_mL', data.get('delta_stats', {})).get('mean_delta', np.nan),
                    # Baseline metrics
                    'SUV_mean_baseline': metrics_a.get('SUV_mean', np.nan),
                    'TPR_mean_baseline': metrics_a.get('TPR_mean', np.nan),
                    'FUR_mean_baseline': metrics_a.get('FUR_mean_per_min', np.nan),
                    # Followup metrics
                    'SUV_mean_followup': metrics_b.get('SUV_mean', np.nan),
                    'TPR_mean_followup': metrics_b.get('TPR_mean', np.nan),
                    'FUR_mean_followup': metrics_b.get('FUR_mean_per_min', np.nan),
                    # Delta metrics
                    'delta_SUV_mean': delta_metrics.get('delta_SUV_mean', np.nan),
                    'delta_TPR_mean': delta_metrics.get('delta_TPR_mean', np.nan),
                    'delta_FUR_mean': delta_metrics.get('delta_FUR_mean_per_min', np.nan),
                    'pct_change_SUV': delta_metrics.get('pct_change_SUV', np.nan),
                    'pct_change_TPR': delta_metrics.get('pct_change_TPR', np.nan),
                    'pct_change_FUR': delta_metrics.get('pct_change_FUR', np.nan),
                    'translation_mm': data['registration']['translation_magnitude_mm']
                })

        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            LONGITUDINAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            summary_file = LONGITUDINAL_OUTPUT_DIR / "delta_summary.csv"
            summary_df.to_csv(summary_file, index=False)
            logger.info(f"\nGroup summary saved: {summary_file}")

            # -------------------------------------------------------------------------
            # ONE-SAMPLE T-TEST: Mean delta different from zero
            # -------------------------------------------------------------------------
            logger.info("\n" + "="*70)
            logger.info("ONE-SAMPLE T-TEST (Mean Delta Different from Zero)")
            logger.info("="*70)

            # Prepare t-test results
            ttest_results = []

            for metric_name, col_name in [
                ('Delta SUV mean', 'delta_SUV_mean'),
                ('Delta TPR mean', 'delta_TPR_mean'),
                ('Delta FUR mean', 'delta_FUR_mean'),
                ('Delta Bq/mL (raw)', 'mean_delta_Bq_mL')
            ]:
                values = summary_df[col_name].dropna().values
                if len(values) >= 3:
                    t_stat, p_value = ttest_1samp(values, 0)
                    n = len(values)
                    mean_val = np.mean(values)
                    std_val = np.std(values, ddof=1)
                    sem_val = std_val / np.sqrt(n)
                    ci_95 = (mean_val - 1.96 * sem_val, mean_val + 1.96 * sem_val)

                    # Direction
                    direction = "increase" if mean_val > 0 else "decrease"

                    logger.info(f"\n{metric_name}:")
                    logger.info(f"  N = {n}")
                    logger.info(f"  Mean = {mean_val:.4f} (direction: {direction})")
                    logger.info(f"  SD = {std_val:.4f}")
                    logger.info(f"  SEM = {sem_val:.4f}")
                    logger.info(f"  95% CI = [{ci_95[0]:.4f}, {ci_95[1]:.4f}]")
                    logger.info(f"  t-statistic = {t_stat:.3f}")
                    logger.info(f"  p-value = {p_value:.4f}")
                    if p_value < 0.05:
                        logger.info(f"  *** SIGNIFICANT at p < 0.05 ***")

                    ttest_results.append({
                        'metric': metric_name,
                        'n': n,
                        'mean': mean_val,
                        'std': std_val,
                        'sem': sem_val,
                        'ci_95_lower': ci_95[0],
                        'ci_95_upper': ci_95[1],
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'significant_p05': p_value < 0.05
                    })
                else:
                    logger.warning(f"  {metric_name}: Not enough data (n={len(values)})")

            # Save t-test results
            if ttest_results:
                ttest_df = pd.DataFrame(ttest_results)
                ttest_file = LONGITUDINAL_OUTPUT_DIR / "ttest_results.csv"
                ttest_df.to_csv(ttest_file, index=False)
                logger.info(f"\nT-test results saved: {ttest_file}")

            # Summary of subjects showing increase vs decrease
            logger.info("\n" + "-"*50)
            logger.info("SUBJECTS DIRECTION OF CHANGE")
            logger.info("-"*50)

            for metric_name, col_name in [
                ('SUV', 'delta_SUV_mean'),
                ('TPR', 'delta_TPR_mean'),
                ('FUR', 'delta_FUR_mean')
            ]:
                values = summary_df[col_name].dropna()
                n_increase = (values > 0).sum()
                n_decrease = (values < 0).sum()
                logger.info(f"{metric_name}: {n_increase} increased, {n_decrease} decreased")

            logger.info("="*70)


if __name__ == "__main__":
    main()
