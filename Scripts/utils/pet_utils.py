"""
PET Quantification utilities for Periodontal Analysis Pipeline.

Handles:
- PET unit validation and conversion
- SUV calculation
- TPR (Tissue-to-Plasma Ratio) calculation
- FUR (Fractional Uptake Rate) calculation
- ROI metric extraction
- ROI resampling for cross-modality analysis
"""

import numpy as np
import nibabel as nib
from nibabel.processing import resample_from_to
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
import json
import logging

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    SUV_MIN_EXPECTED, SUV_MAX_EXPECTED, ROBUST_PERCENTILE
)
from utils.io_utils import get_voxel_dimensions, get_voxel_volume_ml

logger = logging.getLogger(__name__)


# =============================================================================
# UNIT VALIDATION AND CONVERSION
# =============================================================================

def validate_pet_units(pet_data: np.ndarray, pet_json: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Validate and infer PET image units.

    PET images can be in:
    - Bq/mL (raw activity concentration)
    - kBq/mL
    - SUV (already normalized)

    Args:
        pet_data: PET image data array
        pet_json: Optional JSON sidecar with metadata

    Returns:
        Dictionary with unit information and conversion factors
    """
    result = {
        'stated_units': 'unknown',
        'inferred_units': 'unknown',
        'max_value': float(np.nanmax(pet_data)),
        'min_value': float(np.nanmin(pet_data[pet_data > 0])) if np.any(pet_data > 0) else 0,
        'mean_value': float(np.nanmean(pet_data[pet_data > 0])) if np.any(pet_data > 0) else 0,
        'warnings': []
    }

    # Check JSON for stated units
    if pet_json:
        result['stated_units'] = pet_json.get('Units', 'unknown')

    # Infer units from value range
    max_val = result['max_value']

    if max_val < 100:
        # Very low values - likely SUV
        result['inferred_units'] = 'SUV'
        result['conversion_to_bq_ml'] = None  # Cannot convert back without dose/weight
        logger.info(f"PET appears to be in SUV units (max={max_val:.2f})")
    elif max_val < 100000:
        # Moderate values - likely kBq/mL
        result['inferred_units'] = 'kBq/mL'
        result['conversion_to_bq_ml'] = 1000.0
        logger.info(f"PET appears to be in kBq/mL (max={max_val:.2f})")
    else:
        # High values - likely Bq/mL
        result['inferred_units'] = 'Bq/mL'
        result['conversion_to_bq_ml'] = 1.0
        logger.info(f"PET appears to be in Bq/mL (max={max_val:.2f})")

    # Warn if stated and inferred units don't match
    if result['stated_units'] != 'unknown' and result['stated_units'] != result['inferred_units']:
        result['warnings'].append(
            f"Stated units ({result['stated_units']}) may not match "
            f"inferred units ({result['inferred_units']})"
        )

    return result


def convert_to_bq_ml(pet_data: np.ndarray, unit_info: Dict[str, Any]) -> np.ndarray:
    """
    Convert PET data to Bq/mL.

    Args:
        pet_data: PET image data
        unit_info: Dictionary from validate_pet_units()

    Returns:
        PET data in Bq/mL
    """
    if unit_info['inferred_units'] == 'Bq/mL':
        return pet_data.copy()
    elif unit_info['inferred_units'] == 'kBq/mL':
        return pet_data * 1000.0
    elif unit_info['inferred_units'] == 'SUV':
        logger.warning("Cannot convert SUV to Bq/mL without dose/weight info")
        return pet_data.copy()
    else:
        logger.warning(f"Unknown units: {unit_info['inferred_units']}, assuming Bq/mL")
        return pet_data.copy()


# =============================================================================
# SUV CALCULATION
# =============================================================================

def calculate_suv_scaler(weight_kg: float, dose_mbq: float) -> float:
    """
    Calculate SUV scaling factor.

    SUV = C_tissue [Bq/mL] * weight_kg / (dose_Bq)
        = C_tissue [Bq/mL] * weight_kg / (dose_MBq * 1e6)

    Args:
        weight_kg: Body weight in kg
        dose_mbq: Injected dose in MBq

    Returns:
        SUV scaler (multiply Bq/mL by this to get SUV)
    """
    if weight_kg <= 0 or dose_mbq <= 0:
        return np.nan

    # SUV = activity * weight / dose
    # activity in Bq/mL, weight in kg (= 1000g), dose in MBq (= 1e6 Bq)
    # SUV = (Bq/mL) * (kg) / (MBq * 1e6)
    # SUV = (Bq/mL) * (kg) / (Bq)
    # For kg to work as g/mL density, multiply by 1000g/kg, divide by 1000mL/L
    # Simplifies to: SUV = activity_Bq_mL * weight_kg / (dose_MBq * 1e6)

    return weight_kg / (dose_mbq * 1e6)


def calculate_suv(intensity_bq_ml: float, weight_kg: float, dose_mbq: float) -> float:
    """
    Calculate SUV from activity concentration.

    Args:
        intensity_bq_ml: Activity concentration in Bq/mL
        weight_kg: Body weight in kg
        dose_mbq: Injected dose in MBq

    Returns:
        SUV value (dimensionless)
    """
    scaler = calculate_suv_scaler(weight_kg, dose_mbq)
    if np.isnan(scaler):
        return np.nan
    return intensity_bq_ml * scaler


# =============================================================================
# TPR CALCULATION
# =============================================================================

def calculate_tpr(intensity_bq_ml: float, plasma_mean_bq_ml: float) -> float:
    """
    Calculate Tissue-to-Plasma Ratio (TPR).

    TPR = C_tissue / C_plasma_mean

    Both must be in same units (Bq/mL).

    Args:
        intensity_bq_ml: Tissue activity in Bq/mL
        plasma_mean_bq_ml: Mean plasma activity during scan in Bq/mL

    Returns:
        TPR value (dimensionless)
    """
    if plasma_mean_bq_ml <= 0 or np.isnan(plasma_mean_bq_ml):
        return np.nan

    return intensity_bq_ml / plasma_mean_bq_ml


# =============================================================================
# FUR CALCULATION
# =============================================================================

def calculate_fur(intensity_bq_ml: float, plasma_auc_bq_s_ml: float) -> float:
    """
    Calculate Fractional Uptake Rate (FUR).

    FUR = C_tissue(T) / integral_0^T(C_plasma(t) dt)

    Raw units: (Bq/mL) / (Bq*s/mL) = s^-1
    Reported in: min^-1 (multiply by 60)

    Args:
        intensity_bq_ml: Tissue activity at time T in Bq/mL
        plasma_auc_bq_s_ml: Cumulative plasma AUC from 0 to T in Bq*s/mL

    Returns:
        FUR value in min^-1
    """
    if plasma_auc_bq_s_ml <= 0 or np.isnan(plasma_auc_bq_s_ml):
        return np.nan

    # Raw FUR in s^-1
    fur_per_s = intensity_bq_ml / plasma_auc_bq_s_ml

    # Convert to min^-1
    return fur_per_s * 60.0


# =============================================================================
# RESAMPLING UTILITIES
# =============================================================================

def resample_roi_to_pet(roi_img: nib.Nifti1Image, pet_img: nib.Nifti1Image) -> np.ndarray:
    """
    Resample ROI mask from CT space to PET space.

    Uses nearest-neighbor interpolation to preserve binary mask.

    Args:
        roi_img: NIfTI image of binary ROI mask (in CT space)
        pet_img: NIfTI image of PET data (target space)

    Returns:
        Resampled ROI mask as numpy array in PET space
    """
    # Resample using nearest neighbor for binary mask
    resampled_img = resample_from_to(roi_img, pet_img, order=0)  # order=0 = nearest neighbor
    resampled_data = resampled_img.get_fdata()

    # Ensure binary
    resampled_data = (resampled_data > 0.5).astype(np.float32)

    logger.info(f"  Resampled ROI from {roi_img.shape} to {resampled_data.shape}")

    return resampled_data


def check_dimensions_match(pet_data: np.ndarray, roi_mask: np.ndarray) -> bool:
    """
    Check if PET and ROI dimensions match.

    Args:
        pet_data: PET image data array
        roi_mask: ROI mask array

    Returns:
        True if dimensions match, False otherwise
    """
    return pet_data.shape == roi_mask.shape


# =============================================================================
# ROI METRIC EXTRACTION
# =============================================================================

def extract_pet_metrics(pet_data: np.ndarray, roi_mask: np.ndarray,
                        weight_kg: float, dose_mbq: float,
                        plasma_mean_bq_ml: Optional[float] = None,
                        plasma_auc_0_to_t: Optional[float] = None,
                        unit_info: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Extract all PET metrics from an ROI.

    Calculates:
    - Raw intensity (mean, 90th percentile)
    - SUV (mean, 90th percentile)
    - TPR (mean, 90th percentile) if plasma_mean provided
    - FUR (mean, 90th percentile) if plasma_auc provided

    Args:
        pet_data: PET image data
        roi_mask: Binary ROI mask
        weight_kg: Body weight in kg
        dose_mbq: Injected dose in MBq
        plasma_mean_bq_ml: Mean plasma activity during scan (Bq/mL)
        plasma_auc_0_to_t: Cumulative plasma AUC from 0 to T (Bq*s/mL)
        unit_info: Unit information from validate_pet_units()

    Returns:
        Dictionary with all extracted metrics
    """
    result = {
        'intensity_mean_Bq_mL': np.nan,
        'intensity_90th_Bq_mL': np.nan,
        'SUV_mean': np.nan,
        'SUV_90th': np.nan,
        'TPR_mean': np.nan,
        'TPR_90th': np.nan,
        'FUR_mean_per_min': np.nan,
        'FUR_90th_per_min': np.nan,
        'roi_volume_mL': np.nan,
        'roi_voxel_count': 0,
        'warnings': []
    }

    # Get ROI voxels
    roi_values = pet_data[roi_mask > 0]

    if len(roi_values) == 0:
        result['warnings'].append("ROI contains no voxels")
        return result

    # Filter valid values (positive, finite)
    valid_mask = (roi_values > 0) & np.isfinite(roi_values)
    roi_values = roi_values[valid_mask]

    if len(roi_values) == 0:
        result['warnings'].append("ROI contains no valid (positive) voxels")
        return result

    result['roi_voxel_count'] = len(roi_values)

    # Convert to Bq/mL if needed
    if unit_info and unit_info['inferred_units'] == 'kBq/mL':
        roi_values = roi_values * 1000.0

    # Raw intensity metrics
    result['intensity_mean_Bq_mL'] = float(np.mean(roi_values))
    result['intensity_90th_Bq_mL'] = float(np.percentile(roi_values, ROBUST_PERCENTILE))

    # SUV metrics
    suv_scaler = calculate_suv_scaler(weight_kg, dose_mbq)
    if not np.isnan(suv_scaler):
        suv_values = roi_values * suv_scaler
        result['SUV_mean'] = float(np.mean(suv_values))
        result['SUV_90th'] = float(np.percentile(suv_values, ROBUST_PERCENTILE))

        # QC checks
        if result['SUV_mean'] < SUV_MIN_EXPECTED:
            result['warnings'].append(f"SUV_mean unusually low: {result['SUV_mean']:.3f}")
        elif result['SUV_mean'] > SUV_MAX_EXPECTED:
            result['warnings'].append(f"SUV_mean unusually high: {result['SUV_mean']:.3f}")
    else:
        result['warnings'].append("Cannot calculate SUV (missing weight or dose)")

    # TPR metrics
    if plasma_mean_bq_ml and plasma_mean_bq_ml > 0:
        tpr_values = roi_values / plasma_mean_bq_ml
        result['TPR_mean'] = float(np.mean(tpr_values))
        result['TPR_90th'] = float(np.percentile(tpr_values, ROBUST_PERCENTILE))
    else:
        result['warnings'].append("Cannot calculate TPR (missing plasma mean)")

    # FUR metrics
    if plasma_auc_0_to_t and plasma_auc_0_to_t > 0:
        fur_values = (roi_values / plasma_auc_0_to_t) * 60.0  # Convert to min^-1
        result['FUR_mean_per_min'] = float(np.mean(fur_values))
        result['FUR_90th_per_min'] = float(np.percentile(fur_values, ROBUST_PERCENTILE))
    else:
        result['warnings'].append("Cannot calculate FUR (missing plasma AUC)")

    return result


def extract_metrics_for_tooth(pet_data: np.ndarray, tooth_roi: np.ndarray,
                               weight_kg: float, dose_mbq: float,
                               plasma_mean_bq_ml: Optional[float],
                               plasma_auc_0_to_t: Optional[float],
                               tooth_id: int,
                               ct_img: Optional[nib.Nifti1Image] = None) -> Dict[str, Any]:
    """
    Extract PET metrics for a single tooth ROI.

    Args:
        pet_data: PET image data
        tooth_roi: Binary ROI mask for this tooth
        weight_kg: Body weight
        dose_mbq: Injected dose
        plasma_mean_bq_ml: Mean plasma activity
        plasma_auc_0_to_t: Cumulative plasma AUC
        tooth_id: Tooth ID (FDI notation)
        ct_img: Optional NIfTI image for volume calculation

    Returns:
        Dictionary with tooth-level metrics
    """
    metrics = extract_pet_metrics(
        pet_data, tooth_roi, weight_kg, dose_mbq,
        plasma_mean_bq_ml, plasma_auc_0_to_t
    )

    # Add tooth-specific info
    metrics['tooth_id'] = tooth_id

    # Calculate volume
    if ct_img is not None:
        voxel_vol_ml = get_voxel_volume_ml(ct_img)
        metrics['roi_volume_mL'] = float(np.sum(tooth_roi) * voxel_vol_ml)

    # Determine position (anterior/posterior)
    if tooth_id in [11, 12, 13, 21, 22, 23]:
        metrics['tooth_position'] = 'anterior'
    else:
        metrics['tooth_position'] = 'posterior'

    # Determine quadrant
    if 11 <= tooth_id <= 18:
        metrics['quadrant'] = 'upper_right'
    elif 21 <= tooth_id <= 28:
        metrics['quadrant'] = 'upper_left'
    else:
        metrics['quadrant'] = 'unknown'

    return metrics


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_pet_json(json_path: Path) -> Dict[str, Any]:
    """
    Load PET JSON sidecar and extract timing/unit information.

    Args:
        json_path: Path to JSON file

    Returns:
        Dictionary with timing and unit info
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    result = {
        'scan_start_s': data.get('ScanStart', None),
        'scan_duration_s': None,
        'units': data.get('Units', 'unknown'),
        'injected_activity_bq': data.get('InjectedRadioactivity', None),
        'decay_correction': data.get('DecayCorrectionTime', None),
        'time_zero': data.get('TimeZero', None),
        'warnings': []
    }

    # Handle FrameDuration (may be in ms or s)
    frame_duration = data.get('FrameDuration', [1800000])
    if isinstance(frame_duration, list):
        frame_duration = frame_duration[0] if frame_duration else 1800000

    # Convert ms to s if needed
    if frame_duration > 10000:  # Likely milliseconds
        frame_duration = frame_duration / 1000

    result['scan_duration_s'] = frame_duration

    # Handle missing/zero ScanStart
    if result['scan_start_s'] is None or result['scan_start_s'] == 0:
        result['warnings'].append(
            f"ScanStart was {result['scan_start_s']} - using default 1800s (30 min)"
        )
        result['scan_start_s'] = 1800

    return result


def calculate_mean_ct_hu(ct_data: np.ndarray, roi_mask: np.ndarray) -> float:
    """
    Calculate mean CT HU within ROI.

    Note: This is for WITHIN-SUBJECT comparison only.
    CT-AC is not absolutely calibrated.

    Args:
        ct_data: CT image data (HU values)
        roi_mask: Binary ROI mask

    Returns:
        Mean HU value
    """
    roi_values = ct_data[roi_mask > 0]

    if len(roi_values) == 0:
        return np.nan

    return float(np.mean(roi_values))
