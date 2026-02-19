"""
ROI Generation utilities for Periodontal Analysis Pipeline.

Handles:
- Peridental soft tissue shell creation
- HU gating for soft tissue
- Alveolar bone ROI
- Per-tooth ROI subdivision
- Quadrant aggregation
- Metal artifact exclusion
"""

import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Any
from scipy.ndimage import binary_dilation, binary_erosion, label
import logging

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    PRIMARY_DILATION_MM, SENSITIVITY_DILATION_MM,
    SOFT_TISSUE_HU_MIN, SOFT_TISSUE_HU_MAX,
    SOFT_TISSUE_HU_MIN_WIDE, SOFT_TISSUE_HU_MAX_WIDE,
    MIN_ROI_VOXELS, MAXILLA_PROXIMITY_MM,
    QUADRANTS, ALL_UPPER_TEETH
)
from utils.io_utils import get_voxel_dimensions, get_voxel_volume_ml

logger = logging.getLogger(__name__)


# =============================================================================
# PERIDENTAL SOFT TISSUE ROI
# =============================================================================

def create_peridental_roi(teeth_mask: np.ndarray, maxilla_mask: np.ndarray,
                          ct_data: np.ndarray, ct_img: nib.Nifti1Image,
                          dilation_mm: float = PRIMARY_DILATION_MM,
                          metal_mask: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Create peridental soft tissue ROI.

    This is the primary ROI for FDG uptake analysis - a shell of soft tissue
    surrounding the teeth roots.

    Args:
        teeth_mask: Binary mask of teeth
        maxilla_mask: Binary mask of maxilla bone
        ct_data: CT image data (HU values)
        ct_img: NIfTI image for voxel dimensions
        dilation_mm: Dilation distance in mm (default 4mm)
        metal_mask: Optional metal artifact mask for exclusion

    Returns:
        Dictionary with:
            - roi_mask: Binary ROI mask
            - hu_rejection_rate: Fraction of voxels rejected by HU gating
            - pre_hu_voxels: Voxel count before HU gating
            - post_hu_voxels: Voxel count after HU gating
            - warnings: List of warnings
    """
    voxel_dims = get_voxel_dimensions(ct_img)
    voxel_vol_ml = get_voxel_volume_ml(ct_img)
    dilation_voxels = int(np.ceil(dilation_mm / voxel_dims[0]))

    result = {
        'roi_mask': None,
        'hu_rejection_rate': 0,
        'pre_hu_voxels': 0,
        'post_hu_voxels': 0,
        'volume_ml': 0,
        'warnings': [],
        'hu_range_used': (SOFT_TISSUE_HU_MIN, SOFT_TISSUE_HU_MAX)
    }

    # Step 1: Dilate teeth mask
    teeth_dilated = binary_dilation(teeth_mask, iterations=dilation_voxels)

    # Step 2: Create shell (subtract teeth and bone)
    shell = teeth_dilated & ~teeth_mask & ~maxilla_mask
    result['pre_hu_voxels'] = int(np.sum(shell))

    if result['pre_hu_voxels'] == 0:
        result['warnings'].append("Shell has zero voxels after teeth/bone subtraction")
        return result

    # Step 3: HU gating for soft tissue
    soft_tissue_mask = (ct_data >= SOFT_TISSUE_HU_MIN) & (ct_data <= SOFT_TISSUE_HU_MAX)
    shell_gated = shell & soft_tissue_mask
    result['post_hu_voxels'] = int(np.sum(shell_gated))

    # Calculate rejection rate
    result['hu_rejection_rate'] = 1 - (result['post_hu_voxels'] / result['pre_hu_voxels'])

    if result['hu_rejection_rate'] > 0.5:
        result['warnings'].append(
            f"High HU rejection rate: {result['hu_rejection_rate']*100:.1f}% of voxels excluded"
        )

    # Step 3b: Adaptive widening if ROI too small
    if result['post_hu_voxels'] < MIN_ROI_VOXELS:
        logger.warning(f"ROI too small ({result['post_hu_voxels']} voxels), widening HU range")
        result['warnings'].append(
            f"ROI too small after HU gating ({result['post_hu_voxels']} voxels), widening range"
        )
        soft_tissue_mask_wide = (ct_data >= SOFT_TISSUE_HU_MIN_WIDE) & (ct_data <= SOFT_TISSUE_HU_MAX_WIDE)
        shell_gated = shell & soft_tissue_mask_wide
        result['post_hu_voxels'] = int(np.sum(shell_gated))
        result['hu_range_used'] = (SOFT_TISSUE_HU_MIN_WIDE, SOFT_TISSUE_HU_MAX_WIDE)

    # Step 4: Constrain to maxilla-adjacent region
    proximity_voxels = int(np.ceil(MAXILLA_PROXIMITY_MM / voxel_dims[0]))
    maxilla_dilated = binary_dilation(maxilla_mask, iterations=proximity_voxels)
    shell_gated = shell_gated & maxilla_dilated

    # Step 5: STRICT metal exclusion
    if metal_mask is not None and np.sum(metal_mask) > 0:
        pre_metal_voxels = np.sum(shell_gated)
        shell_gated = shell_gated & ~metal_mask
        post_metal_voxels = np.sum(shell_gated)
        metal_excluded = pre_metal_voxels - post_metal_voxels
        if metal_excluded > 0:
            result['warnings'].append(
                f"Metal exclusion removed {metal_excluded} voxels ({metal_excluded/pre_metal_voxels*100:.1f}%)"
            )

    result['roi_mask'] = shell_gated.astype(bool)
    result['volume_ml'] = np.sum(result['roi_mask']) * voxel_vol_ml
    result['post_hu_voxels'] = int(np.sum(result['roi_mask']))

    return result


def create_alveolar_bone_roi(teeth_mask: np.ndarray, maxilla_mask: np.ndarray,
                              ct_img: nib.Nifti1Image,
                              dilation_mm: float = PRIMARY_DILATION_MM,
                              metal_mask: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Create alveolar bone ROI for CT/STIR analysis.

    The alveolar bone is the part of the maxilla immediately surrounding tooth roots.

    Args:
        teeth_mask: Binary mask of teeth
        maxilla_mask: Binary mask of maxilla bone
        ct_img: NIfTI image for voxel dimensions
        dilation_mm: Dilation distance for teeth
        metal_mask: Optional metal artifact mask for exclusion

    Returns:
        Dictionary with roi_mask, volume_ml, and warnings
    """
    voxel_dims = get_voxel_dimensions(ct_img)
    voxel_vol_ml = get_voxel_volume_ml(ct_img)
    dilation_voxels = int(np.ceil(dilation_mm / voxel_dims[0]))

    result = {
        'roi_mask': None,
        'volume_ml': 0,
        'warnings': []
    }

    # Dilate teeth to capture surrounding bone
    teeth_dilated = binary_dilation(teeth_mask, iterations=dilation_voxels)

    # Alveolar bone = maxilla within dilated teeth region
    alveolar_bone = maxilla_mask & teeth_dilated

    # Exclude metal artifacts
    if metal_mask is not None and np.sum(metal_mask) > 0:
        pre_metal = np.sum(alveolar_bone)
        alveolar_bone = alveolar_bone & ~metal_mask
        if np.sum(alveolar_bone) < pre_metal:
            result['warnings'].append(
                f"Metal exclusion removed {pre_metal - np.sum(alveolar_bone)} voxels from bone ROI"
            )

    result['roi_mask'] = alveolar_bone.astype(bool)
    result['volume_ml'] = np.sum(result['roi_mask']) * voxel_vol_ml

    return result


# =============================================================================
# PER-TOOTH AND QUADRANT ROIS
# =============================================================================

def create_tooth_shells(teeth_mask: np.ndarray, peridental_roi: np.ndarray,
                        tooth_instances: Optional[np.ndarray] = None,
                        ct_img: nib.Nifti1Image = None) -> Dict[int, np.ndarray]:
    """
    Create per-tooth ROI subdivisions.

    If tooth instance segmentation is available, creates individual shells
    for each tooth. Otherwise, uses connected component analysis.

    Args:
        teeth_mask: Binary mask of all teeth
        peridental_roi: Full peridental soft tissue ROI
        tooth_instances: Optional instance segmentation with tooth IDs
        ct_img: NIfTI image for voxel dimensions

    Returns:
        Dictionary mapping tooth_id to ROI mask
    """
    tooth_rois = {}

    if tooth_instances is not None and np.max(tooth_instances) > 0:
        # Use provided instance segmentation
        unique_ids = np.unique(tooth_instances)
        unique_ids = unique_ids[unique_ids > 0]  # Exclude background

        voxel_dims = get_voxel_dimensions(ct_img) if ct_img else np.array([1, 1, 1])

        for tooth_id in unique_ids:
            # Get this tooth's mask
            tooth_mask = tooth_instances == tooth_id

            # Dilate to create shell
            dilation_voxels = int(np.ceil(PRIMARY_DILATION_MM / voxel_dims[0]))
            tooth_dilated = binary_dilation(tooth_mask, iterations=dilation_voxels)

            # Intersect with peridental ROI
            tooth_shell = tooth_dilated & peridental_roi

            if np.sum(tooth_shell) > 0:
                tooth_rois[int(tooth_id)] = tooth_shell.astype(bool)

    else:
        # Fallback: use connected components of teeth
        labeled_teeth, num_teeth = label(teeth_mask)

        if ct_img is not None:
            voxel_dims = get_voxel_dimensions(ct_img)
        else:
            voxel_dims = np.array([1, 1, 1])

        for i in range(1, num_teeth + 1):
            tooth_mask = labeled_teeth == i

            # Skip very small components (noise)
            if np.sum(tooth_mask) < 10:
                continue

            # Dilate and intersect
            dilation_voxels = int(np.ceil(PRIMARY_DILATION_MM / voxel_dims[0]))
            tooth_dilated = binary_dilation(tooth_mask, iterations=dilation_voxels)
            tooth_shell = tooth_dilated & peridental_roi

            if np.sum(tooth_shell) > 0:
                # Use component index as temporary ID
                tooth_rois[i] = tooth_shell.astype(bool)

    return tooth_rois


def create_quadrant_rois(tooth_shells: Dict[int, np.ndarray],
                          peridental_roi: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Aggregate per-tooth ROIs into dental quadrants.

    Quadrants (FDI notation):
    - Upper Right Posterior (teeth 14-18)
    - Upper Right Anterior (teeth 11-13)
    - Upper Left Anterior (teeth 21-23)
    - Upper Left Posterior (teeth 24-28)

    Args:
        tooth_shells: Dictionary mapping tooth_id to ROI mask
        peridental_roi: Full peridental ROI (used for whole_upper_jaw)

    Returns:
        Dictionary mapping quadrant name to combined ROI mask
    """
    quadrant_rois = {}

    for quadrant_name, tooth_ids in QUADRANTS.items():
        # Combine shells for teeth in this quadrant
        combined = None

        for tooth_id in tooth_ids:
            if tooth_id in tooth_shells:
                if combined is None:
                    combined = tooth_shells[tooth_id].copy()
                else:
                    combined = combined | tooth_shells[tooth_id]

        if combined is not None:
            quadrant_rois[quadrant_name] = combined

    # Add whole upper jaw
    quadrant_rois['whole_upper_jaw'] = peridental_roi.copy()

    return quadrant_rois


# =============================================================================
# ROI QUALITY CONTROL
# =============================================================================

def validate_roi(roi_mask: np.ndarray, ct_img: nib.Nifti1Image,
                 roi_name: str = "ROI") -> Dict[str, Any]:
    """
    Validate ROI quality.

    Args:
        roi_mask: Binary ROI mask
        ct_img: NIfTI image for voxel dimensions
        roi_name: Name for logging

    Returns:
        Dictionary with validation results
    """
    voxel_vol_ml = get_voxel_volume_ml(ct_img)

    result = {
        'valid': True,
        'voxel_count': int(np.sum(roi_mask)),
        'volume_ml': np.sum(roi_mask) * voxel_vol_ml,
        'warnings': []
    }

    if result['voxel_count'] == 0:
        result['valid'] = False
        result['warnings'].append(f"{roi_name} is empty (0 voxels)")
    elif result['voxel_count'] < MIN_ROI_VOXELS:
        result['warnings'].append(
            f"{roi_name} is small: {result['voxel_count']} voxels < {MIN_ROI_VOXELS} minimum"
        )

    return result


def compare_roi_volumes(baseline_roi: np.ndarray, followup_roi: np.ndarray,
                        ct_img: nib.Nifti1Image,
                        threshold_pct: float = 20.0) -> Dict[str, Any]:
    """
    Compare ROI volumes between timepoints for QC.

    Args:
        baseline_roi: Baseline ROI mask
        followup_roi: Followup ROI mask
        ct_img: NIfTI image for voxel dimensions
        threshold_pct: Threshold for flagging instability

    Returns:
        Dictionary with comparison results
    """
    voxel_vol_ml = get_voxel_volume_ml(ct_img)

    baseline_vol = np.sum(baseline_roi) * voxel_vol_ml
    followup_vol = np.sum(followup_roi) * voxel_vol_ml

    result = {
        'baseline_volume_ml': baseline_vol,
        'followup_volume_ml': followup_vol,
        'pct_change': 0,
        'stable': True,
        'warning': None
    }

    if baseline_vol > 0:
        result['pct_change'] = abs(followup_vol - baseline_vol) / baseline_vol * 100

        if result['pct_change'] > threshold_pct:
            result['stable'] = False
            result['warning'] = (
                f"Volume changed {result['pct_change']:.1f}% "
                f"({baseline_vol:.2f} -> {followup_vol:.2f} mL)"
            )

    return result


def get_roi_statistics(roi_mask: np.ndarray, image_data: np.ndarray,
                       percentile: int = 90) -> Dict[str, float]:
    """
    Calculate statistics for ROI values.

    Args:
        roi_mask: Binary ROI mask
        image_data: Image data array (PET, CT, or STIR)
        percentile: Percentile for robust metric (default 90)

    Returns:
        Dictionary with mean, std, percentile, min, max values
    """
    roi_values = image_data[roi_mask > 0]

    if len(roi_values) == 0:
        return {
            'mean': np.nan,
            'std': np.nan,
            f'p{percentile}': np.nan,
            'min': np.nan,
            'max': np.nan,
            'voxel_count': 0
        }

    # Filter out invalid values
    roi_values = roi_values[np.isfinite(roi_values)]

    if len(roi_values) == 0:
        return {
            'mean': np.nan,
            'std': np.nan,
            f'p{percentile}': np.nan,
            'min': np.nan,
            'max': np.nan,
            'voxel_count': 0
        }

    return {
        'mean': float(np.mean(roi_values)),
        'std': float(np.std(roi_values)),
        f'p{percentile}': float(np.percentile(roi_values, percentile)),
        'min': float(np.min(roi_values)),
        'max': float(np.max(roi_values)),
        'voxel_count': len(roi_values)
    }


# =============================================================================
# ROI GENERATION ORCHESTRATOR
# =============================================================================

def generate_all_rois(teeth_mask: np.ndarray, maxilla_mask: np.ndarray,
                      ct_data: np.ndarray, ct_img: nib.Nifti1Image,
                      metal_mask: Optional[np.ndarray] = None,
                      tooth_instances: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Generate all ROIs for a session.

    Args:
        teeth_mask: Binary mask of teeth
        maxilla_mask: Binary mask of maxilla
        ct_data: CT image data (HU values)
        ct_img: NIfTI image object
        metal_mask: Optional metal artifact mask
        tooth_instances: Optional tooth instance segmentation

    Returns:
        Dictionary with all ROIs and metadata
    """
    result = {
        'peridental_4mm': None,
        'peridental_6mm': None,
        'alveolar_bone': None,
        'tooth_shells': {},
        'quadrant_rois': {},
        'qc_metrics': {},
        'warnings': []
    }

    # Primary peridental ROI (4mm)
    logger.info("  Creating 4mm peridental ROI...")
    peridental_4mm = create_peridental_roi(
        teeth_mask, maxilla_mask, ct_data, ct_img,
        dilation_mm=PRIMARY_DILATION_MM, metal_mask=metal_mask
    )
    result['peridental_4mm'] = peridental_4mm
    result['warnings'].extend(peridental_4mm['warnings'])
    result['qc_metrics']['peridental_4mm_volume_ml'] = peridental_4mm['volume_ml']
    result['qc_metrics']['peridental_4mm_hu_rejection'] = peridental_4mm['hu_rejection_rate']

    # Sensitivity ROI (6mm)
    logger.info("  Creating 6mm peridental ROI (sensitivity)...")
    peridental_6mm = create_peridental_roi(
        teeth_mask, maxilla_mask, ct_data, ct_img,
        dilation_mm=SENSITIVITY_DILATION_MM, metal_mask=metal_mask
    )
    result['peridental_6mm'] = peridental_6mm
    result['qc_metrics']['peridental_6mm_volume_ml'] = peridental_6mm['volume_ml']

    # Alveolar bone ROI
    logger.info("  Creating alveolar bone ROI...")
    alveolar_bone = create_alveolar_bone_roi(
        teeth_mask, maxilla_mask, ct_img,
        dilation_mm=PRIMARY_DILATION_MM, metal_mask=metal_mask
    )
    result['alveolar_bone'] = alveolar_bone
    result['warnings'].extend(alveolar_bone['warnings'])
    result['qc_metrics']['alveolar_bone_volume_ml'] = alveolar_bone['volume_ml']

    # Per-tooth shells
    if peridental_4mm['roi_mask'] is not None:
        logger.info("  Creating per-tooth ROIs...")
        tooth_shells = create_tooth_shells(
            teeth_mask, peridental_4mm['roi_mask'],
            tooth_instances=tooth_instances, ct_img=ct_img
        )
        result['tooth_shells'] = tooth_shells
        result['qc_metrics']['tooth_count'] = len(tooth_shells)

        # Quadrant ROIs
        if len(tooth_shells) > 0:
            logger.info("  Creating quadrant ROIs...")
            quadrant_rois = create_quadrant_rois(
                tooth_shells, peridental_4mm['roi_mask']
            )
            result['quadrant_rois'] = quadrant_rois

    return result
