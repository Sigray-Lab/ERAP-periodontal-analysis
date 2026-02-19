"""
CT Segmentation utilities for Periodontal Analysis Pipeline.

Handles:
- Dental CT segmentation (teeth, maxilla)
- DentalSegmentator integration (nnUNet-based, preferred)
- TotalSegmentator integration (fallback)
- HU-based fallback segmentation
- Metal artifact detection
- Segmentation quality control
"""

import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Any
from scipy import ndimage
from scipy.ndimage import binary_dilation, binary_erosion, label, binary_fill_holes
import logging
import subprocess
import tempfile
import shutil
import os

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    HU_TEETH_MIN, HU_BONE_MIN, HU_BONE_MAX, HU_METAL_THRESHOLD,
    METAL_DILATION_MM,
    TEETH_VOLUME_MIN_ML, TEETH_VOLUME_MAX_ML,
    MAXILLA_VOLUME_MIN_ML, MAXILLA_VOLUME_MAX_ML,
    MIN_TEETH_COUNT
)
from utils.io_utils import get_voxel_dimensions, get_voxel_volume_ml

logger = logging.getLogger(__name__)

# DentalSegmentator model path
DENTAL_SEGMENTATOR_MODEL_PATH = Path(__file__).parent.parent.parent / "Models" / "nnUNet_results"
DENTAL_SEGMENTATOR_DATASET = "Dataset112_DentalSegmentator_v100"

# DentalSegmentator labels
DENTAL_LABELS = {
    'background': 0,
    'upper_skull': 1,
    'mandible': 2,
    'upper_teeth': 3,
    'lower_teeth': 4,
    'mandibular_canal': 5
}


# =============================================================================
# MAIN SEGMENTATION FUNCTION
# =============================================================================

def segment_dental_ct(ct_data: np.ndarray, ct_img: nib.Nifti1Image,
                      subject_id: str, session_id: str,
                      method_override: Optional[str] = None) -> Dict[str, Any]:
    """
    Segment teeth and maxilla from CT with fallback hierarchy.

    CRITICAL: Run independently for EACH timepoint. Do NOT propagate ROIs.

    Fallback hierarchy:
    1. DentalSegmentator (nnUNet-based, best for upper/lower separation)
    2. TotalSegmentator (if DentalSegmentator fails)
    3. HU-threshold based segmentation (final fallback)

    Args:
        ct_data: CT image data array (HU values)
        ct_img: nibabel image object for header/affine
        subject_id: Subject ID for logging
        session_id: Session ID for logging
        method_override: Force specific method ('DentalSegmentator', 'TotalSegmentator', 'HU_threshold')

    Returns:
        Dictionary with:
            - teeth_mask: Binary mask of teeth (UPPER teeth only)
            - maxilla_mask: Binary mask of maxilla bone
            - tooth_instances: Instance segmentation (if available, else None)
            - metal_mask: Binary mask of metal artifact zones
            - method: String indicating segmentation method used
            - qc_metrics: Dictionary of QC metrics
    """
    voxel_dims = get_voxel_dimensions(ct_img)
    voxel_vol_ml = get_voxel_volume_ml(ct_img)

    result = {
        'teeth_mask': None,
        'maxilla_mask': None,
        'tooth_instances': None,
        'metal_mask': None,
        'method': None,
        'qc_metrics': {},
        'warnings': []
    }

    # DentalSegmentator (nnUNet) requires GPU — upsamples to 0.43mm resolution
    # which exceeds CPU memory/time. Only use if explicitly requested.
    if method_override == 'DentalSegmentator':
        try:
            ds_result = run_dental_segmentator(ct_data, ct_img, voxel_dims,
                                                          subject_id=subject_id, session_id=session_id)
            if ds_result is not None:
                teeth_mask = ds_result.get('teeth_mask')
                maxilla_mask = ds_result.get('maxilla_mask')
                qc = validate_segmentation(teeth_mask, maxilla_mask, voxel_vol_ml)
                if qc['valid']:
                    result['teeth_mask'] = teeth_mask
                    result['maxilla_mask'] = maxilla_mask
                    result['method'] = 'DentalSegmentator'
                    result['qc_metrics'] = qc
                    logger.info(f"{subject_id}/{session_id}: DentalSegmentator succeeded")
                else:
                    result['warnings'].append(f"DentalSegmentator QC failed: {qc['reason']}")
        except Exception as e:
            result['warnings'].append(f"DentalSegmentator error: {str(e)}")

    # Try TotalSegmentator (uses dental cluster crop for CPU inference, ~7 min)
    if result['teeth_mask'] is None:
        if method_override is None or method_override == 'TotalSegmentator':
            try:
                ts_result = run_total_segmentator_dental(ct_data, ct_img, voxel_dims)
                if ts_result is not None:
                    teeth_mask = ts_result.get('teeth_mask')
                    maxilla_mask = ts_result.get('maxilla_mask')

                    # Validate segmentation quality
                    qc = validate_segmentation(teeth_mask, maxilla_mask, voxel_vol_ml)

                    if qc['valid']:
                        result['teeth_mask'] = teeth_mask
                        result['maxilla_mask'] = maxilla_mask
                        result['tooth_instances'] = ts_result.get('tooth_instances')
                        result['method'] = 'TotalSegmentator'
                        result['qc_metrics'] = qc
                        logger.info(f"{subject_id}/{session_id}: TotalSegmentator succeeded")
                    else:
                        logger.warning(f"{subject_id}/{session_id}: TotalSegmentator QC failed: {qc['reason']}")
                        result['warnings'].append(f"TotalSegmentator QC failed: {qc['reason']}")

            except ImportError:
                logger.info(f"{subject_id}/{session_id}: TotalSegmentator not installed")
                result['warnings'].append("TotalSegmentator not available")
            except Exception as e:
                logger.warning(f"{subject_id}/{session_id}: TotalSegmentator failed: {e}")
                result['warnings'].append(f"TotalSegmentator error: {str(e)}")

    # Fallback to HU thresholding
    if result['teeth_mask'] is None:
        if method_override is None or method_override == 'HU_threshold':
            logger.info(f"{subject_id}/{session_id}: Using HU threshold segmentation")
            teeth_mask, maxilla_mask = segment_by_hu_threshold(ct_data, voxel_dims)

            qc = validate_segmentation(teeth_mask, maxilla_mask, voxel_vol_ml)

            result['teeth_mask'] = teeth_mask
            result['maxilla_mask'] = maxilla_mask
            result['method'] = 'HU_threshold'
            result['qc_metrics'] = qc

            if not qc['valid']:
                result['warnings'].append(f"HU threshold QC warning: {qc['reason']}")

    # Detect metal artifacts (only near teeth, not entire CT)
    if result['teeth_mask'] is not None:
        result['metal_mask'] = detect_metal_artifacts(ct_data, voxel_dims, result['teeth_mask'])

        # Count affected teeth
        metal_teeth_overlap = result['metal_mask'] & binary_dilation(
            result['teeth_mask'],
            iterations=int(METAL_DILATION_MM / voxel_dims[0])
        )
        result['qc_metrics']['metal_voxels'] = int(np.sum(result['metal_mask']))
        result['qc_metrics']['metal_near_teeth_voxels'] = int(np.sum(metal_teeth_overlap))

    return result


# =============================================================================
# DENTALSEGMENTATOR INTEGRATION (nnUNet-based)
# =============================================================================

def run_dental_segmentator(ct_data: np.ndarray, ct_img: nib.Nifti1Image,
                           voxel_dims: np.ndarray,
                           subject_id: str = "unknown",
                           session_id: str = "unknown",
                           crop_only: bool = False) -> Optional[Dict[str, np.ndarray]]:
    """
    Run DentalSegmentator (nnUNet v2 model) for dental structure segmentation.

    DentalSegmentator provides EXPLICIT upper vs lower teeth separation:
    - Label 3: Upper Teeth (maxillary)
    - Label 4: Lower Teeth (mandibular)
    - Label 1: Upper Skull
    - Label 2: Mandible

    This is preferred over TotalSegmentator because it directly segments
    the UPPER teeth which is what we need for periodontal analysis.

    Args:
        ct_data: CT image data
        ct_img: nibabel image object
        voxel_dims: Voxel dimensions in mm

    Returns:
        Dictionary with teeth_mask (upper only) and maxilla_mask, or None if failed
    """
    import gc

    # Check if model exists
    model_path = DENTAL_SEGMENTATOR_MODEL_PATH / DENTAL_SEGMENTATOR_DATASET
    if not model_path.exists():
        logger.warning(f"DentalSegmentator model not found at {model_path}")
        return None

    # Check if nnUNetv2_predict is available
    nnunet_predict = shutil.which('nnUNetv2_predict')
    if nnunet_predict is None:
        logger.warning("nnUNetv2_predict not found in PATH")
        return None

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # nnUNet requires specific input folder structure
            input_folder = tmpdir / "input"
            output_folder = tmpdir / "output"
            input_folder.mkdir()
            output_folder.mkdir()

            # Crop CT to dental region to speed up processing on CPU
            # DentalSegmentator expects head/dental CT — crop to upper portion
            cropped_img, crop_slices = _crop_ct_to_dental_region(ct_data, ct_img, voxel_dims)

            # Save cropped image for QC inspection
            crop_qc_dir = Path(__file__).parent.parent.parent / "QC" / "dental_segmentator_crops"
            crop_qc_dir.mkdir(parents=True, exist_ok=True)
            crop_qc_file = crop_qc_dir / f"{subject_id}_{session_id}_crop.nii.gz"
            nib.save(cropped_img, str(crop_qc_file))
            logger.info(f"Saved crop QC image: {crop_qc_file}")
            logger.info(f"Crop shape: {cropped_img.shape[:3]}, voxels: {np.prod(cropped_img.shape[:3]):.0f}")

            # Save crop coordinates for mapping back to full CT space
            crop_coords_file = crop_qc_dir / f"{subject_id}_{session_id}_crop_coords.txt"
            with open(crop_coords_file, 'w') as f:
                for dim, s in zip(['x', 'y', 'z'], crop_slices):
                    f.write(f"{dim}: {s.start}-{s.stop}\n")
                f.write(f"original_shape: {ct_data.shape}\n")
                f.write(f"cropped_shape: {cropped_img.shape[:3]}\n")
            logger.info(f"Saved crop coordinates: {crop_coords_file}")

            if crop_only:
                logger.info("crop_only=True, skipping inference")
                return None

            # Save cropped input CT with nnUNet naming convention
            input_file = input_folder / "case_0000.nii.gz"
            nib.save(cropped_img, str(input_file))

            logger.info(f"Running DentalSegmentator (nnUNet) on cropped volume {cropped_img.shape[:3]}...")

            # Set environment variables for nnUNet
            env = os.environ.copy()
            env['nnUNet_results'] = str(DENTAL_SEGMENTATOR_MODEL_PATH)

            # Run nnUNetv2_predict
            cmd = [
                'nnUNetv2_predict',
                '-i', str(input_folder),
                '-o', str(output_folder),
                '-d', DENTAL_SEGMENTATOR_DATASET,
                '-c', '3d_fullres',
                '-f', '0',  # fold 0
                '--verbose'
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=env,
                timeout=28800  # 8 hour timeout for CPU inference
            )

            if result.returncode != 0:
                logger.warning(f"DentalSegmentator failed: {result.stderr}")
                return None

            # Load output segmentation
            output_file = output_folder / "case.nii.gz"
            if not output_file.exists():
                logger.warning(f"DentalSegmentator output not found at {output_file}")
                return None

            seg_img = nib.load(str(output_file))
            seg_data_cropped = seg_img.get_fdata().astype(np.int32)

            # Map cropped segmentation back to full volume
            seg_data = np.zeros(ct_data.shape, dtype=np.int32)
            seg_data[crop_slices] = seg_data_cropped

            # Extract UPPER teeth (label 3) - this is exactly what we need!
            upper_teeth_mask = (seg_data == DENTAL_LABELS['upper_teeth'])
            upper_teeth_voxels = np.sum(upper_teeth_mask)

            logger.info(f"DentalSegmentator: Upper teeth = {upper_teeth_voxels} voxels")

            if upper_teeth_voxels < 100:
                logger.warning("DentalSegmentator found too few upper teeth voxels")
                return None

            # Extract upper skull (label 1) as maxilla proxy
            # The "upper skull" includes maxilla and surrounding bone
            upper_skull_mask = (seg_data == DENTAL_LABELS['upper_skull'])

            # Create maxilla mask: bone near the upper teeth
            # Dilate upper teeth and intersect with upper skull
            dilation_voxels = int(15 / voxel_dims[0])
            teeth_dilated = binary_dilation(upper_teeth_mask, iterations=dilation_voxels)
            maxilla_mask = upper_skull_mask & teeth_dilated

            # If upper skull doesn't provide good maxilla, fall back to HU-based
            if np.sum(maxilla_mask) < 1000:
                logger.info("Upper skull sparse, deriving maxilla from HU")
                bone_hu = (ct_data > HU_BONE_MIN) & (ct_data <= HU_BONE_MAX)
                maxilla_mask = bone_hu & teeth_dilated

            # Keep largest connected component
            labeled, num_features = label(maxilla_mask)
            if num_features > 0:
                component_sizes = ndimage.sum(maxilla_mask, labeled, range(1, num_features + 1))
                largest = np.argmax(component_sizes) + 1
                maxilla_mask = (labeled == largest)

            logger.info(f"DentalSegmentator: Maxilla = {np.sum(maxilla_mask)} voxels")

            # Force garbage collection
            gc.collect()

            return {
                'teeth_mask': upper_teeth_mask.astype(bool),
                'maxilla_mask': maxilla_mask.astype(bool)
            }

    except subprocess.TimeoutExpired:
        logger.warning("DentalSegmentator timed out (8 hours)")
        return None
    except Exception as e:
        logger.warning(f"DentalSegmentator execution failed: {e}")
        return None


def _crop_ct_to_dental_region(ct_data: np.ndarray, ct_img: nib.Nifti1Image,
                               voxel_dims: np.ndarray,
                               margin_mm: float = 30.0) -> Tuple[nib.Nifti1Image, tuple]:
    """
    Crop CT tightly around the dental area for DentalSegmentator CPU inference.

    Strategy: Connected component clustering of high-HU voxels.
      1. Threshold at HU > 2000 (enamel), fallback to HU > 1500
      2. Dilate by 5mm to connect nearby dental structures
      3. Label connected components and keep only the largest cluster
         (actual teeth), ignoring scattered metal streak artifacts
      4. Add margin_mm padding around the cluster bounding box

    This produces tight ~1.2-1.8M voxel crops (~125mm box) instead of
    the previous ~37M voxel crops that included the entire skull.

    Args:
        ct_data: CT data array
        ct_img: nibabel image
        voxel_dims: voxel dimensions in mm
        margin_mm: margin in mm around the dental cluster (default 30mm)

    Returns:
        Tuple of (cropped_nifti_image, crop_slices_tuple)
    """
    # Step 1: Find high-HU voxels (enamel > 2000, fallback to dentin > 1500)
    high_hu = ct_data > 2000
    if high_hu.sum() < 50:
        high_hu = ct_data > 1500
        logger.info("Few enamel voxels (HU>2000), falling back to HU>1500")
    if high_hu.sum() < 50:
        # Last resort: use upper 40% of volume
        logger.warning("No dental structures found, using upper 40% of volume for crop")
        z_min = int(ct_data.shape[2] * 0.6)
        crop_slices = (slice(None), slice(None), slice(z_min, ct_data.shape[2]))
        cropped_data = ct_data[crop_slices]
        affine = ct_img.affine.copy()
        new_origin = affine[:3, :3] @ np.array([0, 0, z_min]) + affine[:3, 3]
        affine[:3, 3] = new_origin
        cropped_img = nib.Nifti1Image(cropped_data, affine, ct_img.header)
        logger.info(f"Fallback crop: {ct_data.shape} -> {cropped_data.shape}")
        return cropped_img, crop_slices

    # Step 2: Dilate by ~5mm to connect nearby dental structures
    dilation_iters = max(1, int(5.0 / voxel_dims[0]))
    dilated = binary_dilation(high_hu, iterations=dilation_iters)

    # Step 3: Label connected components, keep only the largest (teeth cluster)
    labeled, n_components = label(dilated)
    if n_components == 0:
        logger.warning("No connected components found after dilation")
        # Fallback: use bounding box of high_hu directly
        labeled = np.ones_like(high_hu, dtype=np.int32)
        labeled[~high_hu] = 0
        n_components = 1

    sizes = ndimage.sum(dilated, labeled, range(1, n_components + 1))
    largest_label = int(np.argmax(sizes)) + 1
    teeth_cluster = (labeled == largest_label)

    logger.info(f"Dental cluster: {n_components} components, largest={int(sizes[largest_label-1])} voxels "
                f"(total high-HU={high_hu.sum()})")

    # Step 4: Compute bounding box of the teeth cluster + margin
    coords = np.argwhere(teeth_cluster)
    margin_vox = [max(1, int(margin_mm / v)) for v in voxel_dims]

    x_min = max(0, int(coords[:, 0].min()) - margin_vox[0])
    x_max = min(ct_data.shape[0], int(coords[:, 0].max()) + margin_vox[0])
    y_min = max(0, int(coords[:, 1].min()) - margin_vox[1])
    y_max = min(ct_data.shape[1], int(coords[:, 1].max()) + margin_vox[1])
    z_min = max(0, int(coords[:, 2].min()) - margin_vox[2])
    z_max = min(ct_data.shape[2], int(coords[:, 2].max()) + margin_vox[2])

    crop_slices = (slice(x_min, x_max), slice(y_min, y_max), slice(z_min, z_max))
    cropped_data = ct_data[crop_slices]

    # Update affine origin for the crop offset
    affine = ct_img.affine.copy()
    new_origin = affine[:3, :3] @ np.array([x_min, y_min, z_min]) + affine[:3, 3]
    affine[:3, 3] = new_origin

    cropped_img = nib.Nifti1Image(cropped_data, affine, ct_img.header)

    crop_size_mm = [cropped_data.shape[i] * voxel_dims[i] for i in range(3)]
    logger.info(f"Cropped CT from {ct_data.shape} to {cropped_data.shape} "
                f"({cropped_data.size/1e6:.1f}M voxels, "
                f"{crop_size_mm[0]:.0f}x{crop_size_mm[1]:.0f}x{crop_size_mm[2]:.0f}mm)")

    return cropped_img, crop_slices


# =============================================================================
# TOTALSEGMENTATOR INTEGRATION
# =============================================================================

def run_total_segmentator_dental(ct_data: np.ndarray, ct_img: nib.Nifti1Image,
                                  voxel_dims: np.ndarray) -> Optional[Dict[str, np.ndarray]]:
    """
    Run TotalSegmentator with the dedicated TEETH task on a dental cluster crop.

    Uses the dental cluster crop to reduce input volume (~1.5M voxels instead
    of ~147M), making CPU inference feasible (~7 min per subject).

    TotalSegmentator teeth task labels (FDI numbering):
        1: lower_jawbone, 2: upper_jawbone
        11-18: upper right teeth (FDI 11-18)
        19-26: upper left teeth (FDI 21-28)
        27-34: lower left teeth (FDI 31-38)
        35-42: lower right teeth (FDI 41-48)

    Args:
        ct_data: CT image data
        ct_img: nibabel image object
        voxel_dims: Voxel dimensions in mm

    Returns:
        Dictionary with teeth_mask (upper only), maxilla_mask, and
        tooth_instances (per-tooth labels), or None if failed
    """
    try:
        from totalsegmentator.python_api import totalsegmentator
        import gc

        # Crop CT to dental region for fast CPU inference
        cropped_img, crop_slices = _crop_ct_to_dental_region(ct_data, ct_img, voxel_dims)

        logger.info("Running TotalSegmentator 'teeth' task on cropped CT...")

        # Run TotalSegmentator — device='cpu' explicitly to avoid GPU detection
        # Note: TotalSegmentator v2.12.0 has a bug where device=None causes
        # a crash in validate_device_type_api. We patched convert_device_to_string
        # to handle string/None inputs. Passing device='cpu' explicitly.
        result_img = totalsegmentator(
            input=cropped_img,
            output=None,
            task="teeth",
            fast=False,
            quiet=True,
            ml=True,
            device='cpu'
        )

        if result_img is None:
            logger.warning("TotalSegmentator returned None")
            return None

        seg_data_cropped = result_img.get_fdata().astype(np.int32)
        labels_found = np.unique(seg_data_cropped)
        logger.info(f"TotalSegmentator teeth: {len(labels_found)-1} labels found")

        # Map cropped segmentation back to full volume
        seg_data = np.zeros(ct_data.shape, dtype=np.int32)
        seg_data[crop_slices] = seg_data_cropped

        # Upper teeth: labels 11-18 (right), 19-26 (left)
        upper_teeth_labels = list(range(11, 19)) + list(range(19, 27))
        upper_teeth_mask = np.isin(seg_data, upper_teeth_labels)
        upper_teeth_count = np.sum(upper_teeth_mask)
        logger.info(f"TotalSegmentator: Upper teeth = {upper_teeth_count} voxels")

        if upper_teeth_count < 100:
            logger.warning("TotalSegmentator found too few upper teeth voxels")
            # Fall back to all teeth
            all_teeth_labels = upper_teeth_labels + list(range(27, 35)) + list(range(35, 43))
            all_teeth = np.isin(seg_data, all_teeth_labels)
            if np.sum(all_teeth) > upper_teeth_count:
                logger.info("Using all teeth as fallback")
                upper_teeth_mask = all_teeth

        # Upper jawbone (label 2)
        maxilla_mask = (seg_data == 2)
        maxilla_voxels = np.sum(maxilla_mask)

        # If jawbone label is sparse, supplement with HU-based bone near upper teeth
        if maxilla_voxels < 1000 and upper_teeth_count > 0:
            logger.info("Upper jawbone label sparse, supplementing with HU-based bone")
            dilation_voxels = int(15 / voxel_dims[0])
            teeth_dilated = binary_dilation(upper_teeth_mask, iterations=dilation_voxels)
            bone_hu = (ct_data > HU_BONE_MIN) & (ct_data <= HU_BONE_MAX)
            maxilla_mask = (maxilla_mask | (bone_hu & teeth_dilated))

            labeled_m, num_features = label(maxilla_mask)
            if num_features > 0:
                component_sizes = ndimage.sum(maxilla_mask, labeled_m, range(1, num_features + 1))
                largest = np.argmax(component_sizes) + 1
                maxilla_mask = (labeled_m == largest)

        logger.info(f"TotalSegmentator: Maxilla = {np.sum(maxilla_mask)} voxels")

        # Build per-tooth instance map (full volume, upper teeth only)
        tooth_instances = np.zeros(ct_data.shape, dtype=np.int32)
        for lbl in upper_teeth_labels:
            tooth_instances[seg_data == lbl] = lbl

        gc.collect()

        return {
            'teeth_mask': upper_teeth_mask.astype(bool),
            'maxilla_mask': maxilla_mask.astype(bool),
            'tooth_instances': tooth_instances,
            'cropped_img': cropped_img,
            'crop_slices': crop_slices,
            'full_seg_data': seg_data
        }

    except ImportError:
        logger.info("TotalSegmentator not installed")
        return None
    except Exception as e:
        logger.warning(f"TotalSegmentator execution failed: {e}")
        return None


# =============================================================================
# DENTAL REGION MASK (used by HU-fallback to restrict to dental area)
# =============================================================================

def _get_dental_cluster_mask(ct_data: np.ndarray, voxel_dims: np.ndarray,
                              margin_mm: float = 30.0) -> Optional[np.ndarray]:
    """
    Create a binary mask of the dental region using connected component clustering.

    Same algorithm as _crop_ct_to_dental_region but returns a boolean mask
    instead of a cropped image. Used by segment_by_hu_threshold to restrict
    teeth detection to the actual dental area, avoiding metal streak artifacts
    that extend into the skull.

    Args:
        ct_data: CT data array
        voxel_dims: voxel dimensions in mm
        margin_mm: margin around the dental cluster in mm

    Returns:
        Boolean mask of the dental region, or None if no dental structures found
    """
    high_hu = ct_data > 2000
    if high_hu.sum() < 50:
        high_hu = ct_data > 1500
    if high_hu.sum() < 50:
        return None

    dilation_iters = max(1, int(5.0 / voxel_dims[0]))
    dilated = binary_dilation(high_hu, iterations=dilation_iters)

    labeled, n_components = label(dilated)
    if n_components == 0:
        return None

    sizes = ndimage.sum(dilated, labeled, range(1, n_components + 1))
    largest_label = int(np.argmax(sizes)) + 1
    teeth_cluster = (labeled == largest_label)

    # Create a bounding box mask with margin
    coords = np.argwhere(teeth_cluster)
    margin_vox = [max(1, int(margin_mm / v)) for v in voxel_dims]

    mask = np.zeros(ct_data.shape, dtype=bool)
    x_lo = max(0, int(coords[:, 0].min()) - margin_vox[0])
    x_hi = min(ct_data.shape[0], int(coords[:, 0].max()) + margin_vox[0])
    y_lo = max(0, int(coords[:, 1].min()) - margin_vox[1])
    y_hi = min(ct_data.shape[1], int(coords[:, 1].max()) + margin_vox[1])
    z_lo = max(0, int(coords[:, 2].min()) - margin_vox[2])
    z_hi = min(ct_data.shape[2], int(coords[:, 2].max()) + margin_vox[2])
    mask[x_lo:x_hi, y_lo:y_hi, z_lo:z_hi] = True

    logger.info(f"Dental cluster mask: {mask.sum()} voxels "
                f"({x_hi-x_lo}x{y_hi-y_lo}x{z_hi-z_lo} box)")
    return mask


# =============================================================================
# HU-BASED SEGMENTATION
# =============================================================================

def segment_by_hu_threshold(ct_data: np.ndarray, voxel_dims: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Segment teeth and bone using HU thresholds.

    This is the fallback method when automated tools fail.

    CRITICAL: We specifically target UPPER JAW (maxilla) teeth because:
    1. The T2 STIR FOV covers the upper jaw
    2. The FDG-PET FOV is better for upper jaw

    Strategy:
    1. Find oral cavity region anatomically
    2. Identify upper vs lower teeth based on position relative to oral cavity
    3. Keep only UPPER teeth (superior to oral cavity midline)

    Args:
        ct_data: CT image data (HU values)
        voxel_dims: Voxel dimensions in mm

    Returns:
        Tuple of (teeth_mask, maxilla_mask)
    """
    # Step 1: Find the oral cavity region and its z-midline
    oral_region, oral_z_center = find_oral_cavity_region_with_center(ct_data, voxel_dims)

    if np.sum(oral_region) == 0:
        logger.warning("Could not identify oral cavity region, using full volume with caution")
        oral_region = np.ones_like(ct_data, dtype=bool)
        oral_z_center = ct_data.shape[2] // 2

    # Step 1b: Restrict oral_region to dental cluster area to avoid metal
    # streak artifacts extending far into the skull (HU>1500 scattered voxels)
    dental_cluster_mask = _get_dental_cluster_mask(ct_data, voxel_dims, margin_mm=30.0)
    if dental_cluster_mask is not None:
        oral_region = oral_region & dental_cluster_mask
        logger.info(f"Restricted oral region to dental cluster: {oral_region.sum()} voxels")

    # Step 2: Threshold for teeth (high HU - enamel/dentin) ONLY within oral region
    teeth_raw = (ct_data > HU_TEETH_MIN) & oral_region

    # Step 3: Threshold for bone within oral region
    bone_raw = (ct_data > HU_BONE_MIN) & (ct_data <= HU_BONE_MAX) & oral_region

    # Step 4: Clean up teeth mask - remove small isolated components (noise)
    teeth_cleaned = remove_small_components(teeth_raw, min_size_mm3=50, voxel_dims=voxel_dims)

    # Step 5: CRITICAL - Select UPPER teeth only (maxilla, not mandible)
    # Upper teeth are SUPERIOR to the oral cavity center
    # In typical head CT, superior = higher z value (but check orientation)
    upper_teeth = select_upper_teeth(teeth_cleaned, oral_z_center, ct_data.shape, voxel_dims)

    if np.sum(upper_teeth) < 1000:  # Too few voxels - might have wrong orientation
        logger.warning("Upper teeth selection yielded few voxels, trying inverse z selection")
        # Try the other direction
        lower_region = np.zeros_like(teeth_cleaned, dtype=bool)
        lower_region[:, :, :oral_z_center] = True
        upper_teeth_alt = teeth_cleaned & lower_region
        if np.sum(upper_teeth_alt) > np.sum(upper_teeth):
            upper_teeth = upper_teeth_alt
            logger.info("Using inverse z selection for upper teeth")

    # Step 6: Keep largest component of upper teeth
    if np.sum(upper_teeth) > 0:
        upper_teeth = keep_largest_components(upper_teeth, n_components=1)
    else:
        logger.warning("No upper teeth found, using all teeth")
        upper_teeth = keep_largest_components(teeth_cleaned, n_components=1)

    # Step 7: Identify maxilla region around the upper teeth
    maxilla_mask = extract_maxilla_region(bone_raw, upper_teeth, ct_data, voxel_dims)

    logger.info(f"Upper teeth: {np.sum(upper_teeth)} voxels, Maxilla: {np.sum(maxilla_mask)} voxels")

    return upper_teeth, maxilla_mask


def select_upper_teeth(teeth_mask: np.ndarray, oral_z_center: int,
                       shape: tuple, voxel_dims: np.ndarray) -> np.ndarray:
    """
    Select only upper jaw teeth (maxillary teeth).

    Strategy: Find the two main teeth components (upper and lower jaw arches),
    then select the one that is more SUPERIOR (towards nasal cavity).

    Args:
        teeth_mask: Binary mask of all teeth
        oral_z_center: Z-coordinate of oral cavity center
        shape: Image shape
        voxel_dims: Voxel dimensions

    Returns:
        Binary mask of upper teeth only
    """
    # Label all connected components in teeth
    labeled_teeth, num_teeth_components = label(teeth_mask)

    if num_teeth_components == 0:
        return teeth_mask

    # Get the two largest components (should be upper and lower arches)
    component_sizes = ndimage.sum(teeth_mask, labeled_teeth, range(1, num_teeth_components + 1))
    sorted_indices = np.argsort(component_sizes)[::-1]  # Descending

    if num_teeth_components == 1:
        # Only one component - try to split by z
        upper_region = np.zeros(shape, dtype=bool)
        upper_region[:, :, oral_z_center:] = True
        upper_teeth = teeth_mask & upper_region

        # Also try lower z
        lower_region = np.zeros(shape, dtype=bool)
        lower_region[:, :, :oral_z_center] = True
        lower_teeth = teeth_mask & lower_region

        # Return whichever has more voxels (might be wrong orientation)
        if np.sum(upper_teeth) > np.sum(lower_teeth):
            return upper_teeth
        else:
            return lower_teeth

    # Get centroids of the two largest components
    components_info = []
    for i in range(min(2, num_teeth_components)):
        comp_idx = sorted_indices[i] + 1  # +1 because labels start at 1
        comp_mask = (labeled_teeth == comp_idx)
        centroid = ndimage.center_of_mass(comp_mask)
        components_info.append({
            'label': comp_idx,
            'mask': comp_mask,
            'centroid_z': centroid[2],
            'size': component_sizes[sorted_indices[i]]
        })

    # The upper teeth should be at higher z if z increases superiorly (RAS)
    # But we need to check - compare with oral cavity center
    logger.info(f"Two largest teeth components: z={components_info[0]['centroid_z']:.0f} "
                f"({components_info[0]['size']} vox), z={components_info[1]['centroid_z']:.0f} "
                f"({components_info[1]['size']} vox), oral_z={oral_z_center}")

    # Determine which component is upper based on relationship to oral cavity
    # Upper teeth (maxilla) are SUPERIOR to oral cavity
    # Lower teeth (mandible) are INFERIOR to oral cavity

    for comp in components_info:
        comp['dist_from_oral'] = abs(comp['centroid_z'] - oral_z_center)
        comp['is_above_oral'] = comp['centroid_z'] > oral_z_center

    # Select the component that is above the oral cavity center
    # If both are above or both below, select the one closer to oral cavity
    above_oral = [c for c in components_info if c['is_above_oral']]
    below_oral = [c for c in components_info if not c['is_above_oral']]

    if above_oral and below_oral:
        # One above, one below - select the one above (upper jaw)
        upper = above_oral[0]
        logger.info(f"Selected upper teeth (above oral cavity): z={upper['centroid_z']:.0f}")
    elif above_oral:
        # Both above oral cavity — this commonly happens when oral cavity center
        # is between upper and lower teeth. Select the LARGEST component,
        # which is most likely the full dental arch rather than a fragment.
        upper = max(above_oral, key=lambda x: x['size'])
        logger.info(f"Both components above oral, selected largest: z={upper['centroid_z']:.0f} ({upper['size']} vox)")
    else:
        # Both below - CT might have inverted z. Select the largest component.
        upper = max(below_oral, key=lambda x: x['size'])
        logger.info(f"Both components below oral (inverted z?), selected largest: z={upper['centroid_z']:.0f}")

    return upper['mask'].astype(bool)


def find_oral_cavity_region_with_center(ct_data: np.ndarray, voxel_dims: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Identify the oral cavity region and its z-center for upper/lower jaw separation.

    Strategy:
    1. Find air-filled spaces (oral cavity, nasal cavity, airways)
    2. Identify the oral cavity as the large air space with nearby teeth
    3. Return both the search region AND the z-center for jaw separation

    Args:
        ct_data: CT image data (HU values)
        voxel_dims: Voxel dimensions in mm

    Returns:
        Tuple of (Binary mask of the oral/dental region, z-center of oral cavity)
    """
    # Air threshold (very low HU)
    air_mask = ct_data < -800

    # Soft tissue range (to help find facial boundaries)
    soft_tissue = (ct_data > -100) & (ct_data < 200)

    # Find the center of mass of the image (approximate head center)
    z_center = ct_data.shape[2] // 2
    y_center = ct_data.shape[1] // 2
    x_center = ct_data.shape[0] // 2

    # The oral cavity is typically in the anterior-inferior quadrant of the head
    # We'll analyze the air distribution to find it

    # Label connected air regions
    labeled_air, num_air_regions = label(air_mask)

    if num_air_regions == 0:
        logger.warning("No air regions found - unusual CT data")
        # Fallback: use central region
        return create_central_search_region(ct_data.shape, voxel_dims)

    # Find air regions by size
    air_sizes = ndimage.sum(air_mask, labeled_air, range(1, num_air_regions + 1))

    # Get centroids of air regions
    air_centroids = ndimage.center_of_mass(air_mask, labeled_air, range(1, num_air_regions + 1))

    # The oral cavity should be:
    # - In the anterior part of the head (low y values typically)
    # - Below the nasal cavity (lower z for RAS orientation, but depends on scan)
    # - Reasonably large (bigger than small airways)

    # Find candidate oral cavity regions
    min_oral_size = 5000  # minimum voxels for oral cavity
    candidates = []

    for i, (size, centroid) in enumerate(zip(air_sizes, air_centroids), 1):
        if size > min_oral_size:
            candidates.append({
                'label': i,
                'size': size,
                'centroid': centroid
            })

    if not candidates:
        logger.warning("No large air regions found for oral cavity")
        return create_central_search_region(ct_data.shape, voxel_dims), ct_data.shape[2] // 2

    # Sort by z-coordinate to find lower regions (likely oral cavity vs nasal)
    # Note: orientation varies, so we use a heuristic based on size and position

    # For dental CT, the oral cavity should have teeth (high HU) nearby
    # Use this as validation
    best_candidate = None
    best_score = 0

    for cand in candidates:
        # Create a mask for this air region
        region_mask = labeled_air == cand['label']

        # Dilate to include surrounding area
        dilated = binary_dilation(region_mask, iterations=int(30 / voxel_dims[0]))

        # Check for high-HU structures nearby (potential teeth)
        high_hu_nearby = np.sum((ct_data > 1000) & dilated)

        # Score: prefer regions with nearby high-HU (teeth) and moderate size
        score = high_hu_nearby * np.sqrt(cand['size'])

        if score > best_score:
            best_score = score
            best_candidate = cand

    if best_candidate is None:
        return create_central_search_region(ct_data.shape, voxel_dims), ct_data.shape[2] // 2

    # Create the oral region mask
    oral_air = labeled_air == best_candidate['label']

    # Dilate to create search region (teeth are around the oral cavity, not in it)
    dilation_mm = 40  # 40mm search radius around oral cavity
    dilation_voxels = int(dilation_mm / voxel_dims[0])
    oral_region = binary_dilation(oral_air, iterations=dilation_voxels)

    # Also include the bone/teeth that would be adjacent
    # Limit to reasonable z-range (don't go into skull top or neck bottom)
    centroid = best_candidate['centroid']
    z_min = max(0, int(centroid[2] - 60 / voxel_dims[2]))
    z_max = min(ct_data.shape[2], int(centroid[2] + 60 / voxel_dims[2]))

    # Create z-bounded mask
    z_bounded = np.zeros_like(oral_region)
    z_bounded[:, :, z_min:z_max] = True

    oral_region = oral_region & z_bounded

    oral_z_center = int(centroid[2])
    logger.info(f"Oral cavity region identified: centroid z={oral_z_center}, "
                f"search region {np.sum(oral_region)} voxels")

    return oral_region, oral_z_center


def create_central_search_region(shape: tuple, voxel_dims: np.ndarray) -> np.ndarray:
    """
    Create a fallback central search region when oral cavity detection fails.

    Args:
        shape: Image shape
        voxel_dims: Voxel dimensions

    Returns:
        Binary mask of central region
    """
    mask = np.zeros(shape, dtype=bool)

    # Use central 40% of image in each dimension
    x_margin = int(shape[0] * 0.3)
    y_margin = int(shape[1] * 0.3)
    z_margin = int(shape[2] * 0.3)

    mask[x_margin:-x_margin, y_margin:-y_margin, z_margin:-z_margin] = True

    return mask


def keep_largest_components(mask: np.ndarray, n_components: int = 1) -> np.ndarray:
    """
    Keep only the N largest connected components.

    Args:
        mask: Binary mask
        n_components: Number of components to keep

    Returns:
        Cleaned binary mask with only largest components
    """
    labeled, num_features = label(mask)

    if num_features <= n_components:
        return mask

    # Calculate component sizes
    component_sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))

    # Get indices of largest components
    largest_indices = np.argsort(component_sizes)[-n_components:] + 1

    # Create mask with only largest components
    cleaned = np.isin(labeled, largest_indices)

    return cleaned


def remove_small_components(mask: np.ndarray, min_size_mm3: float,
                            voxel_dims: np.ndarray) -> np.ndarray:
    """
    Remove connected components smaller than threshold.

    Args:
        mask: Binary mask
        min_size_mm3: Minimum component size in mm³
        voxel_dims: Voxel dimensions in mm

    Returns:
        Cleaned binary mask
    """
    voxel_vol = np.prod(voxel_dims)
    min_voxels = int(min_size_mm3 / voxel_vol)

    # Label connected components
    labeled, num_features = label(mask)

    # Calculate component sizes
    component_sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))

    # Create mask of components above threshold
    cleaned = np.zeros_like(mask)
    for i, size in enumerate(component_sizes, 1):
        if size >= min_voxels:
            cleaned[labeled == i] = True

    return cleaned


def extract_maxilla_region(bone_mask: np.ndarray, teeth_mask: np.ndarray,
                           ct_data: np.ndarray, voxel_dims: np.ndarray) -> np.ndarray:
    """
    Extract maxilla (upper jaw) bone region.

    Strategy:
    1. Start from teeth mask
    2. Dilate conservatively to include alveolar bone
    3. Keep only bone directly connected/adjacent to teeth

    Args:
        bone_mask: Binary mask of all bone (already restricted to oral region)
        teeth_mask: Binary mask of teeth
        ct_data: CT data for reference
        voxel_dims: Voxel dimensions

    Returns:
        Binary mask of maxilla
    """
    if np.sum(teeth_mask) == 0:
        logger.warning("No teeth found for maxilla extraction")
        # Return empty mask rather than full bone mask
        return np.zeros_like(bone_mask, dtype=bool)

    # Dilate teeth conservatively to create region of interest
    # 10mm should capture alveolar bone but not extend to skull
    dilation_voxels = int(10 / voxel_dims[0])
    teeth_dilated = binary_dilation(teeth_mask, iterations=dilation_voxels)

    # Maxilla is bone within dilated teeth region
    maxilla_candidate = bone_mask & teeth_dilated

    if np.sum(maxilla_candidate) == 0:
        logger.warning("No bone found adjacent to teeth")
        return np.zeros_like(bone_mask, dtype=bool)

    # Keep only the largest connected component
    maxilla_mask = keep_largest_components(maxilla_candidate, n_components=1)

    # Validate: maxilla should not be much larger than teeth
    teeth_vol = np.sum(teeth_mask)
    maxilla_vol = np.sum(maxilla_mask)

    if maxilla_vol > teeth_vol * 10:
        logger.warning(f"Maxilla volume ({maxilla_vol}) >> teeth volume ({teeth_vol}), "
                      "constraining further")
        # Re-do with tighter dilation
        dilation_voxels = int(5 / voxel_dims[0])
        teeth_dilated = binary_dilation(teeth_mask, iterations=dilation_voxels)
        maxilla_candidate = bone_mask & teeth_dilated
        maxilla_mask = keep_largest_components(maxilla_candidate, n_components=1)

    return maxilla_mask.astype(bool)


def refine_teeth_to_upper_jaw(teeth_mask: np.ndarray, maxilla_mask: np.ndarray,
                               voxel_dims: np.ndarray) -> np.ndarray:
    """
    Refine teeth mask to include only upper jaw teeth.

    Args:
        teeth_mask: Full teeth mask (may include mandible)
        maxilla_mask: Maxilla bone mask
        voxel_dims: Voxel dimensions

    Returns:
        Teeth mask for upper jaw only
    """
    # Dilate maxilla slightly to include teeth roots
    dilation_voxels = int(5 / voxel_dims[0])  # 5mm
    maxilla_dilated = binary_dilation(maxilla_mask, iterations=dilation_voxels)

    # Keep teeth that overlap with dilated maxilla
    upper_teeth = teeth_mask & maxilla_dilated

    # If this removes all teeth, fall back to original
    if np.sum(upper_teeth) < np.sum(teeth_mask) * 0.3:
        logger.warning("Upper jaw refinement removed too many teeth, using full mask")
        return teeth_mask

    return upper_teeth


# =============================================================================
# METAL ARTIFACT DETECTION
# =============================================================================

def detect_metal_artifacts(ct_data: np.ndarray, voxel_dims: np.ndarray,
                           teeth_mask: np.ndarray = None) -> np.ndarray:
    """
    Detect metal artifacts from dental implants/fillings.

    Metal artifacts cause PET attenuation correction errors.
    We detect high HU regions NEAR the teeth and dilate for safety margin.

    IMPORTANT: Only detects metal in the dental region to avoid flagging
    metal elsewhere in the CT (e.g., surgical clips, monitoring equipment).

    Args:
        ct_data: CT image data (HU values)
        voxel_dims: Voxel dimensions in mm
        teeth_mask: Optional teeth mask to constrain metal search region

    Returns:
        Binary mask of metal artifact zones (with dilation buffer)
    """
    # Detect metal (very high HU)
    metal_raw = ct_data > HU_METAL_THRESHOLD

    if np.sum(metal_raw) == 0:
        return np.zeros_like(ct_data, dtype=bool)

    # If teeth mask provided, only consider metal NEAR the teeth
    # (within 20mm of teeth region)
    if teeth_mask is not None and np.sum(teeth_mask) > 0:
        search_radius_mm = 20
        search_radius_voxels = int(search_radius_mm / voxel_dims[0])
        teeth_region = binary_dilation(teeth_mask, iterations=search_radius_voxels)
        metal_raw = metal_raw & teeth_region

    if np.sum(metal_raw) == 0:
        return np.zeros_like(ct_data, dtype=bool)

    # Dilate for safety buffer around dental metal
    dilation_voxels = int(METAL_DILATION_MM / voxel_dims[0])
    metal_dilated = binary_dilation(metal_raw, iterations=dilation_voxels)

    return metal_dilated


def identify_metal_affected_teeth(metal_mask: np.ndarray, teeth_mask: np.ndarray,
                                   tooth_instances: Optional[np.ndarray] = None) -> List[int]:
    """
    Identify which teeth are affected by metal artifacts.

    Args:
        metal_mask: Binary mask of metal zones
        teeth_mask: Binary mask of teeth
        tooth_instances: Optional instance segmentation with tooth IDs

    Returns:
        List of affected tooth IDs (or count if no instances)
    """
    overlap = metal_mask & teeth_mask

    if tooth_instances is not None:
        # Find unique tooth IDs in overlap
        affected_ids = np.unique(tooth_instances[overlap])
        affected_ids = affected_ids[affected_ids > 0]  # Remove background
        return affected_ids.tolist()
    else:
        # Return count of affected voxels
        return [int(np.sum(overlap))]


# =============================================================================
# QUALITY CONTROL
# =============================================================================

def validate_segmentation(teeth_mask: np.ndarray, maxilla_mask: np.ndarray,
                          voxel_vol_ml: float) -> Dict[str, Any]:
    """
    Validate segmentation quality.

    Checks:
    - Teeth mask volume in expected range
    - Maxilla mask volume in expected range
    - Teeth count (if instance segmentation available)

    Args:
        teeth_mask: Binary mask of teeth
        maxilla_mask: Binary mask of maxilla
        voxel_vol_ml: Volume of one voxel in mL

    Returns:
        Dictionary with validation results
    """
    qc = {
        'valid': True,
        'reason': None,
        'teeth_volume_ml': 0,
        'maxilla_volume_ml': 0,
        'teeth_voxels': 0,
        'maxilla_voxels': 0
    }

    # Calculate volumes
    if teeth_mask is not None:
        qc['teeth_voxels'] = int(np.sum(teeth_mask))
        qc['teeth_volume_ml'] = qc['teeth_voxels'] * voxel_vol_ml

    if maxilla_mask is not None:
        qc['maxilla_voxels'] = int(np.sum(maxilla_mask))
        qc['maxilla_volume_ml'] = qc['maxilla_voxels'] * voxel_vol_ml

    # Validate teeth volume
    if qc['teeth_volume_ml'] < TEETH_VOLUME_MIN_ML:
        qc['valid'] = False
        qc['reason'] = f"Teeth volume too small: {qc['teeth_volume_ml']:.1f} mL < {TEETH_VOLUME_MIN_ML} mL"
    elif qc['teeth_volume_ml'] > TEETH_VOLUME_MAX_ML:
        qc['valid'] = False
        qc['reason'] = f"Teeth volume too large: {qc['teeth_volume_ml']:.1f} mL > {TEETH_VOLUME_MAX_ML} mL"

    # Validate maxilla volume
    if qc['maxilla_volume_ml'] < MAXILLA_VOLUME_MIN_ML:
        if qc['reason'] is None:
            qc['valid'] = False
            qc['reason'] = f"Maxilla volume too small: {qc['maxilla_volume_ml']:.1f} mL"
    elif qc['maxilla_volume_ml'] > MAXILLA_VOLUME_MAX_ML:
        if qc['reason'] is None:
            qc['valid'] = False
            qc['reason'] = f"Maxilla volume too large: {qc['maxilla_volume_ml']:.1f} mL"

    return qc


def compare_segmentation_volumes(baseline_result: Dict, followup_result: Dict,
                                 subject_id: str, threshold_pct: float = 20.0) -> Dict[str, Any]:
    """
    Compare segmentation volumes between timepoints for QC.

    Flags teeth with >threshold% volume change as "unstable".

    Args:
        baseline_result: Segmentation result from baseline
        followup_result: Segmentation result from followup
        subject_id: Subject ID
        threshold_pct: Percent change threshold for flagging

    Returns:
        Dictionary with comparison results
    """
    comparison = {
        'subject_id': subject_id,
        'teeth_volume_baseline_ml': baseline_result['qc_metrics'].get('teeth_volume_ml', 0),
        'teeth_volume_followup_ml': followup_result['qc_metrics'].get('teeth_volume_ml', 0),
        'maxilla_volume_baseline_ml': baseline_result['qc_metrics'].get('maxilla_volume_ml', 0),
        'maxilla_volume_followup_ml': followup_result['qc_metrics'].get('maxilla_volume_ml', 0),
        'teeth_pct_change': 0,
        'maxilla_pct_change': 0,
        'stable': True,
        'warnings': []
    }

    # Calculate percent changes
    if comparison['teeth_volume_baseline_ml'] > 0:
        comparison['teeth_pct_change'] = abs(
            comparison['teeth_volume_followup_ml'] - comparison['teeth_volume_baseline_ml']
        ) / comparison['teeth_volume_baseline_ml'] * 100

    if comparison['maxilla_volume_baseline_ml'] > 0:
        comparison['maxilla_pct_change'] = abs(
            comparison['maxilla_volume_followup_ml'] - comparison['maxilla_volume_baseline_ml']
        ) / comparison['maxilla_volume_baseline_ml'] * 100

    # Check stability
    if comparison['teeth_pct_change'] > threshold_pct:
        comparison['stable'] = False
        comparison['warnings'].append(
            f"Teeth volume changed {comparison['teeth_pct_change']:.1f}% (threshold: {threshold_pct}%)"
        )

    if comparison['maxilla_pct_change'] > threshold_pct:
        comparison['stable'] = False
        comparison['warnings'].append(
            f"Maxilla volume changed {comparison['maxilla_pct_change']:.1f}% (threshold: {threshold_pct}%)"
        )

    return comparison
