#!/usr/bin/env python3
"""
02_run_geometry_pipeline.py - Batch Geometry ROI Pipeline

Runs the full geometry-based peridental ROI pipeline for all (or selected) subjects:

    Step 1: TotalSegmentator teeth task (produces multilabel with per-tooth FDI labels)
    Step 2: HU-fallback segmentation (always run as secondary method)
    Step 3: Geometry ROI generation (dilate tooth → subtract bone/teeth → peridental shell)
    Step 4: TotalSegmentator head_muscles task (produces tongue mask for future trimming)
    Step 5: Rigid CT→PET co-registration (ANTsPy) + resample masks to PET native space

Output layout:
    DerivedData/segmentations/totalsegmentator_teeth/sub-XXX/ses-YYY/
        totalseg_teeth_multilabel.nii.gz, crop.nii.gz, crop_coords.txt,
        teeth_mask.nii.gz, maxilla_mask.nii.gz
    DerivedData/segmentations/hu_fallback/sub-XXX/ses-YYY/
        teeth_mask.nii.gz, maxilla_mask.nii.gz
    DerivedData/segmentations/totalsegmentator_head_muscles/sub-XXX/ses-YYY/
        head_muscles_multilabel.nii.gz, tongue_mask.nii.gz
    DerivedData/transforms/sub-XXX_ses-YYY/
        ct_to_pet_fwd_0.mat                 (ANTs rigid transform)
        ct_to_pet_inv_0.mat                 (ANTs inverse transform)
        ct_to_pet_meta.json                 (registration metadata)
    QC/registration/
        sub-XXX_ses-YYY_ct_in_PETspace.nii.gz  (resliced CT for co-reg QC)
        sub-XXX_ses-YYY_pet_native.nii.gz      (copy of PET for overlay)
    DerivedData/rois/totalsegmentator_teeth/sub-XXX_ses-YYY/
        tooth_shells_geometry.nii.gz        (labeled, CT space)
        tooth_shells_geometry_PETspace.nii.gz (NN labels, PET space, QC)
        tooth_shells_continuous_PETspace.nii.gz (max-weight composite)
        continuous_masks_PETspace/
            tooth_NN_continuous.nii.gz      (per-tooth float [0,1] masks)
            peridental_upper_jaw_continuous.nii.gz (jaw-level, linear)
            peridental_lower_jaw_continuous.nii.gz (jaw-level, linear)
        peridental_upper_jaw.nii.gz         (binary, CT space)
        peridental_lower_jaw.nii.gz         (binary, CT space)
        tooth_shells_lookup.json

Usage:
    python 02_run_geometry_pipeline.py                           # All subjects, all steps
    python 02_run_geometry_pipeline.py --subjects sub-101 sub-102  # Specific subjects
    python 02_run_geometry_pipeline.py --n-subjects 6              # First N subjects
    python 02_run_geometry_pipeline.py --steps 5                   # Run only Step 5
    python 02_run_geometry_pipeline.py --steps coreg               # Same as --steps 5
    python 02_run_geometry_pipeline.py --steps 3 5                 # Run Steps 3 and 5
    python 02_run_geometry_pipeline.py --force                     # Re-run masks (not registration)
    python 02_run_geometry_pipeline.py --force-registration        # Re-run ANTs registration
    python 02_run_geometry_pipeline.py --skip-head-muscles         # Skip tongue segmentation
"""

import argparse
import gc
import json
import sys
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import nibabel as nib
import pandas as pd
from scipy.ndimage import binary_dilation, map_coordinates
from numpy.linalg import inv

# Add Scripts directory to path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

from config import (
    RAWDATA_DIR, SEGMENTATION_DIR, TOTALSEG_SEG_DIR, HU_SEG_DIR,
    ROI_DIR, TOTALSEG_ROI_DIR, ROI_QC_DIR, LOGNOTES_DIR, DERIVED_DIR,
    SEG_QC_DIR, TOTALSEG_QC_DIR, TRANSFORM_DIR, REG_QC_DIR,
    ensure_directories
)
from utils.io_utils import (
    load_blinding_key, discover_subjects, discover_sessions,
    find_ct_file, find_pet_file, load_nifti, save_nifti,
    get_voxel_dimensions, get_voxel_volume_ml
)
from utils.segmentation_utils import (
    run_total_segmentator_dental, segment_by_hu_threshold,
    detect_metal_artifacts, validate_segmentation
)

# Import ROI generation function from 02b
from importlib.util import spec_from_file_location, module_from_spec
_spec = spec_from_file_location("geometry_roi", script_dir / "02b_geometry_roi_poc.py")
_mod = module_from_spec(_spec)
_spec.loader.exec_module(_mod)
generate_geometry_rois = _mod.generate_geometry_rois
generate_qc_image = _mod.generate_qc_image
ALL_TOOTH_LABELS = _mod.ALL_TOOTH_LABELS
UPPER_JAWBONE_LABEL = _mod.UPPER_JAWBONE_LABEL
LOWER_JAWBONE_LABEL = _mod.LOWER_JAWBONE_LABEL

logger = logging.getLogger(__name__)

# Head muscles segmentation output directory
HEAD_MUSCLES_DIR = DERIVED_DIR / "segmentations" / "totalsegmentator_head_muscles"

# Default dilation
DILATION_MM = 4.0


# =============================================================================
# STEP 1: TotalSegmentator teeth
# =============================================================================

def run_step1_totalseg_teeth(subject_id, session_id, ct_file, ct_data, ct_img,
                              voxel_dims, force=False):
    """Run TotalSegmentator teeth task and save outputs."""
    ts_dir = TOTALSEG_SEG_DIR / subject_id / session_id
    multilabel_file = ts_dir / "totalseg_teeth_multilabel.nii.gz"

    if multilabel_file.exists() and not force:
        logger.info(f"  [Step 1] TotalSeg teeth: EXISTS, skipping")
        return True

    logger.info(f"  [Step 1] Running TotalSegmentator teeth task...")
    try:
        result = run_total_segmentator_dental(ct_data, ct_img, voxel_dims)
        if result is None:
            logger.error(f"  [Step 1] TotalSegmentator returned None")
            return False

        ts_dir.mkdir(parents=True, exist_ok=True)

        # Save teeth and maxilla masks
        save_nifti(result['teeth_mask'].astype(np.uint8), ct_img, ts_dir / "teeth_mask.nii.gz")
        save_nifti(result['maxilla_mask'].astype(np.uint8), ct_img, ts_dir / "maxilla_mask.nii.gz")

        # Save tooth instances
        if result.get('tooth_instances') is not None:
            save_nifti(result['tooth_instances'].astype(np.int16), ct_img,
                      ts_dir / "tooth_instances.nii.gz")

        # Save multilabel in FULL CT space
        if result.get('full_seg_data') is not None:
            save_nifti(result['full_seg_data'].astype(np.int16), ct_img, multilabel_file)

        # Save crop
        if result.get('cropped_img') is not None:
            nib.save(result['cropped_img'], str(ts_dir / "crop.nii.gz"))

        # Save crop coordinates
        if result.get('crop_slices') is not None:
            with open(ts_dir / "crop_coords.txt", 'w') as f:
                for dim, s in zip(['x', 'y', 'z'], result['crop_slices']):
                    f.write(f"{dim}: {s.start}-{s.stop}\n")
                f.write(f"original_shape: {ct_data.shape}\n")

        # Save method info
        qc = validate_segmentation(result['teeth_mask'], result['maxilla_mask'],
                                    get_voxel_volume_ml(ct_img))
        with open(ts_dir / "segmentation_method.txt", 'w') as f:
            f.write(f"Method: TotalSegmentator\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"CT file: {ct_file.name}\n")
            for k, v in qc.items():
                f.write(f"{k}: {v}\n")

        logger.info(f"  [Step 1] TotalSeg teeth: DONE "
                    f"(teeth={qc.get('teeth_volume_ml',0):.1f}mL, "
                    f"maxilla={qc.get('maxilla_volume_ml',0):.1f}mL)")
        gc.collect()
        return True

    except Exception as e:
        logger.error(f"  [Step 1] TotalSegmentator FAILED: {e}")
        return False


# =============================================================================
# STEP 2: HU-fallback segmentation
# =============================================================================

def run_step2_hu_fallback(subject_id, session_id, ct_file, ct_data, ct_img,
                           voxel_dims, force=False):
    """Run HU-threshold segmentation."""
    hu_dir = HU_SEG_DIR / subject_id / session_id
    teeth_file = hu_dir / "teeth_mask.nii.gz"

    if teeth_file.exists() and not force:
        logger.info(f"  [Step 2] HU fallback: EXISTS, skipping")
        return True

    logger.info(f"  [Step 2] Running HU-fallback segmentation...")
    try:
        hu_teeth, hu_maxilla = segment_by_hu_threshold(ct_data, voxel_dims)
        hu_dir.mkdir(parents=True, exist_ok=True)

        save_nifti(hu_teeth.astype(np.uint8), ct_img, teeth_file)
        save_nifti(hu_maxilla.astype(np.uint8), ct_img, hu_dir / "maxilla_mask.nii.gz")

        # Metal artifacts
        metal_mask = detect_metal_artifacts(ct_data, voxel_dims)
        if np.any(metal_mask):
            save_nifti(metal_mask.astype(np.uint8), ct_img, hu_dir / "metal_artifact_mask.nii.gz")

        qc = validate_segmentation(hu_teeth, hu_maxilla, get_voxel_volume_ml(ct_img))
        with open(hu_dir / "segmentation_method.txt", 'w') as f:
            f.write(f"Method: HU_threshold\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"CT file: {ct_file.name}\n")
            for k, v in qc.items():
                f.write(f"{k}: {v}\n")

        logger.info(f"  [Step 2] HU fallback: DONE "
                    f"(teeth={qc.get('teeth_volume_ml',0):.1f}mL, "
                    f"maxilla={qc.get('maxilla_volume_ml',0):.1f}mL)")
        return True

    except Exception as e:
        logger.error(f"  [Step 2] HU fallback FAILED: {e}")
        return False


# =============================================================================
# STEP 3: Geometry ROI generation
# =============================================================================

def run_step3_geometry_rois(subject_id, session_id, dilation_mm=DILATION_MM, force=False, ct_file=None):
    """Generate per-tooth geometry ROIs from TotalSeg multilabel."""
    roi_dir = TOTALSEG_ROI_DIR / f"{subject_id}_{session_id}"
    labeled_file = roi_dir / "tooth_shells_geometry.nii.gz"

    if labeled_file.exists() and not force:
        logger.info(f"  [Step 3] Geometry ROIs: EXISTS, skipping")
        return True

    # Load TotalSeg multilabel
    seg_dir = TOTALSEG_SEG_DIR / subject_id / session_id
    seg_file = seg_dir / "totalseg_teeth_multilabel.nii.gz"
    if not seg_file.exists():
        logger.error(f"  [Step 3] No TotalSeg multilabel: {seg_file}")
        return False

    logger.info(f"  [Step 3] Generating geometry ROIs (dilation={dilation_mm}mm)...")
    seg_data, seg_img = load_nifti(seg_file)
    seg_data = seg_data.astype(np.int32)

    # Load CT for HU-validation of prosthetic labels
    # Use crop CT if multilabel is in crop space, otherwise full CT
    ct_data_for_hu = None
    if ct_file is not None and ct_file.exists():
        crop_file = seg_dir / "crop.nii.gz"
        if crop_file.exists():
            crop_data, _ = load_nifti(crop_file)
            if crop_data.shape == seg_data.shape:
                ct_data_for_hu = crop_data
                logger.info(f"  Using crop CT for HU validation (shapes match: {seg_data.shape})")
            else:
                ct_data_for_hu, _ = load_nifti(ct_file)
        else:
            ct_data_for_hu, _ = load_nifti(ct_file)

    result = generate_geometry_rois(seg_data, seg_img, dilation_mm=dilation_mm,
                                    ct_data=ct_data_for_hu)
    del ct_data_for_hu
    qc = result['qc_metrics']

    # Save outputs
    roi_dir.mkdir(parents=True, exist_ok=True)

    save_nifti(result['labeled_volume'], seg_img, labeled_file)
    save_nifti(result['upper_jaw_roi'].astype(np.uint8), seg_img,
               roi_dir / "peridental_upper_jaw.nii.gz")
    save_nifti(result['lower_jaw_roi'].astype(np.uint8), seg_img,
               roi_dir / "peridental_lower_jaw.nii.gz")

    # Lookup JSON
    lookup = {
        'tooth_ids': sorted([int(f) for f in qc['tooth_volumes_ml'].keys()]),
        'n_teeth': qc['n_teeth_with_roi'],
        'excluded_teeth': {str(k): v for k, v in qc['excluded_teeth'].items()},
        'dilation_mm': dilation_mm,
        'method': 'geometry_only',
        'tooth_volumes_ml': {str(k): round(v, 4) for k, v in qc['tooth_volumes_ml'].items()},
    }
    with open(roi_dir / "tooth_shells_lookup.json", 'w') as f:
        json.dump(lookup, f, indent=2)

    # QC visualization — use crop coordinates to extract matching sub-volumes
    crop_file = seg_dir / "crop.nii.gz"
    coords_file = seg_dir / "crop_coords.txt"
    if crop_file.exists() and coords_file.exists():
        try:
            ct_crop, _ = load_nifti(crop_file)
            # Parse crop coordinates to slice the labeled volume and masks
            crop_slices = []
            with open(coords_file) as f:
                for line in f:
                    if line.startswith(('x:', 'y:', 'z:')):
                        rng = line.split(':')[1].strip()
                        start, stop = rng.split('-')
                        crop_slices.append(slice(int(start), int(stop)))
            if len(crop_slices) == 3:
                sl = tuple(crop_slices)
                labeled_crop = result['labeled_volume'][sl]
                bone_crop = ((seg_data == UPPER_JAWBONE_LABEL) | (seg_data == LOWER_JAWBONE_LABEL))[sl]
                teeth_crop = np.isin(seg_data[sl], ALL_TOOTH_LABELS)
                qc_dir = ROI_QC_DIR / subject_id
                generate_qc_image(ct_crop, labeled_crop, bone_crop, teeth_crop,
                                  qc_dir / f"{session_id}_geometry_roi.png",
                                  f"{subject_id}/{session_id} — Geometry ROIs ({dilation_mm}mm)")
        except Exception as e:
            logger.warning(f"  QC image failed: {e}")

    # Per-tooth volume CSV
    vol_records = []
    for fdi, vol in sorted(qc['tooth_volumes_ml'].items()):
        vol_records.append({
            'subject_id': subject_id, 'session_id': session_id,
            'fdi_tooth': fdi, 'jaw': 'upper' if 11 <= fdi <= 28 else 'lower',
            'roi_volume_ml': round(vol, 4),
            'excluded': fdi in qc['excluded_teeth'],
            'exclusion_reason': qc['excluded_teeth'].get(fdi, ''),
        })
    pd.DataFrame(vol_records).to_csv(
        ROI_QC_DIR / subject_id / f"{session_id}_roi_volumes.csv", index=False)

    logger.info(f"  [Step 3] Geometry ROIs: DONE "
                f"({qc['n_teeth_with_roi']} teeth, "
                f"upper={qc['upper_jaw_volume_ml']:.1f}mL, "
                f"lower={qc['lower_jaw_volume_ml']:.1f}mL)")
    gc.collect()
    return True


# =============================================================================
# STEP 4: TotalSegmentator head_muscles (tongue mask)
# =============================================================================

def run_step4_head_muscles(subject_id, session_id, ct_file, force=False):
    """Run TotalSegmentator head_muscles task for tongue mask."""
    out_dir = HEAD_MUSCLES_DIR / subject_id / session_id
    tongue_file = out_dir / "tongue_mask.nii.gz"

    if tongue_file.exists() and not force:
        logger.info(f"  [Step 4] Head muscles (tongue): EXISTS, skipping")
        return True

    logger.info(f"  [Step 4] Running TotalSegmentator head_muscles task...")
    try:
        from totalsegmentator.python_api import totalsegmentator

        out_dir.mkdir(parents=True, exist_ok=True)

        # Run on full CT (head_muscles does its own cropping internally)
        # ml=True produces a single multilabel file
        # The API saves it as a .nii file at the parent level
        result_img = totalsegmentator(
            input=ct_file,
            output=out_dir,
            task='head_muscles',
            ml=True,
            quiet=True,
            device='cpu'
        )

        # Find the output file — may be saved as ses-XXX.nii at parent level
        # due to ml=True behavior
        ml_file_parent = HEAD_MUSCLES_DIR / subject_id / f"{session_id}.nii"
        ml_file_local = out_dir / "head_muscles.nii.gz"

        if ml_file_parent.exists():
            img = nib.load(str(ml_file_parent))
        elif ml_file_local.exists():
            img = nib.load(str(ml_file_local))
        elif result_img is not None:
            img = result_img
        else:
            # Check for any .nii or .nii.gz in out_dir
            nii_files = list(out_dir.glob("*.nii*"))
            if nii_files:
                img = nib.load(str(nii_files[0]))
            else:
                logger.error(f"  [Step 4] No output file found after head_muscles run")
                return False

        data = img.get_fdata().astype(np.int32)

        # Save multilabel
        nib.save(nib.Nifti1Image(data, img.affine, img.header),
                 str(out_dir / "head_muscles_multilabel.nii.gz"))

        # Extract tongue (label 9)
        tongue = (data == 9).astype(np.uint8)
        n_voxels = int(tongue.sum())
        vox_vol = float(np.prod(img.header.get_zooms())) / 1000
        tongue_vol = n_voxels * vox_vol

        nib.save(nib.Nifti1Image(tongue, img.affine, img.header), str(tongue_file))

        # Clean up misplaced file
        if ml_file_parent.exists():
            ml_file_parent.unlink()

        logger.info(f"  [Step 4] Head muscles: DONE (tongue={tongue_vol:.1f}mL, {n_voxels} voxels)")
        gc.collect()
        return True

    except Exception as e:
        logger.error(f"  [Step 4] Head muscles FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# STEP 5: Rigid CT→PET co-registration (ANTsPy) + resample ROIs to PET space
# =============================================================================

def run_step5_coreg_and_resample(subject_id, session_id, ct_file=None, force=False, force_registration=False):
    """
    Step 5: Rigid CT→PET co-registration using ANTsPy and mask resampling.

    Uses ANTsPy with CT cropped to head/neck region (matching PET FOV ± 50mm)
    for robust optimization. This approach works reliably across all subjects.

    Args:
        force: If True, regenerate PET-space masks even if they exist.
               Does NOT re-run ANTs registration if transforms already exist.
        force_registration: If True, force re-run of ANTs registration.

    Steps:
    1. Crop CT to PET FOV ± margin (removes torso that confuses registration)
    2. Compute rigid body transform (CT→PET) using ANTsPy
    3. Save transform to DerivedData/transforms/
    4. Reslice full CT into PET space for QC
    5. Resample ROI masks with LINEAR interpolation → continuous [0,1] masks
       (preserves partial-volume weighting in PET native space)
    6. Save nearest-neighbor labeled volume for QC overlay

    Output layout:
        DerivedData/transforms/sub-XXX_ses-YYY/
            ct_to_pet_fwd_0.mat               (ANTs rigid transform)
            ct_to_pet_inv_0.mat               (ANTs inverse transform)
            ct_to_pet_meta.json               (registration metadata)
        QC/registration/
            sub-XXX_ses-YYY_ct_in_PETspace.nii.gz   (resliced CT for QC)
            sub-XXX_ses-YYY_pet_native.nii.gz       (copy of PET for overlay)
        DerivedData/rois/totalsegmentator_teeth/sub-XXX_ses-YYY/
            tooth_shells_geometry_PETspace.nii.gz           (NN labels, QC)
            tooth_shells_continuous_PETspace.nii.gz         (max-weight composite)
            continuous_masks_PETspace/
                tooth_NN_continuous.nii.gz                  (per-tooth float [0,1])
                peridental_upper_jaw_continuous.nii.gz      (jaw-level, linear)
                peridental_lower_jaw_continuous.nii.gz      (jaw-level, linear)
    """
    from utils.registration_utils import (
        rigid_register_ct_to_pet, save_transform, load_transform,
        reslice_ct_to_pet, reslice_ct_to_pet_from_files,
        resample_mask_to_pet, resample_labels_to_pet,
    )

    roi_dir = TOTALSEG_ROI_DIR / f"{subject_id}_{session_id}"
    transform_dir = TRANSFORM_DIR / f"{subject_id}_{session_id}"
    meta_file = transform_dir / "ct_to_pet_meta.json"

    # Check if already done
    continuous_file = roi_dir / "tooth_shells_continuous_PETspace.nii.gz"
    if continuous_file.exists() and not force:
        logger.info(f"  [Step 5] Co-registered PET-space ROIs: EXISTS, skipping")
        return True

    # Load ROI labeled volume
    labeled_file = roi_dir / "tooth_shells_geometry.nii.gz"
    if not labeled_file.exists():
        logger.error(f"  [Step 5] No geometry ROI found")
        return False

    labeled_data, labeled_img = load_nifti(labeled_file)

    # Find PET
    session_dir = RAWDATA_DIR / subject_id / session_id
    pet_file = find_pet_file(session_dir)
    if pet_file is None:
        logger.error(f"  [Step 5] No PET file found")
        return False

    pet_data, pet_img = load_nifti(pet_file)
    logger.info(f"  [Step 5] PET: {pet_file.name} shape={pet_img.shape[:3]}")

    # Find CT for registration (use the one passed in, or find it)
    if ct_file is None:
        ct_file = find_ct_file(session_dir, prefer_bone=False)
    if ct_file is None:
        logger.error(f"  [Step 5] No CT file found for registration")
        return False

    ct_data, ct_img = load_nifti(ct_file)
    logger.info(f"  [Step 5] CT: {ct_file.name} shape={ct_img.shape[:3]}")

    # --- 5a: Rigid registration using ANTsPy (or load existing transform) ---
    # Only re-run registration if force_registration is True OR transforms don't exist
    transform_files_exist = list(transform_dir.glob("ct_to_pet_fwd_*.mat"))
    if transform_files_exist and not force_registration:
        logger.info(f"  [Step 5a] Loading existing ANTs transforms from: {transform_dir}")
        transform_files = load_transform(transform_dir, prefix="ct_to_pet")
        # Load metadata
        if meta_file.exists():
            with open(meta_file) as f:
                reg_meta = json.load(f)
            trans_mag = reg_meta.get('translation_magnitude_mm', '?')
            if isinstance(trans_mag, (int, float)):
                logger.info(f"  [Step 5a] Previous registration: translation={trans_mag:.2f}mm")
        reg_result = None  # We'll use transform_files directly
    else:
        if force_registration and transform_files_exist:
            logger.info(f"  [Step 5a] Force re-running ANTs registration (--force-registration)")
        logger.info(f"  [Step 5a] Computing rigid CT→PET registration (ANTsPy, cropped CT)...")
        reg_result, reg_meta = rigid_register_ct_to_pet(ct_img, pet_img, crop_ct=True)

        # Save transform and metadata
        transform_dir.mkdir(parents=True, exist_ok=True)
        save_transform(reg_result, transform_dir, prefix="ct_to_pet")
        with open(meta_file, 'w') as f:
            json.dump(reg_meta, f, indent=2)
        logger.info(f"  [Step 5a] Transforms saved to: {transform_dir}")

        # Get transform files for subsequent operations
        transform_files = reg_result['fwdtransforms']

    # --- 5b: Reslice CT into PET space for QC ---
    REG_QC_DIR.mkdir(parents=True, exist_ok=True)
    ct_pet_qc = REG_QC_DIR / f"{subject_id}_{session_id}_ct_in_PETspace.nii.gz"
    pet_native_qc = REG_QC_DIR / f"{subject_id}_{session_id}_pet_native.nii.gz"

    if not ct_pet_qc.exists() or force:
        logger.info(f"  [Step 5b] Reslicing CT into PET space for QC...")
        if reg_result is not None:
            ct_in_pet = reslice_ct_to_pet(ct_img, pet_img, reg_result)
        else:
            ct_in_pet = reslice_ct_to_pet_from_files(ct_img, pet_img, transform_files)
        nib.save(ct_in_pet, str(ct_pet_qc))
        logger.info(f"  [Step 5b] Saved: {ct_pet_qc.name}")
    else:
        logger.info(f"  [Step 5b] CT in PET space: EXISTS, skipping")

    # Save PET native for overlay comparison
    if not pet_native_qc.exists() or force:
        nib.save(pet_img, str(pet_native_qc))

    # Free full CT data
    del ct_data
    gc.collect()

    # --- 5c: Resample labeled volume → PET space ---
    logger.info(f"  [Step 5c] Resampling tooth shells to PET space (NN + per-tooth linear)...")
    nn_labels_nib, continuous_dict = resample_labels_to_pet(labeled_img, pet_img, transform_files)

    # Save nearest-neighbor labeled (for QC overlay)
    nn_file = roi_dir / "tooth_shells_geometry_PETspace.nii.gz"
    nib.save(nn_labels_nib, str(nn_file))

    # Save per-tooth continuous masks in subdirectory
    cont_dir = roi_dir / "continuous_masks_PETspace"
    cont_dir.mkdir(parents=True, exist_ok=True)
    for fdi, cont_nib in continuous_dict.items():
        cont_file = cont_dir / f"tooth_{fdi:02d}_continuous.nii.gz"
        nib.save(cont_nib, str(cont_file))
    logger.info(f"  [Step 5c] Saved {len(continuous_dict)} per-tooth continuous masks")

    # Also save a summary continuous file (max-weight composite) for quick inspection
    if continuous_dict:
        pet_shape = pet_img.shape[:3]
        composite = np.zeros(pet_shape, dtype=np.float32)
        for fdi, cont_nib in continuous_dict.items():
            w = cont_nib.get_fdata().astype(np.float32)
            composite = np.maximum(composite, w)
        composite_img = nib.Nifti1Image(composite, pet_img.affine)
        nib.save(composite_img, str(continuous_file))

    # --- 5d: Resample jaw masks → PET space (linear) ---
    # Save inside continuous_masks_PETspace/ for consistency with per-tooth masks
    for jaw_name in ["upper_jaw", "lower_jaw"]:
        jaw_file = roi_dir / f"peridental_{jaw_name}.nii.gz"
        if jaw_file.exists():
            jaw_data, jaw_img = load_nifti(jaw_file)
            jaw_pet_nib = resample_mask_to_pet(jaw_img, pet_img, transform_files, is_binary=True)
            out_file = cont_dir / f"peridental_{jaw_name}_continuous.nii.gz"
            nib.save(jaw_pet_nib, str(out_file))
            logger.info(f"  [Step 5d] Saved: {out_file.name}")

    logger.info(f"  [Step 5] Co-registration and PET-space resampling: DONE")
    gc.collect()
    return True


# =============================================================================
# MAIN
# =============================================================================

def parse_steps(step_args):
    """Parse --steps argument into set of step numbers."""
    if step_args is None:
        return {1, 2, 3, 4, 5}  # All steps

    step_map = {
        '1': 1, 'teeth': 1, 'totalseg': 1,
        '2': 2, 'hu': 2, 'hu-fallback': 2,
        '3': 3, 'roi': 3, 'geometry': 3,
        '4': 4, 'tongue': 4, 'head-muscles': 4,
        '5': 5, 'coreg': 5, 'registration': 5,
    }

    steps = set()
    for arg in step_args:
        arg_lower = arg.lower()
        if arg_lower in step_map:
            steps.add(step_map[arg_lower])
        else:
            raise ValueError(f"Unknown step: {arg}. Valid: 1-5, teeth, hu, roi, tongue, coreg")
    return steps


def main():
    parser = argparse.ArgumentParser(description='Batch Geometry ROI Pipeline')
    parser.add_argument('--subjects', nargs='+', type=str, help='Specific subjects to process')
    parser.add_argument('--n-subjects', type=int, help='Process first N subjects only')
    parser.add_argument('--dilation', type=float, default=DILATION_MM,
                        help=f'Dilation distance in mm (default: {DILATION_MM})')
    parser.add_argument('--force', action='store_true',
                        help='Re-run mask generation even if output exists (does NOT re-run ANTs registration)')
    parser.add_argument('--force-registration', action='store_true',
                        help='Force re-run of ANTs CT→PET registration (use with caution)')
    parser.add_argument('--skip-head-muscles', action='store_true',
                        help='Skip tongue segmentation (Step 4)')
    parser.add_argument('--include-tongue-exclusion', action='store_true',
                        help='Also run tongue exclusion (Step 6) after co-registration')
    parser.add_argument('--steps', nargs='+', type=str, default=None,
                        help='Run specific steps only (1-5 or: teeth, hu, roi, tongue, coreg)')
    args = parser.parse_args()

    # Parse steps
    try:
        steps_to_run = parse_steps(args.steps)
    except ValueError as e:
        parser.error(str(e))

    ensure_directories()

    # Logging
    log_dir = LOGNOTES_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"geometry_pipeline_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    logger.info("=" * 70)
    logger.info("GEOMETRY ROI PIPELINE — BATCH RUNNER")
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Dilation: {args.dilation} mm")
    logger.info(f"Steps: {sorted(steps_to_run)}")
    if args.force:
        logger.info("Force mode: ON (will regenerate masks)")
    if args.force_registration:
        logger.info("Force registration: ON (will re-run ANTs)")
    if args.include_tongue_exclusion:
        logger.info("Tongue exclusion: ON (will run Step 6)")
    logger.info(f"Log: {log_file}")
    logger.info("=" * 70)

    # Discover subjects
    blinding_map = load_blinding_key()
    all_subjects = sorted(discover_subjects())

    if args.subjects:
        subjects = [s for s in args.subjects if s in all_subjects]
    elif args.n_subjects:
        subjects = all_subjects[:args.n_subjects]
    else:
        subjects = all_subjects

    logger.info(f"Subjects to process: {len(subjects)} → {', '.join(subjects)}")

    # Track results
    results = []
    total_sessions = 0
    success_count = 0
    skip_count = 0
    fail_count = 0

    for subj_idx, subject_id in enumerate(subjects):
        subject_dir = RAWDATA_DIR / subject_id
        sessions = discover_sessions(subject_dir)

        logger.info(f"\n{'='*60}")
        logger.info(f"[{subj_idx+1}/{len(subjects)}] {subject_id} — {len(sessions)} sessions")
        logger.info(f"{'='*60}")

        for session_id in sessions:
            total_sessions += 1
            session_dir = subject_dir / session_id

            # Get timepoint
            key = (subject_id, session_id)
            timepoint = blinding_map.get(key, "Unknown")
            if timepoint == "Unknown":
                logger.warning(f"  {session_id}: Not in blinding key, skipping")
                fail_count += 1
                continue

            logger.info(f"\n--- {subject_id} / {session_id} ({timepoint}) ---")

            # Find CT
            ct_file = find_ct_file(session_dir, prefer_bone=False)
            if ct_file is None:
                logger.error(f"  No CT file found, skipping")
                fail_count += 1
                results.append({
                    'subject_id': subject_id, 'session_id': session_id,
                    'timepoint': timepoint, 'status': 'no_ct'
                })
                continue

            # Load CT once for steps 1-2
            logger.info(f"  Loading CT: {ct_file.name}")
            ct_data, ct_img = load_nifti(ct_file)
            voxel_dims = get_voxel_dimensions(ct_img)

            # Step 1: TotalSeg teeth
            step1_ok = True
            if 1 in steps_to_run:
                step1_ok = run_step1_totalseg_teeth(
                    subject_id, session_id, ct_file, ct_data, ct_img,
                    voxel_dims, force=args.force)
            else:
                logger.info(f"  [Step 1] Skipped (not in --steps)")

            # Step 2: HU fallback
            step2_ok = True
            if 2 in steps_to_run:
                step2_ok = run_step2_hu_fallback(
                    subject_id, session_id, ct_file, ct_data, ct_img,
                    voxel_dims, force=args.force)
            else:
                logger.info(f"  [Step 2] Skipped (not in --steps)")

            # Free CT memory before ROI generation
            del ct_data
            gc.collect()

            # Check Step 1 result before proceeding to Step 3
            if 3 in steps_to_run and not step1_ok:
                logger.error(f"  CRITICAL: TotalSeg teeth failed, cannot generate geometry ROIs")
                fail_count += 1
                results.append({
                    'subject_id': subject_id, 'session_id': session_id,
                    'timepoint': timepoint, 'status': 'totalseg_failed'
                })
                continue

            # Step 3: Geometry ROIs
            step3_ok = True
            if 3 in steps_to_run:
                step3_ok = run_step3_geometry_rois(
                    subject_id, session_id, dilation_mm=args.dilation, force=args.force,
                    ct_file=ct_file)

                if not step3_ok:
                    logger.error(f"  Geometry ROI generation failed")
                    fail_count += 1
                    results.append({
                        'subject_id': subject_id, 'session_id': session_id,
                        'timepoint': timepoint, 'status': 'roi_failed'
                    })
                    continue
            else:
                logger.info(f"  [Step 3] Skipped (not in --steps)")

            # Step 4: Head muscles (tongue)
            step4_ok = True
            if 4 in steps_to_run:
                if not args.skip_head_muscles:
                    step4_ok = run_step4_head_muscles(
                        subject_id, session_id, ct_file, force=args.force)
                    if not step4_ok:
                        logger.warning(f"  Head muscles failed (non-fatal, continuing)")
                else:
                    logger.info(f"  [Step 4] Skipped (--skip-head-muscles)")
            else:
                logger.info(f"  [Step 4] Skipped (not in --steps)")

            # Step 5: CT→PET co-registration + resample masks
            step5_ok = True
            if 5 in steps_to_run:
                step5_ok = run_step5_coreg_and_resample(
                    subject_id, session_id, ct_file=ct_file,
                    force=args.force, force_registration=args.force_registration)
            else:
                logger.info(f"  [Step 5] Skipped (not in --steps)")

            # Step 6: Tongue exclusion (optional, via --include-tongue-exclusion)
            step6_ok = True
            if args.include_tongue_exclusion and step5_ok:
                logger.info(f"  [Step 6] Running tongue exclusion...")
                try:
                    _te_spec = spec_from_file_location("tongue_exclusion",
                                                        script_dir / "03_create_tongue_exclusion.py")
                    _te_mod = module_from_spec(_te_spec)
                    _te_spec.loader.exec_module(_te_mod)
                    step6_ok = _te_mod.process_session(subject_id, session_id, force=args.force)
                    if step6_ok:
                        logger.info(f"  [Step 6] Tongue exclusion: DONE")
                    else:
                        logger.warning(f"  [Step 6] Tongue exclusion: FAILED (non-fatal)")
                except Exception as e:
                    logger.warning(f"  [Step 6] Tongue exclusion error: {e}")
                    step6_ok = False

            if step5_ok:
                success_count += 1
                status = 'success'
            else:
                fail_count += 1
                status = 'resample_failed'

            results.append({
                'subject_id': subject_id, 'session_id': session_id,
                'timepoint': timepoint, 'status': status,
                'step1_totalseg': step1_ok, 'step2_hu': step2_ok,
                'step3_roi': step3_ok, 'step4_tongue': step4_ok,
                'step5_pet_resample': step5_ok,
                'step6_tongue_exclusion': step6_ok if args.include_tongue_exclusion else None,
            })

            logger.info(f"  ✓ Session complete: {status}")

    # Save results summary
    results_df = pd.DataFrame(results)
    results_file = LOGNOTES_DIR / f"geometry_pipeline_results_{timestamp}.csv"
    results_df.to_csv(results_file, index=False)

    # Print summary
    logger.info(f"\n{'='*70}")
    logger.info("PIPELINE SUMMARY")
    logger.info(f"{'='*70}")
    logger.info(f"Total sessions: {total_sessions}")
    logger.info(f"  Success: {success_count}")
    logger.info(f"  Failed:  {fail_count}")
    logger.info(f"Results: {results_file}")
    logger.info(f"Log: {log_file}")

    if fail_count > 0:
        failed = [r for r in results if r['status'] != 'success']
        logger.info(f"\nFailed sessions:")
        for r in failed:
            logger.info(f"  {r['subject_id']}/{r['session_id']}: {r['status']}")

    logger.info(f"\nCompleted at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
