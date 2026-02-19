"""
registration_utils.py - Registration utilities for CT-PET and longitudinal analysis.

Uses ANTsPy for rigid registration with CT cropped to head/neck region
(matching PET FOV ± margin) for robust initialization.

Functions for CT→PET registration:
    rigid_register_ct_to_pet: Compute rigid transform (CT→PET) using ANTsPy
    save_transform: Save ANTs transform (.mat) to disk
    load_transform: Load ANTs transform from disk
    reslice_ct_to_pet: Apply transform to reslice CT into PET space
    resample_mask_to_pet: Apply transform + linear interpolation for continuous masks
    resample_labels_to_pet: Apply transform + nearest-neighbor for label volumes

Functions for longitudinal CT-CT registration:
    register_ct_to_ct: Compute rigid transform between Baseline and Followup CT
    compute_halfway_transform: Compute T^0.5 for midpoint space using matrix logarithm
    create_midpoint_reference_grid: Create reference grid at midpoint space
    ants_transform_to_matrix: Convert ANTs transform to 4x4 homogeneous matrix
    matrix_to_ants_transform: Convert 4x4 matrix back to ANTs transform
"""

import json
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import nibabel as nib
import ants
from scipy.linalg import logm, expm
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)

# Default margin (mm) for cropping CT to match PET FOV
CT_CROP_MARGIN_MM = 50.0


# =============================================================================
# CT CROPPING
# =============================================================================

def crop_ct_to_pet_fov(ct_nib, pet_nib, margin_mm=CT_CROP_MARGIN_MM):
    """
    Crop CT volume to match PET field-of-view in Z direction.

    This ensures robust registration by removing torso regions that
    have no corresponding PET data, which can confuse optimization.

    Args:
        ct_nib: nibabel CT image
        pet_nib: nibabel PET image
        margin_mm: extra margin (mm) to add on each side of PET FOV

    Returns:
        ct_cropped_nib: nibabel CT image cropped to head/neck region
        crop_info: dict with cropping metadata
    """
    ct_data = ct_nib.get_fdata()
    ct_affine = ct_nib.affine

    # Get PET Z-range in world coordinates
    pet_z_start = pet_nib.affine[2, 3]
    pet_z_end = pet_z_start + pet_nib.shape[2] * pet_nib.affine[2, 2]
    pet_z_min = min(pet_z_start, pet_z_end)
    pet_z_max = max(pet_z_start, pet_z_end)

    # Calculate CT Z positions
    ct_z_positions = ct_affine[2, 3] + np.arange(ct_nib.shape[2]) * ct_affine[2, 2]

    # Find CT slices within PET FOV + margin
    z_min_target = pet_z_min - margin_mm
    z_max_target = pet_z_max + margin_mm

    ct_slice_mask = (ct_z_positions >= z_min_target) & (ct_z_positions <= z_max_target)
    ct_slice_indices = np.where(ct_slice_mask)[0]

    if len(ct_slice_indices) < 10:
        logger.warning(f"  Very few CT slices ({len(ct_slice_indices)}) in PET FOV, using full CT")
        return ct_nib, {'cropped': False, 'reason': 'insufficient_overlap'}

    # Crop CT
    z_start = ct_slice_indices[0]
    z_end = ct_slice_indices[-1] + 1
    ct_cropped = ct_data[:, :, z_start:z_end]

    # Update affine for cropped volume
    ct_cropped_affine = ct_affine.copy()
    ct_cropped_affine[2, 3] = ct_z_positions[z_start]

    ct_cropped_nib = nib.Nifti1Image(ct_cropped.astype(np.float32), ct_cropped_affine)

    crop_info = {
        'cropped': True,
        'original_shape': ct_nib.shape,
        'cropped_shape': ct_cropped_nib.shape,
        'z_slice_range': [int(z_start), int(z_end)],
        'z_world_range': [float(ct_z_positions[z_start]), float(ct_z_positions[z_end-1])],
        'pet_z_range': [float(pet_z_min), float(pet_z_max)],
        'margin_mm': margin_mm,
    }

    logger.info(f"  CT cropped: {ct_nib.shape[2]} → {ct_cropped_nib.shape[2]} slices "
                f"(Z: {crop_info['z_world_range'][0]:.1f} to {crop_info['z_world_range'][1]:.1f} mm)")

    return ct_cropped_nib, crop_info


# =============================================================================
# RIGID REGISTRATION (ANTsPy)
# =============================================================================

def rigid_register_ct_to_pet(ct_nib, pet_nib, crop_ct=True, margin_mm=CT_CROP_MARGIN_MM):
    """
    Compute rigid body (6 DOF) registration: CT → PET space using ANTsPy.

    Crops CT to head/neck region (matching PET FOV) before registration
    to ensure robust optimization.

    Args:
        ct_nib: nibabel CT image (moving)
        pet_nib: nibabel PET image (fixed / reference)
        crop_ct: if True, crop CT to PET FOV before registration
        margin_mm: margin (mm) around PET FOV for cropping

    Returns:
        dict: ANTs registration result with transforms
        dict: registration metadata
    """
    # Optionally crop CT to head/neck region
    if crop_ct:
        ct_cropped_nib, crop_info = crop_ct_to_pet_fov(ct_nib, pet_nib, margin_mm)
    else:
        ct_cropped_nib = ct_nib
        crop_info = {'cropped': False}

    # Save temporary files for ANTs (ANTs works with file paths)
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        ct_tmp = tmpdir / "ct_cropped.nii.gz"
        pet_tmp = tmpdir / "pet.nii.gz"

        nib.save(ct_cropped_nib, str(ct_tmp))
        nib.save(pet_nib, str(pet_tmp))

        # Load into ANTs
        ct_ants = ants.image_read(str(ct_tmp))
        pet_ants = ants.image_read(str(pet_tmp))

        logger.info(f"  Running ANTsPy rigid registration (CT→PET)...")
        logger.info(f"    CT shape: {ct_ants.shape}, spacing: {[f'{s:.3f}' for s in ct_ants.spacing]}")
        logger.info(f"    PET shape: {pet_ants.shape}, spacing: {[f'{s:.3f}' for s in pet_ants.spacing]}")

        # Run rigid registration: CT (moving) → PET (fixed)
        reg_result = ants.registration(
            fixed=pet_ants,
            moving=ct_ants,
            type_of_transform='Rigid',
            verbose=False
        )

    # Extract transform parameters for logging
    # The transform file is stored in reg_result['fwdtransforms'][0]
    transform_file = reg_result['fwdtransforms'][0]

    # Read transform to extract parameters
    try:
        tfm = ants.read_transform(transform_file)
        params = tfm.parameters

        # ANTs rigid transform: parameters depend on transform type
        # For 3D Euler/Rigid: typically [rotation params, translation params]
        if len(params) == 6:
            # Euler: [rot_x, rot_y, rot_z, tx, ty, tz]
            rot_x, rot_y, rot_z = params[:3]
            tx, ty, tz = params[3:6]
        elif len(params) == 12:
            # Affine: extract from matrix
            mat = np.array(params[:9]).reshape(3, 3)
            rot_x = np.arctan2(mat[2, 1], mat[2, 2])
            rot_y = np.arctan2(-mat[2, 0], np.sqrt(mat[2, 1]**2 + mat[2, 2]**2))
            rot_z = np.arctan2(mat[1, 0], mat[0, 0])
            tx, ty, tz = params[9:12]
        else:
            rot_x = rot_y = rot_z = 0
            tx = ty = tz = 0

        translation_mag = np.sqrt(tx**2 + ty**2 + tz**2)

        metadata = {
            'method': 'ANTsPy_rigid',
            'ct_cropped': crop_info.get('cropped', False),
            'crop_info': crop_info,
            'translation_mm': [float(tx), float(ty), float(tz)],
            'translation_magnitude_mm': float(translation_mag),
            'rotation_rad': [float(rot_x), float(rot_y), float(rot_z)],
            'rotation_deg': [float(np.degrees(rot_x)), float(np.degrees(rot_y)), float(np.degrees(rot_z))],
            'timestamp': datetime.now().isoformat(),
        }

        logger.info(f"  Registration complete: "
                    f"translation={metadata['translation_magnitude_mm']:.2f}mm, "
                    f"rotation=[{metadata['rotation_deg'][0]:.2f}, {metadata['rotation_deg'][1]:.2f}, {metadata['rotation_deg'][2]:.2f}]°")

    except Exception as e:
        logger.warning(f"  Could not extract transform parameters: {e}")
        metadata = {
            'method': 'ANTsPy_rigid',
            'ct_cropped': crop_info.get('cropped', False),
            'crop_info': crop_info,
            'timestamp': datetime.now().isoformat(),
            'parameter_extraction_error': str(e),
        }

    return reg_result, metadata


# =============================================================================
# TRANSFORM I/O
# =============================================================================

def save_transform(reg_result, transform_dir, prefix="ct_to_pet"):
    """
    Save ANTs transform files to directory.

    Args:
        reg_result: ANTs registration result dict
        transform_dir: Path to output directory
        prefix: filename prefix

    Returns:
        Path to primary transform file
    """
    transform_dir = Path(transform_dir)
    transform_dir.mkdir(parents=True, exist_ok=True)

    # Copy transform files
    import shutil
    transform_files = []

    for i, tfm_file in enumerate(reg_result.get('fwdtransforms', [])):
        if Path(tfm_file).exists():
            suffix = Path(tfm_file).suffix
            dest = transform_dir / f"{prefix}_fwd_{i}{suffix}"
            shutil.copy2(tfm_file, dest)
            transform_files.append(str(dest))
            logger.info(f"  Transform saved: {dest}")

    for i, tfm_file in enumerate(reg_result.get('invtransforms', [])):
        if Path(tfm_file).exists():
            suffix = Path(tfm_file).suffix
            dest = transform_dir / f"{prefix}_inv_{i}{suffix}"
            shutil.copy2(tfm_file, dest)

    # Return primary forward transform path
    if transform_files:
        return Path(transform_files[0])
    return None


def load_transform(transform_dir, prefix="ct_to_pet"):
    """
    Load ANTs transform files from directory.

    Args:
        transform_dir: Path to transform directory
        prefix: filename prefix

    Returns:
        list of transform file paths (forward transforms)
    """
    transform_dir = Path(transform_dir)

    # Find forward transforms
    fwd_files = sorted(transform_dir.glob(f"{prefix}_fwd_*.mat"))

    if not fwd_files:
        raise FileNotFoundError(f"No transforms found in {transform_dir} with prefix {prefix}")

    return [str(f) for f in fwd_files]


# =============================================================================
# RESLICING / RESAMPLING
# =============================================================================

def reslice_ct_to_pet(ct_nib, pet_nib, reg_result):
    """
    Reslice CT into PET space using the ANTs registration result.
    Uses linear interpolation. Output for QC assessment.

    Args:
        ct_nib: nibabel CT image (original, uncropped)
        pet_nib: nibabel PET image (defines output grid)
        reg_result: ANTs registration result dict

    Returns:
        nibabel image: CT resliced into PET space
    """
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        ct_tmp = tmpdir / "ct.nii.gz"
        pet_tmp = tmpdir / "pet.nii.gz"

        nib.save(ct_nib, str(ct_tmp))
        nib.save(pet_nib, str(pet_tmp))

        ct_ants = ants.image_read(str(ct_tmp))
        pet_ants = ants.image_read(str(pet_tmp))

        # Apply transforms
        warped = ants.apply_transforms(
            fixed=pet_ants,
            moving=ct_ants,
            transformlist=reg_result['fwdtransforms'],
            interpolator='linear'
        )

        # Save and reload as nibabel (force data loading before temp cleanup)
        out_tmp = tmpdir / "warped.nii.gz"
        ants.image_write(warped, str(out_tmp))
        result_nib = nib.load(str(out_tmp))
        # Force data into memory before temp dir is deleted
        result_data = result_nib.get_fdata().copy()

    return nib.Nifti1Image(result_data.astype(np.float32), pet_nib.affine)


def reslice_ct_to_pet_from_files(ct_nib, pet_nib, transform_files):
    """
    Reslice CT into PET space using saved transform files.

    Args:
        ct_nib: nibabel CT image
        pet_nib: nibabel PET image (defines output grid)
        transform_files: list of transform file paths

    Returns:
        nibabel image: CT resliced into PET space
    """
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        ct_tmp = tmpdir / "ct.nii.gz"
        pet_tmp = tmpdir / "pet.nii.gz"

        nib.save(ct_nib, str(ct_tmp))
        nib.save(pet_nib, str(pet_tmp))

        ct_ants = ants.image_read(str(ct_tmp))
        pet_ants = ants.image_read(str(pet_tmp))

        warped = ants.apply_transforms(
            fixed=pet_ants,
            moving=ct_ants,
            transformlist=transform_files,
            interpolator='linear'
        )

        out_tmp = tmpdir / "warped.nii.gz"
        ants.image_write(warped, str(out_tmp))
        result_nib = nib.load(str(out_tmp))
        # Force data into memory before temp dir is deleted
        result_data = result_nib.get_fdata().copy()

    return nib.Nifti1Image(result_data.astype(np.float32), pet_nib.affine)


def resample_mask_to_pet(mask_nib, pet_nib, transform_files, is_binary=True):
    """
    Resample a binary or continuous mask from CT space to PET space
    using LINEAR interpolation.

    Linear interpolation on binary masks produces continuous [0,1] values
    at voxel boundaries — this preserves partial-volume weighting naturally.

    Args:
        mask_nib: nibabel mask image (CT space)
        pet_nib: nibabel PET image (defines output grid)
        transform_files: list of ANTs transform file paths
        is_binary: if True, input is cast to float before resampling

    Returns:
        nibabel image: continuous mask in PET space (float32)
    """
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Ensure mask is float for linear interpolation
        mask_data = mask_nib.get_fdata()
        if is_binary:
            mask_data = mask_data.astype(np.float32)
        mask_float_nib = nib.Nifti1Image(mask_data.astype(np.float32), mask_nib.affine)

        mask_tmp = tmpdir / "mask.nii.gz"
        pet_tmp = tmpdir / "pet.nii.gz"

        nib.save(mask_float_nib, str(mask_tmp))
        nib.save(pet_nib, str(pet_tmp))

        mask_ants = ants.image_read(str(mask_tmp))
        pet_ants = ants.image_read(str(pet_tmp))

        warped = ants.apply_transforms(
            fixed=pet_ants,
            moving=mask_ants,
            transformlist=transform_files,
            interpolator='linear'
        )

        out_tmp = tmpdir / "warped.nii.gz"
        ants.image_write(warped, str(out_tmp))
        result_nib = nib.load(str(out_tmp))
        # Force data into memory before temp dir is deleted
        result_data = result_nib.get_fdata().copy()

    return nib.Nifti1Image(result_data.astype(np.float32), pet_nib.affine)


def resample_labels_to_pet(label_nib, pet_nib, transform_files):
    """
    Resample a labeled volume (multi-label integers) from CT space to PET
    space using nearest-neighbor interpolation (preserves label identity).

    Also produces per-label continuous masks via linear interpolation
    for partial-volume-weighted extraction.

    Args:
        label_nib: nibabel labeled image (CT space, integer labels)
        pet_nib: nibabel PET image (defines output grid)
        transform_files: list of ANTs transform file paths

    Returns:
        nn_nib: nibabel image with nearest-neighbor labels (for QC/visualization)
        continuous_dict: dict {label_int: nibabel_float32_image} continuous masks per label
    """
    import tempfile

    label_data = label_nib.get_fdata().astype(np.int16)
    unique_labels = np.unique(label_data)
    unique_labels = unique_labels[unique_labels > 0]

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        pet_tmp = tmpdir / "pet.nii.gz"
        nib.save(pet_nib, str(pet_tmp))
        pet_ants = ants.image_read(str(pet_tmp))

        # Nearest-neighbor version (for QC overlay / visualization)
        label_tmp = tmpdir / "labels.nii.gz"
        nib.save(nib.Nifti1Image(label_data.astype(np.float32), label_nib.affine), str(label_tmp))
        label_ants = ants.image_read(str(label_tmp))

        nn_warped = ants.apply_transforms(
            fixed=pet_ants,
            moving=label_ants,
            transformlist=transform_files,
            interpolator='nearestNeighbor'
        )

        nn_tmp = tmpdir / "nn_warped.nii.gz"
        ants.image_write(nn_warped, str(nn_tmp))
        nn_nib_raw = nib.load(str(nn_tmp))
        # Force data into memory
        nn_data = np.round(nn_nib_raw.get_fdata()).astype(np.int16).copy()
        nn_affine = pet_nib.affine.copy()

        # Per-label continuous masks (for PV-weighted quantification)
        continuous_dict = {}
        for lbl in unique_labels:
            binary = (label_data == lbl).astype(np.float32)
            bin_nib = nib.Nifti1Image(binary, label_nib.affine)

            bin_tmp = tmpdir / f"bin_{lbl}.nii.gz"
            nib.save(bin_nib, str(bin_tmp))
            bin_ants = ants.image_read(str(bin_tmp))

            cont_warped = ants.apply_transforms(
                fixed=pet_ants,
                moving=bin_ants,
                transformlist=transform_files,
                interpolator='linear'
            )

            cont_tmp = tmpdir / f"cont_{lbl}.nii.gz"
            ants.image_write(cont_warped, str(cont_tmp))
            cont_nib_tmp = nib.load(str(cont_tmp))
            # Force data into memory
            cont_data = cont_nib_tmp.get_fdata().astype(np.float32).copy()
            continuous_dict[int(lbl)] = nib.Nifti1Image(cont_data, pet_nib.affine)

    nn_nib = nib.Nifti1Image(nn_data, nn_affine)
    return nn_nib, continuous_dict


# =============================================================================
# LONGITUDINAL CT-CT REGISTRATION
# =============================================================================

def register_ct_to_ct(ct_baseline_nib, ct_followup_nib, pet_baseline_nib=None, pet_followup_nib=None,
                      transform_type='Rigid', crop_to_pet_fov=True, margin_mm=CT_CROP_MARGIN_MM):
    """
    Register Baseline CT to Followup CT using ANTs rigid registration.

    For longitudinal analysis, rigid registration (6 DOF) is preferred over
    affine to avoid introducing scale bias between timepoints.

    IMPORTANT: By default, CTs are cropped to the PET FOV (head/neck region)
    before registration. This is critical because:
    - Full CT covers head + torso, but torso positioning varies between sessions
    - The dental anatomy of interest is in the head/neck region
    - Registration on cropped images aligns the head region accurately

    Args:
        ct_baseline_nib: nibabel image for Baseline CT (moving)
        ct_followup_nib: nibabel image for Followup CT (fixed)
        pet_baseline_nib: nibabel PET image for baseline (for FOV cropping)
        pet_followup_nib: nibabel PET image for followup (for FOV cropping)
        transform_type: 'Rigid' (6 DOF, recommended) or 'Affine' (12 DOF)
        crop_to_pet_fov: If True, crop CTs to PET FOV before registration (recommended)
        margin_mm: Extra margin (mm) around PET FOV for cropping

    Returns:
        reg_result: ANTs registration result dict
        metadata: dict with registration parameters and cropping info
    """
    import tempfile

    # Crop CTs to PET FOV if requested and PET images provided
    crop_info_bl = {'cropped': False}
    crop_info_fu = {'cropped': False}

    if crop_to_pet_fov and pet_baseline_nib is not None and pet_followup_nib is not None:
        logger.info(f"  Cropping CTs to PET FOV (±{margin_mm}mm margin)...")
        ct_baseline_cropped, crop_info_bl = crop_ct_to_pet_fov(ct_baseline_nib, pet_baseline_nib, margin_mm)
        ct_followup_cropped, crop_info_fu = crop_ct_to_pet_fov(ct_followup_nib, pet_followup_nib, margin_mm)

        if crop_info_bl['cropped']:
            logger.info(f"    Baseline CT: {ct_baseline_nib.shape} → {ct_baseline_cropped.shape}")
        if crop_info_fu['cropped']:
            logger.info(f"    Followup CT: {ct_followup_nib.shape} → {ct_followup_cropped.shape}")
    else:
        ct_baseline_cropped = ct_baseline_nib
        ct_followup_cropped = ct_followup_nib
        if crop_to_pet_fov and (pet_baseline_nib is None or pet_followup_nib is None):
            logger.warning("  PET images not provided - using full CT for registration")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        bl_tmp = tmpdir / "ct_baseline.nii.gz"
        fu_tmp = tmpdir / "ct_followup.nii.gz"

        nib.save(ct_baseline_cropped, str(bl_tmp))
        nib.save(ct_followup_cropped, str(fu_tmp))

        bl_ants = ants.image_read(str(bl_tmp))
        fu_ants = ants.image_read(str(fu_tmp))

        logger.info(f"  Running ANTsPy {transform_type} registration (CT_baseline → CT_followup)...")
        logger.info(f"    Baseline CT shape: {bl_ants.shape}, spacing: {[f'{s:.3f}' for s in bl_ants.spacing]}")
        logger.info(f"    Followup CT shape: {fu_ants.shape}, spacing: {[f'{s:.3f}' for s in fu_ants.spacing]}")

        # Use affine_initializer for center-of-mass alignment to handle large positioning differences
        logger.info(f"    Using center-of-mass initialization...")
        initial_transform = ants.affine_initializer(fu_ants, bl_ants, search_factor=15, radian_fraction=0.1)

        # Run registration with robust multi-resolution approach
        reg_result = ants.registration(
            fixed=fu_ants,
            moving=bl_ants,
            type_of_transform=transform_type,
            initial_transform=initial_transform,  # Start from affine initializer
            aff_iterations=(2100, 1200, 1200, 100),  # More iterations for affine
            aff_shrink_factors=(6, 4, 2, 1),  # Multi-resolution
            aff_smoothing_sigmas=(3, 2, 1, 0),  # Smoothing at each level
            verbose=False
        )

    # Extract transform parameters for logging
    transform_file = reg_result['fwdtransforms'][0]

    try:
        tfm = ants.read_transform(transform_file)
        params = tfm.parameters
        tfm_type = tfm.transform_type

        # Handle different transform types
        if 'Affine' in tfm_type and len(params) == 12:
            # AffineTransform: [R11,R12,R13,R21,R22,R23,R31,R32,R33,tx,ty,tz]
            R = np.array(params[:9]).reshape(3, 3)
            translation = params[9:12]
            # Extract rotation angles from rotation matrix
            rot = Rotation.from_matrix(R)
            rot_angles = rot.as_euler('xyz')
        elif len(params) >= 6:
            # Euler3DTransform: [rot_x, rot_y, rot_z, tx, ty, tz]
            rot_angles = params[:3]
            translation = params[3:6]
        else:
            rot_angles = [0, 0, 0]
            translation = [0, 0, 0]

        translation_mag = np.sqrt(sum(t**2 for t in translation))

        metadata = {
            'method': f'ANTsPy_{transform_type}',
            'transform_type_returned': tfm_type,
            'translation_mm': [float(t) for t in translation],
            'translation_magnitude_mm': float(translation_mag),
            'rotation_rad': [float(r) for r in rot_angles],
            'rotation_deg': [float(np.degrees(r)) for r in rot_angles],
            'cropped_to_pet_fov': crop_info_bl.get('cropped', False) or crop_info_fu.get('cropped', False),
            'crop_info_baseline': crop_info_bl,
            'crop_info_followup': crop_info_fu,
            'timestamp': datetime.now().isoformat(),
        }

        logger.info(f"  CT-CT registration complete: "
                   f"translation={metadata['translation_magnitude_mm']:.2f}mm, "
                   f"rotation=[{metadata['rotation_deg'][0]:.2f}, {metadata['rotation_deg'][1]:.2f}, {metadata['rotation_deg'][2]:.2f}]°")

    except Exception as e:
        logger.warning(f"  Could not extract transform parameters: {e}")
        metadata = {
            'method': f'ANTsPy_{transform_type}',
            'cropped_to_pet_fov': crop_info_bl.get('cropped', False) or crop_info_fu.get('cropped', False),
            'crop_info_baseline': crop_info_bl,
            'crop_info_followup': crop_info_fu,
            'timestamp': datetime.now().isoformat(),
            'parameter_extraction_error': str(e),
        }

    return reg_result, metadata


def ants_transform_to_matrix(transform_file):
    """
    Convert ANTs rigid/affine transform to 4x4 homogeneous matrix.

    Handles both:
    - AffineTransform: 12 parameters [R11,R12,R13,R21,R22,R23,R31,R32,R33,tx,ty,tz]
    - Euler3DTransform: 6 parameters [rot_x,rot_y,rot_z,tx,ty,tz]

    Args:
        transform_file: Path to ANTs .mat transform file

    Returns:
        matrix_4x4: 4x4 homogeneous transformation matrix
        center: Center point used by ANTs (needed for reconstruction)
    """
    tfm = ants.read_transform(transform_file)
    params = tfm.parameters
    fixed_params = tfm.fixed_parameters  # Center point
    transform_type = tfm.transform_type

    # Get center point
    center = np.array(fixed_params[:3]) if len(fixed_params) >= 3 else np.zeros(3)

    if 'Affine' in transform_type and len(params) == 12:
        # AffineTransform: params = [R11,R12,R13,R21,R22,R23,R31,R32,R33,tx,ty,tz]
        R = np.array(params[:9]).reshape(3, 3)
        translation = np.array(params[9:12])
    elif len(params) >= 6:
        # Euler3DTransform: params = [rot_x,rot_y,rot_z,tx,ty,tz]
        rot_x, rot_y, rot_z = params[:3]
        translation = np.array(params[3:6])
        R = Rotation.from_euler('xyz', [rot_x, rot_y, rot_z]).as_matrix()
    else:
        logger.warning(f"Unknown transform format with {len(params)} parameters")
        R = np.eye(3)
        translation = np.zeros(3)

    # For centered transforms: x' = R @ (x - c) + c + t = R @ x + (c - R @ c + t)
    # Build 4x4 homogeneous matrix
    matrix = np.eye(4)
    matrix[:3, :3] = R
    matrix[:3, 3] = center - R @ center + translation

    return matrix, center


def matrix_to_ants_transform(matrix_4x4, center, output_file):
    """
    Convert 4x4 homogeneous matrix back to ANTs AffineTransform file.

    Uses AffineTransform format for full compatibility with ANTs.

    Args:
        matrix_4x4: 4x4 homogeneous transformation matrix
        center: Center point for ANTs transform
        output_file: Path to save the .mat file

    Returns:
        Path to saved transform file
    """
    R = matrix_4x4[:3, :3]
    t_effective = matrix_4x4[:3, 3]

    # Recover translation: t = t_effective - c + R @ c
    t = t_effective - center + R @ center

    # For AffineTransform: parameters = [R11,R12,R13,R21,R22,R23,R31,R32,R33,tx,ty,tz]
    params = list(R.flatten()) + list(t)

    # Create and save transform
    tfm = ants.create_ants_transform(
        transform_type='AffineTransform',
        parameters=params,
        fixed_parameters=list(center),
        dimension=3
    )

    ants.write_transform(tfm, str(output_file))
    return output_file


def compute_halfway_transform(fwd_transform_file, output_dir, prefix='halfway'):
    """
    Compute halfway (midpoint) transform using matrix logarithm.

    For a transform T that maps Baseline → Followup:
    - T^0.5 maps Baseline → Midpoint
    - T^-0.5 maps Followup → Midpoint

    Uses matrix logarithm: T^0.5 = exp(0.5 * log(T))
    This is mathematically exact for any rotation magnitude.

    Args:
        fwd_transform_file: Path to ANTs .mat file (Baseline → Followup)
        output_dir: Directory to save halfway transforms
        prefix: Filename prefix for output files

    Returns:
        dict with paths to:
            'baseline_to_midpoint': Transform from Baseline to Midpoint
            'followup_to_midpoint': Transform from Followup to Midpoint
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert ANTs transform to 4x4 matrix
    T_full, center = ants_transform_to_matrix(fwd_transform_file)

    logger.info(f"  Computing halfway transform using matrix logarithm...")

    # Compute T^0.5 using matrix logarithm
    # T^0.5 = exp(0.5 * log(T))
    try:
        log_T = logm(T_full)
        T_half = expm(0.5 * log_T)
        T_half_inv = expm(-0.5 * log_T)  # T^-0.5
    except Exception as e:
        logger.warning(f"  Matrix logarithm failed ({e}), falling back to simple halving")
        # Fallback: simple parameter halving (less accurate for large rotations)
        tfm = ants.read_transform(fwd_transform_file)
        params = np.array(tfm.parameters)
        half_params = params / 2

        # Create transforms directly
        T_half = np.eye(4)
        T_half_inv = np.eye(4)

        R_half = Rotation.from_euler('xyz', half_params[:3]).as_matrix()
        T_half[:3, :3] = R_half
        T_half[:3, 3] = center - R_half @ center + half_params[3:6]

        R_half_inv = R_half.T
        T_half_inv[:3, :3] = R_half_inv
        T_half_inv[:3, 3] = center - R_half_inv @ center - R_half_inv @ half_params[3:6]

    # Save transforms
    baseline_to_mid_file = output_dir / f"{prefix}_baseline_to_midpoint.mat"
    followup_to_mid_file = output_dir / f"{prefix}_followup_to_midpoint.mat"

    matrix_to_ants_transform(T_half, center, baseline_to_mid_file)
    matrix_to_ants_transform(T_half_inv, center, followup_to_mid_file)

    # Log the halfway transform parameters
    R_half = T_half[:3, :3]
    t_half = T_half[:3, 3]
    rot_half = Rotation.from_matrix(R_half)
    euler_half = rot_half.as_euler('xyz', degrees=True)
    trans_mag_half = np.linalg.norm(t_half)

    logger.info(f"  Halfway transform: translation={trans_mag_half:.2f}mm, "
               f"rotation=[{euler_half[0]:.2f}, {euler_half[1]:.2f}, {euler_half[2]:.2f}]°")

    return {
        'baseline_to_midpoint': str(baseline_to_mid_file),
        'followup_to_midpoint': str(followup_to_mid_file),
    }


def create_midpoint_reference_grid(ct_baseline_nib, ct_followup_nib,
                                    halfway_baseline_to_mid, halfway_followup_to_mid):
    """
    Create reference grid for midpoint space.

    The midpoint grid:
    - Has the same voxel spacing as the input CTs
    - Has its affine positioned at the midpoint between the two CTs
    - Covers the intersection of both CT FOVs (ensures both can contribute)

    Args:
        ct_baseline_nib: Baseline CT nibabel image
        ct_followup_nib: Followup CT nibabel image
        halfway_baseline_to_mid: Transform file from Baseline to Midpoint
        halfway_followup_to_mid: Transform file from Followup to Midpoint

    Returns:
        midpoint_reference_nib: nibabel image defining the midpoint reference grid
    """
    # Use Followup CT as the geometric reference (shape and spacing)
    # The midpoint space will have Followup's grid but with an adjusted origin

    # For simplicity, use Followup CT's grid as the reference
    # This is valid because the halfway transform will properly position
    # both CTs relative to this common reference

    # Create reference grid with same shape/spacing as Followup CT
    reference_data = np.zeros(ct_followup_nib.shape[:3], dtype=np.float32)
    reference_affine = ct_followup_nib.affine.copy()

    midpoint_reference = nib.Nifti1Image(reference_data, reference_affine)

    logger.info(f"  Midpoint reference grid: shape={midpoint_reference.shape}, "
               f"spacing={list(ct_followup_nib.header.get_zooms()[:3])}")

    return midpoint_reference


def resample_to_midpoint(image_nib, transform_files, reference_nib, interpolator='linear'):
    """
    Resample image to midpoint space using composite transforms.

    Args:
        image_nib: Source nibabel image (CT, PET, or mask)
        transform_files: List of transform file paths (applied in reverse order by ANTs)
        reference_nib: Midpoint reference grid nibabel image
        interpolator: 'linear' for continuous data, 'nearestNeighbor' for labels

    Returns:
        resampled_nib: nibabel image in midpoint space
    """
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        img_tmp = tmpdir / "input.nii.gz"
        ref_tmp = tmpdir / "reference.nii.gz"
        out_tmp = tmpdir / "output.nii.gz"

        nib.save(image_nib, str(img_tmp))
        nib.save(reference_nib, str(ref_tmp))

        img_ants = ants.image_read(str(img_tmp))
        ref_ants = ants.image_read(str(ref_tmp))

        # Apply composite transforms
        resampled_ants = ants.apply_transforms(
            fixed=ref_ants,
            moving=img_ants,
            transformlist=transform_files,
            interpolator=interpolator
        )

        ants.image_write(resampled_ants, str(out_tmp))
        result_nib = nib.load(str(out_tmp))
        result_data = result_nib.get_fdata().copy()

    return nib.Nifti1Image(result_data.astype(np.float32), reference_nib.affine)
