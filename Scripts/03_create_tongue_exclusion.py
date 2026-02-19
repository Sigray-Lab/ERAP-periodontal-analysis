#!/usr/bin/env python3
"""
03_create_tongue_exclusion.py - Create Tongue Exclusion Zone Masks

Creates dilated tongue masks (3mm, 5mm, 8mm, 10mm) for all sessions and generates:
1. Dilated tongue masks in CT space
2. Trimmed per-tooth ROIs (subtracting tongue exclusion zone)
3. Trimmed jaw-level ROIs
4. All masks resampled to PET space
5. QC PNGs showing tongue overlay on upper jaw ROIs

Dilation options:
- 3mm: Conservative trim - removes only hottest PVE from tongue/floor of mouth
- 5mm: Moderate-conservative trim
- 8mm: Aggressive trim - removes lingual side entirely, keeps buccal + interproximal
- 10mm: Very aggressive trim

Output:
    DerivedData/rois/totalsegmentator_teeth/sub-XXX_ses-YYY/
        tongue_exclusion_3mm.nii.gz           (CT space, binary)
        tongue_exclusion_5mm.nii.gz           (CT space, binary)
        tongue_exclusion_8mm.nii.gz           (CT space, binary)
        tongue_exclusion_10mm.nii.gz          (CT space, binary)
        continuous_masks_PETspace/
            tooth_XX_trimmed_3mm.nii.gz       (PET space, continuous)
            tooth_XX_trimmed_5mm.nii.gz       (PET space, continuous)
            tooth_XX_trimmed_8mm.nii.gz       (PET space, continuous)
            tooth_XX_trimmed_10mm.nii.gz      (PET space, continuous)
            peridental_upper_jaw_trimmed_3mm.nii.gz
            peridental_upper_jaw_trimmed_5mm.nii.gz
            peridental_upper_jaw_trimmed_8mm.nii.gz
            peridental_upper_jaw_trimmed_10mm.nii.gz
            peridental_lower_jaw_trimmed_3mm.nii.gz
            peridental_lower_jaw_trimmed_5mm.nii.gz
            peridental_lower_jaw_trimmed_8mm.nii.gz
            peridental_lower_jaw_trimmed_10mm.nii.gz
    QC/tongue_exclusion/
        sub-XXX_ses-YYY_tongue_3mm.png
        sub-XXX_ses-YYY_tongue_5mm.png
        sub-XXX_ses-YYY_tongue_8mm.png
        sub-XXX_ses-YYY_tongue_10mm.png

Usage:
    python 03_create_tongue_exclusion.py                    # All sessions
    python 03_create_tongue_exclusion.py --subjects sub-101 # Specific subjects
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import nibabel as nib
from scipy import ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add Scripts directory to path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

from config import (
    RAWDATA_DIR, TOTALSEG_ROI_DIR, TOTALSEG_SEG_DIR, QC_DIR, TRANSFORM_DIR,
    LOGNOTES_DIR, ensure_directories
)
from utils.io_utils import (
    load_nifti, save_nifti, get_voxel_dimensions, load_blinding_key, find_ct_file
)
from utils.registration_utils import load_transform, resample_mask_to_pet

logger = logging.getLogger(__name__)

# Tongue mask location
HEAD_MUSCLES_DIR = TOTALSEG_SEG_DIR.parent / "totalsegmentator_head_muscles"

# QC output directory
TONGUE_QC_DIR = QC_DIR / "tongue_exclusion"


def create_spherical_structuring_element(radius_mm, voxel_dims):
    """Create a 3D spherical structuring element for binary dilation."""
    # Calculate radius in voxels for each dimension
    radius_vox = [int(np.ceil(radius_mm / dim)) for dim in voxel_dims]

    # Create a grid
    z, y, x = np.ogrid[
        -radius_vox[0]:radius_vox[0]+1,
        -radius_vox[1]:radius_vox[1]+1,
        -radius_vox[2]:radius_vox[2]+1
    ]

    # Calculate distance in mm
    dist = np.sqrt(
        (z * voxel_dims[0])**2 +
        (y * voxel_dims[1])**2 +
        (x * voxel_dims[2])**2
    )

    # Create sphere
    sphere = dist <= radius_mm
    return sphere


def dilate_mask(mask_data, radius_mm, voxel_dims):
    """Dilate a binary mask by radius_mm using spherical structuring element."""
    struct_elem = create_spherical_structuring_element(radius_mm, voxel_dims)
    dilated = ndimage.binary_dilation(mask_data, structure=struct_elem)
    return dilated.astype(np.uint8)


def create_qc_png(ct_data, jaw_mask, tongue_mask, dilated_mask, output_path, title):
    """Create a QC PNG showing tongue exclusion overlay on upper jaw ROI."""
    # Find the axial slice with most upper molar ROI voxels
    # Upper molars are in the posterior upper jaw
    jaw_z_sum = np.sum(jaw_mask, axis=(0, 1))
    if jaw_z_sum.max() == 0:
        logger.warning(f"  No jaw mask voxels found for QC")
        return

    # Find slice with most jaw voxels
    best_z = np.argmax(jaw_z_sum)

    # Get slices
    ct_slice = ct_data[:, :, best_z].T
    jaw_slice = jaw_mask[:, :, best_z].T
    tongue_slice = tongue_mask[:, :, best_z].T
    dilated_slice = dilated_mask[:, :, best_z].T

    # Clip CT for display
    ct_display = np.clip(ct_slice, -200, 400)

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # Show CT
    ax.imshow(ct_display, cmap='gray', aspect='equal')

    # Overlay jaw ROI (green outline)
    ax.contour(jaw_slice, levels=[0.5], colors='green', linewidths=1.5)

    # Overlay original tongue (blue fill, transparent)
    tongue_rgba = np.zeros((*tongue_slice.shape, 4))
    tongue_rgba[tongue_slice > 0] = [0, 0, 1, 0.3]
    ax.imshow(tongue_rgba, aspect='equal')

    # Overlay dilated tongue (red outline)
    ax.contour(dilated_slice, levels=[0.5], colors='red', linewidths=2)

    ax.set_title(title, fontsize=14)
    ax.axis('off')

    # Legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='green', linewidth=2, label='Jaw ROI'),
        Patch(facecolor='blue', alpha=0.3, label='Tongue (original)'),
        Line2D([0], [0], color='red', linewidth=2, label='Tongue exclusion zone'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def process_session(subject_id, session_id, force=False):
    """Process one session: create tongue exclusion zones and trimmed ROIs."""

    roi_dir = TOTALSEG_ROI_DIR / f"{subject_id}_{session_id}"
    tongue_seg_dir = HEAD_MUSCLES_DIR / subject_id / session_id
    cont_dir = roi_dir / "continuous_masks_PETspace"
    transform_dir = TRANSFORM_DIR / f"{subject_id}_{session_id}"

    # Check prerequisites
    if not roi_dir.exists():
        logger.warning(f"  No ROI directory, skipping")
        return False

    tongue_file = tongue_seg_dir / "tongue_mask.nii.gz"
    if not tongue_file.exists():
        logger.warning(f"  No tongue mask found at {tongue_file}")
        return False

    if not cont_dir.exists():
        logger.warning(f"  No continuous masks directory, skipping")
        return False

    # Check if already done (all 5 versions must exist: 0mm + 4 dilations)
    out_0mm = roi_dir / "tongue_exclusion_0mm.nii.gz"  # Original, no dilation
    out_3mm = roi_dir / "tongue_exclusion_3mm.nii.gz"
    out_5mm = roi_dir / "tongue_exclusion_5mm.nii.gz"
    out_8mm = roi_dir / "tongue_exclusion_8mm.nii.gz"
    out_10mm = roi_dir / "tongue_exclusion_10mm.nii.gz"
    if out_0mm.exists() and out_3mm.exists() and out_5mm.exists() and out_8mm.exists() and out_10mm.exists() and not force:
        logger.info(f"  Tongue exclusion masks already exist, skipping")
        return True

    # Load tongue mask
    logger.info(f"  Loading tongue mask...")
    tongue_data, tongue_img = load_nifti(tongue_file)
    voxel_dims = get_voxel_dimensions(tongue_img)

    # Load upper jaw mask for reference space (CT space)
    jaw_file = roi_dir / "peridental_upper_jaw.nii.gz"
    if not jaw_file.exists():
        logger.warning(f"  No upper jaw mask found")
        return False
    jaw_data, jaw_img = load_nifti(jaw_file)

    # Create tongue masks: 0mm (original) + dilated versions
    logger.info(f"  Creating tongue masks (0mm original, 3mm, 5mm, 8mm, 10mm dilation)...")
    tongue_0mm = (tongue_data > 0).astype(np.uint8)  # Original, no dilation
    tongue_3mm = dilate_mask(tongue_data > 0, 3.0, voxel_dims)
    tongue_5mm = dilate_mask(tongue_data > 0, 5.0, voxel_dims)
    tongue_8mm = dilate_mask(tongue_data > 0, 8.0, voxel_dims)
    tongue_10mm = dilate_mask(tongue_data > 0, 10.0, voxel_dims)

    # Save masks in CT space
    nib.save(nib.Nifti1Image(tongue_0mm, tongue_img.affine), str(out_0mm))
    nib.save(nib.Nifti1Image(tongue_3mm, tongue_img.affine), str(out_3mm))
    nib.save(nib.Nifti1Image(tongue_5mm, tongue_img.affine), str(out_5mm))
    nib.save(nib.Nifti1Image(tongue_8mm, tongue_img.affine), str(out_8mm))
    nib.save(nib.Nifti1Image(tongue_10mm, tongue_img.affine), str(out_10mm))
    logger.info(f"  Saved: {out_0mm.name}, {out_3mm.name}, {out_5mm.name}, {out_8mm.name}, {out_10mm.name}")

    # Load CT for QC visualization
    session_dir = RAWDATA_DIR / subject_id / session_id
    ct_file = find_ct_file(session_dir, prefer_bone=False)
    if ct_file is not None:
        ct_data, _ = load_nifti(ct_file)

        # Create QC PNGs
        TONGUE_QC_DIR.mkdir(parents=True, exist_ok=True)

        for dilation, dilated, label in [(0, tongue_0mm, "0mm"), (3, tongue_3mm, "3mm"),
                                          (5, tongue_5mm, "5mm"), (8, tongue_8mm, "8mm"),
                                          (10, tongue_10mm, "10mm")]:
            qc_path = TONGUE_QC_DIR / f"{subject_id}_{session_id}_tongue_{label}.png"
            dilation_desc = "original, no dilation" if dilation == 0 else f"{label} dilation"
            create_qc_png(
                ct_data, jaw_data, tongue_data, dilated, qc_path,
                f"{subject_id}/{session_id} - Tongue Exclusion ({dilation_desc})"
            )
            logger.info(f"  QC PNG: {qc_path.name}")

    # Load transforms for PET space resampling
    if not transform_dir.exists():
        logger.warning(f"  No transforms found, cannot create PET-space trimmed masks")
        return True  # Partial success

    transform_files = load_transform(transform_dir, prefix="ct_to_pet")

    # Load PET for reference grid
    from utils.io_utils import find_pet_file
    pet_file = find_pet_file(session_dir)
    if pet_file is None:
        logger.warning(f"  No PET file, cannot create PET-space trimmed masks")
        return True

    _, pet_img = load_nifti(pet_file)

    # Resample tongue exclusion masks to PET space
    logger.info(f"  Resampling tongue exclusion to PET space...")
    tongue_0mm_pet = resample_mask_to_pet(
        nib.Nifti1Image(tongue_0mm.astype(np.float32), tongue_img.affine),
        pet_img, transform_files, is_binary=True
    )
    tongue_3mm_pet = resample_mask_to_pet(
        nib.Nifti1Image(tongue_3mm.astype(np.float32), tongue_img.affine),
        pet_img, transform_files, is_binary=True
    )
    tongue_5mm_pet = resample_mask_to_pet(
        nib.Nifti1Image(tongue_5mm.astype(np.float32), tongue_img.affine),
        pet_img, transform_files, is_binary=True
    )
    tongue_8mm_pet = resample_mask_to_pet(
        nib.Nifti1Image(tongue_8mm.astype(np.float32), tongue_img.affine),
        pet_img, transform_files, is_binary=True
    )
    tongue_10mm_pet = resample_mask_to_pet(
        nib.Nifti1Image(tongue_10mm.astype(np.float32), tongue_img.affine),
        pet_img, transform_files, is_binary=True
    )

    tongue_0mm_pet_data = tongue_0mm_pet.get_fdata()
    tongue_3mm_pet_data = tongue_3mm_pet.get_fdata()
    tongue_5mm_pet_data = tongue_5mm_pet.get_fdata()
    tongue_8mm_pet_data = tongue_8mm_pet.get_fdata()
    tongue_10mm_pet_data = tongue_10mm_pet.get_fdata()

    # Save tongue exclusion masks in PET space (for reference)
    nib.save(tongue_0mm_pet, str(cont_dir / "tongue_exclusion_0mm_PETspace.nii.gz"))
    nib.save(tongue_3mm_pet, str(cont_dir / "tongue_exclusion_3mm_PETspace.nii.gz"))
    nib.save(tongue_5mm_pet, str(cont_dir / "tongue_exclusion_5mm_PETspace.nii.gz"))
    nib.save(tongue_8mm_pet, str(cont_dir / "tongue_exclusion_8mm_PETspace.nii.gz"))
    nib.save(tongue_10mm_pet, str(cont_dir / "tongue_exclusion_10mm_PETspace.nii.gz"))

    # Trim per-tooth continuous masks
    logger.info(f"  Creating trimmed per-tooth masks...")
    tooth_files = sorted(cont_dir.glob("tooth_*_continuous.nii.gz"))
    for tooth_file in tooth_files:
        # Skip if already a trimmed file
        if "trimmed" in tooth_file.name:
            continue

        fdi = tooth_file.stem.split('_')[1]
        tooth_data, tooth_img_t = load_nifti(tooth_file)

        # Apply tongue subtraction (set to 0 where tongue exclusion is >0.5)
        for dilation, tongue_pet_data, label in [
            (0, tongue_0mm_pet_data, "0mm"),
            (3, tongue_3mm_pet_data, "3mm"),
            (5, tongue_5mm_pet_data, "5mm"),
            (8, tongue_8mm_pet_data, "8mm"),
            (10, tongue_10mm_pet_data, "10mm")
        ]:
            trimmed = tooth_data.copy()
            trimmed[tongue_pet_data > 0.5] = 0

            out_file = cont_dir / f"tooth_{fdi}_trimmed_{label}.nii.gz"
            nib.save(nib.Nifti1Image(trimmed.astype(np.float32), pet_img.affine), str(out_file))

    logger.info(f"  Created trimmed masks for {len(tooth_files)} teeth")

    # Trim jaw-level masks
    logger.info(f"  Creating trimmed jaw masks...")
    for jaw_name in ["upper_jaw", "lower_jaw"]:
        jaw_pet_file = cont_dir / f"peridental_{jaw_name}_continuous.nii.gz"
        if not jaw_pet_file.exists():
            continue

        jaw_pet_data, _ = load_nifti(jaw_pet_file)

        for dilation, tongue_pet_data, label in [
            (0, tongue_0mm_pet_data, "0mm"),
            (3, tongue_3mm_pet_data, "3mm"),
            (5, tongue_5mm_pet_data, "5mm"),
            (8, tongue_8mm_pet_data, "8mm"),
            (10, tongue_10mm_pet_data, "10mm")
        ]:
            trimmed = jaw_pet_data.copy()
            trimmed[tongue_pet_data > 0.5] = 0

            out_file = cont_dir / f"peridental_{jaw_name}_trimmed_{label}.nii.gz"
            nib.save(nib.Nifti1Image(trimmed.astype(np.float32), pet_img.affine), str(out_file))

    logger.info(f"  Tongue exclusion processing: DONE")
    return True


def discover_sessions():
    """Discover all sessions that have ROI directories."""
    sessions = []
    for roi_dir in sorted(TOTALSEG_ROI_DIR.glob("sub-*_ses-*")):
        parts = roi_dir.name.split('_')
        if len(parts) >= 2:
            subject_id = parts[0]
            session_id = parts[1]
            sessions.append((subject_id, session_id))
    return sessions


def main():
    parser = argparse.ArgumentParser(description='Create Tongue Exclusion Zone Masks')
    parser.add_argument('--subjects', nargs='+', type=str, help='Specific subjects to process')
    parser.add_argument('--force', action='store_true', help='Re-run even if output exists')
    args = parser.parse_args()

    ensure_directories()
    TONGUE_QC_DIR.mkdir(parents=True, exist_ok=True)

    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOGNOTES_DIR / f"tongue_exclusion_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    logger.info("=" * 70)
    logger.info("TONGUE EXCLUSION ZONE CREATION")
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 70)

    # Discover sessions
    all_sessions = discover_sessions()

    if args.subjects:
        sessions = [(s, sess) for s, sess in all_sessions if s in args.subjects]
    else:
        sessions = all_sessions

    logger.info(f"Sessions to process: {len(sessions)}")

    success_count = 0
    fail_count = 0

    for subject_id, session_id in sessions:
        logger.info(f"\n--- {subject_id} / {session_id} ---")
        try:
            if process_session(subject_id, session_id, force=args.force):
                success_count += 1
            else:
                fail_count += 1
        except Exception as e:
            logger.error(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            fail_count += 1

    logger.info(f"\n{'='*70}")
    logger.info("SUMMARY")
    logger.info(f"{'='*70}")
    logger.info(f"Success: {success_count}")
    logger.info(f"Failed:  {fail_count}")
    logger.info(f"QC PNGs: {TONGUE_QC_DIR}")
    logger.info(f"Log: {log_file}")


if __name__ == "__main__":
    main()
