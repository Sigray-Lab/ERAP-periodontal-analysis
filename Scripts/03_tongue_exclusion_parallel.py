#!/usr/bin/env python3
"""
Parallel tongue exclusion — only 0mm + 3mm dilation.
Runs 4 sessions concurrently for ~4x speedup.
"""
import sys
import logging
import multiprocessing as mp
from pathlib import Path
from datetime import datetime

import numpy as np
import nibabel as nib
from scipy import ndimage

script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

from config import (
    RAWDATA_DIR, TOTALSEG_ROI_DIR, TOTALSEG_SEG_DIR, QC_DIR, TRANSFORM_DIR,
    LOGNOTES_DIR, ensure_directories
)
from utils.io_utils import load_nifti, save_nifti, get_voxel_dimensions, find_ct_file, find_pet_file
from utils.registration_utils import load_transform, resample_mask_to_pet

HEAD_MUSCLES_DIR = TOTALSEG_SEG_DIR.parent / "totalsegmentator_head_muscles"
TONGUE_QC_DIR = QC_DIR / "tongue_exclusion"
DILATIONS = [(0, "0mm"), (3, "3mm")]  # Only 0mm and 3mm


def dilate_mask(mask, distance_mm, voxel_dims):
    if distance_mm <= 0:
        return mask.astype(np.uint8)
    radius_vox = [max(1, int(round(distance_mm / d))) for d in voxel_dims]
    struct = ndimage.generate_binary_structure(3, 1)
    struct = ndimage.iterate_structure(struct, max(radius_vox))
    return ndimage.binary_dilation(mask, structure=struct).astype(np.uint8)


def process_session(subject_id, session_id):
    """Process one session: 0mm + 3mm tongue exclusion only."""
    logger = logging.getLogger(f"{subject_id}_{session_id}")

    roi_dir = TOTALSEG_ROI_DIR / f"{subject_id}_{session_id}"
    tongue_seg_dir = HEAD_MUSCLES_DIR / subject_id / session_id
    cont_dir = roi_dir / "continuous_masks_PETspace"
    transform_dir = TRANSFORM_DIR / f"{subject_id}_{session_id}"
    session_dir = RAWDATA_DIR / subject_id / session_id

    # Prerequisites
    tongue_file = tongue_seg_dir / "tongue_mask.nii.gz"
    for p, name in [(roi_dir, "ROI dir"), (tongue_file, "tongue mask"),
                    (cont_dir, "continuous masks dir")]:
        if not p.exists():
            logger.warning(f"  Missing {name}: {p}")
            return "skipped"

    # Skip if already done (0mm + 3mm CT-space masks exist)
    ct_masks = {label: roi_dir / f"tongue_exclusion_{label}.nii.gz" for _, label in DILATIONS}
    pet_masks = {label: cont_dir / f"tongue_exclusion_{label}_PETspace.nii.gz" for _, label in DILATIONS}
    if all(f.exists() for f in ct_masks.values()) and all(f.exists() for f in pet_masks.values()):
        # Check if trimmed tooth masks also exist
        tooth_files = sorted(cont_dir.glob("tooth_*_continuous.nii.gz"))
        tooth_files = [f for f in tooth_files if "trimmed" not in f.name]
        sample_trimmed = cont_dir / f"{tooth_files[0].stem.replace('_continuous', '')}_trimmed_0mm.nii.gz" if tooth_files else None
        if sample_trimmed and sample_trimmed.exists():
            logger.info(f"  Already done, skipping")
            return "skipped"

    logger.info(f"  Processing...")

    # Load tongue mask
    tongue_data, tongue_img = load_nifti(tongue_file)
    voxel_dims = get_voxel_dimensions(tongue_img)

    # Create dilated masks (CT space)
    tongue_masks = {}
    for dist, label in DILATIONS:
        mask = dilate_mask(tongue_data > 0, float(dist), voxel_dims)
        tongue_masks[label] = mask
        nib.save(nib.Nifti1Image(mask, tongue_img.affine), str(ct_masks[label]))
    logger.info(f"  CT-space tongue masks saved (0mm, 3mm)")

    # QC PNGs
    jaw_file = roi_dir / "peridental_upper_jaw.nii.gz"
    ct_file = find_ct_file(session_dir, prefer_bone=False)
    if jaw_file.exists() and ct_file is not None:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        ct_data, _ = load_nifti(ct_file)
        jaw_data, _ = load_nifti(jaw_file)
        TONGUE_QC_DIR.mkdir(parents=True, exist_ok=True)

        for _, label in DILATIONS:
            best_z = int(np.argmax(np.sum(jaw_data > 0, axis=(0, 1))))
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            for ax, (title_suffix, overlay) in zip(axes, [
                ("CT", None), ("Tongue", tongue_masks[label]), ("Jaw + Tongue", tongue_masks[label])
            ]):
                ct_slice = ct_data[:, :, best_z].T
                ax.imshow(ct_slice, cmap='gray', vmin=-200, vmax=1500, origin='lower')
                if overlay is not None:
                    ax.imshow(np.ma.masked_where(overlay[:, :, best_z].T == 0,
                              overlay[:, :, best_z].T), cmap='Reds', alpha=0.5, origin='lower')
                if title_suffix == "Jaw + Tongue":
                    jaw_slice = jaw_data[:, :, best_z].T
                    ax.imshow(np.ma.masked_where(jaw_slice == 0, jaw_slice),
                              cmap='Blues', alpha=0.3, origin='lower')
                ax.set_title(f"{title_suffix} ({label})")
                ax.axis('off')
            fig.suptitle(f"{subject_id}/{session_id} — Tongue Exclusion ({label})")
            plt.tight_layout()
            plt.savefig(str(TONGUE_QC_DIR / f"{subject_id}_{session_id}_tongue_{label}.png"), dpi=100)
            plt.close()
        logger.info(f"  QC PNGs saved")

    # PET-space resampling (only 2 masks instead of 5)
    if not transform_dir.exists():
        logger.warning(f"  No transforms, cannot create PET-space masks")
        return "partial"

    transform_files = load_transform(transform_dir, prefix="ct_to_pet")
    pet_file = find_pet_file(session_dir)
    if pet_file is None:
        logger.warning(f"  No PET file")
        return "partial"

    _, pet_img = load_nifti(pet_file)

    tongue_pet_data = {}
    for _, label in DILATIONS:
        tongue_pet = resample_mask_to_pet(
            nib.Nifti1Image(tongue_masks[label].astype(np.float32), tongue_img.affine),
            pet_img, transform_files, is_binary=True
        )
        nib.save(tongue_pet, str(pet_masks[label]))
        tongue_pet_data[label] = tongue_pet.get_fdata()
    logger.info(f"  PET-space tongue masks saved")

    # Trim per-tooth continuous masks
    tooth_files = sorted(cont_dir.glob("tooth_*_continuous.nii.gz"))
    tooth_files = [f for f in tooth_files if "trimmed" not in f.name]

    for tooth_file in tooth_files:
        fdi = tooth_file.stem.split('_')[1]
        tooth_data, _ = load_nifti(tooth_file)
        for _, label in DILATIONS:
            trimmed = tooth_data.copy()
            trimmed[tongue_pet_data[label] > 0.5] = 0
            out = cont_dir / f"tooth_{fdi}_trimmed_{label}.nii.gz"
            nib.save(nib.Nifti1Image(trimmed.astype(np.float32), pet_img.affine), str(out))
    logger.info(f"  Trimmed {len(tooth_files)} per-tooth masks")

    # Trim jaw masks
    for jaw_name in ["upper_jaw", "lower_jaw"]:
        jaw_pet_file = cont_dir / f"peridental_{jaw_name}_continuous.nii.gz"
        if not jaw_pet_file.exists():
            continue
        jaw_pet_data, _ = load_nifti(jaw_pet_file)
        for _, label in DILATIONS:
            trimmed = jaw_pet_data.copy()
            trimmed[tongue_pet_data[label] > 0.5] = 0
            out = cont_dir / f"peridental_{jaw_name}_trimmed_{label}.nii.gz"
            nib.save(nib.Nifti1Image(trimmed.astype(np.float32), pet_img.affine), str(out))
    logger.info(f"  Trimmed jaw masks saved")

    logger.info(f"  DONE")
    return "success"


def worker(args):
    subject_id, session_id = args
    logger = logging.getLogger(f"{subject_id}_{session_id}")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            f'%(asctime)s - {subject_id}/{session_id} - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    try:
        result = process_session(subject_id, session_id)
        return (subject_id, session_id, result)
    except Exception as e:
        logger.error(f"  FAILED: {e}", exc_info=True)
        return (subject_id, session_id, f"FAILED: {e}")


def discover_sessions():
    sessions = []
    for roi_dir in sorted(TOTALSEG_ROI_DIR.glob("sub-*_ses-*")):
        parts = roi_dir.name.split('_')
        if len(parts) >= 2:
            sessions.append((parts[0], parts[1]))
    return sessions


def main():
    ensure_directories()
    TONGUE_QC_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOGNOTES_DIR / f"tongue_exclusion_parallel_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    logger = logging.getLogger(__name__)

    sessions = discover_sessions()
    logger.info(f"Tongue exclusion (0mm + 3mm only) — {len(sessions)} sessions, 4 workers")
    logger.info(f"Log: {log_file}")

    start = datetime.now()
    with mp.Pool(min(4, len(sessions))) as pool:
        results = pool.map(worker, sessions)

    success = sum(1 for _, _, s in results if s == "success")
    skipped = sum(1 for _, _, s in results if s == "skipped")
    failed = [r for r in results if str(r[2]).startswith("FAILED")]

    elapsed = (datetime.now() - start).total_seconds() / 60
    logger.info("=" * 70)
    logger.info(f"DONE in {elapsed:.1f} min — Success: {success}, Skipped: {skipped}, Failed: {len(failed)}")
    for subj, sess, err in failed:
        logger.error(f"  {subj}/{sess}: {err}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
