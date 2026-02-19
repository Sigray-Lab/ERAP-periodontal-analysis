#!/usr/bin/env python3
"""
02b_geometry_roi_poc.py - Geometry-Only Peridental ROI Generation (Proof of Concept)

Creates peridental soft tissue ROIs using ONLY geometric operations on
TotalSegmentator multilabel output — NO HU gating required.

Algorithm:
    For each tooth label in the TotalSegmentator output:
    1. Dilate tooth mask by ~3mm (configurable)
    2. Subtract original tooth mask (shell)
    3. Subtract jawbone mask (upper_jawbone label 2 / lower_jawbone label 1)
    4. Subtract ALL other tooth masks (avoid tooth-to-tooth contamination)
    5. Handle overlap: assign contested voxels to nearest tooth centroid
    6. Exclude ROIs within 5mm of prosthetic labels (bridge=8, crown=9, implant=10)

Output:
    DerivedData/rois/<subject>_<session>/
        tooth_shells_geometry.nii.gz     (labeled volume: voxel value = FDI tooth number)
        tooth_shells_lookup.json         (tooth IDs, volumes, exclusion flags)
        peridental_upper_jaw.nii.gz      (union of all upper jaw tooth shells)
        peridental_lower_jaw.nii.gz      (union of all lower jaw tooth shells)
    QC/roi/<subject>/
        <session>_geometry_roi.png       (axial/coronal/sagittal overlay)
        <session>_roi_volumes.csv        (per-tooth volume table)

Usage:
    python 02b_geometry_roi_poc.py --subject sub-103 --session ses-oqbgk
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
from scipy.ndimage import binary_dilation, label, distance_transform_edt

# Add Scripts directory to path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

from config import (
    RAWDATA_DIR, ROI_DIR, ROI_QC_DIR, LOGNOTES_DIR, TOTALSEG_SEG_DIR,
    ensure_directories
)
from utils.io_utils import (
    load_nifti, save_nifti, get_voxel_dimensions, get_voxel_volume_ml
)

logger = logging.getLogger(__name__)

# =============================================================================
# TOTALSEGMENTATOR TEETH LABEL MAP
# =============================================================================

# TotalSegmentator teeth task → FDI mapping
# Labels 11-18 → FDI 11-18 (upper right)
# Labels 19-26 → FDI 21-28 (upper left)
# Labels 27-34 → FDI 31-38 (lower left)
# Labels 35-42 → FDI 41-48 (lower right)

def _fdi_to_label(fdi: int) -> int:
    """Convert FDI tooth number back to TotalSegmentator label."""
    if 11 <= fdi <= 18:
        return fdi
    elif 21 <= fdi <= 28:
        return fdi - 2
    elif 31 <= fdi <= 38:
        return fdi - 4
    elif 41 <= fdi <= 48:
        return fdi - 6
    else:
        return fdi


def _label_to_fdi(label_id: int) -> int:
    """Convert TotalSegmentator label to FDI tooth number."""
    if 11 <= label_id <= 18:
        return label_id  # upper right: label 11→FDI 11, etc.
    elif 19 <= label_id <= 26:
        return label_id + 2  # upper left: label 19→FDI 21, label 26→FDI 28
    elif 27 <= label_id <= 34:
        return label_id + 4  # lower left: label 27→FDI 31, label 34→FDI 38
    elif 35 <= label_id <= 42:
        return label_id + 6  # lower right: label 35→FDI 41, label 42→FDI 48
    else:
        return label_id  # non-tooth label, return as-is

# Jawbone labels
LOWER_JAWBONE_LABEL = 1
UPPER_JAWBONE_LABEL = 2

# Prosthetic labels (must exclude nearby ROIs)
PROSTHETIC_LABELS = [8, 9, 10]  # bridge, crown, implant

# Upper vs lower tooth label ranges
UPPER_TOOTH_LABELS = list(range(11, 27))  # labels 11-26
LOWER_TOOTH_LABELS = list(range(27, 43))  # labels 27-42
ALL_TOOTH_LABELS = UPPER_TOOTH_LABELS + LOWER_TOOTH_LABELS

# Default dilation
DILATION_MM = 3.0
PROSTHETIC_EXCLUSION_MM = 5.0
PROSTHETIC_HU_MIN = 2500  # Minimum mean HU to confirm real metal prosthetic
METAL_ARTIFACT_CAP_HU = 3071  # Scanner HU cap (metal saturation)
METAL_ARTIFACT_CAP_FRACTION = 0.15  # Exclude tooth if >15% voxels at scanner cap


# =============================================================================
# GEOMETRY-ONLY ROI GENERATION
# =============================================================================

def generate_geometry_rois(seg_data: np.ndarray, seg_img: nib.Nifti1Image,
                           dilation_mm: float = DILATION_MM,
                           prosthetic_exclusion_mm: float = PROSTHETIC_EXCLUSION_MM,
                           ct_data: np.ndarray = None
                           ) -> dict:
    """
    Generate per-tooth peridental ROIs using geometry only.

    Args:
        seg_data: TotalSegmentator multilabel array (int)
        seg_img: NIfTI image for affine/header
        dilation_mm: Shell dilation distance in mm
        prosthetic_exclusion_mm: Exclusion radius around prosthetics
        ct_data: CT array (same space as seg_data) for HU-validation of prosthetics

    Returns:
        dict with:
            tooth_shells: {fdi_number: bool_mask}
            excluded_teeth: {fdi_number: reason}
            upper_jaw_roi: bool_mask (union of upper shells)
            lower_jaw_roi: bool_mask (union of lower shells)
            labeled_volume: int array (voxel value = FDI)
            qc_metrics: dict
    """
    voxel_dims = get_voxel_dimensions(seg_img)
    voxel_vol_ml = get_voxel_volume_ml(seg_img)
    dilation_voxels = int(np.ceil(dilation_mm / min(voxel_dims)))

    # Extract bone masks
    upper_bone = (seg_data == UPPER_JAWBONE_LABEL)
    lower_bone = (seg_data == LOWER_JAWBONE_LABEL)
    all_bone = upper_bone | lower_bone

    # Extract prosthetic mask with HU validation
    # TotalSeg labels 8=bridge, 9=crown, 10=implant are often false positives.
    # Only treat as prosthetic if mean HU > PROSTHETIC_HU_MIN (real metal).
    prosthetic_mask = np.zeros(seg_data.shape, dtype=bool)
    prosthetic_labels_found = []
    for plbl in PROSTHETIC_LABELS:
        plbl_mask = (seg_data == plbl)
        if not np.any(plbl_mask):
            continue
        n_vox = int(np.sum(plbl_mask))
        if ct_data is not None:
            mean_hu = float(np.mean(ct_data[plbl_mask]))
            if mean_hu >= PROSTHETIC_HU_MIN:
                prosthetic_mask |= plbl_mask
                prosthetic_labels_found.append(plbl)
                logger.info(f"Prosthetic label {plbl}: {n_vox} vox, mean HU={mean_hu:.0f} — CONFIRMED metal")
            else:
                logger.info(f"Prosthetic label {plbl}: {n_vox} vox, mean HU={mean_hu:.0f} — IGNORED (below {PROSTHETIC_HU_MIN} HU)")
        else:
            # No CT available — fall back to trusting TotalSeg labels
            prosthetic_mask |= plbl_mask
            prosthetic_labels_found.append(plbl)
            logger.warning(f"Prosthetic label {plbl}: {n_vox} vox, no CT for HU validation — trusting label")

    has_prosthetics = np.any(prosthetic_mask)
    if has_prosthetics:
        excl_voxels = int(np.ceil(prosthetic_exclusion_mm / min(voxel_dims)))
        prosthetic_zone = binary_dilation(prosthetic_mask, iterations=excl_voxels)
        logger.info(f"Confirmed prosthetic labels: {prosthetic_labels_found}, "
                     f"exclusion zone: {np.sum(prosthetic_zone)} voxels")
    else:
        prosthetic_zone = np.zeros(seg_data.shape, dtype=bool)
        if any(np.any(seg_data == pl) for pl in PROSTHETIC_LABELS):
            logger.info("All prosthetic labels failed HU validation — no exclusion zone")

    # Build union of ALL tooth masks (for subtraction)
    all_teeth = np.isin(seg_data, ALL_TOOTH_LABELS)

    # Find which tooth labels are present
    present_labels = [int(l) for l in np.unique(seg_data) if l in ALL_TOOTH_LABELS]
    logger.info(f"Teeth found: {len(present_labels)} labels")

    # Step 1: Generate raw shells for each tooth
    raw_shells = {}
    tooth_centroids = {}

    for lbl in present_labels:
        tooth_mask = (seg_data == lbl)
        fdi = _label_to_fdi(lbl)

        # Centroid for overlap resolution
        coords = np.argwhere(tooth_mask)
        tooth_centroids[fdi] = coords.mean(axis=0)

        # Dilate
        dilated = binary_dilation(tooth_mask, iterations=dilation_voxels)

        # Subtract: original tooth, all bone, all OTHER teeth
        shell = dilated & ~tooth_mask & ~all_bone & ~(all_teeth & ~tooth_mask)

        raw_shells[fdi] = shell

    # Step 2: Resolve overlaps using nearest-centroid (Voronoi-like)
    # Build an overlap map
    overlap_count = np.zeros(seg_data.shape, dtype=np.int16)
    for shell in raw_shells.values():
        overlap_count += shell.astype(np.int16)

    contested = (overlap_count > 1)
    n_contested = int(np.sum(contested))
    logger.info(f"Contested voxels (overlap): {n_contested}")

    if n_contested > 0:
        # For contested voxels, assign to nearest tooth centroid
        contested_coords = np.argwhere(contested)

        for idx in range(len(contested_coords)):
            vox = contested_coords[idx]
            best_fdi = None
            best_dist = np.inf

            for fdi, centroid in tooth_centroids.items():
                if raw_shells[fdi][tuple(vox)]:
                    # Distance in mm
                    dist = np.sqrt(np.sum(((vox - centroid) * voxel_dims) ** 2))
                    if dist < best_dist:
                        best_dist = dist
                        best_fdi = fdi

            # Remove this voxel from all shells except best
            for fdi in list(raw_shells.keys()):
                if fdi != best_fdi and raw_shells[fdi][tuple(vox)]:
                    raw_shells[fdi][tuple(vox)] = False

    # Step 3: Mark prosthetic exclusions
    excluded_teeth = {}
    for fdi, shell in raw_shells.items():
        if has_prosthetics and np.any(shell & prosthetic_zone):
            overlap_frac = np.sum(shell & prosthetic_zone) / max(1, np.sum(shell))
            if overlap_frac > 0.1:  # >10% of shell in prosthetic zone
                excluded_teeth[fdi] = f"prosthetic_overlap_{overlap_frac*100:.0f}pct"
                logger.warning(f"FDI {fdi}: EXCLUDED — {overlap_frac*100:.0f}% in prosthetic zone")

    # Step 3b: Metal artifact exclusion (dental fillings)
    # Teeth with large amalgam fillings have many voxels at the scanner HU cap (3071).
    # These cause CT attenuation artifacts that corrupt PET quantification.
    # No dilation zone — fillings are inside the tooth, artifacts are local.
    if ct_data is not None:
        for fdi in list(raw_shells.keys()):
            if fdi in excluded_teeth:
                continue
            # Get the TotalSeg tooth mask (not the shell) for HU check
            lbl = _fdi_to_label(fdi)
            tooth_mask = (seg_data == lbl)
            n_tooth_vox = int(np.sum(tooth_mask))
            if n_tooth_vox == 0:
                continue
            hu_vals = ct_data[tooth_mask]
            n_at_cap = int(np.sum(hu_vals >= METAL_ARTIFACT_CAP_HU))
            cap_frac = n_at_cap / n_tooth_vox
            if cap_frac > METAL_ARTIFACT_CAP_FRACTION:
                excluded_teeth[fdi] = f"metal_artifact_{cap_frac*100:.0f}pct_at_cap"
                logger.warning(f"FDI {fdi}: EXCLUDED — {cap_frac*100:.0f}% voxels at HU cap ({n_at_cap}/{n_tooth_vox}), metal filling artifact")
            elif n_at_cap > 0:
                logger.info(f"FDI {fdi}: {cap_frac*100:.1f}% at HU cap ({n_at_cap}/{n_tooth_vox}) — below threshold, kept")

    # Step 4: Build labeled volume and jaw aggregates
    labeled_vol = np.zeros(seg_data.shape, dtype=np.int16)
    upper_jaw = np.zeros(seg_data.shape, dtype=bool)
    lower_jaw = np.zeros(seg_data.shape, dtype=bool)

    tooth_volumes = {}

    for fdi, shell in raw_shells.items():
        if fdi in excluded_teeth:
            continue

        labeled_vol[shell] = fdi
        vol_ml = float(np.sum(shell)) * voxel_vol_ml
        tooth_volumes[fdi] = vol_ml

        # Determine upper (FDI 11-28) vs lower (FDI 31-48)
        if 11 <= fdi <= 28:
            upper_jaw |= shell
        elif 31 <= fdi <= 48:
            lower_jaw |= shell

    # QC metrics
    qc = {
        'n_teeth_total': len(present_labels),
        'n_teeth_with_roi': len(tooth_volumes),
        'n_teeth_excluded': len(excluded_teeth),
        'n_upper_teeth': sum(1 for f in tooth_volumes if 11 <= f <= 28),
        'n_lower_teeth': sum(1 for f in tooth_volumes if 31 <= f <= 48),
        'upper_jaw_volume_ml': float(np.sum(upper_jaw)) * voxel_vol_ml,
        'lower_jaw_volume_ml': float(np.sum(lower_jaw)) * voxel_vol_ml,
        'total_roi_volume_ml': float(np.sum(labeled_vol > 0)) * voxel_vol_ml,
        'contested_voxels': n_contested,
        'dilation_mm': dilation_mm,
        'tooth_volumes_ml': tooth_volumes,
        'excluded_teeth': excluded_teeth,
    }

    return {
        'tooth_shells': raw_shells,
        'excluded_teeth': excluded_teeth,
        'upper_jaw_roi': upper_jaw,
        'lower_jaw_roi': lower_jaw,
        'labeled_volume': labeled_vol,
        'qc_metrics': qc,
    }


# =============================================================================
# QC VISUALIZATION
# =============================================================================

def generate_qc_image(ct_data: np.ndarray, labeled_vol: np.ndarray,
                      bone_mask: np.ndarray, teeth_mask: np.ndarray,
                      output_path: Path, title: str) -> None:
    """Generate QC PNG showing geometry ROIs overlaid on CT."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        import matplotlib.patches as mpatches

        # Find best slices based on ROI content
        roi_mask = labeled_vol > 0

        # Axial: slice with most ROI
        roi_per_z = np.sum(roi_mask, axis=(0, 1))
        best_z = int(np.argmax(roi_per_z)) if roi_per_z.max() > 0 else ct_data.shape[2] // 2

        # Coronal
        roi_per_y = np.sum(roi_mask, axis=(0, 2))
        best_y = int(np.argmax(roi_per_y)) if roi_per_y.max() > 0 else ct_data.shape[1] // 2

        # Sagittal
        roi_per_x = np.sum(roi_mask, axis=(1, 2))
        best_x = int(np.argmax(roi_per_x)) if roi_per_x.max() > 0 else ct_data.shape[0] // 2

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        slices_info = [
            (0, ct_data[:, :, best_z].T, labeled_vol[:, :, best_z].T,
             bone_mask[:, :, best_z].T, teeth_mask[:, :, best_z].T, f'Axial z={best_z}'),
            (1, ct_data[:, best_y, :].T, labeled_vol[:, best_y, :].T,
             bone_mask[:, best_y, :].T, teeth_mask[:, best_y, :].T, f'Coronal y={best_y}'),
            (2, ct_data[best_x, :, :].T, labeled_vol[best_x, :, :].T,
             bone_mask[best_x, :, :].T, teeth_mask[best_x, :, :].T, f'Sagittal x={best_x}'),
        ]

        # Top row: CT with ROI contours
        for col, ct_sl, roi_sl, bone_sl, teeth_sl, label_str in slices_info:
            ax = axes[0, col]
            ax.imshow(ct_sl, cmap='gray', origin='lower', vmin=-200, vmax=2000)
            if np.any(roi_sl > 0):
                ax.contour(roi_sl > 0, colors='lime', linewidths=1.5)
            if np.any(bone_sl):
                ax.contour(bone_sl, colors='orange', linewidths=0.8, alpha=0.6)
            if np.any(teeth_sl):
                ax.contour(teeth_sl, colors='cyan', linewidths=0.5, alpha=0.8)
            ax.set_title(label_str, fontsize=11)
            ax.axis('off')

        # Bottom row: colored per-tooth ROIs
        unique_fdi = np.unique(labeled_vol)
        unique_fdi = unique_fdi[unique_fdi > 0]
        n_teeth = len(unique_fdi)

        # Create a colormap for teeth
        if n_teeth > 0:
            cmap = plt.cm.get_cmap('tab20', max(n_teeth, 2))

        for col, ct_sl, roi_sl, bone_sl, teeth_sl, label_str in slices_info:
            ax = axes[1, col]
            ax.imshow(ct_sl, cmap='gray', origin='lower', vmin=-200, vmax=2000, alpha=0.5)

            # Overlay each tooth ROI in a different color
            overlay = np.zeros((*ct_sl.shape, 4))
            for idx, fdi in enumerate(unique_fdi):
                tooth_sl = (roi_sl == fdi)
                if np.any(tooth_sl):
                    color = cmap(idx % 20)
                    overlay[tooth_sl] = [color[0], color[1], color[2], 0.6]

            ax.imshow(overlay, origin='lower')
            ax.set_title(f'{label_str} (per-tooth)', fontsize=11)
            ax.axis('off')

        # Legend
        legend_elements = [
            mpatches.Patch(color='lime', label='ROI boundary'),
            mpatches.Patch(color='orange', label='Bone'),
            mpatches.Patch(color='cyan', label='Teeth'),
        ]
        fig.legend(handles=legend_elements, loc='upper right', fontsize=10)

        plt.suptitle(title, fontsize=13, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 0.95, 0.95])

        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"QC image saved: {output_path}")

    except ImportError:
        logger.warning("Matplotlib not available, skipping QC image")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Geometry-Only Peridental ROI Generation')
    parser.add_argument('--subject', type=str, required=True)
    parser.add_argument('--session', type=str, required=True)
    parser.add_argument('--dilation', type=float, default=DILATION_MM,
                        help=f'Dilation distance in mm (default: {DILATION_MM})')
    args = parser.parse_args()

    # Setup
    ensure_directories()

    log_dir = LOGNOTES_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"geometry_roi_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger.info(f"Log: {log_file}")

    subject_id = args.subject
    session_id = args.session

    logger.info("=" * 60)
    logger.info(f"GEOMETRY-ONLY ROI GENERATION")
    logger.info(f"Subject: {subject_id}, Session: {session_id}")
    logger.info(f"Dilation: {args.dilation} mm")
    logger.info("=" * 60)

    # --- Load TotalSegmentator multilabel ---
    # Check QC dir first (where POC outputs were saved)
    seg_dir = TOTALSEG_SEG_DIR / subject_id / session_id
    seg_file = seg_dir / "totalseg_teeth_multilabel.nii.gz"

    if not seg_file.exists():
        logger.error(f"TotalSegmentator output not found: {seg_file}")
        sys.exit(1)

    logger.info(f"Loading segmentation: {seg_file}")
    seg_data, seg_img = load_nifti(seg_file)
    seg_data = seg_data.astype(np.int32)

    # Also load the cropped CT for QC visualization
    crop_file = seg_dir / "crop.nii.gz"

    ct_data = None
    if crop_file.exists():
        logger.info(f"Loading cropped CT: {crop_file}")
        ct_data, _ = load_nifti(crop_file)
    else:
        logger.warning("No cropped CT found for QC visualization")

    # --- Load PET ---
    pet_dir = RAWDATA_DIR / subject_id / session_id / "pet"
    pet_files = list(pet_dir.glob("*_pet.nii")) + list(pet_dir.glob("*_pet.nii.gz"))
    pet_data = None
    pet_img = None
    if pet_files:
        logger.info(f"Loading PET: {pet_files[0]}")
        pet_data, pet_img = load_nifti(pet_files[0])
    else:
        logger.warning("No PET file found")

    # --- Generate ROIs ---
    result = generate_geometry_rois(seg_data, seg_img, dilation_mm=args.dilation)
    qc = result['qc_metrics']

    logger.info(f"\n--- ROI Summary ---")
    logger.info(f"Teeth with ROI: {qc['n_teeth_with_roi']} / {qc['n_teeth_total']}")
    logger.info(f"Excluded: {qc['n_teeth_excluded']}")
    logger.info(f"Upper jaw ROI: {qc['upper_jaw_volume_ml']:.2f} mL ({qc['n_upper_teeth']} teeth)")
    logger.info(f"Lower jaw ROI: {qc['lower_jaw_volume_ml']:.2f} mL ({qc['n_lower_teeth']} teeth)")
    logger.info(f"Total ROI: {qc['total_roi_volume_ml']:.2f} mL")
    logger.info(f"Contested voxels resolved: {qc['contested_voxels']}")

    for fdi, vol in sorted(qc['tooth_volumes_ml'].items()):
        logger.info(f"  FDI {fdi}: {vol:.3f} mL")

    for fdi, reason in qc['excluded_teeth'].items():
        logger.warning(f"  FDI {fdi}: EXCLUDED ({reason})")

    # --- Save outputs ---
    output_dir = ROI_DIR / f"{subject_id}_{session_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Labeled volume (voxel value = FDI number)
    save_nifti(result['labeled_volume'], seg_img,
               output_dir / "tooth_shells_geometry.nii.gz")
    logger.info(f"Saved labeled volume: {output_dir / 'tooth_shells_geometry.nii.gz'}")

    # Upper jaw union
    save_nifti(result['upper_jaw_roi'].astype(np.uint8), seg_img,
               output_dir / "peridental_upper_jaw.nii.gz")

    # Lower jaw union
    save_nifti(result['lower_jaw_roi'].astype(np.uint8), seg_img,
               output_dir / "peridental_lower_jaw.nii.gz")

    # Lookup JSON
    lookup = {
        'tooth_ids': sorted([int(f) for f in qc['tooth_volumes_ml'].keys()]),
        'n_teeth': qc['n_teeth_with_roi'],
        'excluded_teeth': {str(k): v for k, v in qc['excluded_teeth'].items()},
        'dilation_mm': args.dilation,
        'method': 'geometry_only',
        'tooth_volumes_ml': {str(k): round(v, 4) for k, v in qc['tooth_volumes_ml'].items()},
    }
    with open(output_dir / "tooth_shells_lookup.json", 'w') as f:
        json.dump(lookup, f, indent=2)

    # --- QC visualization ---
    if ct_data is not None:
        bone_mask = (seg_data == UPPER_JAWBONE_LABEL) | (seg_data == LOWER_JAWBONE_LABEL)
        teeth_mask = np.isin(seg_data, ALL_TOOTH_LABELS)

        qc_dir = ROI_QC_DIR / subject_id
        qc_path = qc_dir / f"{session_id}_geometry_roi.png"

        generate_qc_image(
            ct_data, result['labeled_volume'], bone_mask, teeth_mask,
            qc_path, f"{subject_id} / {session_id} — Geometry ROIs ({args.dilation}mm dilation)"
        )

    # --- Per-tooth volume CSV ---
    vol_records = []
    for fdi, vol in sorted(qc['tooth_volumes_ml'].items()):
        jaw = 'upper' if 11 <= fdi <= 28 else 'lower'
        vol_records.append({
            'subject_id': subject_id,
            'session_id': session_id,
            'fdi_tooth': fdi,
            'jaw': jaw,
            'roi_volume_ml': round(vol, 4),
            'excluded': fdi in qc['excluded_teeth'],
            'exclusion_reason': qc['excluded_teeth'].get(fdi, ''),
        })

    vol_df = pd.DataFrame(vol_records)
    qc_vol_path = ROI_QC_DIR / subject_id / f"{session_id}_roi_volumes.csv"
    qc_vol_path.parent.mkdir(parents=True, exist_ok=True)
    vol_df.to_csv(qc_vol_path, index=False)
    logger.info(f"Volume CSV: {qc_vol_path}")

    # --- PET SUV extraction ---
    if pet_data is not None:
        logger.info("\n--- PET SUV Extraction ---")

        # Check if PET and segmentation are in the same space
        # The segmentation is in CROPPED CT space. PET is in full brain space.
        # We need to resample PET to the cropped CT space or vice versa.
        # For now, let's check dimensions.
        seg_shape = seg_data.shape
        pet_shape = pet_data.shape

        logger.info(f"Seg shape: {seg_shape}, PET shape: {pet_shape}")

        if seg_shape != pet_shape:
            logger.info("Resampling PET to segmentation space...")
            try:
                from scipy.ndimage import map_coordinates
                from numpy.linalg import inv

                # Compute PET voxel → world → seg voxel transform
                seg_affine = seg_img.affine
                pet_affine = pet_img.affine

                # World-to-voxel for PET
                pet_vox2world = pet_affine
                seg_vox2world = seg_affine
                seg_world2vox = inv(seg_vox2world)

                # For each voxel in seg space, find corresponding PET voxel
                # Create coordinate grids
                ii, jj, kk = np.mgrid[0:seg_shape[0], 0:seg_shape[1], 0:seg_shape[2]]
                seg_vox = np.stack([ii.ravel(), jj.ravel(), kk.ravel(),
                                    np.ones(ii.size)], axis=0)

                # seg voxel → world
                world_coords = seg_vox2world @ seg_vox

                # world → PET voxel
                pet_world2vox = inv(pet_vox2world)
                pet_vox = pet_world2vox @ world_coords

                # Sample PET at these coordinates
                pet_resampled = map_coordinates(
                    pet_data, pet_vox[:3], order=1, mode='constant', cval=0
                ).reshape(seg_shape)

                logger.info(f"PET resampled to seg space: {pet_resampled.shape}")
                logger.info(f"PET range in ROI region: {pet_resampled[result['labeled_volume'] > 0].min():.2f} - "
                            f"{pet_resampled[result['labeled_volume'] > 0].max():.2f}")

            except Exception as e:
                logger.error(f"PET resampling failed: {e}")
                pet_resampled = None
        else:
            pet_resampled = pet_data

        if pet_resampled is not None:
            suv_records = []

            for fdi in sorted(qc['tooth_volumes_ml'].keys()):
                if fdi in qc['excluded_teeth']:
                    continue

                shell = result['tooth_shells'][fdi]
                values = pet_resampled[shell]
                values = values[np.isfinite(values)]

                if len(values) == 0:
                    continue

                rec = {
                    'subject_id': subject_id,
                    'session_id': session_id,
                    'fdi_tooth': fdi,
                    'jaw': 'upper' if 11 <= fdi <= 28 else 'lower',
                    'suv_mean': float(np.mean(values)),
                    'suv_std': float(np.std(values)),
                    'suv_median': float(np.median(values)),
                    'suv_p90': float(np.percentile(values, 90)),
                    'suv_max': float(np.max(values)),
                    'suv_min': float(np.min(values)),
                    'n_voxels': len(values),
                    'roi_volume_ml': qc['tooth_volumes_ml'][fdi],
                }
                suv_records.append(rec)

            # Jaw-level aggregates
            for jaw_name, jaw_roi in [('upper_jaw', result['upper_jaw_roi']),
                                       ('lower_jaw', result['lower_jaw_roi'])]:
                if np.any(jaw_roi):
                    values = pet_resampled[jaw_roi]
                    values = values[np.isfinite(values)]
                    if len(values) > 0:
                        suv_records.append({
                            'subject_id': subject_id,
                            'session_id': session_id,
                            'fdi_tooth': jaw_name,
                            'jaw': jaw_name.replace('_jaw', ''),
                            'suv_mean': float(np.mean(values)),
                            'suv_std': float(np.std(values)),
                            'suv_median': float(np.median(values)),
                            'suv_p90': float(np.percentile(values, 90)),
                            'suv_max': float(np.max(values)),
                            'suv_min': float(np.min(values)),
                            'n_voxels': len(values),
                            'roi_volume_ml': float(np.sum(jaw_roi)) * get_voxel_volume_ml(seg_img),
                        })

            if suv_records:
                suv_df = pd.DataFrame(suv_records)
                suv_path = output_dir / "pet_suv_per_tooth.csv"
                suv_df.to_csv(suv_path, index=False)
                logger.info(f"PET SUV saved: {suv_path}")

                # Print summary
                tooth_only = suv_df[suv_df['fdi_tooth'].apply(lambda x: isinstance(x, int))]
                if len(tooth_only) > 0:
                    logger.info(f"\nPET SUV summary (per-tooth):")
                    logger.info(f"  Mean SUV range: {tooth_only['suv_mean'].min():.3f} - {tooth_only['suv_mean'].max():.3f}")
                    logger.info(f"  Overall mean: {tooth_only['suv_mean'].mean():.3f}")

                jaw_rows = suv_df[suv_df['fdi_tooth'].apply(lambda x: isinstance(x, str))]
                for _, row in jaw_rows.iterrows():
                    logger.info(f"  {row['fdi_tooth']}: SUVmean={row['suv_mean']:.3f}, "
                                f"n={row['n_voxels']}, vol={row['roi_volume_ml']:.2f}mL")

    logger.info(f"\n{'='*60}")
    logger.info("DONE")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
