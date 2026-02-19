#!/usr/bin/env python3
"""
02_roi_generation.py - Peridental ROI Generation Script

This script generates ROIs from CT segmentation masks for all subjects/sessions.

CRITICAL: ROIs are generated INDEPENDENTLY for each timepoint using that
timepoint's native segmentation. Do NOT propagate ROIs between timepoints.

Usage:
    cd Periodontal_Analysis/Scripts
    python 02_roi_generation.py

    # Or with options:
    python 02_roi_generation.py --subject sub-101
    python 02_roi_generation.py --force  # Re-run even if output exists

Prerequisites:
    - 01_segmentation.py must have been run first

Output:
    DerivedData/rois/sub-XXX_ses-XXXXX/
        - peridental_soft_tissue_4mm.nii.gz
        - peridental_soft_tissue_6mm.nii.gz
        - alveolar_bone.nii.gz
        - metal_artifact_mask.nii.gz (if applicable)
        - tooth_shells/ (per-tooth ROIs if available)
    QC/roi/
        - hu_rejection_rates.csv
        - roi_volumes.csv
"""

import argparse
import sys
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

# Add Scripts directory to path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

from config import (
    RAWDATA_DIR, SEGMENTATION_DIR, ROI_DIR, ROI_QC_DIR, LOGNOTES_DIR,
    VOLUME_QC_DIR, VOLUME_STABILITY_THRESHOLD,
    ensure_directories
)
from utils.io_utils import (
    load_blinding_key, discover_subjects, discover_sessions,
    find_ct_file, load_nifti, save_nifti, get_voxel_volume_ml
)
from utils.roi_utils import (
    generate_all_rois, compare_roi_volumes, validate_roi
)


def setup_logging(log_dir: Path) -> logging.Logger:
    """Setup logging to file and console."""
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"roi_generation_log_{timestamp}.txt"

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


def generate_roi_qc_image(ct_data: np.ndarray, peridental_roi: np.ndarray,
                          alveolar_bone_roi: np.ndarray, teeth_mask: np.ndarray,
                          output_path: Path, title: str) -> None:
    """
    Generate QC visualization of ROIs overlaid on CT.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # Find best slice (most ROI voxels)
        roi_per_slice = np.sum(peridental_roi, axis=(0, 1))
        if np.max(roi_per_slice) > 0:
            best_slice = np.argmax(roi_per_slice)
        else:
            best_slice = ct_data.shape[2] // 2

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Axial - zoomed to ROI
        ax = axes[0]
        ct_slice = ct_data[:, :, best_slice].T
        ax.imshow(ct_slice, cmap='gray', origin='lower', vmin=-200, vmax=1500)

        if np.sum(peridental_roi[:, :, best_slice]) > 0:
            ax.contour(peridental_roi[:, :, best_slice].T, colors='lime', linewidths=1.5)
        if np.sum(alveolar_bone_roi[:, :, best_slice]) > 0:
            ax.contour(alveolar_bone_roi[:, :, best_slice].T, colors='orange', linewidths=1)
        if np.sum(teeth_mask[:, :, best_slice]) > 0:
            ax.contour(teeth_mask[:, :, best_slice].T, colors='cyan', linewidths=0.5)

        ax.set_title(f'Axial (z={best_slice})')
        ax.axis('off')

        # Coronal
        roi_per_coronal = np.sum(peridental_roi, axis=(0, 2))
        best_coronal = np.argmax(roi_per_coronal) if np.max(roi_per_coronal) > 0 else ct_data.shape[1] // 2

        ax = axes[1]
        ax.imshow(ct_data[:, best_coronal, :].T, cmap='gray', origin='lower', vmin=-200, vmax=1500)
        if np.sum(peridental_roi[:, best_coronal, :]) > 0:
            ax.contour(peridental_roi[:, best_coronal, :].T, colors='lime', linewidths=1.5)
        if np.sum(alveolar_bone_roi[:, best_coronal, :]) > 0:
            ax.contour(alveolar_bone_roi[:, best_coronal, :].T, colors='orange', linewidths=1)
        ax.set_title(f'Coronal (y={best_coronal})')
        ax.axis('off')

        # Sagittal
        mid_sag = ct_data.shape[0] // 2
        ax = axes[2]
        ax.imshow(ct_data[mid_sag, :, :].T, cmap='gray', origin='lower', vmin=-200, vmax=1500)
        if np.sum(peridental_roi[mid_sag, :, :]) > 0:
            ax.contour(peridental_roi[mid_sag, :, :].T, colors='lime', linewidths=1.5)
        if np.sum(alveolar_bone_roi[mid_sag, :, :]) > 0:
            ax.contour(alveolar_bone_roi[mid_sag, :, :].T, colors='orange', linewidths=1)
        ax.set_title(f'Sagittal (x={mid_sag})')
        ax.axis('off')

        # Legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='lime', linewidth=2, label='Peridental (soft tissue)'),
            Line2D([0], [0], color='orange', linewidth=2, label='Alveolar bone'),
            Line2D([0], [0], color='cyan', linewidth=1, label='Teeth'),
        ]
        fig.legend(handles=legend_elements, loc='upper right', fontsize=10)

        plt.suptitle(title, fontsize=12)
        plt.tight_layout()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    except ImportError:
        logging.warning("Matplotlib not available, skipping QC image generation")


def process_subject_session(subject_id: str, session_id: str, timepoint: str,
                            ct_file: Path, force: bool = False,
                            logger: logging.Logger = None) -> dict:
    """
    Generate ROIs for a single subject/session.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # Define paths
    seg_dir = SEGMENTATION_DIR / subject_id / session_id
    output_dir = ROI_DIR / f"{subject_id}_{session_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check prerequisites
    teeth_file = seg_dir / "teeth_mask.nii.gz"
    maxilla_file = seg_dir / "maxilla_mask.nii.gz"

    if not teeth_file.exists() or not maxilla_file.exists():
        logger.warning(f"  {session_id}: Segmentation not found, skipping")
        return {'status': 'missing_segmentation', 'output_dir': output_dir}

    # Check if already processed
    peridental_output = output_dir / "peridental_soft_tissue_4mm.nii.gz"
    if peridental_output.exists() and not force:
        logger.info(f"  {session_id}: Already processed, skipping (use --force to re-run)")
        return {'status': 'skipped', 'output_dir': output_dir}

    # Load data
    logger.info(f"  {session_id} ({timepoint}): Loading segmentation and CT...")
    ct_data, ct_img = load_nifti(ct_file)
    teeth_mask, _ = load_nifti(teeth_file)
    maxilla_mask, _ = load_nifti(maxilla_file)

    teeth_mask = teeth_mask > 0
    maxilla_mask = maxilla_mask > 0

    # Load metal mask if exists
    metal_file = seg_dir / "metal_artifact_mask.nii.gz"
    metal_mask = None
    if metal_file.exists():
        metal_mask, _ = load_nifti(metal_file)
        metal_mask = metal_mask > 0

    # Load tooth instances if available
    tooth_instances = None
    instances_file = seg_dir / "tooth_instances.nii.gz"
    if instances_file.exists():
        tooth_instances, _ = load_nifti(instances_file)

    # Generate all ROIs
    logger.info(f"    Generating ROIs...")
    roi_result = generate_all_rois(
        teeth_mask, maxilla_mask, ct_data, ct_img,
        metal_mask=metal_mask, tooth_instances=tooth_instances
    )

    # Save ROIs
    logger.info(f"    Saving ROIs to {output_dir}")

    # Primary peridental ROI
    if roi_result['peridental_4mm']['roi_mask'] is not None:
        save_nifti(
            roi_result['peridental_4mm']['roi_mask'].astype(np.uint8),
            ct_img, output_dir / "peridental_soft_tissue_4mm.nii.gz"
        )

    # Sensitivity ROI
    if roi_result['peridental_6mm']['roi_mask'] is not None:
        save_nifti(
            roi_result['peridental_6mm']['roi_mask'].astype(np.uint8),
            ct_img, output_dir / "peridental_soft_tissue_6mm.nii.gz"
        )

    # Alveolar bone
    if roi_result['alveolar_bone']['roi_mask'] is not None:
        save_nifti(
            roi_result['alveolar_bone']['roi_mask'].astype(np.uint8),
            ct_img, output_dir / "alveolar_bone.nii.gz"
        )

    # Metal mask (copy from segmentation)
    if metal_mask is not None and np.sum(metal_mask) > 0:
        save_nifti(
            metal_mask.astype(np.uint8),
            ct_img, output_dir / "metal_artifact_mask.nii.gz"
        )

    # Per-tooth shells - save as single labeled NIfTI instead of individual files
    # This reduces storage from ~700MB to ~3MB per session
    if roi_result['tooth_shells']:
        # Create labeled volume where each voxel value = tooth_id
        labeled_shells = np.zeros(ct_data.shape, dtype=np.int16)
        for tooth_id, shell in roi_result['tooth_shells'].items():
            labeled_shells[shell > 0] = tooth_id

        save_nifti(
            labeled_shells,
            ct_img, output_dir / "tooth_shells_labeled.nii.gz"
        )
        # Also save a lookup JSON with tooth IDs
        tooth_ids = list(roi_result['tooth_shells'].keys())
        import json
        with open(output_dir / "tooth_shells_lookup.json", 'w') as f:
            json.dump({'tooth_ids': tooth_ids, 'n_teeth': len(tooth_ids)}, f)

    # Log metrics
    qc = roi_result['qc_metrics']
    logger.info(f"    Peridental 4mm: {qc.get('peridental_4mm_volume_ml', 0):.2f} mL "
                f"(HU rejection: {qc.get('peridental_4mm_hu_rejection', 0)*100:.1f}%)")
    logger.info(f"    Peridental 6mm: {qc.get('peridental_6mm_volume_ml', 0):.2f} mL")
    logger.info(f"    Alveolar bone: {qc.get('alveolar_bone_volume_ml', 0):.2f} mL")
    logger.info(f"    Tooth shells created: {qc.get('tooth_count', 0)}")

    for w in roi_result['warnings']:
        logger.warning(f"    {w}")

    # Generate QC image
    qc_dir = ROI_QC_DIR / subject_id
    qc_image_path = qc_dir / f"{session_id}_roi.png"

    peridental_roi = roi_result['peridental_4mm']['roi_mask']
    alveolar_roi = roi_result['alveolar_bone']['roi_mask']

    if peridental_roi is not None and alveolar_roi is not None:
        title = f"{subject_id} / {session_id} ({timepoint}) - ROIs"
        generate_roi_qc_image(
            ct_data, peridental_roi, alveolar_roi, teeth_mask,
            qc_image_path, title
        )
        logger.info(f"    QC image saved to {qc_image_path}")

    return {
        'status': 'success',
        'output_dir': output_dir,
        'qc_metrics': qc,
        'warnings': roi_result['warnings'],
        'peridental_4mm_vol': qc.get('peridental_4mm_volume_ml', 0),
        'peridental_6mm_vol': qc.get('peridental_6mm_volume_ml', 0),
        'alveolar_bone_vol': qc.get('alveolar_bone_volume_ml', 0),
        'tooth_count': qc.get('tooth_count', 0),
        'hu_rejection_rate': qc.get('peridental_4mm_hu_rejection', 0)
    }


def main():
    """Main ROI generation pipeline."""
    parser = argparse.ArgumentParser(description='Peridental ROI Generation Pipeline')
    parser.add_argument('--subject', type=str, help='Process only this subject')
    parser.add_argument('--force', action='store_true', help='Re-run even if output exists')
    args = parser.parse_args()

    # Setup
    ensure_directories()
    logger = setup_logging(LOGNOTES_DIR)

    logger.info("=" * 60)
    logger.info("PERIODONTAL ANALYSIS - ROI GENERATION")
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    # Load blinding key
    try:
        blinding_map = load_blinding_key()
        logger.info(f"Loaded blinding key with {len(blinding_map)} entries")
    except FileNotFoundError as e:
        logger.error(f"ERROR: {e}")
        sys.exit(1)

    # Discover subjects
    subjects = discover_subjects()
    if args.subject:
        if args.subject in subjects:
            subjects = [args.subject]
        else:
            logger.error(f"Subject {args.subject} not found")
            sys.exit(1)

    logger.info(f"Processing {len(subjects)} subjects")

    # Track results
    all_results = []
    volume_data = []
    hu_rejection_data = []

    # Process each subject
    for subject_id in subjects:
        logger.info(f"\n{'='*40}")
        logger.info(f"SUBJECT: {subject_id}")
        logger.info(f"{'='*40}")

        subject_dir = RAWDATA_DIR / subject_id
        sessions = discover_sessions(subject_dir)

        subject_results = {}

        for session_id in sessions:
            session_dir = subject_dir / session_id

            # Get timepoint
            key = (subject_id, session_id)
            timepoint = blinding_map.get(key, "Unknown")
            if timepoint == "Unknown":
                logger.warning(f"  {session_id}: Not in blinding key, skipping")
                continue

            # Find CT file - use same CT as segmentation
            # Read from segmentation_method.txt to ensure consistency
            seg_method_file = SEGMENTATION_DIR / subject_id / session_id / "segmentation_method.txt"
            ct_file = None
            if seg_method_file.exists():
                with open(seg_method_file, 'r') as f:
                    for line in f:
                        if line.startswith('CT file:'):
                            ct_filename = line.split(':', 1)[1].strip()
                            ct_file = session_dir / "ct" / ct_filename
                            if not ct_file.exists():
                                ct_file = None
                            break

            if ct_file is None:
                # Fallback to find_ct_file with same preference as segmentation
                ct_file = find_ct_file(session_dir, prefer_bone=True)

            if ct_file is None:
                logger.warning(f"  {session_id}: No CT file found, skipping")
                continue

            # Process
            result = process_subject_session(
                subject_id, session_id, timepoint, ct_file,
                force=args.force, logger=logger
            )

            result['subject_id'] = subject_id
            result['session_id'] = session_id
            result['timepoint'] = timepoint

            all_results.append(result)
            subject_results[timepoint] = result

            # Collect volume data
            if result['status'] == 'success':
                volume_data.append({
                    'subject_id': subject_id,
                    'session_id': session_id,
                    'timepoint': timepoint,
                    'peridental_4mm_ml': result.get('peridental_4mm_vol', np.nan),
                    'peridental_6mm_ml': result.get('peridental_6mm_vol', np.nan),
                    'alveolar_bone_ml': result.get('alveolar_bone_vol', np.nan),
                    'tooth_count': result.get('tooth_count', 0)
                })

                hu_rejection_data.append({
                    'subject_id': subject_id,
                    'session_id': session_id,
                    'timepoint': timepoint,
                    'hu_rejection_rate': result.get('hu_rejection_rate', np.nan)
                })

        # Compare baseline vs followup
        if 'Baseline' in subject_results and 'Followup' in subject_results:
            bl = subject_results['Baseline']
            fu = subject_results['Followup']

            if bl['status'] == 'success' and fu['status'] == 'success':
                pct_change = 0
                if bl.get('peridental_4mm_vol', 0) > 0:
                    pct_change = abs(fu.get('peridental_4mm_vol', 0) - bl.get('peridental_4mm_vol', 0)) / bl.get('peridental_4mm_vol', 0) * 100

                if pct_change > VOLUME_STABILITY_THRESHOLD:
                    logger.warning(f"  Volume stability WARNING: {pct_change:.1f}% change in peridental ROI")

    # Save summary data
    ROI_QC_DIR.mkdir(parents=True, exist_ok=True)

    if volume_data:
        vol_df = pd.DataFrame(volume_data)
        vol_file = ROI_QC_DIR / "roi_volumes.csv"
        # Merge with existing data to avoid overwriting when run per-subject
        if vol_file.exists():
            existing = pd.read_csv(vol_file)
            # Remove rows for subjects we just processed, then append new
            processed_subjects = vol_df['subject_id'].unique()
            existing = existing[~existing['subject_id'].isin(processed_subjects)]
            vol_df = pd.concat([existing, vol_df], ignore_index=True)
        vol_df.to_csv(vol_file, index=False)
        logger.info(f"\nROI volumes saved to: {vol_file}")

    if hu_rejection_data:
        hu_df = pd.DataFrame(hu_rejection_data)
        hu_file = ROI_QC_DIR / "hu_rejection_rates.csv"
        # Merge with existing data to avoid overwriting when run per-subject
        if hu_file.exists():
            existing = pd.read_csv(hu_file)
            processed_subjects = hu_df['subject_id'].unique()
            existing = existing[~existing['subject_id'].isin(processed_subjects)]
            hu_df = pd.concat([existing, hu_df], ignore_index=True)
        hu_df.to_csv(hu_file, index=False)
        logger.info(f"HU rejection rates saved to: {hu_file}")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    success_count = sum(1 for r in all_results if r['status'] == 'success')
    skip_count = sum(1 for r in all_results if r['status'] == 'skipped')
    missing_count = sum(1 for r in all_results if r['status'] == 'missing_segmentation')

    logger.info(f"Sessions processed: {success_count}")
    logger.info(f"Sessions skipped: {skip_count}")
    logger.info(f"Sessions with missing segmentation: {missing_count}")

    if volume_data:
        vol_df = pd.DataFrame(volume_data)
        logger.info(f"\nPeridental 4mm volume range: {vol_df['peridental_4mm_ml'].min():.2f} - {vol_df['peridental_4mm_ml'].max():.2f} mL")
        logger.info(f"Mean peridental 4mm volume: {vol_df['peridental_4mm_ml'].mean():.2f} mL")

    logger.info("\n" + "=" * 60)
    logger.info(f"ROI generation completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
