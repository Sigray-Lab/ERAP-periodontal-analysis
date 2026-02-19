#!/usr/bin/env python3
"""
01_segmentation.py - CT Dental Segmentation Script

This script segments teeth and maxilla from CT images for all subjects/sessions.

CRITICAL: Segmentation is performed INDEPENDENTLY for each timepoint.
Do NOT propagate or warp ROIs between timepoints.

Usage:
    cd Periodontal_Analysis/Scripts
    python 01_segmentation.py

    # Or with options:
    python 01_segmentation.py --subject sub-101
    python 01_segmentation.py --method HU_threshold
    python 01_segmentation.py --force  # Re-run even if output exists

Output:
    DerivedData/segmentations/sub-XXX/ses-XXXXX/     (primary masks)
    DerivedData/segmentations/totalsegmentator_teeth/ (TotalSegmentator output)
    DerivedData/segmentations/hu_fallback/            (HU-threshold output)
    QC/dental_segmentator_crops/sub-XXX/ses-YYY/      (TotalSegmentator QC)
    QC/HU_segmentation/sub-XXX/                       (HU-threshold QC)
"""

import argparse
import sys
import logging
from pathlib import Path
from datetime import datetime
import json

import numpy as np
import nibabel as nib
import pandas as pd

# Add Scripts directory to path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

from config import (
    RAWDATA_DIR, SEGMENTATION_DIR, TOTALSEG_SEG_DIR, HU_SEG_DIR,
    SEG_QC_DIR, TOTALSEG_QC_DIR, LOGNOTES_DIR,
    ensure_directories
)
from utils.io_utils import (
    load_blinding_key, discover_subjects, discover_sessions,
    find_ct_file, load_nifti, save_nifti, get_voxel_volume_ml
)
from utils.segmentation_utils import (
    segment_dental_ct, compare_segmentation_volumes, validate_segmentation,
    run_total_segmentator_dental, segment_by_hu_threshold,
    detect_metal_artifacts
)
from utils.io_utils import get_voxel_dimensions


def setup_logging(log_dir: Path) -> logging.Logger:
    """Setup logging to file and console."""
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"segmentation_log_{timestamp}.txt"

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


def generate_segmentation_qc_image(ct_data: np.ndarray, teeth_mask: np.ndarray,
                                    maxilla_mask: np.ndarray, metal_mask: np.ndarray,
                                    output_path: Path, title: str) -> None:
    """
    Generate QC visualization of segmentation overlaid on CT.

    Args:
        ct_data: CT image data
        teeth_mask: Teeth segmentation mask
        maxilla_mask: Maxilla segmentation mask
        metal_mask: Metal artifact mask
        output_path: Path to save PNG
        title: Title for the figure
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap

        # Find slice with most teeth
        teeth_per_slice = np.sum(teeth_mask, axis=(0, 1))
        if np.max(teeth_per_slice) > 0:
            best_slice = np.argmax(teeth_per_slice)
        else:
            best_slice = ct_data.shape[2] // 2

        # Also get coronal and sagittal views
        teeth_per_coronal = np.sum(teeth_mask, axis=(0, 2))
        best_coronal = np.argmax(teeth_per_coronal) if np.max(teeth_per_coronal) > 0 else ct_data.shape[1] // 2

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Axial view
        ax = axes[0]
        ax.imshow(ct_data[:, :, best_slice].T, cmap='gray', origin='lower',
                  vmin=-200, vmax=2000)
        if teeth_mask is not None and np.sum(teeth_mask[:, :, best_slice]) > 0:
            ax.contour(teeth_mask[:, :, best_slice].T, colors='cyan', linewidths=1)
        if maxilla_mask is not None and np.sum(maxilla_mask[:, :, best_slice]) > 0:
            ax.contour(maxilla_mask[:, :, best_slice].T, colors='yellow', linewidths=0.5)
        if metal_mask is not None and np.sum(metal_mask[:, :, best_slice]) > 0:
            ax.contour(metal_mask[:, :, best_slice].T, colors='red', linewidths=1, linestyles='dashed')
        ax.set_title(f'Axial (z={best_slice})')
        ax.axis('off')

        # Coronal view
        ax = axes[1]
        ax.imshow(ct_data[:, best_coronal, :].T, cmap='gray', origin='lower',
                  vmin=-200, vmax=2000)
        if teeth_mask is not None and np.sum(teeth_mask[:, best_coronal, :]) > 0:
            ax.contour(teeth_mask[:, best_coronal, :].T, colors='cyan', linewidths=1)
        if maxilla_mask is not None and np.sum(maxilla_mask[:, best_coronal, :]) > 0:
            ax.contour(maxilla_mask[:, best_coronal, :].T, colors='yellow', linewidths=0.5)
        if metal_mask is not None and np.sum(metal_mask[:, best_coronal, :]) > 0:
            ax.contour(metal_mask[:, best_coronal, :].T, colors='red', linewidths=1, linestyles='dashed')
        ax.set_title(f'Coronal (y={best_coronal})')
        ax.axis('off')

        # Sagittal view (mid-line)
        mid_sag = ct_data.shape[0] // 2
        ax = axes[2]
        ax.imshow(ct_data[mid_sag, :, :].T, cmap='gray', origin='lower',
                  vmin=-200, vmax=2000)
        if teeth_mask is not None and np.sum(teeth_mask[mid_sag, :, :]) > 0:
            ax.contour(teeth_mask[mid_sag, :, :].T, colors='cyan', linewidths=1)
        if maxilla_mask is not None and np.sum(maxilla_mask[mid_sag, :, :]) > 0:
            ax.contour(maxilla_mask[mid_sag, :, :].T, colors='yellow', linewidths=0.5)
        ax.set_title(f'Sagittal (x={mid_sag})')
        ax.axis('off')

        # Legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='cyan', linewidth=2, label='Teeth'),
            Line2D([0], [0], color='yellow', linewidth=2, label='Maxilla'),
            Line2D([0], [0], color='red', linewidth=2, linestyle='dashed', label='Metal')
        ]
        fig.legend(handles=legend_elements, loc='upper right', fontsize=10)

        plt.suptitle(title, fontsize=12)
        plt.tight_layout()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    except ImportError:
        logging.warning("Matplotlib not available, skipping QC image generation")


def _save_segmentation_outputs(teeth_mask, maxilla_mask, metal_mask, ct_img,
                                output_dir, method_name, qc_metrics, warnings,
                                ct_file, tooth_instances=None):
    """Save segmentation masks and metadata to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)

    save_nifti(teeth_mask.astype(np.uint8), ct_img, output_dir / "teeth_mask.nii.gz")
    save_nifti(maxilla_mask.astype(np.uint8), ct_img, output_dir / "maxilla_mask.nii.gz")

    if metal_mask is not None and np.sum(metal_mask) > 0:
        save_nifti(metal_mask.astype(np.uint8), ct_img, output_dir / "metal_artifact_mask.nii.gz")

    if tooth_instances is not None and np.any(tooth_instances > 0):
        save_nifti(tooth_instances.astype(np.int16), ct_img, output_dir / "tooth_instances.nii.gz")

    method_file = output_dir / "segmentation_method.txt"
    with open(method_file, 'w') as f:
        f.write(f"Method: {method_name}\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"CT file: {ct_file.name}\n")
        for key, val in qc_metrics.items():
            f.write(f"{key}: {val}\n")
        if warnings:
            f.write("Warnings:\n")
            for w in warnings:
                f.write(f"  - {w}\n")


def process_subject_session(subject_id: str, session_id: str, timepoint: str,
                            ct_file: Path, method_override: str = None,
                            force: bool = False, logger: logging.Logger = None) -> dict:
    """
    Process segmentation for a single subject/session.

    Runs BOTH TotalSegmentator and HU-threshold, saving each to its own
    subfolder. The primary output (used by downstream ROI/quant steps) is
    TotalSegmentator if it succeeds, otherwise HU fallback.

    Output layout:
        DerivedData/segmentations/totalsegmentator_teeth/sub-XXX/ses-YYY/
        DerivedData/segmentations/hu_fallback/sub-XXX/ses-YYY/
        DerivedData/segmentations/sub-XXX/ses-YYY/  (primary = best available)
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # Define output directories
    primary_dir = SEGMENTATION_DIR / subject_id / session_id
    ts_dir = TOTALSEG_SEG_DIR / subject_id / session_id
    hu_dir = HU_SEG_DIR / subject_id / session_id

    # Check if already processed
    teeth_output = primary_dir / "teeth_mask.nii.gz"
    if teeth_output.exists() and not force:
        logger.info(f"  {session_id}: Already processed, skipping (use --force to re-run)")
        return {'status': 'skipped', 'output_dir': primary_dir}

    # Load CT
    logger.info(f"  {session_id} ({timepoint}): Loading CT from {ct_file.name}")
    ct_data, ct_img = load_nifti(ct_file)
    voxel_dims = get_voxel_dimensions(ct_img)
    voxel_vol_ml = get_voxel_volume_ml(ct_img)
    logger.info(f"    CT shape: {ct_data.shape}, range: [{ct_data.min():.0f}, {ct_data.max():.0f}] HU")

    # Metal detection (shared)
    metal_mask = detect_metal_artifacts(ct_data, voxel_dims)

    # =========================================================================
    # 1. TotalSegmentator (preferred method)
    # =========================================================================
    ts_result = None
    ts_qc = None
    if method_override is None or method_override == 'TotalSegmentator':
        try:
            logger.info(f"    Running TotalSegmentator teeth task...")
            ts_result = run_total_segmentator_dental(ct_data, ct_img, voxel_dims)
            if ts_result is not None:
                ts_qc = validate_segmentation(
                    ts_result['teeth_mask'], ts_result['maxilla_mask'], voxel_vol_ml
                )
                _save_segmentation_outputs(
                    ts_result['teeth_mask'], ts_result['maxilla_mask'], metal_mask,
                    ct_img, ts_dir, 'TotalSegmentator', ts_qc, [],
                    ct_file, ts_result.get('tooth_instances')
                )

                # Save crop, full multilabel segmentation, and crop coordinates
                if ts_result.get('cropped_img') is not None:
                    nib.save(ts_result['cropped_img'], str(ts_dir / "crop.nii.gz"))
                if ts_result.get('full_seg_data') is not None:
                    save_nifti(ts_result['full_seg_data'].astype(np.int16), ct_img,
                              ts_dir / "totalseg_teeth_multilabel.nii.gz")
                if ts_result.get('crop_slices') is not None:
                    with open(ts_dir / "crop_coords.txt", 'w') as f:
                        for dim, s in zip(['x', 'y', 'z'], ts_result['crop_slices']):
                            f.write(f"{dim}: {s.start}-{s.stop}\n")
                        f.write(f"original_shape: {ct_data.shape}\n")

                logger.info(f"    TotalSegmentator: teeth={ts_qc.get('teeth_volume_ml',0):.1f}mL, "
                           f"maxilla={ts_qc.get('maxilla_volume_ml',0):.1f}mL")

                # QC image for TotalSegmentator
                ts_qc_dir = TOTALSEG_QC_DIR / subject_id / session_id
                ts_qc_dir.mkdir(parents=True, exist_ok=True)
                generate_segmentation_qc_image(
                    ct_data, ts_result['teeth_mask'], ts_result['maxilla_mask'],
                    metal_mask, ts_qc_dir / "totalseg_segmentation.png",
                    f"{subject_id}/{session_id} ({timepoint}) - TotalSegmentator"
                )
            else:
                logger.warning(f"    TotalSegmentator returned None")
        except ImportError:
            logger.info(f"    TotalSegmentator not installed, skipping")
        except Exception as e:
            logger.warning(f"    TotalSegmentator failed: {e}")

    # =========================================================================
    # 2. HU-threshold (always run as secondary/fallback)
    # =========================================================================
    hu_teeth = None
    hu_maxilla = None
    hu_qc = None
    if method_override is None or method_override == 'HU_threshold':
        logger.info(f"    Running HU-threshold segmentation...")
        hu_teeth, hu_maxilla = segment_by_hu_threshold(ct_data, voxel_dims)
        hu_qc = validate_segmentation(hu_teeth, hu_maxilla, voxel_vol_ml)
        _save_segmentation_outputs(
            hu_teeth, hu_maxilla, metal_mask,
            ct_img, hu_dir, 'HU_threshold', hu_qc, [],
            ct_file
        )
        logger.info(f"    HU-threshold: teeth={hu_qc.get('teeth_volume_ml',0):.1f}mL, "
                   f"maxilla={hu_qc.get('maxilla_volume_ml',0):.1f}mL")

        # QC image for HU
        hu_qc_dir = SEG_QC_DIR / subject_id
        hu_qc_dir.mkdir(parents=True, exist_ok=True)
        generate_segmentation_qc_image(
            ct_data, hu_teeth, hu_maxilla,
            metal_mask, hu_qc_dir / f"{session_id}_segmentation.png",
            f"{subject_id}/{session_id} ({timepoint}) - HU threshold"
        )

    # =========================================================================
    # 3. Select primary method and save to main output dir
    # =========================================================================
    primary_method = None
    primary_teeth = None
    primary_maxilla = None
    primary_qc = {}
    primary_warnings = []
    tooth_instances = None

    # Prefer TotalSegmentator
    if ts_result is not None and ts_qc is not None and ts_qc.get('valid', False):
        primary_method = 'TotalSegmentator'
        primary_teeth = ts_result['teeth_mask']
        primary_maxilla = ts_result['maxilla_mask']
        primary_qc = ts_qc
        tooth_instances = ts_result.get('tooth_instances')
    elif hu_teeth is not None and hu_qc is not None:
        primary_method = 'HU_threshold'
        primary_teeth = hu_teeth
        primary_maxilla = hu_maxilla
        primary_qc = hu_qc
        primary_warnings.append("TotalSegmentator unavailable, using HU fallback")
    else:
        logger.error(f"    Both segmentation methods FAILED")
        return {'status': 'failed', 'output_dir': primary_dir, 'error': 'No segmentation produced'}

    _save_segmentation_outputs(
        primary_teeth, primary_maxilla, metal_mask,
        ct_img, primary_dir, primary_method, primary_qc, primary_warnings,
        ct_file, tooth_instances
    )

    logger.info(f"    Primary method: {primary_method}")
    if primary_qc.get('metal_voxels', 0) > 0:
        logger.info(f"    Metal artifacts: {primary_qc['metal_voxels']} voxels")
    if primary_warnings:
        for w in primary_warnings:
            logger.warning(f"    {w}")

    return {
        'status': 'success',
        'output_dir': primary_dir,
        'method': primary_method,
        'qc_metrics': primary_qc,
        'warnings': primary_warnings,
        'ct_file': str(ct_file)
    }


def main():
    """Main segmentation pipeline."""
    parser = argparse.ArgumentParser(description='CT Dental Segmentation Pipeline')
    parser.add_argument('--subject', type=str, help='Process only this subject (e.g., sub-101)')
    parser.add_argument('--method', type=str, choices=['TotalSegmentator', 'HU_threshold'],
                        help='Force specific segmentation method')
    parser.add_argument('--force', action='store_true', help='Re-run even if output exists')
    parser.add_argument('--prefer-bone-ct', action='store_true', default=False,
                        help='Prefer bone reconstruction CT if available (default: False - bone CT often has limited FOV)')
    args = parser.parse_args()

    # Setup
    ensure_directories()
    logger = setup_logging(LOGNOTES_DIR)

    logger.info("=" * 60)
    logger.info("PERIODONTAL ANALYSIS - CT SEGMENTATION")
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
            logger.error(f"Subject {args.subject} not found in {RAWDATA_DIR}")
            sys.exit(1)

    logger.info(f"Processing {len(subjects)} subjects: {', '.join(subjects)}")

    # Track results
    all_results = []
    failed_sessions = []
    subject_comparisons = []

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

            # Get timepoint from blinding key
            key = (subject_id, session_id)
            timepoint = blinding_map.get(key, "Unknown")
            if timepoint == "Unknown":
                logger.warning(f"  {session_id}: Not found in blinding key, skipping")
                continue

            # Find CT file
            ct_file = find_ct_file(session_dir, prefer_bone=args.prefer_bone_ct)
            if ct_file is None:
                logger.warning(f"  {session_id}: No CT file found, skipping")
                failed_sessions.append(f"{subject_id}/{session_id}: No CT file")
                continue

            # Process
            result = process_subject_session(
                subject_id, session_id, timepoint, ct_file,
                method_override=args.method, force=args.force, logger=logger
            )

            result['subject_id'] = subject_id
            result['session_id'] = session_id
            result['timepoint'] = timepoint

            all_results.append(result)
            subject_results[timepoint] = result

            if result['status'] == 'failed':
                failed_sessions.append(f"{subject_id}/{session_id}: {result.get('error', 'Unknown error')}")

        # Compare baseline vs followup for this subject
        if 'Baseline' in subject_results and 'Followup' in subject_results:
            baseline = subject_results['Baseline']
            followup = subject_results['Followup']

            if baseline['status'] == 'success' and followup['status'] == 'success':
                comparison = compare_segmentation_volumes(
                    {'qc_metrics': baseline['qc_metrics']},
                    {'qc_metrics': followup['qc_metrics']},
                    subject_id
                )
                subject_comparisons.append(comparison)

                if not comparison['stable']:
                    logger.warning(f"  Volume stability WARNING for {subject_id}:")
                    for w in comparison['warnings']:
                        logger.warning(f"    {w}")

    # Save volume comparison results
    if subject_comparisons:
        comparison_df = pd.DataFrame(subject_comparisons)
        comparison_file = SEG_QC_DIR / "volume_comparison_all_subjects.csv"
        # Merge with existing data to avoid overwriting when run per-subject
        if comparison_file.exists():
            existing = pd.read_csv(comparison_file)
            processed_subjects = comparison_df['subject_id'].unique()
            existing = existing[~existing['subject_id'].isin(processed_subjects)]
            comparison_df = pd.concat([existing, comparison_df], ignore_index=True)
        comparison_df.to_csv(comparison_file, index=False)
        logger.info(f"\nVolume comparison saved to: {comparison_file}")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    success_count = sum(1 for r in all_results if r['status'] == 'success')
    skip_count = sum(1 for r in all_results if r['status'] == 'skipped')
    fail_count = sum(1 for r in all_results if r['status'] == 'failed')

    logger.info(f"Sessions processed: {success_count}")
    logger.info(f"Sessions skipped (already done): {skip_count}")
    logger.info(f"Sessions failed: {fail_count}")

    if failed_sessions:
        logger.info("\nFailed sessions:")
        for f in failed_sessions:
            logger.info(f"  - {f}")

    # Count unstable subjects
    unstable = [c for c in subject_comparisons if not c['stable']]
    if unstable:
        logger.warning(f"\nSubjects with unstable segmentation volumes: {len(unstable)}")
        for c in unstable:
            logger.warning(f"  - {c['subject_id']}: teeth {c['teeth_pct_change']:.1f}%, maxilla {c['maxilla_pct_change']:.1f}%")

    logger.info("\n" + "=" * 60)
    logger.info(f"Segmentation pipeline completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    # Return exit code based on failures
    if fail_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
