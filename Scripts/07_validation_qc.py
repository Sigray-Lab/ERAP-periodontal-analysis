#!/usr/bin/env python3
"""
07_validation_qc.py - Validation and QC Aggregation Script

This script:
1. Aggregates all QC flags from previous steps
2. Validates biomarkers at baseline (healthy vs unhealthy)
3. Generates comprehensive QC reports

Usage:
    cd Periodontal_Analysis/Scripts
    python 07_validation_qc.py

Prerequisites:
    - 05_pet_quantification.py (for metrics)

Output:
    QC/QC_flags_report.csv
    QC/QC_summary_report.txt
    Outputs/validation_correlations.csv
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
    OUTPUTS_DIR, QC_DIR, LOGNOTES_DIR, DERIVED_DIR,
    TOOTH_METRICS_FILE, VALIDATION_FILE,
    ensure_directories
)
from utils.io_utils import load_clinical_ratings
from utils.qc_utils import (
    validate_baseline_biomarkers, save_qc_reports,
    check_volume_stability, create_qc_flag
)


def setup_logging(log_dir: Path) -> logging.Logger:
    """Setup logging to file and console."""
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"validation_qc_log_{timestamp}.txt"

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


def load_existing_qc_data() -> list:
    """Load QC flags from previous pipeline steps."""
    all_flags = []

    # Check for segmentation QC
    seg_qc_dir = QC_DIR / "HU_segmentation"
    if seg_qc_dir.exists():
        vol_comparison = seg_qc_dir / "volume_comparison_all_subjects.csv"
        if vol_comparison.exists():
            df = pd.read_csv(vol_comparison)
            for _, row in df.iterrows():
                if not row.get('stable', True):
                    all_flags.append(create_qc_flag(
                        row['subject_id'],
                        'Baseline vs Followup',
                        'roi_volume_unstable',
                        value=f"teeth: {row.get('teeth_pct_change', 0):.1f}%",
                        details='Segmentation volume changed significantly'
                    ))

    # Check for ROI QC
    roi_qc_dir = QC_DIR / "roi"
    if roi_qc_dir.exists():
        hu_rejection = roi_qc_dir / "hu_rejection_rates.csv"
        if hu_rejection.exists():
            df = pd.read_csv(hu_rejection)
            for _, row in df.iterrows():
                if row.get('hu_rejection_rate', 0) > 0.5:
                    all_flags.append(create_qc_flag(
                        row['subject_id'],
                        row['session_id'],
                        'hu_rejection_high',
                        value=f"{row['hu_rejection_rate']*100:.1f}%"
                    ))

    return all_flags


def check_tooth_volume_stability(metrics_df: pd.DataFrame) -> list:
    """
    Check per-tooth volume stability between timepoints.

    Returns list of QC flags for unstable teeth.
    """
    flags = []

    # Need ROI volume column
    if 'roi_volume_mL' not in metrics_df.columns:
        return flags

    # Group by subject and tooth
    for subject_id in metrics_df['subject_id'].unique():
        subject_data = metrics_df[metrics_df['subject_id'] == subject_id]

        for tooth_id in subject_data['tooth_id'].unique():
            tooth_data = subject_data[subject_data['tooth_id'] == tooth_id]

            baseline = tooth_data[tooth_data['timepoint'] == 'Baseline']
            followup = tooth_data[tooth_data['timepoint'] == 'Followup']

            if len(baseline) == 0 or len(followup) == 0:
                continue

            baseline_vol = baseline['roi_volume_mL'].values[0]
            followup_vol = followup['roi_volume_mL'].values[0]

            flag = check_volume_stability(
                baseline_vol, followup_vol, subject_id, f"Tooth {tooth_id}"
            )

            if flag:
                flags.append(flag)

    return flags


def check_suv_outliers(metrics_df: pd.DataFrame) -> list:
    """Check for SUV outliers in the data."""
    flags = []

    if 'SUV_mean' not in metrics_df.columns:
        return flags

    for _, row in metrics_df.iterrows():
        suv = row.get('SUV_mean', np.nan)
        if np.isnan(suv):
            continue

        if suv < 0.1:
            flags.append(create_qc_flag(
                row['subject_id'],
                row.get('session_id', row.get('timepoint', '')),
                'suv_outlier',
                value=f"{suv:.3f}",
                details=f"Tooth {row.get('tooth_id', '?')}: SUV very low"
            ))
        elif suv > 20:
            flags.append(create_qc_flag(
                row['subject_id'],
                row.get('session_id', row.get('timepoint', '')),
                'suv_outlier',
                value=f"{suv:.3f}",
                details=f"Tooth {row.get('tooth_id', '?')}: SUV very high"
            ))

    return flags


def main():
    """Main validation and QC pipeline."""
    parser = argparse.ArgumentParser(description='Validation and QC Pipeline')
    parser.add_argument('--subject', type=str, help='Process only this subject (unused, for pipeline compatibility)')
    parser.add_argument('--force', action='store_true', help='Force re-run (unused, for pipeline compatibility)')
    args = parser.parse_args()

    # Setup
    ensure_directories()
    logger = setup_logging(LOGNOTES_DIR)

    logger.info("=" * 60)
    logger.info("PERIODONTAL ANALYSIS - VALIDATION AND QC")
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    # Collect all QC flags
    all_flags = []

    # Load existing QC data from previous steps
    logger.info("\nLoading QC data from previous steps...")
    existing_flags = load_existing_qc_data()
    all_flags.extend(existing_flags)
    logger.info(f"  Loaded {len(existing_flags)} existing flags")

    # Load metrics
    tooth_metrics_file = OUTPUTS_DIR / TOOTH_METRICS_FILE
    metrics_df = None

    if tooth_metrics_file.exists():
        metrics_df = pd.read_csv(tooth_metrics_file)
        logger.info(f"Loaded tooth metrics: {len(metrics_df)} rows")

        # Check SUV outliers
        logger.info("\nChecking for SUV outliers...")
        suv_flags = check_suv_outliers(metrics_df)
        all_flags.extend(suv_flags)
        logger.info(f"  Found {len(suv_flags)} SUV outlier flags")

        # Check tooth volume stability
        logger.info("\nChecking tooth volume stability...")
        vol_flags = check_tooth_volume_stability(metrics_df)
        all_flags.extend(vol_flags)
        logger.info(f"  Found {len(vol_flags)} volume stability flags")

    else:
        logger.warning(f"Tooth metrics file not found: {tooth_metrics_file}")
        logger.warning("Run 05_pet_quantification.py first")

    # Load clinical ratings
    try:
        clinical_df = load_clinical_ratings()
        logger.info(f"Loaded clinical ratings: {len(clinical_df)} entries")
    except FileNotFoundError as e:
        logger.warning(f"Clinical ratings not found: {e}")
        clinical_df = None

    # Baseline validation
    validation_result = None

    if metrics_df is not None and clinical_df is not None:
        logger.info("\n" + "=" * 40)
        logger.info("BASELINE VALIDATION")
        logger.info("=" * 40)

        validation_result = validate_baseline_biomarkers(metrics_df, clinical_df)

        if validation_result['valid']:
            logger.info("Baseline validation: PASSED")
        else:
            logger.warning("Baseline validation: FAILED")
            all_flags.append(create_qc_flag(
                'ALL', 'Baseline', 'validation_failed',
                details='Biomarkers do not distinguish healthy/unhealthy teeth'
            ))

        # Log detailed results
        for m in validation_result.get('metrics_tested', []):
            status = "PASS" if m['passed'] else "FAIL"
            logger.info(f"  {m['metric']}: {status}")
            logger.info(f"    Healthy: {m['healthy_mean']:.3f} (n={m['healthy_n']})")
            logger.info(f"    Unhealthy: {m['unhealthy_mean']:.3f} (n={m['unhealthy_n']})")
            logger.info(f"    Cohen's d: {m['cohens_d']:.2f}, p={m['mann_whitney_p']:.3f}")

        if validation_result['warnings']:
            for w in validation_result['warnings']:
                logger.warning(f"  {w}")

        # Save validation results
        if validation_result['metrics_tested']:
            val_df = pd.DataFrame(validation_result['metrics_tested'])
            val_file = OUTPUTS_DIR / VALIDATION_FILE
            val_df.to_csv(val_file, index=False)
            logger.info(f"\nValidation results saved to: {val_file}")

    # Generate QC reports
    logger.info("\n" + "=" * 40)
    logger.info("GENERATING QC REPORTS")
    logger.info("=" * 40)

    flags_path, summary_path = save_qc_reports(
        all_flags, QC_DIR, metrics_df, validation_result
    )

    logger.info(f"QC flags saved to: {flags_path}")
    logger.info(f"QC summary saved to: {summary_path}")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("QC SUMMARY")
    logger.info("=" * 60)

    if all_flags:
        flags_df = pd.DataFrame(all_flags)
        severity_counts = flags_df['severity'].value_counts()

        logger.info(f"Total QC flags: {len(all_flags)}")
        for severity, count in severity_counts.items():
            logger.info(f"  {severity}: {count}")
    else:
        logger.info("No QC flags raised - all data passed checks!")

    logger.info("\n" + "=" * 60)
    logger.info(f"Validation/QC completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
