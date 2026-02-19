#!/usr/bin/env python3
"""
04_plasma_processing.py - Input Function Processing Script

This script processes combined IDIF + plasma input function data for all sessions.

Usage:
    cd Periodontal_Analysis/Scripts
    python 04_plasma_processing.py

    # Or with options:
    python 04_plasma_processing.py --subject sub-101
    python 04_plasma_processing.py --force

Output:
    DerivedData/input_functions/sub-XXX_ses-{Timepoint}_if_processed.csv
    DerivedData/session_info.csv
    QC/plasma/*.png
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
    RAWDATA_DIR, INPUT_FUNC_DIR, PLASMA_QC_DIR, DERIVED_DIR, LOGNOTES_DIR,
    DEFAULT_SCAN_START_S, DEFAULT_SCAN_DURATION_S, TISSUE_TIME_S,
    ensure_directories
)
from utils.io_utils import (
    load_blinding_key, load_ecrf_data, get_suv_parameters,
    discover_subjects, find_input_function_file, find_pet_json
)
from utils.plasma_utils import (
    load_input_function, process_input_function,
    validate_input_function, generate_if_qc_plot
)
from utils.pet_utils import load_pet_json


def setup_logging(log_dir: Path) -> logging.Logger:
    """Setup logging to file and console."""
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"plasma_processing_log_{timestamp}.txt"

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


def process_session(subject_id: str, timepoint: str,
                    suv_params: dict, logger: logging.Logger,
                    force: bool = False) -> dict:
    """
    Process input function for a single session.

    Args:
        subject_id: Subject ID
        timepoint: 'Baseline' or 'Followup'
        suv_params: SUV parameters from eCRF
        logger: Logger instance
        force: Re-run even if output exists

    Returns:
        Dictionary with processed data
    """
    result = {
        'status': 'failed',
        'subject_id': subject_id,
        'timepoint': timepoint,
        'weight_kg': suv_params.get('weight_kg'),
        'dose_mbq': suv_params.get('injected_mbq'),
        'plasma_mean_Bq_mL': np.nan,
        'plasma_auc_window_Bq_s_mL': np.nan,
        'plasma_auc_0_to_T_Bq_s_mL': np.nan,
        'scan_start_s': DEFAULT_SCAN_START_S,
        'scan_end_s': DEFAULT_SCAN_START_S + DEFAULT_SCAN_DURATION_S,
        'n_idif_samples': 0,
        'n_plasma_samples': 0,
        'if_quality': 'unknown',
        'warnings': []
    }

    # Check for output file
    output_file = INPUT_FUNC_DIR / f"{subject_id}_ses-{timepoint}_if_processed.csv"
    if output_file.exists() and not force:
        logger.info(f"  {timepoint}: Already processed, skipping")
        result['status'] = 'skipped'
        return result

    # Find input function file
    if_file = find_input_function_file(subject_id, timepoint)
    if if_file is None:
        logger.warning(f"  {timepoint}: No input function file found")
        result['warnings'].append("Input function file not found")
        return result

    logger.info(f"  {timepoint}: Loading {if_file.name}")

    # Load input function
    if_data = load_input_function(if_file)
    result['n_idif_samples'] = if_data['n_idif_samples']
    result['n_plasma_samples'] = if_data['n_plasma_samples']
    result['warnings'].extend(if_data['warnings'])

    # Validate
    validation = validate_input_function(if_data)
    result['if_quality'] = validation['quality']

    if not validation['valid']:
        logger.warning(f"    Input function validation failed: {validation['issues']}")
        result['warnings'].extend(validation['issues'])
        return result

    logger.info(f"    IDIF samples: {if_data['n_idif_samples']}, Plasma samples: {if_data['n_plasma_samples']}")

    # Get scan timing from PET JSON if available
    scan_start_s = DEFAULT_SCAN_START_S
    scan_duration_s = DEFAULT_SCAN_DURATION_S

    # Try to find PET JSON for timing info
    # (This requires finding the session directory, which we may not have here)
    # For now, use defaults - the actual timing will be obtained in quantification step

    # Process input function
    processed = process_input_function(
        if_data,
        scan_start_s=scan_start_s,
        scan_end_s=scan_start_s + scan_duration_s,
        tissue_time_s=TISSUE_TIME_S
    )

    result['plasma_mean_Bq_mL'] = processed['plasma_mean_Bq_mL']
    result['plasma_auc_window_Bq_s_mL'] = processed['plasma_auc_window_Bq_s_mL']
    result['plasma_auc_0_to_T_Bq_s_mL'] = processed['plasma_auc_0_to_T_Bq_s_mL']
    result['scan_start_s'] = processed['scan_start_s']
    result['scan_end_s'] = processed['scan_end_s']
    result['n_samples_in_window'] = processed['n_samples_in_window']
    result['warnings'].extend(processed['warnings'])

    logger.info(f"    Plasma mean: {processed['plasma_mean_Bq_mL']:.0f} Bq/mL")
    logger.info(f"    Plasma AUC (window): {processed['plasma_auc_window_Bq_s_mL']:.0f} Bq*s/mL")
    logger.info(f"    Plasma AUC (0-T): {processed['plasma_auc_0_to_T_Bq_s_mL']:.0f} Bq*s/mL")

    if processed['warnings']:
        for w in processed['warnings']:
            logger.warning(f"    {w}")

    # Save processed data
    INPUT_FUNC_DIR.mkdir(parents=True, exist_ok=True)

    # Save detailed CSV
    processed_df = pd.DataFrame({
        'time_s': if_data['combined_times'],
        'activity_Bq_mL': if_data['combined_activities']
    })
    processed_df.to_csv(output_file, index=False)
    logger.info(f"    Saved to {output_file}")

    # Generate QC plot
    qc_dir = PLASMA_QC_DIR / subject_id
    qc_plot_path = qc_dir / f"{timepoint}_input_function.png"
    title = f"{subject_id} / {timepoint} - Input Function"
    generate_if_qc_plot(if_data, processed, qc_plot_path, title)
    logger.info(f"    QC plot saved to {qc_plot_path}")

    result['status'] = 'success'
    return result


def main():
    """Main input function processing pipeline."""
    parser = argparse.ArgumentParser(description='Input Function Processing Pipeline')
    parser.add_argument('--subject', type=str, help='Process only this subject')
    parser.add_argument('--force', action='store_true', help='Re-run even if output exists')
    args = parser.parse_args()

    # Setup
    ensure_directories()
    logger = setup_logging(LOGNOTES_DIR)

    logger.info("=" * 60)
    logger.info("PERIODONTAL ANALYSIS - INPUT FUNCTION PROCESSING")
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    # Load blinding key
    try:
        blinding_map = load_blinding_key()
        logger.info(f"Loaded blinding key with {len(blinding_map)} entries")
    except FileNotFoundError as e:
        logger.error(f"ERROR: {e}")
        sys.exit(1)

    # Load eCRF data
    try:
        ecrf_df = load_ecrf_data()
        logger.info(f"Loaded eCRF data")
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
    session_info = []

    # Process each subject
    for subject_id in subjects:
        logger.info(f"\n{'='*40}")
        logger.info(f"SUBJECT: {subject_id}")
        logger.info(f"{'='*40}")

        for timepoint in ['Baseline', 'Followup']:
            # Get SUV parameters
            suv_params = get_suv_parameters(ecrf_df, subject_id, timepoint)

            if suv_params.get('weight_kg') is None or suv_params.get('injected_mbq') is None:
                logger.warning(f"  {timepoint}: Missing SUV parameters, skipping")
                continue

            # Process
            result = process_session(
                subject_id, timepoint, suv_params, logger, force=args.force
            )

            all_results.append(result)

            # Add to session info
            if result['status'] == 'success':
                session_info.append({
                    'subject_id': subject_id,
                    'timepoint': timepoint,
                    'weight_kg': result['weight_kg'],
                    'dose_mbq': result['dose_mbq'],
                    'plasma_mean_Bq_mL': result['plasma_mean_Bq_mL'],
                    'plasma_auc_window_Bq_s_mL': result['plasma_auc_window_Bq_s_mL'],
                    'plasma_auc_0_to_T_Bq_s_mL': result['plasma_auc_0_to_T_Bq_s_mL'],
                    'scan_start_s': result['scan_start_s'],
                    'scan_end_s': result['scan_end_s'],
                    'n_idif_samples': result['n_idif_samples'],
                    'n_plasma_samples': result['n_plasma_samples'],
                    'if_quality': result['if_quality']
                })

    # Save session info summary
    if session_info:
        session_df = pd.DataFrame(session_info)
        session_file = DERIVED_DIR / "session_info.csv"
        session_df.to_csv(session_file, index=False)
        logger.info(f"\nSession info saved to: {session_file}")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    success_count = sum(1 for r in all_results if r['status'] == 'success')
    skip_count = sum(1 for r in all_results if r['status'] == 'skipped')
    fail_count = sum(1 for r in all_results if r['status'] == 'failed')

    logger.info(f"Sessions processed: {success_count}")
    logger.info(f"Sessions skipped: {skip_count}")
    logger.info(f"Sessions failed: {fail_count}")

    # Quality summary
    if session_info:
        quality_counts = pd.DataFrame(session_info)['if_quality'].value_counts()
        logger.info(f"\nInput function quality:")
        for quality, count in quality_counts.items():
            logger.info(f"  {quality}: {count}")

    logger.info("\n" + "=" * 60)
    logger.info(f"Plasma processing completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
