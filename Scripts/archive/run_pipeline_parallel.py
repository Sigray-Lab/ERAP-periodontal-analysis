#!/usr/bin/env python3
"""
run_pipeline_parallel.py - Parallel Pipeline Orchestrator

This script runs the periodontal analysis pipeline with parallel subject processing.
Uses multiprocessing to process subjects concurrently for CPU-intensive steps.

Usage:
    cd Periodontal_Analysis/Scripts

    # Run full pipeline with parallelization (auto-detect cores)
    python run_pipeline_parallel.py

    # Specify number of parallel workers
    python run_pipeline_parallel.py --workers 4

    # Run specific steps
    python run_pipeline_parallel.py --steps 1 2 5

    # Force re-run all
    python run_pipeline_parallel.py --force

Pipeline Steps:
    1. Segmentation (01_segmentation.py) - PARALLEL
    2. ROI Generation (02_roi_generation.py) - PARALLEL
    3. Registration (03_registration.py) - [NOT YET IMPLEMENTED]
    4. Plasma Processing (04_plasma_processing.py) - PARALLEL
    5. PET Quantification (05_pet_quantification.py) - PARALLEL
    6. STIR Quantification (06_stir_quantification.py) - [NOT YET IMPLEMENTED]
    7. Validation & QC (07_validation_qc.py) - SEQUENTIAL (aggregation)

Output:
    LogNotes/pipeline_parallel_<timestamp>.log
"""

import argparse
import subprocess
import sys
import os
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Tuple
from multiprocessing import Pool, cpu_count
from functools import partial

# Add Scripts directory to path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

from config import LOGNOTES_DIR, RAWDATA_DIR, ensure_directories
from utils.io_utils import discover_subjects


# Pipeline step definitions
PIPELINE_STEPS = {
    1: {
        'name': 'Segmentation',
        'script': '01_segmentation.py',
        'description': 'CT dental structure segmentation',
        'implemented': True,
        'parallelizable': False,  # SEQUENTIAL - TotalSegmentator uses too much RAM
        'memory_intensive': True
    },
    2: {
        'name': 'ROI Generation',
        'script': '02_roi_generation.py',
        'description': 'Generate peridental soft tissue ROIs',
        'implemented': True,
        'parallelizable': True
    },
    3: {
        'name': 'Registration',
        'script': '03_registration.py',
        'description': 'Multi-modal image registration',
        'implemented': False,
        'parallelizable': True
    },
    4: {
        'name': 'Plasma Processing',
        'script': '04_plasma_processing.py',
        'description': 'Process input functions for TPR/FUR',
        'implemented': True,
        'parallelizable': True
    },
    5: {
        'name': 'PET Quantification',
        'script': '05_pet_quantification.py',
        'description': 'Extract SUV, TPR, FUR metrics',
        'implemented': True,
        'parallelizable': True
    },
    6: {
        'name': 'STIR Quantification',
        'script': '06_stir_quantification.py',
        'description': 'Extract STIR z-scores',
        'implemented': False,
        'parallelizable': True
    },
    7: {
        'name': 'Validation & QC',
        'script': '07_validation_qc.py',
        'description': 'Aggregate QC flags and validate biomarkers',
        'implemented': True,
        'parallelizable': False  # Needs all data aggregated
    }
}


def setup_logging(log_dir: Path) -> logging.Logger:
    """Setup logging to file and console."""
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"pipeline_parallel_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Pipeline log file: {log_file}")
    return logger


def run_step_for_subject(args: Tuple[int, str, bool]) -> Tuple[str, bool, str]:
    """
    Run a pipeline step for a single subject.

    Args:
        args: Tuple of (step_num, subject_id, force)

    Returns:
        Tuple of (subject_id, success, error_message)
    """
    step_num, subject_id, force = args
    step = PIPELINE_STEPS[step_num]
    script_path = script_dir / step['script']

    cmd = [sys.executable, str(script_path), '--subject', subject_id]
    if force:
        cmd.append('--force')

    try:
        result = subprocess.run(
            cmd,
            cwd=script_dir,
            capture_output=True,
            text=True,
            timeout=1800  # 30 minute timeout per subject
        )

        if result.returncode != 0:
            return (subject_id, False, f"Exit code {result.returncode}")
        return (subject_id, True, "")

    except subprocess.TimeoutExpired:
        return (subject_id, False, "Timeout (30 min)")
    except Exception as e:
        return (subject_id, False, str(e))


def run_step_parallel(step_num: int, subjects: List[str], force: bool,
                      n_workers: int, logger: logging.Logger) -> Tuple[List[str], List[str]]:
    """
    Run a pipeline step for all subjects in parallel.

    Args:
        step_num: Step number
        subjects: List of subject IDs
        force: Force re-run
        n_workers: Number of parallel workers
        logger: Logger instance

    Returns:
        Tuple of (successful_subjects, failed_subjects)
    """
    step = PIPELINE_STEPS[step_num]

    logger.info(f"\n{'='*60}")
    logger.info(f"STEP {step_num}: {step['name']} (PARALLEL)")
    logger.info(f"Description: {step['description']}")
    logger.info(f"Subjects: {len(subjects)}")
    logger.info(f"Workers: {n_workers}")
    logger.info(f"{'='*60}")

    if not step['implemented']:
        logger.warning(f"Step {step_num} is not yet implemented - skipping")
        return subjects, []

    # Create argument tuples for each subject
    work_args = [(step_num, subj, force) for subj in subjects]

    # Run in parallel
    successful = []
    failed = []

    with Pool(processes=n_workers) as pool:
        results = pool.map(run_step_for_subject, work_args)

    for subject_id, success, error_msg in results:
        if success:
            successful.append(subject_id)
            logger.info(f"  {subject_id}: SUCCESS")
        else:
            failed.append(subject_id)
            logger.error(f"  {subject_id}: FAILED - {error_msg}")

    logger.info(f"\nStep {step_num} completed: {len(successful)} success, {len(failed)} failed")

    return successful, failed


def run_step_sequential(step_num: int, force: bool, logger: logging.Logger,
                        subjects: List[str] = None) -> Tuple[List[str], List[str]]:
    """
    Run a pipeline step sequentially (for aggregation steps or memory-intensive steps).

    Args:
        step_num: Step number
        force: Force re-run
        logger: Logger instance
        subjects: Optional list of subjects (for per-subject sequential runs)

    Returns:
        Tuple of (successful_subjects, failed_subjects) or ([], []) for aggregation
    """
    step = PIPELINE_STEPS[step_num]
    script_path = script_dir / step['script']

    logger.info(f"\n{'='*60}")
    logger.info(f"STEP {step_num}: {step['name']} (SEQUENTIAL)")
    logger.info(f"Description: {step['description']}")
    if step.get('memory_intensive'):
        logger.info(f"Note: Running sequentially due to high memory usage")
    logger.info(f"{'='*60}")

    if not step['implemented']:
        logger.warning(f"Step {step_num} is not yet implemented - skipping")
        return (subjects or [], [])

    # If subjects provided, run per-subject sequentially
    if subjects and step.get('memory_intensive'):
        successful = []
        failed = []

        for i, subject_id in enumerate(subjects, 1):
            logger.info(f"\n  [{i}/{len(subjects)}] Processing {subject_id}...")

            cmd = [sys.executable, str(script_path), '--subject', subject_id]
            if force:
                cmd.append('--force')

            try:
                result = subprocess.run(
                    cmd,
                    cwd=script_dir,
                    capture_output=True,
                    text=True,
                    timeout=1800  # 30 min timeout
                )

                if result.returncode == 0:
                    successful.append(subject_id)
                    logger.info(f"    {subject_id}: SUCCESS")
                else:
                    failed.append(subject_id)
                    logger.error(f"    {subject_id}: FAILED (exit code {result.returncode})")

            except subprocess.TimeoutExpired:
                failed.append(subject_id)
                logger.error(f"    {subject_id}: FAILED (timeout)")
            except Exception as e:
                failed.append(subject_id)
                logger.error(f"    {subject_id}: FAILED ({e})")

        logger.info(f"\nStep {step_num} completed: {len(successful)} success, {len(failed)} failed")
        return (successful, failed)

    # Otherwise run as aggregation (no per-subject)
    cmd = [sys.executable, str(script_path)]
    if force:
        cmd.append('--force')

    try:
        result = subprocess.run(
            cmd,
            cwd=script_dir,
            capture_output=False,
            text=True
        )

        if result.returncode != 0:
            logger.error(f"Step {step_num} failed with return code {result.returncode}")
            return ([], ['aggregation'])

        logger.info(f"Step {step_num} completed successfully")
        return ([], [])

    except Exception as e:
        logger.error(f"Error running step {step_num}: {e}")
        return ([], ['aggregation'])


def main():
    """Main parallel pipeline orchestrator."""
    parser = argparse.ArgumentParser(
        description='Parallel Periodontal Analysis Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_pipeline_parallel.py                    # Run all with auto workers
    python run_pipeline_parallel.py --workers 4       # Use 4 parallel workers
    python run_pipeline_parallel.py --steps 1 2 5     # Run specific steps
    python run_pipeline_parallel.py --force           # Force re-run all
        """
    )

    parser.add_argument(
        '--steps',
        type=int,
        nargs='+',
        choices=list(PIPELINE_STEPS.keys()),
        help='Specific steps to run (default: all implemented)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=None,
        help='Number of parallel workers (default: CPU count - 2)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-run even if outputs exist'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all pipeline steps and exit'
    )
    parser.add_argument(
        '--subjects',
        type=str,
        nargs='+',
        help='Process only these subjects (e.g., --subjects sub-101 sub-102)'
    )

    args = parser.parse_args()

    # Just list steps and exit
    if args.list:
        print("\n" + "="*70)
        print("PERIODONTAL ANALYSIS PIPELINE (PARALLEL)")
        print("="*70)
        print("\nAvailable steps:\n")
        for step_num, step in PIPELINE_STEPS.items():
            status = "✓" if step['implemented'] else "✗"
            parallel = "PARALLEL" if step.get('parallelizable', False) else "SEQ"
            impl_str = "" if step['implemented'] else " [NOT IMPLEMENTED]"
            print(f"  {step_num}. [{status}] {step['name']} ({parallel}): {step['description']}{impl_str}")
        print("\n" + "="*70)
        sys.exit(0)

    # Setup
    ensure_directories()
    logger = setup_logging(LOGNOTES_DIR)

    # Determine number of workers
    n_workers = args.workers
    if n_workers is None:
        # Conservative default: 6 workers to avoid system freeze
        # User can override with --workers flag
        n_workers = min(6, max(1, cpu_count() - 4))

    # Discover subjects
    if args.subjects:
        # Use specified subjects
        all_subjects = discover_subjects()
        subjects = [s for s in args.subjects if s in all_subjects]
        if len(subjects) != len(args.subjects):
            missing = set(args.subjects) - set(subjects)
            logger.warning(f"Subjects not found: {missing}")
    else:
        subjects = discover_subjects()

    # Determine which steps to run
    if args.steps:
        steps_to_run = args.steps
    else:
        steps_to_run = [n for n, s in PIPELINE_STEPS.items() if s['implemented']]

    # Pipeline header
    logger.info("\n" + "="*70)
    logger.info("PERIODONTAL ANALYSIS PIPELINE (PARALLEL)")
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*70)

    logger.info(f"\nSubjects: {len(subjects)}")
    logger.info(f"Workers: {n_workers}")
    logger.info(f"Steps to run: {steps_to_run}")
    if args.force:
        logger.info("Force mode: ON")

    # Track results
    all_failed = {}

    # Run pipeline steps
    for step_num in steps_to_run:
        step = PIPELINE_STEPS[step_num]

        if step.get('parallelizable', False):
            # Run in parallel
            successful, failed = run_step_parallel(
                step_num, subjects, args.force, n_workers, logger
            )
            if failed:
                all_failed[step_num] = failed
        else:
            # Run sequentially (memory-intensive or aggregation steps)
            successful, failed = run_step_sequential(
                step_num, args.force, logger, subjects if step.get('memory_intensive') else None
            )
            if failed:
                all_failed[step_num] = failed

    # Summary
    logger.info("\n" + "="*70)
    logger.info("PIPELINE SUMMARY")
    logger.info("="*70)

    for step_num in steps_to_run:
        step = PIPELINE_STEPS[step_num]
        if step_num in all_failed:
            failed_count = len(all_failed[step_num])
            logger.info(f"  Step {step_num} ({step['name']}): {failed_count} FAILED")
        else:
            logger.info(f"  Step {step_num} ({step['name']}): SUCCESS")

    if all_failed:
        logger.error(f"\nFailed subjects by step:")
        for step_num, failed_subjs in all_failed.items():
            logger.error(f"  Step {step_num}: {failed_subjs}")
    else:
        logger.info("\nAll steps completed successfully!")

    logger.info(f"\nPipeline finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*70)

    # Exit with error code if any step failed
    sys.exit(1 if all_failed else 0)


if __name__ == "__main__":
    main()
