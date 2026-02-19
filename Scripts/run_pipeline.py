#!/usr/bin/env python3
"""
run_pipeline.py - Master Pipeline Orchestrator

Runs the ERAP Periodontal Analysis pipeline from RawData to statistical outputs.

Usage:
    cd Periodontal_Analysis/Scripts

    # Run full pipeline (skips existing outputs)
    python run_pipeline.py

    # Run specific steps
    python run_pipeline.py --steps 1 2 3

    # Run for specific subject
    python run_pipeline.py --subject sub-101

    # Force re-run (ignore existing outputs)
    python run_pipeline.py --force

    # Dry run (show what would be executed)
    python run_pipeline.py --dry-run

Pipeline Steps:
    1. Input Functions    (01_process_input_functions.py)
    2. Geometry Pipeline  (02_run_geometry_pipeline.py)
    3. Tongue Exclusion   (03_create_tongue_exclusion.py)
    4. Batch Quantify     (04_batch_quantify.py)
    5. Statistics         (05_statistical_analysis.py) - runs all trimming levels

Output:
    LogNotes/pipeline_run_<timestamp>.log
"""

import argparse
import subprocess
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

# Add Scripts directory to path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

from config import LOGNOTES_DIR, ensure_directories


# Pipeline step definitions - UPDATED for geometry-based pipeline
PIPELINE_STEPS = {
    1: {
        'name': 'Input Functions',
        'script': '01_process_input_functions.py',
        'description': 'Process aorta IDIF + plasma samples → interpolated input function',
        'implemented': True,
        'supports_subject': True,
        'supports_force': True
    },
    2: {
        'name': 'Geometry Pipeline',
        'script': '02_run_geometry_pipeline.py',
        'description': 'TotalSeg teeth → geometry ROIs → tongue mask → CT-PET registration',
        'implemented': True,
        'supports_subject': True,
        'supports_force': True
    },
    3: {
        'name': 'Tongue Exclusion',
        'script': '03_create_tongue_exclusion.py',
        'description': 'Create trimmed ROIs (0mm original + 3mm, 5mm, 8mm, 10mm tongue exclusion)',
        'implemented': True,
        'supports_subject': True,
        'supports_force': True
    },
    4: {
        'name': 'Batch Quantify',
        'script': '04_batch_quantify.py',
        'description': 'Extract SUV, TPR, FUR metrics for all teeth and trimming levels',
        'implemented': True,
        'supports_subject': False,
        'supports_force': True
    },
    5: {
        'name': 'Statistical Analysis',
        'script': '05_statistical_analysis.py',
        'description': 'Run paired t-tests and LMM for all trimming levels',
        'implemented': True,
        'supports_subject': False,
        'supports_force': False,
        'multi_run': ['0mm', '3mm', '5mm', '8mm', '10mm']  # Run for each trimming level
    }
}


def setup_logging(log_dir: Path) -> logging.Logger:
    """Setup logging to file and console."""
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"pipeline_run_{timestamp}.log"

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


def run_step(step_num: int, subject: Optional[str] = None,
             force: bool = False, dry_run: bool = False,
             logger: logging.Logger = None) -> bool:
    """
    Run a single pipeline step.

    Args:
        step_num: Step number (1-5)
        subject: Optional subject ID to process
        force: Force re-run even if outputs exist
        dry_run: Print command without executing
        logger: Logger instance

    Returns:
        True if successful, False otherwise
    """
    if step_num not in PIPELINE_STEPS:
        logger.error(f"Invalid step number: {step_num}")
        return False

    step = PIPELINE_STEPS[step_num]

    if not step['implemented']:
        logger.warning(f"Step {step_num} ({step['name']}) is not yet implemented - skipping")
        return True

    script_path = script_dir / step['script']

    if not script_path.exists():
        logger.error(f"Script not found: {script_path}")
        return False

    # Handle multi-run steps (like statistics for each trimming level)
    if 'multi_run' in step:
        all_success = True
        for variant in step['multi_run']:
            cmd = [sys.executable, str(script_path), '--trimming', variant]

            logger.info(f"\n{'='*60}")
            logger.info(f"STEP {step_num}: {step['name']} (trimming={variant})")
            logger.info(f"Command: {' '.join(cmd)}")
            logger.info(f"{'='*60}")

            if dry_run:
                logger.info("[DRY RUN] Would execute above command")
                continue

            try:
                result = subprocess.run(cmd, cwd=script_dir, capture_output=False, text=True)
                if result.returncode != 0:
                    logger.error(f"Step {step_num} (trimming={variant}) failed")
                    all_success = False
            except Exception as e:
                logger.error(f"Error running step {step_num} (trimming={variant}): {e}")
                all_success = False

        return all_success

    # Standard single-run step
    cmd = [sys.executable, str(script_path)]

    if subject and step.get('supports_subject', False):
        cmd.extend(['--subject', subject])

    if force and step.get('supports_force', False):
        cmd.append('--force')

    logger.info(f"\n{'='*60}")
    logger.info(f"STEP {step_num}: {step['name']}")
    logger.info(f"Description: {step['description']}")
    logger.info(f"Script: {step['script']}")
    logger.info(f"Command: {' '.join(cmd)}")
    logger.info(f"{'='*60}")

    if dry_run:
        logger.info("[DRY RUN] Would execute above command")
        return True

    try:
        result = subprocess.run(
            cmd,
            cwd=script_dir,
            capture_output=False,
            text=True
        )

        if result.returncode != 0:
            logger.error(f"Step {step_num} failed with return code {result.returncode}")
            return False

        logger.info(f"Step {step_num} completed successfully")
        return True

    except Exception as e:
        logger.error(f"Error running step {step_num}: {e}")
        return False


def print_pipeline_overview():
    """Print overview of all pipeline steps."""
    print("\n" + "="*70)
    print("ERAP PERIODONTAL ANALYSIS PIPELINE")
    print("="*70)
    print("\nPipeline steps:\n")

    for step_num, step in PIPELINE_STEPS.items():
        status = "✓" if step['implemented'] else "✗"
        impl_str = "" if step['implemented'] else " [NOT IMPLEMENTED]"
        print(f"  {step_num}. [{status}] {step['name']}")
        print(f"       {step['description']}{impl_str}")
        if 'multi_run' in step:
            print(f"       (runs for: {', '.join(step['multi_run'])})")
        print()

    print("="*70)
    print("\nData flow:")
    print("  RawData/ → 01 → 02 → 03 → 04 → 05 → Outputs/")
    print("="*70)


def main():
    """Main pipeline orchestrator."""
    parser = argparse.ArgumentParser(
        description='ERAP Periodontal Analysis Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_pipeline.py                    # Run all steps (skip existing)
    python run_pipeline.py --steps 1 2 3      # Run specific steps
    python run_pipeline.py --subject sub-101  # Process single subject
    python run_pipeline.py --force            # Force re-run all
    python run_pipeline.py --dry-run          # Show what would run
    python run_pipeline.py --list             # List all steps
        """
    )

    parser.add_argument(
        '--steps',
        type=int,
        nargs='+',
        choices=list(PIPELINE_STEPS.keys()),
        help='Specific steps to run (default: all)'
    )
    parser.add_argument(
        '--subject',
        type=str,
        help='Process only this subject (e.g., sub-101)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-run even if outputs exist'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print commands without executing'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all pipeline steps and exit'
    )
    parser.add_argument(
        '--continue-on-error',
        action='store_true',
        help='Continue to next step even if current step fails'
    )

    args = parser.parse_args()

    # Just list steps and exit
    if args.list:
        print_pipeline_overview()
        sys.exit(0)

    # Setup
    ensure_directories()
    logger = setup_logging(LOGNOTES_DIR)

    # Determine which steps to run
    if args.steps:
        steps_to_run = args.steps
    else:
        steps_to_run = [n for n, s in PIPELINE_STEPS.items() if s['implemented']]

    # Pipeline header
    logger.info("\n" + "="*70)
    logger.info("ERAP PERIODONTAL ANALYSIS PIPELINE")
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*70)

    logger.info(f"\nSteps to run: {steps_to_run}")
    if args.subject:
        logger.info(f"Subject filter: {args.subject}")
    if args.force:
        logger.info("Force mode: ON")
    if args.dry_run:
        logger.info("Dry run mode: ON")

    # Run pipeline steps
    results = {}
    failed_steps = []

    for step_num in steps_to_run:
        success = run_step(
            step_num,
            subject=args.subject,
            force=args.force,
            dry_run=args.dry_run,
            logger=logger
        )

        results[step_num] = success

        if not success:
            failed_steps.append(step_num)
            if not args.continue_on_error:
                logger.error(f"\nPipeline stopped due to failure at step {step_num}")
                break

    # Summary
    logger.info("\n" + "="*70)
    logger.info("PIPELINE SUMMARY")
    logger.info("="*70)

    for step_num in steps_to_run:
        step = PIPELINE_STEPS[step_num]
        if step_num in results:
            status = "SUCCESS" if results[step_num] else "FAILED"
        else:
            status = "SKIPPED"

        logger.info(f"  Step {step_num} ({step['name']}): {status}")

    if failed_steps:
        logger.error(f"\nFailed steps: {failed_steps}")
        logger.info("\nTo re-run failed steps:")
        logger.info(f"  python run_pipeline.py --steps {' '.join(map(str, failed_steps))}")
    else:
        logger.info("\nAll steps completed successfully!")

    logger.info(f"\nPipeline finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*70)

    sys.exit(1 if failed_steps else 0)


if __name__ == "__main__":
    main()
