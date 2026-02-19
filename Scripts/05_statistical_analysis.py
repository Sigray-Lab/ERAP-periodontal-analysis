#!/usr/bin/env python3
"""
05_statistical_analysis.py - Statistical Analysis for ERAP Periodontal Metrics

Performs:
1. Jaw-level paired analysis (upper jaw focus, harmonized data)
2. Tooth-level linear mixed models
3. Effect size calculations
4. Sensitivity analysis (3mm vs 8mm trimming)

Output:
    Outputs/statistical_analysis/by_trimming/{trimming}/
        jaw_paired_tests.csv
        tooth_lmm_results.csv
        tooth_lmm_summary.txt
        effect_sizes.csv

Usage:
    python 05_statistical_analysis.py                    # Default: upper jaw, 0mm trimming
    python 05_statistical_analysis.py --trimming 3mm    # Use 3mm trimming
    python 05_statistical_analysis.py --include-lower   # Include lower jaw
    python 05_statistical_analysis.py --include-tooth-type  # Add tooth type to LMM
"""

import argparse
import sys
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

# Add Scripts directory to path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

from config import (
    OUTPUTS_DIR, LOGNOTES_DIR, ensure_directories,
    CROSS_SECTIONAL_OUTPUT_DIR, STATS_OUTPUT_DIR
)

logger = logging.getLogger(__name__)

# Metrics to analyze
METRICS = ['SUV_mean', 'SUV_p90', 'TPR_mean', 'TPR_p90', 'FUR_mean_per_min', 'FUR_p90_per_min']


# =============================================================================
# PAIRED CONSISTENCY RULE
# =============================================================================

def apply_paired_consistency(tooth_df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure paired consistency: only include teeth present in BOTH timepoints.

    This is critical for valid paired analysis.
    """
    # Get teeth present at each timepoint for each subject
    paired_teeth = []

    for subject_id in tooth_df['subject_id'].unique():
        subj_data = tooth_df[tooth_df['subject_id'] == subject_id]

        baseline_teeth = set(subj_data[subj_data['timepoint'] == 'Baseline']['fdi_tooth'].unique())
        followup_teeth = set(subj_data[subj_data['timepoint'] == 'Followup']['fdi_tooth'].unique())

        # Intersection: teeth present in both
        shared_teeth = baseline_teeth & followup_teeth

        # Filter to shared teeth only
        subj_paired = subj_data[subj_data['fdi_tooth'].isin(shared_teeth)]
        paired_teeth.append(subj_paired)

    if paired_teeth:
        return pd.concat(paired_teeth, ignore_index=True)
    return pd.DataFrame()


# =============================================================================
# JAW-LEVEL PAIRED ANALYSIS
# =============================================================================

def jaw_level_paired_analysis(jaw_df: pd.DataFrame, jaw: str = 'upper',
                               trimming: str = '0mm') -> pd.DataFrame:
    """
    Perform paired pre/post analysis for jaw-level metrics.

    Uses harmonized data (teeth present in both sessions).

    Args:
        jaw_df: DataFrame with jaw-level metrics
        jaw: 'upper', 'lower', or 'both'
        trimming: 'none', '0mm', '3mm', '5mm', '8mm', or '10mm'

    Returns:
        DataFrame with statistical results
    """
    # Filter data
    df = jaw_df[
        (jaw_df['harmonized'] == True) &
        (jaw_df['trimming'] == trimming)
    ].copy()

    if jaw != 'both':
        df = df[df['jaw'] == jaw]

    if len(df) == 0:
        logger.warning(f"No data found for jaw={jaw}, trimming={trimming}, harmonized=True")
        return pd.DataFrame()

    results = []

    for metric in METRICS:
        if metric not in df.columns or df[metric].isna().all():
            continue

        # Pivot to wide format: one row per subject
        pivot = df.pivot_table(
            index='subject_id',
            columns='timepoint',
            values=metric,
            aggfunc='first'
        )

        # Require both timepoints
        if 'Baseline' not in pivot.columns or 'Followup' not in pivot.columns:
            continue

        complete = pivot.dropna()
        n = len(complete)

        if n < 3:
            logger.warning(f"Insufficient pairs for {metric}: {n}")
            continue

        baseline = complete['Baseline'].values
        followup = complete['Followup'].values
        diff = followup - baseline

        # Normality test on differences
        if n >= 3:
            _, normality_p = stats.shapiro(diff)
        else:
            normality_p = np.nan

        # Paired t-test
        t_stat, t_pval = stats.ttest_rel(baseline, followup)

        # Wilcoxon signed-rank (non-parametric)
        try:
            w_stat, w_pval = stats.wilcoxon(baseline, followup)
        except ValueError:
            # All differences are zero
            w_stat, w_pval = np.nan, np.nan

        # Effect size (Cohen's d for paired samples)
        sd_diff = np.std(diff, ddof=1)
        cohens_d = np.mean(diff) / sd_diff if sd_diff > 0 else np.nan

        # Percent change
        pct_change = 100 * np.mean(diff) / np.mean(baseline) if np.mean(baseline) != 0 else np.nan

        results.append({
            'metric': metric,
            'jaw': jaw,
            'trimming': trimming,
            'n_pairs': n,
            'baseline_mean': np.mean(baseline),
            'baseline_sd': np.std(baseline, ddof=1),
            'followup_mean': np.mean(followup),
            'followup_sd': np.std(followup, ddof=1),
            'mean_diff': np.mean(diff),
            'sd_diff': sd_diff,
            'pct_change': pct_change,
            'normality_p': normality_p,
            't_statistic': t_stat,
            't_pvalue': t_pval,
            'wilcoxon_statistic': w_stat,
            'wilcoxon_pvalue': w_pval,
            'cohens_d': cohens_d,
        })

    return pd.DataFrame(results)


# =============================================================================
# TOOTH-LEVEL LINEAR MIXED MODEL
# =============================================================================

def tooth_level_lmm(tooth_df: pd.DataFrame, outcome: str = 'SUV_mean',
                    jaw: str = 'upper', trimming: str = '0mm',
                    include_tooth_type: bool = False) -> dict:
    """
    Fit Linear Mixed Model for tooth-level repeated measures.

    Model: outcome ~ Timepoint + (1|SubjectID)
    Optional: outcome ~ Timepoint + ToothType + (1|SubjectID)

    Args:
        tooth_df: DataFrame with tooth-level metrics
        outcome: Metric column name
        jaw: 'upper', 'lower', or 'both'
        trimming: 'none', '0mm', '3mm', '5mm', '8mm', or '10mm'
        include_tooth_type: Include tooth_type as fixed effect

    Returns:
        dict with model summary and results
    """
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        logger.error("statsmodels not installed. Run: pip install statsmodels")
        return {'error': 'statsmodels not installed'}

    # Filter data
    df = tooth_df[tooth_df['trimming'] == trimming].copy()

    if jaw != 'both':
        df = df[df['jaw'] == jaw]

    # Apply paired consistency
    df = apply_paired_consistency(df)

    if len(df) == 0:
        return {'error': 'No data after paired consistency filtering'}

    # Drop rows with missing outcome
    df = df.dropna(subset=[outcome])

    if len(df) < 10:
        return {'error': f'Insufficient data: {len(df)} observations'}

    # Ensure proper types
    df['Timepoint'] = pd.Categorical(
        df['timepoint'],
        categories=['Baseline', 'Followup'],
        ordered=True
    )

    # Build formula
    if include_tooth_type and 'tooth_type' in df.columns:
        formula = f"{outcome} ~ Timepoint + C(tooth_type)"
    else:
        formula = f"{outcome} ~ Timepoint"

    # Fit mixed model
    try:
        model = smf.mixedlm(formula, df, groups=df['subject_id'])
        result = model.fit(method='powell')  # More robust optimizer

        return {
            'outcome': outcome,
            'jaw': jaw,
            'trimming': trimming,
            'n_observations': len(df),
            'n_subjects': df['subject_id'].nunique(),
            'n_teeth': df['fdi_tooth'].nunique(),
            'formula': formula,
            'summary': result.summary().as_text(),
            'fixed_effects': result.fe_params.to_dict(),
            'pvalues': result.pvalues.to_dict(),
            'converged': result.converged,
        }
    except Exception as e:
        return {'error': str(e)}


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Statistical Analysis for ERAP Periodontal Metrics')
    parser.add_argument('--trimming', choices=['none', '0mm', '3mm', '5mm', '8mm', '10mm'], default='0mm',
                        help='Tongue trimming distance (default: 0mm = original tongue mask, no dilation)')
    parser.add_argument('--include-lower', action='store_true',
                        help='Include lower jaw in analysis (default: upper only)')
    parser.add_argument('--include-tooth-type', action='store_true',
                        help='Include tooth type in LMM')
    args = parser.parse_args()

    ensure_directories()
    STATS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOGNOTES_DIR / f"statistical_analysis_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    logger.info("=" * 70)
    logger.info("STATISTICAL ANALYSIS — ERAP Periodontal Metrics")
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Trimming: {args.trimming}")
    logger.info(f"Jaw: {'upper + lower' if args.include_lower else 'upper only'}")
    logger.info("=" * 70)

    # Load data from cross-sectional output directory
    tooth_file = CROSS_SECTIONAL_OUTPUT_DIR / 'tooth_level_metrics.csv'
    jaw_file = CROSS_SECTIONAL_OUTPUT_DIR / 'jaw_level_metrics.csv'

    if not tooth_file.exists() or not jaw_file.exists():
        logger.error("Output files not found. Run 04_batch_quantify.py first.")
        sys.exit(1)

    tooth_df = pd.read_csv(tooth_file)
    jaw_df = pd.read_csv(jaw_file)

    logger.info(f"Loaded tooth-level: {len(tooth_df)} rows")
    logger.info(f"Loaded jaw-level: {len(jaw_df)} rows")

    # Determine jaw(s) to analyze
    jaws = ['upper', 'lower'] if args.include_lower else ['upper']

    # Create output directory for this trimming level
    trimming_output_dir = STATS_OUTPUT_DIR / "by_trimming" / args.trimming
    trimming_output_dir.mkdir(parents=True, exist_ok=True)

    # ==========================================================================
    # 1. JAW-LEVEL PAIRED ANALYSIS
    # ==========================================================================
    logger.info("\n" + "=" * 50)
    logger.info("1. JAW-LEVEL PAIRED ANALYSIS")
    logger.info("=" * 50)

    jaw_results = []
    for jaw in jaws:
        result = jaw_level_paired_analysis(jaw_df, jaw=jaw, trimming=args.trimming)
        if len(result) > 0:
            jaw_results.append(result)

    if jaw_results:
        jaw_results_df = pd.concat(jaw_results, ignore_index=True)
        jaw_output = trimming_output_dir / 'jaw_paired_tests.csv'
        jaw_results_df.to_csv(jaw_output, index=False)
        logger.info(f"Saved: {jaw_output}")

        # Print summary
        for _, row in jaw_results_df.iterrows():
            sig = "***" if row['t_pvalue'] < 0.001 else "**" if row['t_pvalue'] < 0.01 else "*" if row['t_pvalue'] < 0.05 else ""
            logger.info(f"  {row['metric']} ({row['jaw']}): "
                       f"diff={row['mean_diff']:.4f} ({row['pct_change']:.1f}%), "
                       f"p={row['t_pvalue']:.4f}{sig}, d={row['cohens_d']:.3f}")
    else:
        logger.warning("No jaw-level results generated")

    # ==========================================================================
    # 2. TOOTH-LEVEL LINEAR MIXED MODELS
    # ==========================================================================
    logger.info("\n" + "=" * 50)
    logger.info("2. TOOTH-LEVEL LINEAR MIXED MODELS")
    logger.info("=" * 50)

    lmm_results = []
    lmm_summaries = []

    for jaw in jaws:
        for outcome in METRICS:
            result = tooth_level_lmm(
                tooth_df, outcome=outcome, jaw=jaw, trimming=args.trimming,
                include_tooth_type=args.include_tooth_type
            )

            if 'error' in result:
                logger.warning(f"  {outcome} ({jaw}): {result['error']}")
                continue

            lmm_results.append({
                'outcome': result['outcome'],
                'jaw': result['jaw'],
                'trimming': result['trimming'],
                'n_observations': result['n_observations'],
                'n_subjects': result['n_subjects'],
                'formula': result['formula'],
                'timepoint_effect': result['fixed_effects'].get('Timepoint[T.Followup]', np.nan),
                'timepoint_pvalue': result['pvalues'].get('Timepoint[T.Followup]', np.nan),
                'converged': result['converged'],
            })

            lmm_summaries.append(f"\n{'='*60}\n{outcome} ({jaw})\n{'='*60}\n{result['summary']}")

            # Log result
            tp_effect = result['fixed_effects'].get('Timepoint[T.Followup]', np.nan)
            tp_pval = result['pvalues'].get('Timepoint[T.Followup]', np.nan)
            sig = "***" if tp_pval < 0.001 else "**" if tp_pval < 0.01 else "*" if tp_pval < 0.05 else ""
            logger.info(f"  {outcome} ({jaw}): effect={tp_effect:.4f}, p={tp_pval:.4f}{sig}")

    if lmm_results:
        lmm_df = pd.DataFrame(lmm_results)
        lmm_csv = trimming_output_dir / 'tooth_lmm_results.csv'
        lmm_df.to_csv(lmm_csv, index=False)
        logger.info(f"Saved: {lmm_csv}")

        # Save full summaries
        lmm_txt = trimming_output_dir / 'tooth_lmm_summary.txt'
        with open(lmm_txt, 'w') as f:
            f.write('\n'.join(lmm_summaries))
        logger.info(f"Saved: {lmm_txt}")
    else:
        logger.warning("No LMM results generated")

    # ==========================================================================
    # 3. EFFECT SIZES SUMMARY
    # ==========================================================================
    logger.info("\n" + "=" * 50)
    logger.info("3. EFFECT SIZES SUMMARY")
    logger.info("=" * 50)

    if jaw_results:
        effect_df = jaw_results_df[['metric', 'jaw', 'trimming', 'n_pairs',
                                     'mean_diff', 'pct_change', 'cohens_d',
                                     't_pvalue', 'wilcoxon_pvalue']].copy()
        effect_df = effect_df.rename(columns={
            't_pvalue': 'p_parametric',
            'wilcoxon_pvalue': 'p_nonparametric'
        })

        effect_output = trimming_output_dir / 'effect_sizes.csv'
        effect_df.to_csv(effect_output, index=False)
        logger.info(f"Saved: {effect_output}")

    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("ANALYSIS COMPLETE")
    logger.info(f"Results saved to: {trimming_output_dir}")
    logger.info(f"Log: {log_file}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
