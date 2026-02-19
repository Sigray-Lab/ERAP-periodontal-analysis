"""
Quality Control utilities for Periodontal Analysis Pipeline.

Handles:
- QC flag generation
- Validation checks
- QC report generation
- Visualization helpers
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import logging

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    VOLUME_STABILITY_THRESHOLD, SUV_MIN_EXPECTED, SUV_MAX_EXPECTED,
    MIN_ROI_VOXELS, MIN_TEETH_COUNT
)

logger = logging.getLogger(__name__)


# =============================================================================
# QC FLAG DEFINITIONS
# =============================================================================

QC_FLAG_DEFINITIONS = {
    'segmentation_failed': {
        'severity': 'CRITICAL',
        'description': 'CT segmentation failed - no teeth detected',
        'action': 'Exclude subject from analysis'
    },
    'segmentation_quality': {
        'severity': 'WARNING',
        'description': 'Low quality segmentation',
        'action': 'Review segmentation manually'
    },
    'roi_volume_unstable': {
        'severity': 'WARNING',
        'description': 'ROI volume changed >20% between timepoints',
        'action': 'Exclude affected teeth from analysis'
    },
    'metal_artifact': {
        'severity': 'INFO',
        'description': 'Metal artifacts detected near teeth',
        'action': 'Note percentage affected in report'
    },
    'pet_ct_motion': {
        'severity': 'WARNING',
        'description': 'Large PET-CT translation detected',
        'action': 'Review registration'
    },
    'roi_too_small': {
        'severity': 'WARNING',
        'description': 'ROI contains fewer than minimum voxels',
        'action': 'Check segmentation'
    },
    'hu_rejection_high': {
        'severity': 'WARNING',
        'description': 'High fraction of voxels rejected by HU gating',
        'action': 'Check HU range settings'
    },
    'suv_outlier': {
        'severity': 'WARNING',
        'description': 'SUV value outside expected range',
        'action': 'Review PET data'
    },
    'plasma_sparse': {
        'severity': 'WARNING',
        'description': 'Few plasma samples in scan window',
        'action': 'TPR/FUR may be less reliable'
    },
    'stir_missing': {
        'severity': 'INFO',
        'description': 'STIR image not found',
        'action': 'STIR metrics will be NA'
    },
    'validation_failed': {
        'severity': 'WARNING',
        'description': 'Baseline validation shows no healthy/unhealthy difference',
        'action': 'Biomarker may lack sensitivity'
    }
}


# =============================================================================
# QC FLAG GENERATION
# =============================================================================

def create_qc_flag(subject_id: str, session_id: str, flag_type: str,
                   value: Any = None, details: str = None) -> Dict[str, Any]:
    """
    Create a QC flag entry.

    Args:
        subject_id: Subject ID
        session_id: Session ID
        flag_type: Flag type from QC_FLAG_DEFINITIONS
        value: Optional value that triggered the flag
        details: Optional additional details

    Returns:
        Dictionary with flag information
    """
    flag_def = QC_FLAG_DEFINITIONS.get(flag_type, {
        'severity': 'UNKNOWN',
        'description': flag_type,
        'action': 'Review'
    })

    return {
        'subject_id': subject_id,
        'session_id': session_id,
        'flag_type': flag_type,
        'severity': flag_def['severity'],
        'description': flag_def['description'],
        'value': str(value) if value is not None else '',
        'details': details or '',
        'action': flag_def['action'],
        'timestamp': datetime.now().isoformat()
    }


def check_segmentation_qc(seg_result: Dict, subject_id: str, session_id: str) -> List[Dict]:
    """
    Generate QC flags for segmentation results.

    Args:
        seg_result: Segmentation result dictionary
        subject_id: Subject ID
        session_id: Session ID

    Returns:
        List of QC flag dictionaries
    """
    flags = []

    qc = seg_result.get('qc_metrics', {})

    # Check if segmentation succeeded
    if seg_result.get('teeth_mask') is None:
        flags.append(create_qc_flag(
            subject_id, session_id, 'segmentation_failed',
            details=seg_result.get('error', 'Unknown error')
        ))
        return flags

    # Check teeth volume
    teeth_vol = qc.get('teeth_volume_ml', 0)
    if teeth_vol < 5:
        flags.append(create_qc_flag(
            subject_id, session_id, 'segmentation_quality',
            value=f"{teeth_vol:.1f} mL",
            details='Teeth volume below minimum'
        ))
    elif teeth_vol > 50:
        flags.append(create_qc_flag(
            subject_id, session_id, 'segmentation_quality',
            value=f"{teeth_vol:.1f} mL",
            details='Teeth volume above maximum'
        ))

    # Check for metal artifacts
    metal_voxels = qc.get('metal_near_teeth_voxels', 0)
    if metal_voxels > 0:
        flags.append(create_qc_flag(
            subject_id, session_id, 'metal_artifact',
            value=f"{metal_voxels} voxels",
            details='Metal artifacts near teeth'
        ))

    return flags


def check_roi_qc(roi_result: Dict, subject_id: str, session_id: str) -> List[Dict]:
    """
    Generate QC flags for ROI generation results.

    Args:
        roi_result: ROI generation result dictionary
        subject_id: Subject ID
        session_id: Session ID

    Returns:
        List of QC flag dictionaries
    """
    flags = []

    qc = roi_result.get('qc_metrics', {})

    # Check ROI volume
    peridental_vol = qc.get('peridental_4mm_volume_ml', 0)
    if peridental_vol < 0.1:
        flags.append(create_qc_flag(
            subject_id, session_id, 'roi_too_small',
            value=f"{peridental_vol:.2f} mL",
            details='Peridental ROI too small'
        ))

    # Check HU rejection rate
    hu_rejection = qc.get('peridental_4mm_hu_rejection', 0)
    if hu_rejection > 0.5:
        flags.append(create_qc_flag(
            subject_id, session_id, 'hu_rejection_high',
            value=f"{hu_rejection*100:.1f}%",
            details='High fraction of voxels rejected by HU gating'
        ))

    return flags


def check_pet_qc(metrics: Dict, subject_id: str, session_id: str) -> List[Dict]:
    """
    Generate QC flags for PET quantification results.

    Args:
        metrics: PET metrics dictionary
        subject_id: Subject ID
        session_id: Session ID

    Returns:
        List of QC flag dictionaries
    """
    flags = []

    # Check SUV values
    suv_mean = metrics.get('SUV_mean', np.nan)
    if not np.isnan(suv_mean):
        if suv_mean < SUV_MIN_EXPECTED:
            flags.append(create_qc_flag(
                subject_id, session_id, 'suv_outlier',
                value=f"{suv_mean:.3f}",
                details=f'SUV_mean below {SUV_MIN_EXPECTED}'
            ))
        elif suv_mean > SUV_MAX_EXPECTED:
            flags.append(create_qc_flag(
                subject_id, session_id, 'suv_outlier',
                value=f"{suv_mean:.3f}",
                details=f'SUV_mean above {SUV_MAX_EXPECTED}'
            ))

    return flags


def check_volume_stability(baseline_vol: float, followup_vol: float,
                           subject_id: str, roi_name: str) -> Optional[Dict]:
    """
    Check ROI volume stability between timepoints.

    Args:
        baseline_vol: Baseline ROI volume
        followup_vol: Followup ROI volume
        subject_id: Subject ID
        roi_name: Name of ROI (e.g., tooth ID)

    Returns:
        QC flag if unstable, None otherwise
    """
    if baseline_vol <= 0 or followup_vol <= 0:
        return None

    pct_change = abs(followup_vol - baseline_vol) / baseline_vol * 100

    if pct_change > VOLUME_STABILITY_THRESHOLD:
        return create_qc_flag(
            subject_id, f"Baseline vs Followup", 'roi_volume_unstable',
            value=f"{pct_change:.1f}%",
            details=f'{roi_name}: {baseline_vol:.2f} -> {followup_vol:.2f} mL'
        )

    return None


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_baseline_biomarkers(metrics_df: pd.DataFrame,
                                  clinical_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate that biomarkers show expected differences at baseline.

    Key question: Do unhealthy teeth show higher FDG/STIR than healthy?

    Args:
        metrics_df: DataFrame with tooth-level metrics
        clinical_df: DataFrame with clinical ratings

    Returns:
        Dictionary with validation results
    """
    from scipy.stats import mannwhitneyu

    result = {
        'valid': True,
        'metrics_tested': [],
        'warnings': []
    }

    # Filter to baseline only
    baseline = metrics_df[metrics_df['timepoint'] == 'Baseline'].copy()

    if len(baseline) == 0:
        result['valid'] = False
        result['warnings'].append("No baseline data for validation")
        return result

    # Get clinical categories
    if 'clinical_category' not in baseline.columns:
        # Merge with clinical data
        baseline = baseline.merge(
            clinical_df[['subject_id', 'session_id', 'tooth_id', 'rating_category']],
            on=['subject_id', 'session_id', 'tooth_id'],
            how='left'
        )
        baseline['clinical_category'] = baseline['rating_category']

    # Split into healthy vs unhealthy
    healthy = baseline[baseline['clinical_category'] == 'healthy']
    unhealthy = baseline[baseline['clinical_category'] == 'unhealthy']

    if len(healthy) < 3 or len(unhealthy) < 3:
        result['warnings'].append(
            f"Insufficient data for validation: healthy={len(healthy)}, unhealthy={len(unhealthy)}"
        )
        return result

    # Test each metric
    metrics_to_test = ['SUV_mean', 'TPR_mean', 'FUR_mean_per_min']

    for metric in metrics_to_test:
        if metric not in baseline.columns:
            continue

        h_vals = healthy[metric].dropna()
        u_vals = unhealthy[metric].dropna()

        if len(h_vals) < 3 or len(u_vals) < 3:
            continue

        # Mann-Whitney U test (unhealthy > healthy)
        stat, pval = mannwhitneyu(u_vals, h_vals, alternative='greater')

        # Cohen's d effect size
        pooled_std = np.sqrt((np.var(h_vals) + np.var(u_vals)) / 2)
        if pooled_std > 0:
            cohens_d = (np.mean(u_vals) - np.mean(h_vals)) / pooled_std
        else:
            cohens_d = 0

        metric_result = {
            'metric': metric,
            'healthy_mean': float(np.mean(h_vals)),
            'healthy_n': len(h_vals),
            'unhealthy_mean': float(np.mean(u_vals)),
            'unhealthy_n': len(u_vals),
            'mann_whitney_p': float(pval),
            'cohens_d': float(cohens_d),
            'passed': pval < 0.1 and cohens_d > 0.3
        }

        result['metrics_tested'].append(metric_result)

        if not metric_result['passed']:
            result['warnings'].append(
                f"{metric}: No significant healthy/unhealthy difference "
                f"(p={pval:.3f}, d={cohens_d:.2f})"
            )

    # Overall validation
    passed_count = sum(1 for m in result['metrics_tested'] if m['passed'])
    if passed_count == 0:
        result['valid'] = False
        result['warnings'].append("No metrics passed baseline validation")

    return result


# =============================================================================
# QC REPORT GENERATION
# =============================================================================

def generate_qc_summary(flags_df: pd.DataFrame, metrics_df: pd.DataFrame = None,
                        validation_result: Dict = None) -> str:
    """
    Generate human-readable QC summary report.

    Args:
        flags_df: DataFrame of QC flags
        metrics_df: Optional metrics DataFrame for statistics
        validation_result: Optional validation results

    Returns:
        String with formatted report
    """
    lines = []
    lines.append("=" * 80)
    lines.append("PERIODONTAL ANALYSIS - QUALITY CONTROL SUMMARY REPORT")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 80)
    lines.append("")

    # Overview
    if len(flags_df) > 0:
        n_subjects = flags_df['subject_id'].nunique()
        n_critical = len(flags_df[flags_df['severity'] == 'CRITICAL'])
        n_warning = len(flags_df[flags_df['severity'] == 'WARNING'])
        n_info = len(flags_df[flags_df['severity'] == 'INFO'])

        lines.append("OVERVIEW")
        lines.append("-" * 40)
        lines.append(f"Subjects with flags: {n_subjects}")
        lines.append(f"Total flags: {len(flags_df)}")
        lines.append(f"  CRITICAL: {n_critical}")
        lines.append(f"  WARNING: {n_warning}")
        lines.append(f"  INFO: {n_info}")
        lines.append("")

        # Flags by type
        lines.append("FLAGS BY TYPE")
        lines.append("-" * 40)
        type_counts = flags_df['flag_type'].value_counts()
        for flag_type, count in type_counts.items():
            severity = QC_FLAG_DEFINITIONS.get(flag_type, {}).get('severity', '?')
            lines.append(f"  {flag_type} [{severity}]: {count}")
        lines.append("")

        # Critical flags detail
        if n_critical > 0:
            lines.append("CRITICAL FLAGS (require action)")
            lines.append("-" * 40)
            critical = flags_df[flags_df['severity'] == 'CRITICAL']
            for _, row in critical.iterrows():
                lines.append(f"  {row['subject_id']}/{row['session_id']}: {row['description']}")
                if row['details']:
                    lines.append(f"    -> {row['details']}")
            lines.append("")
    else:
        lines.append("No QC flags raised - all data passed checks.")
        lines.append("")

    # Validation results
    if validation_result:
        lines.append("BASELINE VALIDATION")
        lines.append("-" * 40)

        if validation_result['valid']:
            lines.append("Status: PASSED")
        else:
            lines.append("Status: FAILED")

        for m in validation_result.get('metrics_tested', []):
            status = "PASS" if m['passed'] else "FAIL"
            lines.append(f"  {m['metric']}: {status}")
            lines.append(f"    Healthy: {m['healthy_mean']:.3f} (n={m['healthy_n']})")
            lines.append(f"    Unhealthy: {m['unhealthy_mean']:.3f} (n={m['unhealthy_n']})")
            lines.append(f"    Cohen's d: {m['cohens_d']:.2f}, p={m['mann_whitney_p']:.3f}")

        if validation_result['warnings']:
            lines.append("\nValidation warnings:")
            for w in validation_result['warnings']:
                lines.append(f"  - {w}")
        lines.append("")

    # Metrics summary
    if metrics_df is not None and len(metrics_df) > 0:
        lines.append("METRICS SUMMARY")
        lines.append("-" * 40)

        for col in ['SUV_mean', 'SUV_90th', 'TPR_mean', 'FUR_mean_per_min']:
            if col in metrics_df.columns:
                vals = metrics_df[col].dropna()
                if len(vals) > 0:
                    lines.append(f"  {col}: {vals.min():.3f} - {vals.max():.3f} (mean: {vals.mean():.3f})")
        lines.append("")

    lines.append("=" * 80)

    return '\n'.join(lines)


def save_qc_reports(flags: List[Dict], output_dir: Path,
                    metrics_df: pd.DataFrame = None,
                    validation_result: Dict = None) -> Tuple[Path, Path]:
    """
    Save QC reports (CSV and text summary).

    Args:
        flags: List of QC flag dictionaries
        output_dir: Output directory
        metrics_df: Optional metrics DataFrame
        validation_result: Optional validation results

    Returns:
        Tuple of (flags_csv_path, summary_txt_path)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save flags CSV
    flags_df = pd.DataFrame(flags) if flags else pd.DataFrame()
    flags_path = output_dir / "QC_flags_report.csv"
    flags_df.to_csv(flags_path, index=False)

    # Generate and save summary
    summary = generate_qc_summary(flags_df, metrics_df, validation_result)
    summary_path = output_dir / "QC_summary_report.txt"
    with open(summary_path, 'w') as f:
        f.write(summary)

    return flags_path, summary_path
