"""
Input Function / Plasma utilities for Periodontal Analysis Pipeline.

Handles:
- Loading combined IDIF + plasma input function data
- Processing input function for TPR and FUR calculations
- Interpolation across IDIF-plasma gap
- QC of input function data
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional, Any, List
from scipy.interpolate import interp1d
import logging

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    DEFAULT_SCAN_START_S, DEFAULT_SCAN_DURATION_S, TISSUE_TIME_S
)

logger = logging.getLogger(__name__)


# =============================================================================
# INPUT FUNCTION LOADING
# =============================================================================

def load_input_function(if_path: Path) -> Dict[str, Any]:
    """
    Load combined IDIF + plasma input function TSV file.

    The file contains:
    - IDIF from aorta (0-10 min, typically)
    - Manual whole blood samples (ignored)
    - Manual plasma samples (20-90 min, typically)

    Args:
        if_path: Path to input function TSV file

    Returns:
        Dictionary with:
            - idif_times: Time points for IDIF (seconds)
            - idif_activities: Activity values for IDIF (Bq/mL)
            - plasma_times: Time points for plasma samples (seconds)
            - plasma_activities: Activity values for plasma (Bq/mL)
            - combined_times: All valid times sorted
            - combined_activities: All valid activities sorted
            - warnings: List of warnings
    """
    df = pd.read_csv(if_path, sep='\t')

    result = {
        'idif_times': np.array([]),
        'idif_activities': np.array([]),
        'plasma_times': np.array([]),
        'plasma_activities': np.array([]),
        'combined_times': np.array([]),
        'combined_activities': np.array([]),
        'n_idif_samples': 0,
        'n_plasma_samples': 0,
        'warnings': []
    }

    # Column names may vary - handle common variations
    time_col = None
    activity_col = None
    roi_col = None

    for col in df.columns:
        col_lower = col.lower()
        if 'time' in col_lower:
            time_col = col
        elif 'radioactivity' in col_lower or 'activity' in col_lower:
            activity_col = col
        elif 'roi' in col_lower:
            roi_col = col

    if time_col is None or activity_col is None:
        result['warnings'].append(f"Could not identify time/activity columns in {if_path.name}")
        return result

    # Extract IDIF (aorta) samples
    if roi_col:
        idif_mask = df[roi_col].str.lower() == 'aorta'
        idif_df = df[idif_mask].copy()
    else:
        # Assume all early samples are IDIF
        idif_df = df[df[time_col] < 600].copy()  # First 10 minutes

    # Filter valid IDIF values
    idif_df = idif_df[idif_df[activity_col].notna()]
    idif_df = idif_df[idif_df[activity_col] > 0]  # Remove zeros/negatives

    if len(idif_df) > 0:
        result['idif_times'] = idif_df[time_col].values.astype(float)
        result['idif_activities'] = idif_df[activity_col].values.astype(float)
        result['n_idif_samples'] = len(idif_df)

    # Extract plasma samples
    if roi_col:
        plasma_mask = df[roi_col].str.lower() == 'plasma'
        plasma_df = df[plasma_mask].copy()
    else:
        # Assume later samples are plasma (after 10 min gap)
        plasma_df = df[df[time_col] >= 1200].copy()

    # Filter valid plasma values
    plasma_df = plasma_df[plasma_df[activity_col].notna()]
    plasma_df = plasma_df[plasma_df[activity_col] > 0]

    if len(plasma_df) > 0:
        result['plasma_times'] = plasma_df[time_col].values.astype(float)
        result['plasma_activities'] = plasma_df[activity_col].values.astype(float)
        result['n_plasma_samples'] = len(plasma_df)

    # Combine IDIF + plasma for full curve
    if result['n_idif_samples'] > 0 and result['n_plasma_samples'] > 0:
        all_times = np.concatenate([result['idif_times'], result['plasma_times']])
        all_activities = np.concatenate([result['idif_activities'], result['plasma_activities']])

        # Sort by time
        sort_idx = np.argsort(all_times)
        result['combined_times'] = all_times[sort_idx]
        result['combined_activities'] = all_activities[sort_idx]
    elif result['n_idif_samples'] > 0:
        result['combined_times'] = result['idif_times']
        result['combined_activities'] = result['idif_activities']
        result['warnings'].append("No plasma samples found - using IDIF only")
    elif result['n_plasma_samples'] > 0:
        result['combined_times'] = result['plasma_times']
        result['combined_activities'] = result['plasma_activities']
        result['warnings'].append("No IDIF samples found - using plasma only")
    else:
        result['warnings'].append("No valid input function data found")

    # QC checks
    if result['n_idif_samples'] < 5:
        result['warnings'].append(f"Few IDIF samples: {result['n_idif_samples']}")

    if result['n_plasma_samples'] < 2:
        result['warnings'].append(f"Few plasma samples: {result['n_plasma_samples']}")

    # Check for gap between IDIF and plasma
    if result['n_idif_samples'] > 0 and result['n_plasma_samples'] > 0:
        idif_end = np.max(result['idif_times'])
        plasma_start = np.min(result['plasma_times'])
        gap = plasma_start - idif_end

        if gap > 1200:  # >20 min gap
            result['warnings'].append(f"Large gap between IDIF and plasma: {gap/60:.1f} min")

    return result


# =============================================================================
# INPUT FUNCTION PROCESSING
# =============================================================================

def process_input_function(if_data: Dict[str, Any],
                           scan_start_s: float = DEFAULT_SCAN_START_S,
                           scan_end_s: float = None,
                           tissue_time_s: float = TISSUE_TIME_S) -> Dict[str, Any]:
    """
    Process input function for TPR and FUR calculations.

    Calculates:
    - TPR denominator: Mean plasma during scan window
    - FUR denominator: Cumulative AUC from 0 to tissue time

    Args:
        if_data: Dictionary from load_input_function()
        scan_start_s: Scan start time in seconds (default 1800s = 30 min)
        scan_end_s: Scan end time (default: scan_start + 1800s)
        tissue_time_s: Tissue measurement time for FUR (default 2700s = 45 min)

    Returns:
        Dictionary with processed values and QC info
    """
    if scan_end_s is None:
        scan_end_s = scan_start_s + DEFAULT_SCAN_DURATION_S

    result = {
        'plasma_mean_Bq_mL': np.nan,
        'plasma_auc_window_Bq_s_mL': np.nan,
        'plasma_auc_0_to_T_Bq_s_mL': np.nan,
        'scan_start_s': scan_start_s,
        'scan_end_s': scan_end_s,
        'tissue_time_s': tissue_time_s,
        'n_samples_in_window': 0,
        'interpolation_used': False,
        'warnings': list(if_data.get('warnings', []))
    }

    times = if_data.get('combined_times', np.array([]))
    activities = if_data.get('combined_activities', np.array([]))

    if len(times) < 2:
        result['warnings'].append("Insufficient input function data for processing")
        return result

    # Create interpolation function
    # Use cubic spline, but be careful at boundaries
    try:
        # For robust interpolation, use pchip or cubic with bounds handling
        f = interp1d(times, activities, kind='linear', fill_value='extrapolate')
        result['interpolation_used'] = True
    except Exception as e:
        result['warnings'].append(f"Interpolation failed: {e}")
        return result

    # === TPR denominator (mean during scan window) ===
    # Count samples actually in window
    samples_in_window = np.sum((times >= scan_start_s) & (times <= scan_end_s))
    result['n_samples_in_window'] = int(samples_in_window)

    if samples_in_window < 2:
        result['warnings'].append(
            f"Only {samples_in_window} samples in scan window [{scan_start_s/60:.0f}-{scan_end_s/60:.0f} min]"
        )

    # Calculate AUC over scan window using trapezoidal integration
    window_times = np.linspace(scan_start_s, scan_end_s, 100)
    window_activities = f(window_times)

    # Ensure non-negative
    window_activities = np.maximum(window_activities, 0)

    auc_window = np.trapz(window_activities, window_times)
    result['plasma_auc_window_Bq_s_mL'] = float(auc_window)

    # Mean plasma concentration in window
    scan_duration = scan_end_s - scan_start_s
    result['plasma_mean_Bq_mL'] = float(auc_window / scan_duration)

    # === FUR denominator (cumulative AUC 0 to T) ===
    # Integrate from t=0 to tissue measurement time
    fur_times = np.linspace(0, tissue_time_s, 200)
    fur_activities = f(fur_times)
    fur_activities = np.maximum(fur_activities, 0)

    auc_0_to_T = np.trapz(fur_activities, fur_times)
    result['plasma_auc_0_to_T_Bq_s_mL'] = float(auc_0_to_T)

    # QC: Check for reasonable values
    if result['plasma_mean_Bq_mL'] < 100:
        result['warnings'].append(
            f"Low plasma mean: {result['plasma_mean_Bq_mL']:.1f} Bq/mL (expected >1000)"
        )
    elif result['plasma_mean_Bq_mL'] > 100000:
        result['warnings'].append(
            f"High plasma mean: {result['plasma_mean_Bq_mL']:.1f} Bq/mL (check units)"
        )

    return result


def interpolate_at_time(if_data: Dict[str, Any], target_time: float) -> float:
    """
    Interpolate input function at a specific time point.

    Args:
        if_data: Dictionary from load_input_function()
        target_time: Time in seconds

    Returns:
        Interpolated activity value (Bq/mL)
    """
    times = if_data.get('combined_times', np.array([]))
    activities = if_data.get('combined_activities', np.array([]))

    if len(times) < 2:
        return np.nan

    # Simple linear interpolation
    if target_time <= times[0]:
        return float(activities[0])
    if target_time >= times[-1]:
        return float(activities[-1])

    # Find bracketing interval
    idx = np.searchsorted(times, target_time)
    t0, t1 = times[idx-1], times[idx]
    v0, v1 = activities[idx-1], activities[idx]

    # Linear interpolation
    frac = (target_time - t0) / (t1 - t0)
    return float(v0 + frac * (v1 - v0))


# =============================================================================
# QC AND VISUALIZATION
# =============================================================================

def generate_if_qc_plot(if_data: Dict[str, Any], processed: Dict[str, Any],
                        output_path: Path, title: str) -> None:
    """
    Generate QC plot of input function with processing markers.

    Args:
        if_data: Raw input function data
        processed: Processed input function data
        output_path: Path to save PNG
        title: Plot title
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Full time course
        ax = axes[0]

        # Plot IDIF
        if if_data['n_idif_samples'] > 0:
            ax.scatter(if_data['idif_times'] / 60, if_data['idif_activities'],
                      c='blue', label=f'IDIF (n={if_data["n_idif_samples"]})', s=30)

        # Plot plasma
        if if_data['n_plasma_samples'] > 0:
            ax.scatter(if_data['plasma_times'] / 60, if_data['plasma_activities'],
                      c='red', label=f'Plasma (n={if_data["n_plasma_samples"]})', s=50, marker='s')

        # Plot interpolated curve
        if len(if_data['combined_times']) > 1:
            interp_times = np.linspace(
                np.min(if_data['combined_times']),
                np.max(if_data['combined_times']),
                200
            )
            f = interp1d(if_data['combined_times'], if_data['combined_activities'],
                        kind='linear', fill_value='extrapolate')
            ax.plot(interp_times / 60, f(interp_times), 'k--', alpha=0.5, label='Interpolated')

        # Mark scan window
        scan_start = processed['scan_start_s'] / 60
        scan_end = processed['scan_end_s'] / 60
        ax.axvspan(scan_start, scan_end, alpha=0.2, color='green', label='Scan window')

        ax.set_xlabel('Time (min)')
        ax.set_ylabel('Activity (Bq/mL)')
        ax.set_title('Full Time Course')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # Zoom to scan window
        ax = axes[1]

        if if_data['n_plasma_samples'] > 0:
            window_mask = (if_data['plasma_times'] >= processed['scan_start_s'] - 600) & \
                         (if_data['plasma_times'] <= processed['scan_end_s'] + 600)
            if np.any(window_mask):
                ax.scatter(if_data['plasma_times'][window_mask] / 60,
                          if_data['plasma_activities'][window_mask],
                          c='red', s=80, marker='s', label='Plasma samples')

        # Mark mean plasma level
        if not np.isnan(processed['plasma_mean_Bq_mL']):
            ax.axhline(processed['plasma_mean_Bq_mL'], color='green', linestyle='-',
                      linewidth=2, label=f'Mean: {processed["plasma_mean_Bq_mL"]:.0f} Bq/mL')

        ax.axvspan(scan_start, scan_end, alpha=0.2, color='green')

        ax.set_xlabel('Time (min)')
        ax.set_ylabel('Activity (Bq/mL)')
        ax.set_title(f'Scan Window ({scan_start:.0f}-{scan_end:.0f} min)')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # Set reasonable x-limits for zoom
        ax.set_xlim(scan_start - 15, scan_end + 15)

        plt.suptitle(title, fontsize=12)
        plt.tight_layout()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    except ImportError:
        logger.warning("Matplotlib not available, skipping IF QC plot")


def validate_input_function(if_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate input function data quality.

    Args:
        if_data: Dictionary from load_input_function()

    Returns:
        Dictionary with validation results
    """
    result = {
        'valid': True,
        'quality': 'good',
        'issues': []
    }

    # Check sample counts
    if if_data['n_idif_samples'] == 0:
        result['valid'] = False
        result['issues'].append("No IDIF samples")
    elif if_data['n_idif_samples'] < 5:
        result['quality'] = 'poor'
        result['issues'].append(f"Few IDIF samples ({if_data['n_idif_samples']})")

    if if_data['n_plasma_samples'] == 0:
        result['valid'] = False
        result['issues'].append("No plasma samples")
    elif if_data['n_plasma_samples'] < 3:
        result['quality'] = 'poor'
        result['issues'].append(f"Few plasma samples ({if_data['n_plasma_samples']})")

    # Check for reasonable peak in IDIF
    if if_data['n_idif_samples'] > 0:
        peak_activity = np.max(if_data['idif_activities'])
        if peak_activity < 10000:
            result['quality'] = 'poor' if result['quality'] != 'poor' else result['quality']
            result['issues'].append(f"Low IDIF peak: {peak_activity:.0f} Bq/mL")

    # Check time coverage
    if len(if_data['combined_times']) > 0:
        time_range = np.max(if_data['combined_times']) - np.min(if_data['combined_times'])
        if time_range < 1800:  # Less than 30 min
            result['quality'] = 'poor'
            result['issues'].append(f"Short time coverage: {time_range/60:.0f} min")

    if result['valid'] and len(result['issues']) == 0:
        result['quality'] = 'good'
    elif result['valid']:
        result['quality'] = 'acceptable' if len(result['issues']) < 2 else 'poor'

    return result
