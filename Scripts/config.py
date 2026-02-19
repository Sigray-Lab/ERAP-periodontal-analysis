"""
Configuration module for Periodontal Analysis Pipeline.

Defines paths, constants, and parameters for reproducible analysis.
All paths are relative to enable portability.
"""

from pathlib import Path
from typing import Dict, List, Tuple

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

# Script location determines all other paths
SCRIPT_DIR = Path(__file__).parent                    # Periodontal_Analysis/Scripts/
ANALYSIS_DIR = SCRIPT_DIR.parent                      # Periodontal_Analysis/
PROJECT_ROOT = ANALYSIS_DIR.parent                    # ERAP_FDG_ONH_periodontium_analysis/

# Shared data directories (READ-ONLY)
RAWDATA_DIR = PROJECT_ROOT / "RawData"
BLINDKEY_DIR = PROJECT_ROOT / "BlindKey"
INPUT_FUNC_RAW_DIR = RAWDATA_DIR / "InputFunctions"  # Raw aorta+plasma TSV files

# Analysis-specific output directories
OUTPUTS_DIR = ANALYSIS_DIR / "Outputs"
DERIVED_DIR = ANALYSIS_DIR / "DerivedData"
QC_DIR = ANALYSIS_DIR / "QC"
LOGNOTES_DIR = ANALYSIS_DIR / "LogNotes"

# Outputs subdirectories (organized by analysis type)
CROSS_SECTIONAL_OUTPUT_DIR = OUTPUTS_DIR / "cross_sectional"
LONGITUDINAL_OUTPUT_DIR = OUTPUTS_DIR / "longitudinal"
STATS_OUTPUT_DIR = OUTPUTS_DIR / "statistical_analysis"

# Derived data subdirectories
SEGMENTATION_DIR = DERIVED_DIR / "segmentations"
TOTALSEG_SEG_DIR = SEGMENTATION_DIR / "totalsegmentator_teeth"
HU_SEG_DIR = SEGMENTATION_DIR / "hu_fallback"
ROI_DIR = DERIVED_DIR / "rois"
TOTALSEG_ROI_DIR = ROI_DIR / "totalsegmentator_teeth"
HU_ROI_DIR = ROI_DIR / "hu_fallback"
TRANSFORM_DIR = DERIVED_DIR / "transforms"
INPUT_FUNC_DIR = DERIVED_DIR / "input_functions"
LONGITUDINAL_DIR = DERIVED_DIR / "longitudinal"

# QC subdirectories
SEG_QC_DIR = QC_DIR / "HU_segmentation"
TOTALSEG_QC_DIR = QC_DIR / "dental_segmentator_crops"
REG_QC_DIR = QC_DIR / "registration"
ROI_QC_DIR = QC_DIR / "roi"
VOLUME_QC_DIR = QC_DIR / "volume_stability"
PLASMA_QC_DIR = QC_DIR / "plasma"
VIS_DIR = QC_DIR / "visualizations"
LONGITUDINAL_QC_DIR = QC_DIR / "longitudinal"


def ensure_directories():
    """Create all output directories if they don't exist."""
    dirs = [
        OUTPUTS_DIR, DERIVED_DIR, QC_DIR, LOGNOTES_DIR,
        SEGMENTATION_DIR, TOTALSEG_SEG_DIR, HU_SEG_DIR,
        ROI_DIR, TOTALSEG_ROI_DIR, HU_ROI_DIR,
        TRANSFORM_DIR, INPUT_FUNC_DIR,
        SEG_QC_DIR, TOTALSEG_QC_DIR, REG_QC_DIR, ROI_QC_DIR,
        VOLUME_QC_DIR, PLASMA_QC_DIR, VIS_DIR,
        LONGITUDINAL_DIR, LONGITUDINAL_QC_DIR
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


# =============================================================================
# DATA FILE PATTERNS
# =============================================================================

# CT file patterns (prefer standard reconstruction for consistency)
# Note: Some subjects have _run-1_ suffix (e.g., sub-112, sub-114) and Thorax vs Torax spelling
CT_PATTERN_PRIMARY = "*_chunk-ToraxBrain_rec-stnd1.25mm_ct.nii"
CT_PATTERN_BONE = "*_chunk-ToraxBrain_rec-bone1.25mm_ct.nii"
CT_PATTERN_FALLBACK = "*_chunk-Tor*Brain_rec-*_ct.nii"  # Handles Torax/Thorax typos
CT_PATTERN_RUN = "*_chunk-*Brain_rec-*_run-*_ct.nii"  # Handles _run-1_ suffix (sub-112, sub-114)

# PET file pattern
PET_PATTERN = "*_chunk-brain_rec-StaticMoCo_trc-18FFDG_pet.nii"

# STIR/T2w file pattern (confirmed as STIR from JSON metadata)
STIR_PATTERN = "*_chunk-teeth_T2w.nii"

# Input function pattern
INPUT_FUNC_PATTERN = "{subject_id}_ses-{timepoint}_desc-IF_tacs.tsv"

# Clinical ratings file
CLINICAL_RATINGS_FILE = RAWDATA_DIR / "Teeth_ratings" / "Blinded_Scoring_inflammation_ProbsM.xlsx"

# Blinding key
BLINDING_KEY_FILE = BLINDKEY_DIR / "Blinding_key.csv"

# eCRF data
ECRF_PATTERN = "K8ERAPKIH22001_DATA_*.csv"


# =============================================================================
# SEGMENTATION PARAMETERS
# =============================================================================

# HU thresholds for CT segmentation
HU_TEETH_MIN = 1500          # Lower bound for teeth (enamel/dentin)
HU_BONE_MIN = 300            # Lower bound for bone
HU_BONE_MAX = 1500           # Upper bound for bone (below teeth)
HU_METAL_THRESHOLD = 2000    # Threshold for metal artifact detection

# Metal artifact dilation (in mm)
METAL_DILATION_MM = 5.0

# Expected ranges for QC
TEETH_VOLUME_MIN_ML = 5.0    # Minimum expected teeth mask volume
TEETH_VOLUME_MAX_ML = 50.0   # Maximum expected teeth mask volume
MAXILLA_VOLUME_MIN_ML = 10.0
MAXILLA_VOLUME_MAX_ML = 100.0
MIN_TEETH_COUNT = 8          # Minimum expected teeth in upper jaw


# =============================================================================
# ROI GENERATION PARAMETERS
# =============================================================================

# Peridental shell dilation distances (in mm)
PRIMARY_DILATION_MM = 4.0    # Primary analysis
SENSITIVITY_DILATION_MM = 6.0  # Sensitivity analysis

# Soft tissue HU range (wide for low-dose CT-AC)
SOFT_TISSUE_HU_MIN = -100
SOFT_TISSUE_HU_MAX = 200

# Adaptive widening if ROI too small
SOFT_TISSUE_HU_MIN_WIDE = -150
SOFT_TISSUE_HU_MAX_WIDE = 250

# Minimum ROI voxel count
MIN_ROI_VOXELS = 100

# Maxilla proximity constraint (mm)
MAXILLA_PROXIMITY_MM = 6.0

# Volume stability threshold for QC (percent change)
VOLUME_STABILITY_THRESHOLD = 20.0


# =============================================================================
# TOOTH IDENTIFICATION (FDI NOTATION)
# =============================================================================

# Upper jaw teeth (FDI notation)
UPPER_RIGHT_TEETH = [18, 17, 16, 15, 14, 13, 12, 11]  # Q1
UPPER_LEFT_TEETH = [21, 22, 23, 24, 25, 26, 27, 28]   # Q2
ALL_UPPER_TEETH = UPPER_RIGHT_TEETH + UPPER_LEFT_TEETH

# Anterior vs Posterior
ANTERIOR_TEETH = [13, 12, 11, 21, 22, 23]  # Incisors + canines
POSTERIOR_TEETH = [18, 17, 16, 15, 14, 24, 25, 26, 27, 28]  # Premolars + molars

# Quadrant definitions for aggregation
QUADRANTS = {
    'upper_right_posterior': [14, 15, 16, 17, 18],
    'upper_right_anterior': [11, 12, 13],
    'upper_left_anterior': [21, 22, 23],
    'upper_left_posterior': [24, 25, 26, 27, 28],
}

# Tooth type classification from FDI numbers
TOOTH_TYPE_MAP = {
    # Incisors (central=1, lateral=2)
    11: 'incisor', 12: 'incisor', 21: 'incisor', 22: 'incisor',
    31: 'incisor', 32: 'incisor', 41: 'incisor', 42: 'incisor',
    # Canines (3)
    13: 'canine', 23: 'canine', 33: 'canine', 43: 'canine',
    # Premolars (4-5)
    14: 'premolar', 15: 'premolar', 24: 'premolar', 25: 'premolar',
    34: 'premolar', 35: 'premolar', 44: 'premolar', 45: 'premolar',
    # Molars (6-8)
    16: 'molar', 17: 'molar', 18: 'molar',
    26: 'molar', 27: 'molar', 28: 'molar',
    36: 'molar', 37: 'molar', 38: 'molar',
    46: 'molar', 47: 'molar', 48: 'molar',
}


def get_tooth_type(fdi: int) -> str:
    """Get tooth type from FDI number."""
    return TOOTH_TYPE_MAP.get(fdi, 'unknown')


# =============================================================================
# CLINICAL RATINGS MAPPING
# =============================================================================

# Rating scale (from Excel: 1-4, with some NaN for missing teeth)
# Need to map to healthy/unhealthy categories
RATING_SCALE = {
    1: 'healthy',          # Assumed: 1 = no inflammation
    2: 'mild',             # Mild inflammation
    3: 'moderate',         # Moderate inflammation
    4: 'severe',           # Severe inflammation
}

# For binary classification
RATING_HEALTHY = [1]
RATING_UNHEALTHY = [2, 3, 4]
RATING_MISSING = [None, float('nan')]


# =============================================================================
# PET QUANTIFICATION PARAMETERS
# =============================================================================

# PET scan timing (typical values, can be overridden from JSON)
DEFAULT_SCAN_START_S = 1800   # 30 minutes post-injection
DEFAULT_SCAN_DURATION_S = 1800  # 30 minutes scan duration

# Tissue measurement time for FUR (mid-scan)
TISSUE_TIME_S = 2700  # 45 minutes post-injection (mid-point of 30-60 min scan)

# SUV outlier thresholds for QC
SUV_MIN_EXPECTED = 0.1
SUV_MAX_EXPECTED = 20.0

# Percentile for robust metrics
ROBUST_PERCENTILE = 90


# =============================================================================
# STIR QUANTIFICATION PARAMETERS
# =============================================================================

# Z-score threshold for edema detection
EDEMA_Z_THRESHOLD = 2.0

# Minimum reference ROI voxels
MIN_REFERENCE_VOXELS = 100


# =============================================================================
# REGISTRATION PARAMETERS
# =============================================================================

# Maximum acceptable translation (mm) before flagging
MAX_PET_CT_TRANSLATION_MM = 5.0

# Registration metric
REGISTRATION_METRIC = 'mutual_information'


# =============================================================================
# LONGITUDINAL ANALYSIS PARAMETERS
# =============================================================================

# Gaussian smoothing FWHM for delta images (mm)
LONGITUDINAL_SMOOTHING_FWHM_MM = 3.0

# Default tongue trimming for longitudinal analysis
# '0mm' = original tongue mask (no dilation), '3mm', '5mm', '8mm', '10mm' = dilated
LONGITUDINAL_DEFAULT_TRIMMING = '0mm'


# =============================================================================
# OUTPUT FILE NAMES
# =============================================================================

# Main output files
MAIN_METRICS_FILE = "periodontal_metrics.csv"
TOOTH_METRICS_FILE = "tooth_level_metrics.csv"
QUADRANT_METRICS_FILE = "quadrant_level_metrics.csv"
STATISTICS_FILE = "pre_post_statistics.csv"
TARGETED_STATS_FILE = "targeted_disease_statistics.csv"
VALIDATION_FILE = "validation_correlations.csv"

# QC files
QC_FLAGS_FILE = "QC_flags_report.csv"
QC_SUMMARY_FILE = "QC_summary_report.txt"
UNSTABLE_TEETH_FILE = "unstable_teeth.csv"

# Derived data files
SESSION_INFO_FILE = "session_info.csv"


# =============================================================================
# LOGGING
# =============================================================================

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_output_path(filename: str) -> Path:
    """Get full path for output file."""
    return OUTPUTS_DIR / filename


def get_derived_path(subdir: str, filename: str) -> Path:
    """Get full path for derived data file."""
    return DERIVED_DIR / subdir / filename


def get_qc_path(subdir: str, filename: str) -> Path:
    """Get full path for QC file."""
    return QC_DIR / subdir / filename
