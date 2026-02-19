"""
I/O utility functions for Periodontal Analysis Pipeline.

Handles:
- File discovery (subjects, sessions, images)
- NIfTI loading/saving with proper scaling
- Blinding key and eCRF data loading
- Clinical ratings loading
"""

import json
import numpy as np
import nibabel as nib
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from glob import glob

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    RAWDATA_DIR, BLINDKEY_DIR, BLINDING_KEY_FILE, CLINICAL_RATINGS_FILE,
    CT_PATTERN_PRIMARY, CT_PATTERN_BONE, CT_PATTERN_FALLBACK,
    PET_PATTERN, STIR_PATTERN, ECRF_PATTERN,
    RATING_SCALE, RATING_HEALTHY, RATING_UNHEALTHY,
    ALL_UPPER_TEETH
)


# =============================================================================
# BLINDING AND METADATA
# =============================================================================

def load_blinding_key(project_root: Optional[Path] = None) -> Dict[Tuple[str, str], str]:
    """
    Load the blinding key CSV and create mapping from (subject, session) to timepoint.

    Args:
        project_root: Optional project root path (uses config default if None)

    Returns:
        Dictionary mapping (subject_id, blinded_session_code) to timepoint
    """
    if project_root is None:
        blinding_file = BLINDING_KEY_FILE
    else:
        blinding_file = project_root / "BlindKey" / "Blinding_key.csv"

    if not blinding_file.exists():
        raise FileNotFoundError(f"Blinding key not found at: {blinding_file}")

    df = pd.read_csv(blinding_file)

    # Create mapping: (participant_id, ses-{Blind.code}) -> Session
    mapping = {}
    for _, row in df.iterrows():
        subject_id = row['participant_id']
        blinded_code = f"ses-{row['Blind.code']}"
        timepoint = row['Session']
        mapping[(subject_id, blinded_code)] = timepoint

    return mapping


def load_ecrf_data(rawdata_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Load the eCRF data CSV containing weight and injected dose information.

    Args:
        rawdata_dir: Optional RawData directory path

    Returns:
        DataFrame with eCRF data
    """
    if rawdata_dir is None:
        rawdata_dir = RAWDATA_DIR

    ecrf_dir = rawdata_dir / "eCRF_data"
    ecrf_files = list(ecrf_dir.glob(ECRF_PATTERN))

    if not ecrf_files:
        raise FileNotFoundError(f"No eCRF data files found matching {ECRF_PATTERN} in {ecrf_dir}")

    # Use most recent file if multiple exist
    ecrf_file = sorted(ecrf_files)[-1]
    df = pd.read_csv(ecrf_file, encoding='utf-8-sig')

    return df


def get_suv_parameters(ecrf_df: pd.DataFrame, subject_id: str, timepoint: str) -> Dict[str, float]:
    """
    Extract SUV calculation parameters from eCRF data.

    Args:
        ecrf_df: eCRF DataFrame
        subject_id: Subject ID (e.g., 'sub-101')
        timepoint: 'Baseline' or 'Followup'

    Returns:
        Dictionary with 'weight_kg' and 'injected_mbq' keys
    """
    # Extract numeric subject ID
    subj_num = int(subject_id.replace('sub-', ''))

    # Find the row for this subject
    row = ecrf_df[ecrf_df['subject_id'] == subj_num]
    if len(row) == 0:
        return {'weight_kg': None, 'injected_mbq': None, 'error': f'Subject {subj_num} not found in eCRF'}

    row = row.iloc[0]

    # Determine which PET session (pet_1 = Baseline, pet_2 = Followup)
    if timepoint == 'Baseline':
        weight_col = 'weight_kg_pet_1'
        dose_col = 'injected_mbq_pet_1'
    else:  # Followup
        weight_col = 'weight_kg_pet_2'
        dose_col = 'injected_mbq_pet_2'

    # Get values, handling potential string/numeric issues
    weight = row.get(weight_col)
    dose = row.get(dose_col)

    # Clean weight value (may have comma as decimal separator)
    if pd.notna(weight):
        if isinstance(weight, str):
            weight = float(weight.replace(',', '.'))
        else:
            weight = float(weight)
    else:
        weight = None

    # Clean dose value
    if pd.notna(dose):
        if isinstance(dose, str):
            dose = float(dose.replace(',', '.'))
        else:
            dose = float(dose)
    else:
        dose = None

    return {'weight_kg': weight, 'injected_mbq': dose}


def load_clinical_ratings(ratings_file: Optional[Path] = None) -> pd.DataFrame:
    """
    Load clinical periodontal ratings from Excel file.

    Transforms wide format (T_11, T_12, ...) to long format for analysis.

    Args:
        ratings_file: Path to Excel file (uses config default if None)

    Returns:
        DataFrame with columns: subject_id, session_id, tooth_id, rating, rating_category
    """
    if ratings_file is None:
        ratings_file = CLINICAL_RATINGS_FILE

    if not ratings_file.exists():
        raise FileNotFoundError(f"Clinical ratings file not found at: {ratings_file}")

    df = pd.read_excel(ratings_file)

    # Identify tooth columns (T_11, T_12, etc.)
    tooth_cols = [col for col in df.columns if col.startswith('T_')]

    # Melt to long format
    id_cols = ['Subject_ID', 'Folder_name', 'Blind_Code']
    long_df = df.melt(
        id_vars=id_cols,
        value_vars=tooth_cols,
        var_name='tooth_col',
        value_name='rating'
    )

    # Extract tooth ID from column name (T_11 -> 11)
    long_df['tooth_id'] = long_df['tooth_col'].str.replace('T_', '').astype(int)

    # Map rating to category
    def categorize_rating(r):
        if pd.isna(r):
            return 'missing'
        r = int(r)
        if r in RATING_HEALTHY:
            return 'healthy'
        elif r in RATING_UNHEALTHY:
            return 'unhealthy'
        else:
            return 'unknown'

    long_df['rating_category'] = long_df['rating'].apply(categorize_rating)

    # Clean up column names
    long_df = long_df.rename(columns={
        'Subject_ID': 'subject_id',
        'Folder_name': 'session_id'
    })

    # Select and order columns
    long_df = long_df[['subject_id', 'session_id', 'tooth_id', 'rating', 'rating_category']]

    return long_df


# =============================================================================
# FILE DISCOVERY
# =============================================================================

def discover_subjects(rawdata_dir: Optional[Path] = None) -> List[str]:
    """
    Discover all subject directories in RawData.

    Args:
        rawdata_dir: Optional RawData directory path

    Returns:
        Sorted list of subject IDs (e.g., ['sub-101', 'sub-102', ...])
    """
    if rawdata_dir is None:
        rawdata_dir = RAWDATA_DIR

    subjects = []
    for item in rawdata_dir.iterdir():
        if item.is_dir() and item.name.startswith('sub-') and 'ScalarVolume' not in item.name:
            subjects.append(item.name)

    return sorted(subjects)


def discover_sessions(subject_dir: Path) -> List[str]:
    """
    Discover all session directories for a given subject.

    Args:
        subject_dir: Path to subject directory

    Returns:
        List of session IDs (e.g., ['ses-fnfgs', 'ses-qbimm'])
    """
    sessions = []
    for item in subject_dir.iterdir():
        if item.is_dir() and item.name.startswith('ses-') and 'ScalarVolume' not in item.name:
            sessions.append(item.name)

    return sorted(sessions)


def find_ct_file(session_dir: Path, prefer_bone: bool = False) -> Optional[Path]:
    """
    Find CT file in session directory.

    Uses standard reconstruction by default for consistency across subjects.
    Bone reconstruction can be preferred but falls back to standard if bone CT
    has smaller FOV (fewer slices).

    Args:
        session_dir: Path to session directory (e.g., sub-101/ses-fnfgs/)
        prefer_bone: If True, prefer bone reconstruction if available

    Returns:
        Path to CT file or None if not found
    """
    import nibabel as nib

    ct_dir = session_dir / "ct"
    if not ct_dir.exists():
        return None

    def get_ct_file(pattern):
        matches = list(ct_dir.glob(pattern))
        matches = [m for m in matches if 'ScalarVolume' not in str(m)]
        if matches:
            nii_files = [m for m in matches if not str(m).endswith('.gz')]
            return nii_files[0] if nii_files else matches[0]
        return None

    bone_file = get_ct_file(CT_PATTERN_BONE)
    stnd_file = get_ct_file(CT_PATTERN_PRIMARY)

    if prefer_bone and bone_file and stnd_file:
        # Check if bone CT has similar FOV to standard CT
        # If bone CT has significantly fewer slices, use standard instead
        try:
            bone_img = nib.load(str(bone_file))
            stnd_img = nib.load(str(stnd_file))
            bone_slices = bone_img.shape[2]
            stnd_slices = stnd_img.shape[2]

            if bone_slices < stnd_slices * 0.7:  # Bone has <70% of standard slices
                logging.getLogger(__name__).info(
                    f"Bone CT has smaller FOV ({bone_slices} vs {stnd_slices} slices), using standard CT"
                )
                return stnd_file
            return bone_file
        except Exception:
            return bone_file
    elif prefer_bone and bone_file:
        return bone_file
    elif stnd_file:
        return stnd_file
    elif bone_file:
        return bone_file

    # Fallback pattern (handles Torax/Thorax spelling variations)
    fallback = get_ct_file(CT_PATTERN_FALLBACK)
    if fallback:
        return fallback

    # Run pattern (handles _run-1_ suffix in sub-112, sub-114)
    from config import CT_PATTERN_RUN
    run_file = get_ct_file(CT_PATTERN_RUN)
    return run_file


def find_pet_file(session_dir: Path) -> Optional[Path]:
    """
    Find PET file in session directory.

    Args:
        session_dir: Path to session directory

    Returns:
        Path to PET file or None if not found
    """
    pet_dir = session_dir / "pet"
    if not pet_dir.exists():
        return None

    matches = list(pet_dir.glob(PET_PATTERN))
    matches = [m for m in matches if 'ScalarVolume' not in str(m) and 'mask' not in m.name.lower()]

    if matches:
        # Prefer .nii over .nii.gz
        nii_files = [m for m in matches if not str(m).endswith('.gz')]
        if nii_files:
            return nii_files[0]
        return matches[0]

    # Fallback: any PET file
    for f in pet_dir.glob("*_pet.nii*"):
        if 'ScalarVolume' not in str(f) and 'mask' not in f.name.lower():
            return f

    return None


def find_pet_json(session_dir: Path, subject_id: str, timepoint: str,
                  rawdata_dir: Optional[Path] = None) -> Optional[Path]:
    """
    Find PET JSON sidecar file.

    Checks updated JSON folder first, then original location.

    Args:
        session_dir: Path to session directory
        subject_id: Subject ID
        timepoint: 'Baseline' or 'Followup'
        rawdata_dir: Optional RawData directory

    Returns:
        Path to JSON file or None if not found
    """
    if rawdata_dir is None:
        rawdata_dir = RAWDATA_DIR

    # Check updated JSON folder first
    updated_dir = rawdata_dir / "json_side_cars_updated"
    if updated_dir.exists():
        pattern = f"{subject_id}_ses-{timepoint}_trc-18FFDG_rec-StaticMoCo_chunk-1_pet.json"
        updated_json = updated_dir / pattern
        if updated_json.exists():
            return updated_json

    # Fallback to original location
    pet_dir = session_dir / "pet"
    if pet_dir.exists():
        for f in pet_dir.glob("*_pet.json"):
            if 'ScalarVolume' not in str(f):
                return f

    return None


def find_stir_file(session_dir: Path) -> Optional[Path]:
    """
    Find STIR/T2w file in session directory.

    Note: Files are named T2w but are actually STIR sequences (confirmed from JSON).

    Args:
        session_dir: Path to session directory

    Returns:
        Path to STIR file or None if not found
    """
    anat_dir = session_dir / "anat"
    if not anat_dir.exists():
        return None

    matches = list(anat_dir.glob(STIR_PATTERN))
    matches = [m for m in matches if 'ScalarVolume' not in str(m)]

    if matches:
        # Prefer .nii over .nii.gz
        nii_files = [m for m in matches if not str(m).endswith('.gz')]
        if nii_files:
            return nii_files[0]
        return matches[0]

    return None


def find_input_function_file(subject_id: str, timepoint: str,
                             rawdata_dir: Optional[Path] = None) -> Optional[Path]:
    """
    Find input function TSV file for a subject/session.

    Args:
        subject_id: Subject ID (e.g., 'sub-101')
        timepoint: 'Baseline' or 'Followup'
        rawdata_dir: Optional RawData directory

    Returns:
        Path to input function file or None if not found
    """
    if rawdata_dir is None:
        rawdata_dir = RAWDATA_DIR

    if_dir = rawdata_dir / "InputFunctions"
    if not if_dir.exists():
        return None

    # Pattern: sub-101_ses-Baseline_desc-IF_tacs.tsv
    filename = f"{subject_id}_ses-{timepoint}_desc-IF_tacs.tsv"
    if_file = if_dir / filename

    if if_file.exists():
        return if_file

    return None


# =============================================================================
# NIFTI I/O
# =============================================================================

def load_nifti(filepath: Path) -> Tuple[np.ndarray, nib.Nifti1Image]:
    """
    Load a NIfTI image with proper scaling applied.

    Args:
        filepath: Path to NIfTI file

    Returns:
        Tuple of (scaled data array, nibabel image object)
    """
    img = nib.load(str(filepath))
    # get_fdata() handles scl_slope and scl_inter automatically
    data = img.get_fdata(dtype=np.float32)
    return data, img


def save_nifti(data: np.ndarray, reference_img: nib.Nifti1Image,
               output_path: Path, compress: bool = True) -> None:
    """
    Save data as NIfTI file using reference image for header/affine.

    Args:
        data: Data array to save
        reference_img: Reference image for header and affine
        output_path: Output file path
        compress: If True, save as .nii.gz (default)
    """
    # Create new image with same affine and header
    new_img = nib.Nifti1Image(data.astype(np.float32),
                               reference_img.affine,
                               reference_img.header)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Add .gz extension if compressing and not already present
    if compress and not str(output_path).endswith('.gz'):
        output_path = Path(str(output_path) + '.gz')

    nib.save(new_img, str(output_path))


def get_voxel_dimensions(img: nib.Nifti1Image) -> np.ndarray:
    """
    Get voxel dimensions from NIfTI header.

    Args:
        img: nibabel image object

    Returns:
        Array of voxel dimensions [x, y, z] in mm
    """
    return np.array(img.header.get_zooms()[:3])


def get_voxel_volume_ml(img: nib.Nifti1Image) -> float:
    """
    Get voxel volume in mL.

    Args:
        img: nibabel image object

    Returns:
        Voxel volume in mL
    """
    voxel_dims = get_voxel_dimensions(img)
    voxel_volume_mm3 = np.prod(voxel_dims)
    return voxel_volume_mm3 / 1000.0  # mm³ to mL


def load_json(filepath: Path) -> Dict[str, Any]:
    """
    Load JSON file.

    Args:
        filepath: Path to JSON file

    Returns:
        Dictionary with JSON contents
    """
    with open(filepath, 'r') as f:
        return json.load(f)


# =============================================================================
# DATA VALIDATION
# =============================================================================

def validate_image_shapes(images: Dict[str, np.ndarray]) -> bool:
    """
    Validate that all images have the same shape.

    Args:
        images: Dictionary of image name -> data array

    Returns:
        True if all shapes match, False otherwise
    """
    shapes = [img.shape for img in images.values()]
    if len(set(shapes)) > 1:
        return False
    return True


def check_image_orientation(img: nib.Nifti1Image) -> str:
    """
    Get image orientation string (e.g., 'RAS', 'LPS').

    Args:
        img: nibabel image object

    Returns:
        Orientation string
    """
    return ''.join(nib.aff2axcodes(img.affine))
