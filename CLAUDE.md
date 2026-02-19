# ERAP Periodontal FDG-PET/MRI Analysis Pipeline

## 1. Project Purpose

This project extracts multi-modal imaging metrics from the **periodontal/peridental region** (gingiva, alveolar bone) in the ERAP clinical trial evaluating rapamycin treatment in early-stage Alzheimer's disease. Rapamycin has shown anti-inflammatory effects on periodontitis in mouse models, and this exploratory analysis aims to detect potential treatment effects on oral inflammatory markers using the existing imaging data.

### Scientific Context

**The imaging challenge:**
- The periodontal ligament (PDL) is ~0.2-0.4mm thick — far below PET resolution (~4-6mm FWHM)
- We therefore target the broader **peridental soft tissue compartment** (gingiva, mucosa) which:
  - Is resolvable at available imaging resolution
  - Contains the relevant inflammatory biology (macrophages, neutrophils)
  - Provides sufficient voxels for stable quantification

**Available data (per subject, 2 timepoints):**
- FDG-PET (30-60 min static acquisition, 1mm voxels)
- CT (attenuation scan from PET/CT, sufficient for dental segmentation)
- MRI STIR (upper jaw protocol, edema-sensitive)
- Combined input function: IDIF from aorta (0-10 min) + manual plasma samples (20-90 min)
- Clinical periodontal ratings (tooth-by-tooth: healthy/unhealthy/missing/unrateable)

### Key Metrics Extracted

| Metric | Description | Primary Use |
|--------|-------------|-------------|
| **SUVmean** | Mean SUV over ROI | Standard PET metric |
| **SUV_90th** | 90th percentile SUV in ROI | Robust "hot" metric |
| **TPRmean** | Mean Tissue-to-Plasma Ratio | Blood-normalized uptake |
| **TPR_90th** | 90th percentile TPR | Focal inflammation |
| **FURmean** | Mean Fractional Uptake Rate | Metabolic rate approximation |
| **FUR_90th** | 90th percentile FUR | Focal metabolic activity |

---

## 2. Current Pipeline Status (as of 2026-01-30)

### Pipeline Complete

**All 13 subjects processed** (sub-101 through sub-114, excluding sub-106 which doesn't exist).

| Step | Script | Status | Sessions |
|------|--------|--------|----------|
| Input Functions | `01_process_input_functions.py` | **COMPLETE** | 26 |
| Geometry Pipeline | `02_run_geometry_pipeline.py` | **COMPLETE** | 26 |
| Tongue Exclusion | `03_create_tongue_exclusion.py` | **COMPLETE** | 26 |
| Batch Quantification | `04_batch_quantify.py` | **COMPLETE** | 26 |
| Statistical Analysis | `05_statistical_analysis.py` | **COMPLETE** | All 4 trimming levels |

### Output Summary

| Output | Location | Rows |
|--------|----------|------|
| **Tooth-level metrics** | `Outputs/tooth_level_metrics.csv` | 2125 rows |
| **Jaw-level metrics** | `Outputs/jaw_level_metrics.csv` | 520 rows |
| **Statistical results** | `Outputs/statistical_results/` | 16 files (4 per trimming level) |

### Tongue Trimming Options

Four trimming levels available: **3mm, 5mm, 8mm, 10mm**

Statistical analysis shows that **larger trimming (8-10mm) improves sensitivity** for detecting Baseline→Followup changes, likely due to better exclusion of tongue spillover signal.

---

## 3. Quick Start: Running the Pipeline

### Prerequisites

```bash
pip install nibabel numpy pandas scipy matplotlib seaborn scikit-image antspyx statsmodels
pip install TotalSegmentator  # Requires ~7 min/session on CPU for teeth task
```

### Run Full Pipeline

```bash
cd Periodontal_Analysis/Scripts

# Run full pipeline (skips existing outputs)
python run_pipeline.py

# Run specific steps only
python run_pipeline.py --steps 1 2 3

# Run for specific subject
python run_pipeline.py --subject sub-101

# Force re-run all steps
python run_pipeline.py --force

# See what would run without executing
python run_pipeline.py --dry-run

# List all steps
python run_pipeline.py --list
```

### Pipeline Steps

| Step | Script | Description |
|------|--------|-------------|
| 1 | `01_process_input_functions.py` | IDIF + plasma → interpolated input function |
| 2 | `02_run_geometry_pipeline.py` | TotalSeg → geometry ROIs → CT-PET registration |
| 3 | `03_create_tongue_exclusion.py` | 3/5/8/10mm tongue trimming |
| 4 | `04_batch_quantify.py` | Extract SUV, TPR, FUR metrics |
| 5 | `05_statistical_analysis.py` | Paired t-tests and LMM (runs for all trimming levels) |

### Run Individual Scripts

```bash
# Geometry pipeline only
python 02_run_geometry_pipeline.py
python 02_run_geometry_pipeline.py --steps 5              # Only registration
python 02_run_geometry_pipeline.py --subject sub-101      # Specific subject

# Tongue exclusion
python 03_create_tongue_exclusion.py
python 03_create_tongue_exclusion.py --subject sub-101

# Batch quantification
python 04_batch_quantify.py

# Statistical analysis (for specific trimming level)
python 05_statistical_analysis.py --trimming 10mm
python 05_statistical_analysis.py --trimming 8mm --jaw both
```

---

## 4. Directory Structure

```
ERAP_FDG_ONH_periodontium_analysis/
│
├── RawData/                                   # READ ONLY
│   ├── sub-XXX/ses-XXXXX/                     # Subject/session folders
│   │   ├── pet/  *_pet.nii                    # FDG-PET (Bq/mL)
│   │   ├── ct/   *_ct.nii                     # CT attenuation scan
│   │   └── stir/ *_stir.nii                   # MRI STIR (upper jaw)
│   ├── eCRF_data/                             # Clinical data (weight, dose)
│   ├── InputFunctions/                        # IDIF + plasma samples
│   └── json_side_cars_updated/                # PET timing metadata
│
├── BlindKey/Blinding_key.csv                  # Session → Timepoint mapping
│
└── Periodontal_Analysis/
    ├── CLAUDE.md                              # This file
    ├── Scripts/
    │   ├── run_pipeline.py                    # MASTER PIPELINE WRAPPER
    │   ├── config.py                          # Paths and constants
    │   ├── 01_process_input_functions.py      # Input function processing
    │   ├── 02_run_geometry_pipeline.py        # Geometry pipeline (Steps 1-5)
    │   ├── 02b_geometry_roi_poc.py            # Per-tooth ROI generation
    │   ├── 03_create_tongue_exclusion.py      # Tongue trimming
    │   ├── 04_batch_quantify.py               # Batch quantification
    │   ├── 05_statistical_analysis.py         # Statistical analysis
    │   ├── 07_validation_qc.py                # QC validation
    │   ├── utils/                             # Utility modules
    │   └── archive/                           # Legacy scripts (not used)
    │
    ├── Outputs/
    │   ├── tooth_level_metrics.csv            # Per-tooth metrics (2125 rows)
    │   ├── jaw_level_metrics.csv              # Jaw-level metrics (520 rows)
    │   └── statistical_results/               # Statistical output files
    │       ├── effect_sizes_*mm_tongue_trim.csv
    │       ├── jaw_level_paired_tests_*mm_tongue_trim.csv
    │       ├── tooth_level_lmm_results_*mm_tongue_trim.csv
    │       └── tooth_level_lmm_summary_*mm_tongue_trim.txt
    │
    ├── DerivedData/
    │   ├── input_functions/                   # 26 processed IF files
    │   ├── segmentations/
    │   │   ├── totalsegmentator_teeth/        # TotalSeg teeth output
    │   │   ├── totalsegmentator_head_muscles/ # Tongue masks
    │   │   └── hu_fallback/                   # HU-threshold backup
    │   ├── rois/totalsegmentator_teeth/       # Per-tooth ROIs
    │   │   └── sub-XXX_ses-YYY/
    │   │       ├── tooth_shells_*.nii.gz
    │   │       ├── tongue_exclusion_*.nii.gz
    │   │       └── continuous_masks_PETspace/
    │   └── transforms/                        # ANTs CT→PET transforms
    │
    ├── QC/
    │   ├── plasma/                            # Input function QC plots
    │   ├── roi/                               # ROI visualization PNGs
    │   ├── tongue_exclusion/                  # Tongue trimming QC (104 images)
    │   ├── registration/                      # CT in PET space (NIfTI)
    │   └── archive/                           # Legacy QC files
    │
    └── LogNotes/                              # Pipeline logs
```

---

## 5. Segmentation Details

### Dual-Method Design

The pipeline **always runs both** TotalSegmentator and HU-threshold, saving each to its own subfolder.

**TotalSegmentator (Primary):**
- v2.12.0 `teeth` task with dental cluster crop (~7 min/session on CPU)
- Per-tooth FDI labels (11-48) + jawbones (1=lower, 2=upper)
- `head_muscles` task extracts tongue mask (label 9)

**HU-threshold (Fallback):**
- Teeth: HU > 1500, restricted to dental cluster
- Upper jaw: largest connected component above oral cavity

### Metal & Prosthetic Exclusion

1. **Prosthetic exclusion (TotalSeg labels 8/9/10)**: HU-validated — only excluded if mean HU ≥ 2500
2. **Metal filling artifact check**: Teeth with >15% voxels at HU cap (≥3071) are excluded

---

## 6. Tongue Exclusion System

### The Problem

Upper molar ROIs extend lingually toward the tongue, which has high FDG uptake. This "Tongue Effect" contaminates molar metrics with spillover from tongue muscle metabolism.

### The Solution

Dilated tongue masks are created and subtracted from dental ROIs:

| Dilation | Purpose |
|----------|---------|
| **3mm** | Conservative trim — removes only the hottest PVE |
| **5mm** | Moderate trim |
| **8mm** | Aggressive trim — removes lingual side, keeps buccal |
| **10mm** | Maximum trim — strongest tongue exclusion |

**Output files:**
```
DerivedData/rois/totalsegmentator_teeth/sub-XXX_ses-YYY/
├── tongue_exclusion_3mm.nii.gz
├── tongue_exclusion_5mm.nii.gz
├── tongue_exclusion_8mm.nii.gz
├── tongue_exclusion_10mm.nii.gz
└── continuous_masks_PETspace/
    ├── tooth_XX_trimmed_3mm.nii.gz
    ├── tooth_XX_trimmed_5mm.nii.gz
    ├── tooth_XX_trimmed_8mm.nii.gz
    └── tooth_XX_trimmed_10mm.nii.gz
```

---

## 7. Cross-Session Tooth Harmonization

### The Problem

Jaw-level ROIs can differ between Baseline and Followup due to different teeth being segmented.

### The Solution

**Harmonized** jaw metrics use only teeth present in BOTH sessions:

| Column | Description |
|--------|-------------|
| `harmonized` | `True` = shared teeth only, `False` = all teeth |
| `n_shared_teeth` | Count of teeth in intersection |

---

## 8. Output CSV Structure

### tooth_level_metrics.csv (2125 rows)

| Column | Description |
|--------|-------------|
| `subject_id` | e.g., "sub-101" |
| `session_id` | e.g., "ses-fnfgs" |
| `timepoint` | "Baseline" or "Followup" |
| `fdi_tooth` | FDI tooth number (11-48) |
| `jaw` | "upper" or "lower" |
| `trimming` | "none", "3mm", "5mm", "8mm", or "10mm" |
| `n_voxels` | Voxel count in ROI |
| `roi_volume_ml` | Effective volume |
| `SUV_mean`, `SUV_p90` | Standardized uptake values |
| `TPR_mean`, `TPR_p90` | Tissue-to-plasma ratios |
| `FUR_mean_per_min`, `FUR_p90_per_min` | Fractional uptake rates |

### jaw_level_metrics.csv (520 rows)

Same columns as tooth-level, plus `harmonized` and `n_shared_teeth`.

---

## 9. Statistical Analysis

### Methods

- **Jaw-level**: Paired t-tests (Baseline vs Followup, n=13)
- **Tooth-level**: Linear Mixed Models with subject as random effect

### Key Results (10mm trimming, upper jaw)

| Metric | % Change | p-value | Cohen's d |
|--------|----------|---------|-----------|
| TPR_mean | +9.7% | **0.022*** | 0.73 |
| TPR_p90 | +9.0% | **0.022*** | 0.73 |
| FUR_mean | +7.3% | **0.032*** | 0.67 |

10 of 13 subjects showed increased FUR/TPR from Baseline to Followup.

---

## 10. QC Flags

### Subjects to Monitor

| Subject | Flag | Notes |
|---------|------|-------|
| sub-113 | YELLOW | Lower followup plasma activity — may inflate TPR increase |
| sub-112, sub-111 | INFO | Decreased metrics (possible non-responders) |
| sub-106 | RED | Missing from dataset |

---

## 11. Formula Reference

### SUV (Standardized Uptake Value)
```
SUV = (C_tissue [Bq/mL] × body_weight [kg]) / (injected_dose [MBq] × 10⁶)
```

### TPR (Tissue-to-Plasma Ratio)
```
TPR = C_tissue [kBq/mL] / C_plasma_mean [kBq/mL]
Where: C_plasma_mean = mean plasma activity during scan window (30-60 min)
```

### FUR (Fractional Uptake Rate)
```
FUR = C_tissue(T) [kBq/mL] / ∫₀ᵀ C_plasma(t) dt [kBq·s/mL]
Reported as: min⁻¹
```

---

## 12. Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Geometry-only ROIs (no HU gating)** | Eliminates HU threshold sensitivity |
| **4mm dilation for peridental shell** | Captures PDL space; enamel is metabolically dead |
| **CT→PET rigid registration (ANTsPy)** | PET/CT misalignment up to ~1.5cm in some sessions |
| **Linear interpolation for masks** | Continuous [0,1] masks preserve partial-volume weighting |
| **Four tongue trimming options** | Allows sensitivity analysis (3/5/8/10mm) |
| **Cross-session harmonization** | Only compare teeth present in BOTH sessions |
| **HU-validated prosthetic exclusion** | TotalSeg prosthetic labels are often false positives |

---

## 13. Troubleshooting

| Problem | Solution |
|---------|----------|
| "No CT found" for subject | Check RawData folder; some subjects lack CT |
| TPR/FUR are NaN | Check eCRF data and input functions exist |
| "No continuous masks directory" | Run geometry pipeline step 5 |
| "No tongue mask found" | Run step 4 (head_muscles) |
| TotalSegmentator errors | Check TotalSegmentator installation and version |

---

## 14. References

1. **FUR methodology**: Hunter et al. PMC6424227
2. **SUV noise**: Boellaard et al. PMC3417317
3. **TotalSegmentator**: Wasserthal et al. Radiology: AI 2023
