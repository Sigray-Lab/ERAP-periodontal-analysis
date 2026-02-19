# ERAP Periodontal FDG-PET/MRI Analysis Pipeline

Analysis pipeline for quantifying periodontal inflammation using FDG-PET/MRI imaging data from the **ERAP clinical trial**.

## Background

Rapamycin has shown anti-inflammatory effects on periodontitis in mouse models. This exploratory imaging analysis tests whether those effects are detectable in humans using FDG-PET, targeting the peridental soft tissue compartment (gingiva, mucosa, alveolar bone) surrounding each tooth.

**Imaging challenge:** The periodontal ligament is ~0.2–0.4 mm thick — far below PET resolution (~4–6 mm FWHM). We therefore quantify the broader peridental region using geometry-based ROIs derived from CT dental segmentation.

## Pipeline Overview

| Step | Script | Description |
|------|--------|-------------|
| 1 | `01_process_input_functions.py` | Combine aortic IDIF (0–10 min) + manual plasma samples (20–90 min) into a continuous input function |
| 2 | `02_run_geometry_pipeline.py` | TotalSegmentator dental segmentation → per-tooth geometry ROIs → CT-to-PET registration |
| 3 | `03_create_tongue_exclusion.py` | Create tongue exclusion masks (0/3/5/8/10 mm dilation) to remove lingual spillover |
| 4 | `04_batch_quantify.py` | Extract SUV, TPR, and FUR metrics per tooth and per jaw |
| 5 | `05_statistical_analysis.py` | Paired t-tests (jaw-level) and linear mixed models (tooth-level) |
| 6 | `06_longitudinal_delta.py` | PET↔PET registration and voxelwise pre–post delta analysis |
| 7 | `07_validation_qc.py` | QC validation checks |

### Quantitative Metrics

| Metric | Description |
|--------|-------------|
| **SUV** (mean, p90) | Standardized Uptake Value — body-weight-normalized tissue activity |
| **FUR** (mean, p90) | Fractional Uptake Rate — metabolic rate approximation (min⁻¹) |

## Quick Start

### Prerequisites

```bash
pip install nibabel numpy pandas scipy matplotlib seaborn scikit-image antspyx statsmodels
pip install TotalSegmentator  # ~7 min/session on CPU for teeth task
```

### Running the Pipeline

```bash
cd Scripts/

# Full pipeline (skips existing outputs)
python run_pipeline.py

# Force re-run all steps
python run_pipeline.py --force

# Run specific subject
python run_pipeline.py --subject sub-101

# Individual scripts
python 04_batch_quantify.py --force
python 06_longitudinal_delta.py --force --trimming 0mm
python 05_statistical_analysis.py --trimming 0mm
```

## Data Requirements

The pipeline expects BIDS-like input data (not included in this repository):

```
RawData/
├── sub-XXX/ses-XXXXX/
│   ├── pet/  *_pet.nii       # FDG-PET (Bq/mL, 30–60 min static)
│   ├── ct/   *_ct.nii        # CT attenuation scan
│   └── stir/ *_stir.nii      # MRI STIR (upper jaw)
├── eCRF_data/                 # Clinical data (weight, dose)
├── InputFunctions/            # IDIF + plasma samples
└── json_side_cars_updated/    # PET timing metadata
```

## Output Structure

```
Outputs/
├── cross_sectional/           # Per-timepoint metrics (04_batch_quantify.py)
│   ├── tooth_level_metrics.csv
│   └── jaw_level_metrics.csv
├── longitudinal/              # Pre–post delta analysis (06_longitudinal_delta.py)
│   ├── delta_summary.csv
│   └── ttest_results.csv
└── statistical_analysis/      # Statistical tests (05_statistical_analysis.py)
    └── by_trimming/{0,3,5,8,10}mm/
        ├── jaw_paired_tests.csv
        ├── tooth_lmm_results.csv
        └── effect_sizes.csv
```

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Geometry-only ROIs (no HU gating) | Eliminates HU threshold sensitivity |
| 4 mm dilation for peridental shell | Captures periodontal space; enamel is metabolically inert |
| PET↔PET direct registration | Robust soft-tissue alignment for longitudinal comparison |
| 0 mm tongue trimming default | Original tongue mask without dilation; conservative approach |
| Cross-session tooth harmonization | Only teeth present in both timepoints are compared |
| Continuous [0,1] masks in PET space | Preserves partial-volume weighting after resampling |

## Tongue Exclusion

Upper molar ROIs extend lingually toward the tongue, which has high FDG uptake. The pipeline creates dilated tongue masks (0/3/5/8/10 mm) that are subtracted from dental ROIs. The default is 0 mm (original segmentation mask, no dilation). Statistical sensitivity analysis across trimming levels is supported.

## Development

This pipeline was developed collaboratively using Claude Code (Anthropic's AI coding agent) and is maintained by the [Sigray Lab](https://github.com/Sigray-Lab) at Karolinska Institutet.

## References

1. Hunter et al., "Quantification of FDG uptake using fractional uptake rate," *PMC6424227*
2. Boellaard et al., "SUV variability in PET/CT," *PMC3417317*
3. Wasserthal et al., "TotalSegmentator," *Radiology: AI*, 2023

## License

This project is part of the ERAP clinical trial. Raw imaging data are not included in this repository due to patient privacy regulations.
