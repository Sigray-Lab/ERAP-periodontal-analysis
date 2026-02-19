# RawData Requirements

Expected raw data directory structure for the ERAP Periodontal FDG-PET/MRI Analysis Pipeline.

> **Important:** `RawData/` is a **sibling directory** to this repository, not inside it.
> The pipeline expects the following layout:
>
> ```
> ERAP_FDG_ONH_periodontium_analysis/
> ├── RawData/                    ← Raw data (not version-controlled)
> └── Periodontal_Analysis/       ← This repository
> ```
>
> `config.py` resolves this automatically via `PROJECT_ROOT / "RawData"`.

---

## Subjects and Sessions

| Item | Value |
|------|-------|
| Subjects | 13 (sub-101 through sub-114; sub-106 absent) |
| Sessions per subject | 2 (Baseline and Followup) |
| Total sessions | 26 |
| Session naming | Blinded codes (e.g., `ses-qbimm`), mapped via `BlindKey/Blinding_key.csv` |

---

## Directory Layout

```
RawData/
├── BlindKey/
│   └── Blinding_key.csv
│
├── sub-101/
│   ├── ses-qbimm/                          (blinded session code)
│   │   ├── pet/
│   │   │   └── *_chunk-brain_rec-StaticMoCo_trc-18FFDG_pet.nii
│   │   ├── ct/
│   │   │   └── *_chunk-ToraxBrain_rec-stnd1.25mm_ct.nii
│   │   └── anat/
│   │       └── *_chunk-teeth_T2w.nii
│   └── ses-fnfgs/
│       └── ... (same structure)
├── sub-102/
│   └── ...
├── ... (sub-103 through sub-114, excluding sub-106)
│
├── InputFunctions/
├── BloodPlasma/
├── eCRF_data/
├── json_side_cars_updated/
├── Teeth_ratings/
├── SUV_info/
├── Cerebellum_tacs/
└── bmd_ct/
```

---

## Per-Session Imaging Files

### PET (`sub-XXX/ses-XXXXX/pet/`)

| Field | Value |
|-------|-------|
| Filename pattern | `*_chunk-brain_rec-StaticMoCo_trc-18FFDG_pet.nii` |
| Format | NIfTI, float32 |
| Units | Bq/mL |
| Resolution | ~1 mm isotropic |
| Acquisition | Static 30-minute FDG-PET (30-60 min post-injection) |
| Size | ~140 MB per file |
| Count | 26 (1 per session) |
| Used by | `04_batch_quantify.py`, `06_longitudinal_delta.py` |

### CT (`sub-XXX/ses-XXXXX/ct/`)

| Field | Value |
|-------|-------|
| Primary filename | `*_chunk-ToraxBrain_rec-stnd1.25mm_ct.nii` (standard reconstruction) |
| Fallback filename | `*_chunk-ToraxBrain_rec-bone1.25mm_ct.nii` (bone reconstruction) |
| Format | NIfTI, int16 |
| Units | Hounsfield Units (HU) |
| Resolution | 1.25 mm |
| Size | ~281 MB per file |
| Count | 26 (1 per session; some subjects have both standard + bone) |
| Used by | `02_run_geometry_pipeline.py` (segmentation), `06_longitudinal_delta.py` |

**Filename variations:**
- Spelling: `ToraxBrain` or `ThoraxBrain` (handled by glob patterns)
- Run suffix: `_run-1_` present for sub-112 and sub-114
- Pipeline priority: standard reconstruction preferred; bone used as fallback

### Anatomy / STIR (`sub-XXX/ses-XXXXX/anat/`)

| Field | Value |
|-------|-------|
| Filename pattern | `*_chunk-teeth_T2w.nii` |
| Format | NIfTI, float32 |
| Note | Named "T2w" but is actually a STIR sequence (confirmed via JSON sidecar) |
| Resolution | Upper jaw FOV |
| Size | ~16 MB per file |
| Count | 26 (1 per session) |
| Used by | Future edema analysis (not currently in main pipeline) |

---

## Shared Data Directories

### BlindKey (`BlindKey/`)

**File:** `Blinding_key.csv`

| Column | Description | Example |
|--------|-------------|---------|
| `participant_id` | Subject ID | `sub-101` |
| `sex` | Biological sex | `M` / `F` |
| `age_inclusion` | Age at inclusion | `45` |
| `Session` | Timepoint label | `Baseline` / `Followup` |
| `Blind.code` | Blinded session code | `qbimm` |
| `ct.dcm.IDs` | CT DICOM reference | (internal use) |
| `mr.dcm.IDs` | MR DICOM reference | (internal use) |

- **Format:** Quoted CSV, 26 rows (13 subjects x 2 sessions)
- **Used by:** All pipeline scripts (session unblinding)

### InputFunctions (`InputFunctions/`)

**Pattern:** `{subject_id}_ses-{Baseline|Followup}_desc-IF_tacs.tsv`

| Column | Description | Units |
|--------|-------------|-------|
| `Time(s)` | Time post-injection | seconds |
| `ROI` | Source region | `aorta` or `plasma` |
| `Radioactivity(Bq/mL)` | Activity concentration | Bq/mL |

- **Format:** Tab-separated values, 26 files
- **Content:** Combined image-derived input function (aorta IDIF, 0-10 min) + manual plasma samples (20-90 min)
- **Used by:** `01_process_input_functions.py` (TPR/FUR denominators)

### BloodPlasma (`BloodPlasma/`)

**Pattern:** `{subject_id}_ses-{Baseline|Followup}_recording-manual_blood.{tsv,json}`

**TSV columns:**

| Column | Description | Units |
|--------|-------------|-------|
| `time` | Sampling time post-injection | seconds |
| `whole_blood_radioactivity` | Whole blood activity | Bq/mL |
| `plasma_radioactivity` | Plasma activity | Bq/mL |

**JSON fields:** `PlasmaAvail`, `WholeBloodAvail`, `MetaboliteAvail`, `DispersionCorrected`

- **Format:** TSV + JSON pairs, ~51 files total
- **Content:** 4-5 manual blood samples per session
- **Used by:** `01_process_input_functions.py`

### eCRF Data (`eCRF_data/`)

**Pattern:** `K8ERAPKIH22001_DATA_*.csv` (most recent file used)

**Key columns for PET quantification:**

| Column | Description | Units |
|--------|-------------|-------|
| `weight_kg_pet_1` | Body weight at Baseline PET | kg |
| `injected_mbq_pet_1` | Injected FDG dose at Baseline | MBq |
| `weight_kg_pet_2` | Body weight at Followup PET | kg |
| `injected_mbq_pet_2` | Injected FDG dose at Followup | MBq |

- **Format:** CSV with 800+ columns (REDCap export)
- **Content:** Demographics, clinical data, PET parameters, blood sample times, adverse events
- **Used by:** `04_batch_quantify.py` (SUV calculation: weight and dose)

### PET JSON Sidecars (`json_side_cars_updated/`)

**Pattern:** `{subject_id}_ses-{Baseline|Followup}_trc-18FFDG_rec-StaticMoCo_chunk-1_pet.json`

**Key fields:**

| Field | Description | Typical value |
|-------|-------------|---------------|
| `ScanStart` | Acquisition start (post-injection) | ~1800 s |
| `FrameDuration` | Frame length | ~1800 s |
| `FrameTimesStart` | Frame start times array | `[1800]` |
| `Units` | PET data units | `Bq/mL` |

- **Format:** JSON, 26 files
- **Used by:** `04_batch_quantify.py` (scan timing for TPR/FUR)

### Clinical Ratings (`Teeth_ratings/`)

**File:** `Blinded_Scoring_inflammation_ProbsM.xlsx`

| Column | Description |
|--------|-------------|
| `Subject_ID` | Subject ID (e.g., `sub-101`) |
| `Folder_name` | Blinded session folder |
| `T_11` ... `T_48` | Per-tooth inflammation scores (FDI notation) |

**Scoring:** 1 = healthy, 2 = mild, 3 = moderate, 4 = severe, NaN = missing

- **Format:** Excel (.xlsx), 1 file covering all subjects/sessions
- **Used by:** `05_statistical_analysis.py` (clinical correlation)

### SUV Scaling Factors (`SUV_info/`)

**File:** `session_scaling_factors.csv`

| Column | Description | Units |
|--------|-------------|-------|
| `subject_id` | Subject ID | |
| `session_blinded` | Blinded session code | |
| `session_unblinded` | `Baseline` / `Followup` | |
| `injected_MBq` | Injected FDG dose | MBq |
| `body_weight_kg` | Body weight | kg |
| `CER_AUC` | Cerebellum AUC | Bq*s/mL |
| `plasma_brain_chunk_AUC` | Plasma AUC during brain scan | kBq*s/mL |

- **Format:** Semicolon-separated CSV, 26 rows
- **Used by:** Validation / cross-checking of pipeline-computed values

### Cerebellum TACs (`Cerebellum_tacs/`)

**Pattern:** `{subject_id}_ses-{Baseline|Followup}_label-cerebellum_tacs.tsv`

| Column | Description | Units |
|--------|-------------|-------|
| `Frame` | Frame number | 0-indexed |
| `ROI` | Region | `cerebellum` |
| `Mean(Bq/mL)` | Mean activity | Bq/mL |
| `Std(Bq/mL)` | Standard deviation | Bq/mL |
| `Volume(voxels)` | ROI size | voxels |
| `FrameStart(s)` | Frame start time | seconds |

- **Format:** TSV, 26 files
- **Used by:** Reference tissue quantification (future use)

### BMD CT Images (`bmd_ct/`)

**Structure:** `bmd_ct/sub-XXX/ses-{Baseline|Followup}/ct/`

**Pattern:** `{subject_id}_ses-{timepoint}_desc-BMD_rec-stnd{1.25|2.5}mm_ct.{nii.gz,json}`

- **Format:** Gzipped NIfTI + JSON pairs, 2 resolutions per session
- **Count:** ~104 files (13 subjects x 2 sessions x 2 resolutions x 2 file types)
- **Used by:** Bone mineral density analysis (separate from periodontal pipeline)

---

## Summary Table

| Data Source | Format | Files | Per Session | Typical Size |
|-------------|--------|-------|-------------|--------------|
| PET images | NIfTI (.nii) | 26 | 1 | ~140 MB |
| CT images | NIfTI (.nii) | 26 | 1 | ~281 MB |
| STIR/T2w images | NIfTI (.nii) | 26 | 1 | ~16 MB |
| Blinding key | CSV | 1 | -- | 1.5 KB |
| Input functions | TSV | 26 | 1 | ~1 KB |
| Blood plasma | TSV + JSON | ~51 | 2 | <1 KB |
| eCRF data | CSV | 1 | -- | 49 KB |
| PET JSON sidecars | JSON | 26 | 1 | ~2 KB |
| Clinical ratings | Excel | 1 | -- | 19 KB |
| SUV scaling factors | CSV | 1 | -- | 2 KB |
| Cerebellum TACs | TSV | 26 | 1 | ~1 KB |
| BMD CT images | NIfTI.gz + JSON | ~104 | 4 | Variable |

---

## Notes

1. **Session codes are blinded.** Always use `BlindKey/Blinding_key.csv` to map `ses-{code}` to Baseline/Followup.
2. **CT file disambiguation.** Some subjects have both standard and bone CT reconstructions. The pipeline prefers standard (`rec-stnd1.25mm`); bone is used as fallback. See `utils/io_utils.py:find_ct_file()`.
3. **Missing subject.** sub-106 is absent from all datasets.
4. **eCRF versioning.** Multiple dated exports may exist; the pipeline uses the most recent file matching `K8ERAPKIH22001_DATA_*.csv`.
5. **European decimal format.** Some eCRF numeric fields use commas as decimal separators (e.g., `"72,5"` for 72.5 kg). The pipeline handles this.
6. **File discovery.** Use the helper functions in `Scripts/utils/io_utils.py` (`find_ct_file()`, `find_pet_file()`, `find_input_function_file()`, etc.) which handle all filename variations.
