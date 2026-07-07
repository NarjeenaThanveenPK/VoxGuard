# Dataset — ASVspoof 2021 Logical Access

Raw audio files are not included in this repository due to file size (~15 GB).

## Download Instructions

### Step 1 — Download evaluation audio

Go to: https://zenodo.org/records/4837263

Download the evaluation FLAC files.

### Step 2 — Download metadata

Go to: https://www.asvspoof.org/index2021.html

Download: `LA-keys-full.tar.gz`

Extract so this path exists:

```
data/LA-keys-full/keys/LA/CM/trial_metadata.txt
```
### Step 3 — Run the sorting script

Open `deepfake_model.ipynb` and run Cell 3.
It reads the metadata and copies files into:
- `data/real/` — 2,000 bonafide (real) files
- `data/asvspoof_fake/` — 2,000 spoofed (fake) files

**Note:** The label is at column index 5 in the metadata (not index 4).
This was discovered through debugging and is already fixed in the notebook.

## Dataset Details

| Attribute | Value |
|-----------|-------|
| Name | ASVspoof 2021 Logical Access (LA) |
| Source | Zenodo DOI: 10.5281/zenodo.4837263 |
| Real samples used | 2,000 |
| Fake samples used | 2,000 |
| Total | 4,000 |
| Format | FLAC |
| Sample rate | 16,000 Hz |
| Attack systems | 19 (A01–A19) — TTS and Voice Conversion |