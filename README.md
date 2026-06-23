# music-ml-classification

A production-style machine learning pipeline for three distinct music understanding tasks: composer classification from symbolic MIDI data, next-sequence prediction for musical continuity, and multi-label audio tagging. Built entirely from scratch over 20 days, exploring classical feature engineering, transformer-based embeddings, and ensemble learning strategies across both symbolic and audio music representations.

---

## Tasks

### Task 1 — Composer Classification (Symbolic / Multiclass)
Given a MIDI file, predict which composer wrote the piece.

**Approach:** Hybrid pipeline combining 300+ hand-engineered symbolic music features with Aria transformer embeddings (3-slice: first/middle/last 512 tokens concatenated into a 1536-dim representation). The feature set covers pitch class distributions, Krumhansl-Schmuckler key profiles, inter-onset interval statistics, rhythmic entropy, polyphony density, melodic contour, ornament ratios, pitch trigram frequency, interval transition matrices, and section-level descriptors (early/mid/late thirds). These are fused with Aria embeddings and fed into a weighted ensemble of HistGradientBoosting, Random Forest, ExtraTrees, Logistic Regression, and XGBoost classifiers with Dirichlet-optimized blend weights.

**Key design decisions:**
- Aria embeddings extracted from 3 non-overlapping 512-token slices (beginning, middle, end) to capture structural variation across a piece, not just a fixed window
- 600-candidate Dirichlet random search over blend weights with validation-based selection
- Full dataset retraining after weight tuning to maximize generalization
- Variance-thresholded feature selection and balanced class weighting throughout

**Result:** 0.9684 accuracy (baseline: 0.251 — 3.86× improvement)

---

### Task 2 — Next Sequence Prediction (Symbolic / Binary)
Given two bars of MIDI, predict whether the second bar immediately follows the first in a real piece of music.

**Approach:** Feature-rich binary classifier built on harmonic continuity analysis between bar boundaries. Features include circle-of-fifths distance between terminal/initial chords, dominant resolution scoring (perfect/plagal/step cadences), key compatibility (diatonic overlap fraction), voice leading smoothness, pitch class overlap (Jaccard), semitone distance, tritone detection, common-tone ratio, and melodic direction at boundary points. Paired with full statistical summaries (mean/std/min/max/median/IQR) of pitch, duration, velocity, IOI, polyphony, and interval sequences for both bars, extracted from multi-resolution head/tail windows (first/last 4, 8, 12, 16, 24 notes and 20/33/50% time fractions).

**Key design decisions:**
- Harmonic continuity features designed from music theory first principles (voice leading, resolution, circle of fifths) rather than purely statistical
- Multi-resolution windowing captures both local boundary context and global bar-level statistics simultaneously
- Pitch class transition matrices encode melodic tendency patterns within each bar

**Result:** 0.9285 accuracy (baseline: 0.624 — 1.49× improvement)

---

### Task 3 — Music Tagging (Audio / Multilabel)
Given an AI-synthesized audio clip, predict a set of genre/mood tags (rock, jazz, pop, electronic, chill, blues, punk, dance, oldies, country).

**Approach:** Ensemble of frozen pretrained audio foundation models with lightweight MLP heads trained on top. Models used:
- **MERT-v1-95M** — Music-specific transformer pretrained with masked audio modeling, 95M parameters
- **MERT-v1-330M** — Larger variant, 330M parameters, higher-capacity audio representations
- **CLAP (laion/clap-htsat-unfused)** — Contrastive Language-Audio Pretraining, cross-modal audio-text model
- **MuQ-large-msd-iter** (optional) — Music understanding model trained on Million Song Dataset

Each model generates fixed embeddings (frozen, no fine-tuning). Multiple MLP heads are trained per embedding space with different random seeds, optimized with Focal Loss (to handle class imbalance) + AdamW + Cosine Annealing LR schedule with early stopping. Final predictions are a softmax-temperature-weighted blend across all model+ensemble combinations, where weights are proportional to each model's validation mAP (temperature=0.05 for sharp weight sharpening toward best performers). Embeddings are cached to disk after first extraction for fast iteration.

**Key design decisions:**
- Frozen embeddings dramatically outperform training from scratch on small datasets — the key insight driving the entire Task 3 approach
- Ensemble diversity across model families (MERT acoustic, CLAP cross-modal, MuQ domain-matched) reduces correlated errors
- Concatenated embedding space (MERT-95M + MERT-330M + CLAP + MuQ) trains separate MLPs that capture cross-model synergies
- Softmax blend weighting (vs. equal weighting) learned from validation mAP ensures better-performing models dominate the final prediction

**Result:** 0.7069 mAP (baseline: 0.270 — 2.6× improvement)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Input (MIDI / Audio)                  │
└───────────────┬─────────────────────┬───────────────────┘
                │                     │
    ┌───────────▼──────────┐ ┌────────▼────────────────┐
    │  Symbolic Pipeline   │ │    Audio Pipeline        │
    │  (Tasks 1 & 2)       │ │    (Task 3)              │
    │                      │ │                          │
    │  miditoolkit parse   │ │  MERT-95M  (frozen)      │
    │  300+ hand features  │ │  MERT-330M (frozen)      │
    │  Aria embeddings     │ │  CLAP      (frozen)      │
    │  (3-slice, 1536-dim) │ │  MuQ       (frozen)      │
    └───────────┬──────────┘ └────────┬────────────────-┘
                │                     │
    ┌───────────▼──────────┐ ┌────────▼────────────────┐
    │  Weighted Ensemble   │ │  MLP Heads per model     │
    │  HGB + RF + Extra    │ │  Focal Loss + AdamW      │
    │  Trees + LR + XGB    │ │  Cosine Annealing LR     │
    │  Dirichlet weights   │ │  Early stopping          │
    └───────────┬──────────┘ └────────┬────────────────-┘
                │                     │
                └──────────┬──────────┘
                           │
              ┌────────────▼──────────────┐
              │  Softmax-weighted blend   │
              │  (val mAP → weights)      │
              └────────────┬──────────────┘
                           │
                    Final Predictions
```

---

## Tech Stack

Python 3.11 · PyTorch · scikit-learn · Hugging Face Transformers · librosa · miditoolkit · XGBoost · joblib · NumPy

**Models:** Aria (loubb/aria-medium-embedding) · MERT-v1-95M/330M (m-a-p) · CLAP (laion/clap-htsat-unfused) · MuQ (OpenMuQ/MuQ-large-msd-iter)

---

## Setup

```bash
git clone https://github.com/maneesh-bijjula/music-ml-classification.git
cd music-ml-classification
pip install torch transformers scikit-learn librosa miditoolkit xgboost joblib numpy
# Optional: pip install muq
python music_ml_pipeline.py
```

Set the `ASSIGNMENT1_STUDENT_ROOT` environment variable to point to your data directory, or place data in `./student_files/`.

---

## Key Findings

Frozen pretrained audio embeddings (MERT, CLAP) dramatically outperform models trained from scratch on small audio datasets — the central empirical finding of this project. On Task 3, the baseline CNN trained from scratch on MelSpectrograms achieved 0.270 mAP; the frozen MERT+CLAP ensemble achieved 0.7069 mAP, a 2.6× improvement, without any fine-tuning of the foundation models themselves.

For symbolic tasks (Tasks 1 & 2), combining transformer embeddings (Aria) with domain-specific hand-engineered features consistently outperformed either approach alone — suggesting that music-theoretic features (key profiles, harmonic continuity, voice leading) encode information that transformers do not automatically learn from token sequences alone.

---

## Results Summary

| Task | Metric | Baseline | This Pipeline |
|------|--------|----------|---------------|
| Task 1: Composer Classification | Accuracy | 0.251 | **0.9684** |
| Task 2: Next Sequence Prediction | Accuracy | 0.624 | **0.9285** |
| Task 3: Music Tagging | mAP | 0.270 | **0.7069** |
