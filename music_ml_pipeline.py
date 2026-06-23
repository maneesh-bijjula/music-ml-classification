import ast
import gc
import os
import random
import warnings
from collections import Counter
from pathlib import Path

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import joblib
import librosa
import miditoolkit
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from torch.utils.data import DataLoader, Dataset
try:
    from transformers import AutoModel, AutoProcessor, ClapModel, ClapProcessor
except ModuleNotFoundError:
    AutoModel = AutoProcessor = ClapModel = ClapProcessor = None


warnings.filterwarnings("ignore", category=UserWarning)


ROOT = Path(__file__).resolve().parent / "student_files"
CACHE_DIR = Path(__file__).resolve().parent / ".cache_assignment1"
CACHE_DIR.mkdir(exist_ok=True)

TAGS = [
    "rock",
    "oldies",
    "jazz",
    "pop",
    "dance",
    "blues",
    "punk",
    "chill",
    "electronic",
    "country",
]

_KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88], dtype=np.float64)
_KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17], dtype=np.float64)


def discover_student_root():
    candidates = []
    env_root = os.environ.get("ASSIGNMENT1_STUDENT_ROOT")
    if env_root:
        candidates.append(Path(env_root).expanduser())

    file_root = Path(__file__).resolve().parent
    home = Path.home()
    candidates.extend(
        [
            ROOT,
            file_root / "student_files",
            file_root / "student_files_latest" / "student_files",
            home / "Downloads" / "student_files_latest" / "student_files",
            home / "Downloads" / "student_files",
        ]
    )

    required_dirs = [
        "task1_composer_classification",
        "task2_next_sequence_prediction",
        "task3_audio_classification",
    ]
    for candidate in candidates:
        if all((candidate / dirname).exists() for dirname in required_dirs):
            return candidate

    return ROOT


ROOT = discover_student_root()


def require_transformers():
    if AutoModel is None or AutoProcessor is None or ClapModel is None or ClapProcessor is None:
        raise ModuleNotFoundError(
            "Task 3 requires the `transformers` package in the active Python environment."
        )


def read_python_literal(path):
    with open(path, "r") as f:
        return ast.literal_eval(f.read())


def write_submission_predictions(predictions, outpath, normalize_audio_paths=False):
    serializable = predictions
    if normalize_audio_paths:
        serializable = {
            (k[2:] if isinstance(k, str) and k.startswith("./") else k): v
            for k, v in predictions.items()
        }
    with open(outpath, "w") as f:
        f.write(repr(serializable) + "\n")


def summarize(arr):
    arr = np.asarray(arr, dtype=np.float32)
    return [
        float(arr.mean()),
        float(arr.std()),
        float(arr.min()),
        float(arr.max()),
        float(np.median(arr)),
    ]


def summarize7(arr):
    arr = np.asarray(arr, dtype=np.float32)
    if len(arr) == 0:
        return [0.0] * 7
    return [
        float(arr.mean()),
        float(arr.std()),
        float(arr.min()),
        float(arr.max()),
        float(np.median(arr)),
        float(np.percentile(arr, 25)),
        float(np.percentile(arr, 75)),
    ]


def safe_entropy(hist):
    hist = np.asarray(hist, dtype=np.float64)
    total = hist.sum()
    if total <= 0:
        return 0.0
    probs = hist[hist > 0] / total
    return float(-(probs * np.log2(probs)).sum())


def ks_key_correlations(pch):
    pch = np.asarray(pch, dtype=np.float64)
    if pch.sum() <= 0:
        return np.zeros(24, dtype=np.float32)
    pch = pch / pch.sum()
    corrs = []
    for root in range(12):
        corrs.append(np.corrcoef(pch, np.roll(_KS_MAJOR, root))[0, 1])
        corrs.append(np.corrcoef(pch, np.roll(_KS_MINOR, root))[0, 1])
    return np.nan_to_num(np.asarray(corrs, dtype=np.float32))


def best_ks_scale_mask(pch):
    corrs = ks_key_correlations(pch)
    if len(corrs) == 0:
        return np.zeros(12, dtype=np.float32)
    best_idx = int(np.argmax(corrs))
    root = best_idx // 2
    scale = _MAJOR_SCALE if best_idx % 2 == 0 else _MINOR_SCALE
    mask = np.zeros(12, dtype=np.float32)
    for step in scale:
        mask[(root + step) % 12] = 1.0
    return mask


def safe_corr(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if len(a) == 0 or len(b) == 0 or a.std() <= 1e-6 or b.std() <= 1e-6:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def normalized_hist(values, length):
    values = np.asarray(values)
    if values.size == 0:
        return np.zeros(length, dtype=np.float32)
    hist = np.bincount(np.clip(values.astype(int), 0, length - 1), minlength=length).astype(np.float32)
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


def log_duration_bins(values, length=12):
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return np.zeros(length, dtype=np.float32)
    bins = np.clip(np.round(np.log2(np.maximum(values, 1.0) + 1.0)).astype(int), 0, length - 1)
    return normalized_hist(bins, length)


def accuracy1(groundtruth, predictions):
    return sum(int(predictions[k] == groundtruth[k]) for k in groundtruth) / len(groundtruth)


def accuracy2(groundtruth, predictions):
    return sum(int(predictions[k] == groundtruth[k]) for k in groundtruth) / len(groundtruth)


def accuracy3(groundtruth, predictions):
    preds, targets = [], []
    for k in groundtruth:
        preds.append([predictions[k][tag] for tag in TAGS])
        targets.append([1 if tag in groundtruth[k] else 0 for tag in TAGS])
    return average_precision_score(targets, preds, average="macro")


def _circle_of_fifths_dist(pc1, pc2):
    cof = [0, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10, 5]
    pos1 = cof.index(int(pc1) % 12)
    pos2 = cof.index(int(pc2) % 12)
    return min(abs(pos1 - pos2), 12 - abs(pos1 - pos2))


def _dominant_resolution_score(src_pcs, tgt_pcs):
    if not src_pcs or not tgt_pcs:
        return 0.0
    src_root = min(src_pcs) % 12
    tgt_root = min(tgt_pcs) % 12
    if (tgt_root - src_root) % 12 == 5:
        return 1.0
    if (tgt_root - src_root) % 12 == 7:
        return 0.7
    if (tgt_root - src_root) % 12 == 2:
        return 0.5
    return 0.0


_MAJOR_SCALE = {0, 2, 4, 5, 7, 9, 11}
_MINOR_SCALE = {0, 2, 3, 5, 7, 8, 10}


def _key_compatibility(src_pcs, tgt_pcs):
    if not src_pcs or not tgt_pcs:
        return 0.0
    tgt_root = min(tgt_pcs) % 12
    diatonic = {(tgt_root + i) % 12 for i in _MAJOR_SCALE}
    return sum(1 for pc in src_pcs if pc % 12 in diatonic) / max(1, len(src_pcs))


def harmonic_continuity_features(src_last_chord, tgt_first_chord):
    src_pcs = set((src_last_chord.astype(int) % 12).tolist()) if len(src_last_chord) else set()
    tgt_pcs = set((tgt_first_chord.astype(int) % 12).tolist()) if len(tgt_first_chord) else set()
    if not src_pcs or not tgt_pcs:
        return np.zeros(8, dtype=np.float32)
    src_root = int(min(src_pcs))
    tgt_root = int(min(tgt_pcs))
    cof_dist = _circle_of_fifths_dist(src_root, tgt_root)
    dom_res = _dominant_resolution_score(src_pcs, tgt_pcs)
    key_compat = _key_compatibility(src_pcs, tgt_pcs)
    pc_overlap = len(src_pcs & tgt_pcs) / max(1, len(src_pcs | tgt_pcs))
    semitone_dist = min((tgt_root - src_root) % 12, (src_root - tgt_root) % 12)
    voice_leading = sum(min((sp - tp) % 12 for tp in tgt_pcs) for sp in src_pcs) / max(1, len(src_pcs))
    tritone = 1.0 if semitone_dist == 6 else 0.0
    common_tones = len(src_pcs & tgt_pcs) / max(1, len(src_pcs))
    return np.array(
        [cof_dist, dom_res, key_compat, pc_overlap, semitone_dist, voice_leading, tritone, common_tones],
        dtype=np.float32,
    )


# =========================
# Task 1 exact standalone code
# =========================

"""
Task 1 v17 - fast cached-Aria + LR stabilizer
Key idea:
  1. 3 Aria embeddings per piece (first/middle/last 512 tokens) concatenated → 1536-dim
  2. Adds no-extra-extraction Aria slice statistics: mean/std/max/min and section deltas
  3. Uses a small fast ensemble plus LR stabilizer so it can finish before the deadline
  4. Writes several prediction variants so we can test without rerunning

Install:
!pip install transformers==4.44.0 git+https://github.com/EleutherAI/aria-utils.git miditoolkit xgboost joblib scikit-learn
"""

import ast
import warnings
from collections import Counter
from pathlib import Path

import joblib
import miditoolkit
import numpy as np
import torch
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("XGBoost not found: !pip install xgboost")

warnings.filterwarnings("ignore")

ROOT      = Path("/content/student_files/task1_composer_classification")
CACHE_DIR = Path("/content/task1_cache_v14")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT    = "/content/predictions1.json"

ARIA_EMB_MODEL = "loubb/aria-medium-embedding"
ARIA_MAX_LEN   = 2048
ARIA_SLICE_LEN = 512   # tokens per slice
ARIA_EMB_DIM   = 512   # per slice
ARIA_TOTAL_DIM = 512 * 3  # 3 slices concatenated = 1536

_KS_MAJOR    = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88], dtype=np.float64)
_KS_MINOR    = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17], dtype=np.float64)
_MAJOR_SCALE = {0,2,4,5,7,9,11}
_MINOR_SCALE = {0,2,3,5,7,8,10}


def read_literal(path):
    with open(path) as f:
        return ast.literal_eval(f.read())

def write_predictions(predictions, outpath):
    with open(outpath, "w") as f:
        f.write(repr(predictions) + "\n")

def summarize7(arr):
    arr = np.asarray(arr, dtype=np.float32)
    if len(arr) == 0: return [0.0]*7
    return [float(arr.mean()), float(arr.std()), float(arr.min()), float(arr.max()),
            float(np.median(arr)), float(np.percentile(arr,25)), float(np.percentile(arr,75))]

def safe_entropy(hist):
    hist = np.asarray(hist, dtype=np.float64)
    total = hist.sum()
    if total <= 0: return 0.0
    probs = hist[hist>0]/total
    return float(-(probs*np.log2(probs)).sum())

def ks_key_correlations(pch):
    pch = np.asarray(pch, dtype=np.float64)
    if pch.sum() <= 0: return np.zeros(24, dtype=np.float32)
    pch = pch/pch.sum()
    corrs = []
    for root in range(12):
        corrs.append(np.corrcoef(pch, np.roll(_KS_MAJOR,root))[0,1])
        corrs.append(np.corrcoef(pch, np.roll(_KS_MINOR,root))[0,1])
    return np.nan_to_num(np.asarray(corrs, dtype=np.float32))

def best_ks_scale_mask(pch):
    corrs = ks_key_correlations(pch)
    if len(corrs)==0: return np.zeros(12, dtype=np.float32)
    best_idx = int(np.argmax(corrs))
    root = best_idx//2
    scale = _MAJOR_SCALE if best_idx%2==0 else _MINOR_SCALE
    mask = np.zeros(12, dtype=np.float32)
    for step in scale: mask[(root+step)%12] = 1.0
    return mask


# ── Aria multi-slice embedding extraction ────────────────────────────────────

def load_aria_model(device):
    print("Loading Aria embedding model...")
    config = AutoConfig.from_pretrained(ARIA_EMB_MODEL, trust_remote_code=True)
    if not hasattr(config, "initializer_range"):
        config.initializer_range = 0.02
    model = AutoModelForCausalLM.from_pretrained(
        ARIA_EMB_MODEL, config=config, trust_remote_code=True
    ).to(device)
    model.eval()
    # Aria's remote tokenizer imports BatchEncoding from an older transformers path.
    import transformers.tokenization_utils as tokenization_utils
    from transformers.tokenization_utils_base import BatchEncoding
    tokenization_utils.BatchEncoding = BatchEncoding
    tokenizer = AutoTokenizer.from_pretrained(ARIA_EMB_MODEL, trust_remote_code=True)
    print(f"  Aria loaded on {device}")
    return model, tokenizer


def extract_single_embedding(input_ids, model, device):
    """Extract embedding from a single tokenized slice."""
    with torch.no_grad():
        outputs = model.forward(input_ids=input_ids.to(device))
    return outputs[0].squeeze().cpu().numpy().astype(np.float32)


def extract_aria_multislice(midi_path, model, tokenizer, device):
    """
    Extract 3 Aria embeddings from different parts of the piece:
    - Slice 1: first ARIA_SLICE_LEN tokens (opening)
    - Slice 2: middle ARIA_SLICE_LEN tokens (development)
    - Slice 3: last ARIA_SLICE_LEN tokens (ending)
    Concatenate → 1536-dim total
    """
    try:
        prompt = tokenizer.encode_from_file(str(midi_path), return_tensors="pt")
        input_ids = prompt.input_ids  # shape: (1, total_len)
        total_len = input_ids.shape[1]
        eos_id = tokenizer._convert_token_to_id(tokenizer.eos_token)

        def get_slice(ids, start, length):
            end = min(start + length, ids.shape[1])
            slc = ids[:, start:end].clone()
            # Ensure EOS at end
            if slc.shape[1] < length:
                # pad with zeros and put EOS at last real position
                pass
            slc[:, -1] = eos_id
            return slc

        # Slice 1: first tokens
        s1 = get_slice(input_ids, 0, ARIA_SLICE_LEN)
        emb1 = extract_single_embedding(s1, model, device)

        # Slice 2: middle tokens
        mid = max(0, total_len//2 - ARIA_SLICE_LEN//2)
        s2 = get_slice(input_ids, mid, ARIA_SLICE_LEN)
        emb2 = extract_single_embedding(s2, model, device)

        # Slice 3: last tokens
        last = max(0, total_len - ARIA_SLICE_LEN)
        s3 = get_slice(input_ids, last, ARIA_SLICE_LEN)
        emb3 = extract_single_embedding(s3, model, device)

        # Handle shape — could be 1D (512,) or needs reshaping
        for e in [emb1, emb2, emb3]:
            if e.shape != (ARIA_EMB_DIM,):
                e = e.flatten()[:ARIA_EMB_DIM]

        return np.concatenate([emb1, emb2, emb3]).astype(np.float32)

    except Exception as e:
        print(f"  Aria failed for {midi_path}: {e}")
        return np.zeros(ARIA_TOTAL_DIM, dtype=np.float32)


def extract_all_aria_embeddings(paths, root, model, tokenizer, device, cache_path):
    cache = joblib.load(cache_path) if Path(cache_path).exists() else {}
    missing = [p for p in paths if p not in cache]
    if missing:
        print(f"  Extracting {len(missing)} Aria multi-slice embeddings...")
        for idx, rel_path in enumerate(missing, 1):
            midi_path = Path(root) / rel_path
            cache[rel_path] = extract_aria_multislice(midi_path, model, tokenizer, device)
            if idx % 100 == 0 or idx == len(missing):
                print(f"  {idx}/{len(missing)}  shape: {cache[rel_path].shape}")
                joblib.dump(cache, cache_path)
    else:
        print(f"  All {len(paths)} Aria embeddings cached")
    return np.stack([cache[p] for p in paths]).astype(np.float32)


def expand_aria_features(aria_embs):
    """Reuse the cached three slices and expose section-level trajectory features."""
    aria_embs = np.asarray(aria_embs, dtype=np.float32)
    slices = aria_embs.reshape(len(aria_embs), 3, ARIA_EMB_DIM)
    first = slices[:, 0, :]
    middle = slices[:, 1, :]
    last = slices[:, 2, :]
    mean = slices.mean(axis=1)
    std = slices.std(axis=1)
    maxv = slices.max(axis=1)
    minv = slices.min(axis=1)
    end_minus_start = last - first
    middle_minus_edges = middle - 0.5 * (first + last)
    return np.concatenate(
        [aria_embs, mean, std, maxv, minv, end_minus_start, middle_minus_edges],
        axis=1,
    ).astype(np.float32)


# ── v10 feature extraction (unchanged) ───────────────────────────────────────

def extract_v10_features(midi, notes):
    if not notes:
        return np.zeros(441, dtype=np.float32)

    pitches    = np.array([n.pitch    for n in notes], dtype=np.float32)
    starts     = np.array([n.start    for n in notes], dtype=np.float32)
    ends       = np.array([n.end      for n in notes], dtype=np.float32)
    durations  = np.maximum(1, ends-starts)
    velocities = np.array([n.velocity for n in notes], dtype=np.float32)
    iois       = np.diff(starts)
    pitch_diffs     = np.diff(pitches)
    abs_pitch_diffs = np.abs(pitch_diffs)
    span       = max(1.0, float(ends.max()-starts.min()))
    polyphony  = np.array(list(Counter(starts.astype(int)).values()), dtype=np.float32)
    tempos     = np.array([t.tempo for t in midi.tempo_changes], dtype=np.float32) if midi.tempo_changes else np.array([120.0], dtype=np.float32)

    vel_std      = float(velocities.std())
    vel_zero_var = 1.0 if vel_std<0.5 else 0.0
    vel_feats    = [vel_std, vel_zero_var, float(velocities.max()-velocities.min()),
                    float(len(np.unique(velocities)))/128.0] + \
                   [float(np.percentile(velocities,p)) for p in [10,25,50,75,90]]

    if len(iois)>0:
        ioi_cv  = float(np.std(iois)/(np.mean(iois)+1e-6))
        med_ioi = float(np.median(iois))
        ioi_feats = [float(np.percentile(iois,p)) for p in [10,25,50,75,90]] + \
                    [ioi_cv, float((iois<med_ioi*0.5).mean()), float((iois>med_ioi*2).mean())]
    else:
        ioi_feats = [0.0]*8; ioi_cv=0.0

    pitch_range  = float(pitches.max()-pitches.min())
    bass_notes   = float((pitches<48).mean())
    treble_notes = float((pitches>72).mean())
    mid_notes    = float(((pitches>=48)&(pitches<=72)).mean())
    btg = float(np.mean(pitches[pitches>np.median(pitches)])-np.mean(pitches[pitches<=np.median(pitches)])) if len(pitches)>1 else 0.0
    register_feats = [pitch_range, bass_notes, treble_notes, mid_notes, btg]

    sn  = float((durations<np.percentile(durations,10)).mean())
    vsn = float((durations<50).mean())
    ornament_feats = [sn, vsn]

    third = span/3.0
    section_feats = []
    for lo,hi in [(0,third),(third,2*third),(2*third,span)]:
        mask = (starts>=lo)&(starts<hi)
        if mask.sum()>0:
            section_feats.extend([float(pitches[mask].mean()),float(velocities[mask].mean()),float(velocities[mask].std()),float(mask.sum())/max(1,len(notes))])
        else:
            section_feats.extend([0.0,0.0,0.0,0.0])

    chord_sizes = np.array(list(Counter(starts.astype(int)).values()), dtype=np.float32)
    mono_ratio  = float((chord_sizes==1).mean())
    duo_ratio   = float((chord_sizes==2).mean())
    poly_ratio  = float((chord_sizes>=3).mean())
    voice_feats = [mono_ratio, duo_ratio, poly_ratio]

    repeated_ratio = float((pitch_diffs==0).mean()) if len(pitch_diffs) else 0.0
    trill_ratio    = float((np.abs(pitch_diffs)<=2).mean()) if len(pitch_diffs) else 0.0

    interval_feats = [
        float((abs_pitch_diffs==0).mean()), float((abs_pitch_diffs<=2).mean()),
        float(((abs_pitch_diffs>=3)&(abs_pitch_diffs<=4)).mean()),
        float(((abs_pitch_diffs>=5)&(abs_pitch_diffs<=6)).mean()),
        float((abs_pitch_diffs==7).mean()), float((abs_pitch_diffs>=8).mean()),
    ] if len(abs_pitch_diffs) else [0.0]*6

    pcs = pitches.astype(int)%12
    pitch_classes = np.bincount(pcs, minlength=12).astype(np.float32)
    if pitch_classes.sum(): pitch_classes /= pitch_classes.sum()
    octaves = np.bincount(np.clip(pitches.astype(int)//12,0,10), minlength=11).astype(np.float32)
    if octaves.sum(): octaves /= octaves.sum()
    duration_bins = np.bincount(np.clip(np.round(np.log2(durations+1)).astype(int),0,15), minlength=16).astype(np.float32)
    if duration_bins.sum(): duration_bins /= duration_bins.sum()
    ioi_bins = np.bincount(np.clip(np.round(np.log2(np.maximum(iois,1)+1)).astype(int),0,15), minlength=16).astype(np.float32) if len(iois) else np.zeros(16,dtype=np.float32)
    if ioi_bins.sum(): ioi_bins /= ioi_bins.sum()

    pc_trans = np.zeros((12,12), dtype=np.float32)
    if len(pitches)>1:
        left=pitches[:-1].astype(int)%12; right=pitches[1:].astype(int)%12
        for s,d in zip(left,right): pc_trans[s,d]+=1
        pc_trans /= max(1.0, pc_trans.sum())

    ks_corrs   = ks_key_correlations(pitch_classes)
    scale_mask = best_ks_scale_mask(pitch_classes)
    dissonance = float(sum(pitch_classes[pc] for pc in range(12) if scale_mask[pc]==0.0))

    scalar_feats = []
    for arr in [pitches, durations, velocities,
                iois if len(iois) else [0.0],
                pitch_diffs if len(pitch_diffs) else [0.0],
                abs_pitch_diffs if len(abs_pitch_diffs) else [0.0],
                polyphony, tempos]:
        scalar_feats.extend(summarize7(arr))

    base = (
        np.asarray(scalar_feats + [
            len(notes), span, len(notes)/span,
            float(polyphony.mean()), float(polyphony.max()),
            safe_entropy(pitch_classes), safe_entropy(octaves),
            safe_entropy(duration_bins), safe_entropy(ioi_bins),
            repeated_ratio, trill_ratio, dissonance,
            float(len(notes)/span), ioi_cv,
        ], dtype=np.float32).tolist()
        + pitch_classes.tolist() + octaves.tolist()
        + duration_bins.tolist() + ioi_bins.tolist()
        + pc_trans.reshape(-1).tolist() + ks_corrs.tolist()
        + vel_feats + ioi_feats + register_feats + ornament_feats
        + section_feats + voice_feats + interval_feats
        + [vel_zero_var*pitch_range, vel_zero_var*mono_ratio,
           vel_std*ioi_cv, pitch_range*poly_ratio,
           bass_notes*poly_ratio, treble_notes*sn,
           float(np.mean(velocities))*vel_std]
    )

    # Rich features
    trigram_counts = Counter()
    if len(pcs)>2:
        for idx in range(len(pcs)-2):
            trigram_counts[(pcs[idx],pcs[idx+1],pcs[idx+2])]+=1
    total_t = sum(trigram_counts.values()) or 1
    top_trigrams = [c/total_t for _,c in trigram_counts.most_common(24)]
    top_trigrams += [0.0]*(24-len(top_trigrams))

    interval_steps = np.diff(pitches.astype(int))
    interval_bins  = np.bincount(np.clip(np.abs(interval_steps),0,36), minlength=37).astype(np.float32)
    if interval_bins.sum(): interval_bins /= interval_bins.sum()

    rel_intervals = np.clip(interval_steps,-12,11).astype(np.int32)
    int_trans = np.zeros((24,24), dtype=np.float32)
    if len(rel_intervals)>1:
        s=rel_intervals[:-1]+12; d=rel_intervals[1:]+12
        for l,r in zip(s,d): int_trans[l,r]+=1.0
        int_trans /= max(1.0, int_trans.sum())

    if len(iois):
        ioi_std=float(np.std(iois)); ioi_mean=float(np.mean(iois))
        ioi_cv2=float(ioi_std/(ioi_mean+1e-6))
        ioi_ent=safe_entropy(np.bincount(np.clip(np.round(np.log2(np.maximum(iois,1)+1)).astype(int),0,31),minlength=32).astype(np.float32))
        lioi=np.diff(iois)
        lioi_bins=np.bincount(np.clip(np.round(np.log2(np.maximum(np.abs(lioi),1)+1)).astype(int),0,15),minlength=16).astype(np.float32)
        if lioi_bins.sum(): lioi_bins/=lioi_bins.sum()
    else:
        ioi_std=ioi_mean=ioi_cv2=ioi_ent=0.0
        lioi_bins=np.zeros(16,dtype=np.float32)

    mel_ent = safe_entropy(np.bincount(np.clip(rel_intervals+12,0,23),minlength=24).astype(np.float32) if len(rel_intervals) else np.zeros(24,dtype=np.float32))
    dur_prof = np.bincount(np.clip(np.round(np.log2(durations+1)).astype(int),0,23),minlength=24).astype(np.float32)
    if dur_prof.sum(): dur_prof/=dur_prof.sum()
    ioi_prof = np.bincount(np.clip(np.round(np.log2(np.maximum(iois,1)+1)).astype(int),0,23),minlength=24).astype(np.float32) if len(iois) else np.zeros(24,dtype=np.float32)
    if ioi_prof.sum(): ioi_prof/=ioi_prof.sum()

    rep_pairs = float(sum(c>1 for c in Counter(tuple(pcs[i:i+4]) for i in range(len(pcs)-3)).values())) if len(pcs)>3 else 0.0

    rich_extra = np.concatenate([
        np.asarray(top_trigrams, dtype=np.float32),
        interval_bins, dur_prof, ioi_prof,
        np.asarray([rep_pairs,
                    float((durations>=np.percentile(durations,75)).mean()),
                    float((velocities>=np.percentile(velocities,80)).mean()),
                    float((pitches<48).mean()), float((pitches>72).mean()),
                    mel_ent, ioi_ent, ioi_std, ioi_mean, ioi_cv2], dtype=np.float32),
        lioi_bins, int_trans.reshape(-1),
    ])

    return np.concatenate([np.asarray(base, dtype=np.float32), rich_extra]).astype(np.float32)


# ── Main model ────────────────────────────────────────────────────────────────

class Task1V14Model:
    def __init__(self):
        self.feat_cache_path  = CACHE_DIR / "v10_feats.joblib"
        self.aria_train_path  = CACHE_DIR / "aria_train_3slice.joblib"
        self.aria_test_path   = CACHE_DIR / "aria_test_3slice.joblib"
        self.feat_cache       = joblib.load(self.feat_cache_path) if self.feat_cache_path.exists() else {}

        self.base_models = [
            ("hgb_fast", HistGradientBoostingClassifier(max_depth=5, learning_rate=0.05, max_iter=420, random_state=42)),
            ("rf_fast", RandomForestClassifier(n_estimators=650, max_features="sqrt", class_weight="balanced", random_state=42, n_jobs=-1)),
            ("extra_fast", ExtraTreesClassifier(n_estimators=900, max_features="sqrt", class_weight="balanced", random_state=42, n_jobs=-1)),
            ("lr_fast", make_pipeline(StandardScaler(), LogisticRegression(max_iter=1200, C=1.5, class_weight="balanced", n_jobs=-1))),
        ]
        if HAS_XGB:
            self.base_models.append((
                "xgb_fast", XGBClassifier(
                    n_estimators=420, max_depth=4, learning_rate=0.04,
                    subsample=0.85, colsample_bytree=0.78,
                    eval_metric="mlogloss", random_state=42, n_jobs=-1,
                )
            ))

        self.base_fitted   = []
        self.rich_pipeline = None
        self.blend_alpha   = 0.0
        self.base_weights  = None
        self.device        = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def _load_notes(self, rel_path):
        midi = miditoolkit.MidiFile(str(ROOT / rel_path))
        notes = []
        for inst in midi.instruments: notes.extend(inst.notes)
        notes.sort(key=lambda n: (n.start, n.pitch))
        return midi, notes

    def get_v10_features(self, rel_path):
        if rel_path not in self.feat_cache:
            midi, notes = self._load_notes(rel_path)
            self.feat_cache[rel_path] = extract_v10_features(midi, notes)
        return self.feat_cache[rel_path]

    def build_combined_matrix(self, paths, aria_embs):
        v10 = np.stack([self.get_v10_features(p) for p in paths])
        aria_expanded = expand_aria_features(aria_embs)
        return np.concatenate([v10, aria_expanded], axis=1)

    def _weighted_average(self, prob_list, weights):
        weights = np.asarray(weights, dtype=np.float32)
        weights = weights / max(float(weights.sum()), 1e-8)
        out = None
        for weight, probs in zip(weights, prob_list):
            out = probs * weight if out is None else out + probs * weight
        return out

    def tune_base_weights(self, prob_list, y_val):
        candidates = []
        n = len(prob_list)
        candidates.append(("equal", np.ones(n, dtype=np.float32)))

        if n == 5:
            candidates.extend([
                ("tree_boost_lr", np.array([0.18, 0.18, 0.27, 0.08, 0.29], dtype=np.float32)),
                ("lr_stable", np.array([0.20, 0.18, 0.24, 0.18, 0.20], dtype=np.float32)),
                ("extra_xgb_lr", np.array([0.12, 0.16, 0.30, 0.12, 0.30], dtype=np.float32)),
                ("balanced", np.array([0.20, 0.20, 0.20, 0.20, 0.20], dtype=np.float32)),
            ])
            alpha = np.array([1.5, 1.8, 2.2, 1.2, 2.1], dtype=np.float32)
        elif n == 4:
            candidates.extend([
                ("tree_boost", np.array([0.20, 0.22, 0.30, 0.28], dtype=np.float32)),
                ("extra_xgb", np.array([0.12, 0.18, 0.35, 0.35], dtype=np.float32)),
                ("balanced", np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)),
            ])
            alpha = np.array([1.4, 1.8, 2.3, 2.3], dtype=np.float32)
        elif n == 3:
            candidates.extend([
                ("tree_heavy", np.array([0.20, 0.35, 0.45], dtype=np.float32)),
                ("balanced", np.array([0.33, 0.33, 0.34], dtype=np.float32)),
            ])
            alpha = np.array([1.4, 2.2, 2.4], dtype=np.float32)
        else:
            alpha = np.ones(n, dtype=np.float32)

        rng = np.random.default_rng(2026)
        for i in range(600):
            candidates.append((f"rand_{i}", rng.dirichlet(alpha).astype(np.float32)))

        best_name, best_weights, best_acc = None, None, -1.0
        for name, weights in candidates:
            blended = self._weighted_average(prob_list, weights)
            acc = accuracy_score(y_val, blended.argmax(1))
            if acc > best_acc:
                best_name, best_weights, best_acc = name, weights.copy(), acc

        best_weights = best_weights / best_weights.sum()
        print(f"  Selected base weights ({best_name}) val_acc={best_acc:.4f}: {best_weights.tolist()}")
        return best_weights

    def train(self, train_json):
        labels = read_literal(train_json)
        paths  = list(labels)
        y      = np.array([labels[p] for p in paths])

        # v10 features
        print(f"Extracting v10 features for {len(paths)} files...")
        _ = [self.get_v10_features(p) for p in paths]
        joblib.dump(self.feat_cache, self.feat_cache_path)

        # Aria 3-slice embeddings
        print("Extracting Aria 3-slice embeddings...")
        aria_model, aria_tokenizer = load_aria_model(self.device)
        aria_train = extract_all_aria_embeddings(
            paths, ROOT, aria_model, aria_tokenizer, self.device, self.aria_train_path)
        del aria_model; torch.cuda.empty_cache() if self.device.type=="cuda" else None
        print(f"Aria 3-slice dim: {aria_train.shape[1]}")

        X = self.build_combined_matrix(paths, aria_train)
        print(f"Combined feature shape: {X.shape}")
        sw = compute_sample_weight("balanced", y)

        print("Skipping 5-fold CV for deadline-speed run.")

        Xb_tr, Xb_val, y_tr, y_val = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
        sw_tr = compute_sample_weight("balanced", y_tr)

        base_prob_list = []
        self.base_fitted = []
        for name, model in self.base_models:
            print(f"  Training {name}...")
            fitted = clone(model)
            if "hgb" in name or "xgb" in name:
                fitted.fit(Xb_tr, y_tr, sample_weight=sw_tr)
            else:
                fitted.fit(Xb_tr, y_tr)
            current = fitted.predict_proba(Xb_val)
            base_prob_list.append(current)
            print(f"    val_acc={accuracy_score(y_val, fitted.predict(Xb_val)):.4f}")
            self.base_fitted.append((name, fitted))
        self.base_weights = self.tune_base_weights(base_prob_list, y_val)
        base_probs = self._weighted_average(base_prob_list, self.base_weights)
        print(f"  Base ensemble val_acc={accuracy_score(y_val, base_probs.argmax(1)):.4f}")

        self.rich_pipeline = None
        self.blend_alpha = 0.0
        print("Skipping rich ensemble for deadline-speed run.")

        print("Retraining on full data...")
        self.base_fitted = []
        for name, model in self.base_models:
            print(f"  {name}...")
            fitted = clone(model)
            if "hgb" in name or "xgb" in name:
                fitted.fit(X, y, sample_weight=sw)
            else:
                fitted.fit(X, y)
            self.base_fitted.append((name, fitted))
        print("Done!")

    def predict(self, test_json, outpath):
        entries = read_literal(test_json)
        keys    = list(entries)
        print(f"Predicting {len(keys)} test files...")

        print("Extracting Aria test embeddings...")
        aria_model, aria_tokenizer = load_aria_model(self.device)
        aria_test = extract_all_aria_embeddings(
            keys, ROOT, aria_model, aria_tokenizer, self.device, self.aria_test_path)
        del aria_model; torch.cuda.empty_cache() if self.device.type=="cuda" else None

        X_test = self.build_combined_matrix(keys, aria_test)
        n_cls  = len(set(read_literal(ROOT / "train.json").values()))

        base_prob_list = []
        for _, model in self.base_fitted:
            current = model.predict_proba(X_test)
            base_prob_list.append(current)
        base_probs = self._weighted_average(base_prob_list, self.base_weights)
        rich_probs = base_probs if self.rich_pipeline is None else self.rich_pipeline.predict_proba(X_test)
        probs = (1.0-self.blend_alpha)*base_probs + self.blend_alpha*rich_probs
        preds = probs.argmax(axis=1)
        write_predictions({k: int(p) for k,p in zip(keys,preds)}, outpath)
        print(f"Wrote {outpath}")

        outpath = Path(outpath)
        equal_base = self._weighted_average(base_prob_list, np.ones(len(base_prob_list), dtype=np.float32))
        variants = {
            "predictions1_v17_equalbase.json": equal_base,
            "predictions1_v17_weightedbase.json": base_probs,
            "predictions1_v17_treeheavy.json": self._weighted_average(
                base_prob_list,
                np.array([0.12, 0.16, 0.30, 0.12, 0.30], dtype=np.float32)[: len(base_prob_list)],
            ),
        }
        primary_preds = preds
        for filename, variant_probs in variants.items():
            variant_preds = variant_probs.argmax(axis=1)
            variant = {k: int(p) for k,p in zip(keys, variant_preds)}
            variant_path = outpath.with_name(filename)
            write_predictions(variant, variant_path)
            print(f"Wrote {variant_path} ({int((variant_preds != primary_preds).sum())} predictions differ from primary)")


if __name__ == "__main__":
    # !pip install transformers==4.44.0 git+https://github.com/EleutherAI/aria-utils.git miditoolkit xgboost
    model = Task1V14Model()
    model.train(ROOT / "train.json")
    model.predict(ROOT / "test.json", OUTPUT)



class Task2SequenceModel:
    def __init__(self):
        self.root = ROOT / "task2_next_sequence_prediction"
        self.feature_cache_path = CACHE_DIR / "task2_sequence_features_v12.joblib"
        self.segment_cache = joblib.load(self.feature_cache_path) if self.feature_cache_path.exists() else {}
        self.models = [
            ("hgb_a", HistGradientBoostingClassifier(max_depth=8, learning_rate=0.04, max_iter=450, random_state=42)),
            ("hgb_b", HistGradientBoostingClassifier(max_depth=10, learning_rate=0.03, max_iter=550, random_state=7)),
            ("extra", ExtraTreesClassifier(n_estimators=1400, random_state=42, n_jobs=-1)),
            ("rf", RandomForestClassifier(n_estimators=1000, random_state=42, n_jobs=-1, class_weight="balanced_subsample")),
            ("lr", make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000, C=2.5))),
        ]
        self.fitted = []
        self.ensemble_weights = None
        self.decision_threshold = 0.5
        self.consistency_alpha = 0.0

    def segment_features(self, rel_path):
        if rel_path in self.segment_cache:
            return self.segment_cache[rel_path]

        midi = miditoolkit.MidiFile(str(self.root / rel_path))
        notes = []
        for inst in midi.instruments:
            notes.extend(inst.notes)
        notes.sort(key=lambda n: (n.start, n.pitch, n.end, n.velocity))
        if not notes:
            empty = {
                "base": np.zeros(110, dtype=np.float32),
                "first_notes": np.zeros((8, 4), dtype=np.float32),
                "last_notes": np.zeros((8, 4), dtype=np.float32),
                "head_windows": {},
                "tail_windows": {},
                "pcs": np.zeros(12, dtype=np.float32),
                "transitions": np.zeros((12, 12), dtype=np.float32),
                "first_chord": np.zeros(0, dtype=np.float32),
                "last_chord": np.zeros(0, dtype=np.float32),
                "last_pitch": 0.0,
                "first_pitch": 0.0,
                "last_vel": 64.0,
                "first_vel": 64.0,
                "last_dur": 1.0,
                "first_dur": 1.0,
                "last_tempo": 120.0,
                "end_melody_dir": 0.0,
                "start_melody_dir": 0.0,
                "end_vel_mean": 64.0,
                "start_vel_mean": 64.0,
            }
            self.segment_cache[rel_path] = empty
            return empty

        pitches = np.array([n.pitch for n in notes], dtype=np.float32)
        starts = np.array([n.start for n in notes], dtype=np.float32)
        ends = np.array([n.end for n in notes], dtype=np.float32)
        durations = np.maximum(1, ends - starts)
        velocities = np.array([n.velocity for n in notes], dtype=np.float32)
        iois = np.diff(starts)
        pitch_classes = np.bincount((pitches.astype(int) % 12), minlength=12).astype(np.float32)
        if pitch_classes.sum():
            pitch_classes /= pitch_classes.sum()
        onset_counts = Counter(starts.astype(int).tolist())
        polyphony = np.array(list(onset_counts.values()), dtype=np.float32)
        span = max(1.0, float(ends.max() - starts.min()))
        pitch_diffs = np.diff(pitches)
        abs_pitch_diffs = np.abs(pitch_diffs)

        descriptors = []
        prev_start = starts[0]
        for note in notes:
            descriptors.append([note.pitch, max(1, note.end - note.start), max(0, note.start - prev_start), note.velocity])
            prev_start = note.start
        descriptors = np.asarray(descriptors, dtype=np.float32)

        first_notes = np.zeros((8, 4), dtype=np.float32)
        last_notes = np.zeros((8, 4), dtype=np.float32)
        first_notes[: min(8, len(descriptors))] = descriptors[: min(8, len(descriptors))]
        last_notes[-min(8, len(descriptors)) :] = descriptors[-min(8, len(descriptors)) :]

        first_onset = starts.min()
        last_onset = starts.max()
        first_chord = np.array([n.pitch for n in notes if n.start == first_onset], dtype=np.float32)
        last_chord = np.array([n.pitch for n in notes if n.start == last_onset], dtype=np.float32)

        transitions = np.zeros((12, 12), dtype=np.float32)
        if len(pitches) > 1:
            left = pitches[:-1].astype(int) % 12
            right = pitches[1:].astype(int) % 12
            for src, dst in zip(left, right):
                transitions[src, dst] += 1
            transitions /= max(1.0, transitions.sum())

        thirds = span / 3.0
        early_density = float(((starts >= 0) & (starts < thirds)).sum()) / max(1, len(notes))
        mid_density = float(((starts >= thirds) & (starts < 2 * thirds)).sum()) / max(1, len(notes))
        late_density = float((starts >= 2 * thirds).sum()) / max(1, len(notes))
        contour = [
            float((pitch_diffs > 0).mean()) if len(pitch_diffs) else 0.0,
            float((pitch_diffs < 0).mean()) if len(pitch_diffs) else 0.0,
            float((pitch_diffs == 0).mean()) if len(pitch_diffs) else 0.0,
            float((abs_pitch_diffs <= 2).mean()) if len(abs_pitch_diffs) else 0.0,
            float((abs_pitch_diffs > 7).mean()) if len(abs_pitch_diffs) else 0.0,
        ]

        feats = []
        for arr in [
            pitches,
            durations,
            velocities,
            iois if len(iois) else [0.0],
            polyphony,
            first_chord,
            last_chord,
            pitch_diffs if len(pitch_diffs) else [0.0],
            abs_pitch_diffs if len(abs_pitch_diffs) else [0.0],
        ]:
            feats.extend(summarize(arr))
        feats.extend(
            [
                len(notes),
                span,
                len(notes) / span,
                float(polyphony.mean()) if len(polyphony) else 0.0,
                float(polyphony.max()) if len(polyphony) else 0.0,
                early_density,
                mid_density,
                late_density,
            ]
        )
        feats.extend(contour)
        feats.extend(pitch_classes.tolist())
        feats.extend(transitions.reshape(-1)[:36].tolist())

        tempos = [t.tempo for t in midi.tempo_changes] if midi.tempo_changes else [120.0]
        last_tempo = float(tempos[-1])
        end_pitches = pitches[-min(4, len(pitches)) :]
        start_pitches = pitches[: min(4, len(pitches))]
        end_mel_dir = float(np.mean(np.diff(end_pitches))) if len(end_pitches) > 1 else 0.0
        start_mel_dir = float(np.mean(np.diff(start_pitches))) if len(start_pitches) > 1 else 0.0
        end_vel_mean = float(velocities[-min(8, len(velocities)) :].mean())
        start_vel_mean = float(velocities[: min(8, len(velocities))].mean())

        def make_window(mask_or_indices):
            if isinstance(mask_or_indices, slice):
                idx = np.arange(len(notes))[mask_or_indices]
            else:
                idx = np.asarray(mask_or_indices)
            idx = idx.astype(int)
            if idx.size == 0:
                return {
                    "summary": np.zeros(37, dtype=np.float32),
                    "chroma": np.zeros(12, dtype=np.float32),
                    "dur_hist": np.zeros(12, dtype=np.float32),
                    "ioi_hist": np.zeros(12, dtype=np.float32),
                    "pitch_seq": np.zeros(12, dtype=np.float32),
                    "ioi_seq": np.zeros(8, dtype=np.float32),
                    "chord": np.zeros(0, dtype=np.float32),
                }
            wp = pitches[idx]
            ws = starts[idx]
            wd = durations[idx]
            wv = velocities[idx]
            wioi = np.diff(ws) if len(ws) > 1 else np.zeros(0, dtype=np.float32)
            wdiff = np.diff(wp) if len(wp) > 1 else np.zeros(0, dtype=np.float32)
            wabs = np.abs(wdiff)
            chroma = np.bincount((wp.astype(int) % 12), minlength=12).astype(np.float32)
            if chroma.sum():
                chroma /= chroma.sum()
            pitch_seq = wp[: min(12, len(wp))]
            pitch_seq = np.pad(pitch_seq, (0, 12 - len(pitch_seq))).astype(np.float32)
            if len(wp) > 12:
                pitch_seq = wp[-12:].astype(np.float32)
            ioi_seq = wioi[: min(8, len(wioi))]
            ioi_seq = np.pad(ioi_seq, (0, 8 - len(ioi_seq))).astype(np.float32)
            first_or_last_onset = ws[0] if idx[0] == 0 else ws[-1]
            chord = np.array([n.pitch for n in notes if n.start == first_or_last_onset], dtype=np.float32)
            summary = np.asarray(
                summarize(wp)
                + summarize(wd)
                + summarize(wv)
                + summarize(wioi if len(wioi) else [0.0])
                + summarize(wdiff if len(wdiff) else [0.0])
                + summarize(wabs if len(wabs) else [0.0])
                + [
                    float(len(idx)),
                    float(wp[-1] - wp[0]) if len(wp) > 1 else 0.0,
                    float((wdiff > 0).mean()) if len(wdiff) else 0.0,
                    float((wdiff < 0).mean()) if len(wdiff) else 0.0,
                    float((wabs <= 2).mean()) if len(wabs) else 0.0,
                    float((wabs >= 7).mean()) if len(wabs) else 0.0,
                    float(wp.max() - wp.min()) if len(wp) else 0.0,
                ],
                dtype=np.float32,
            )
            return {
                "summary": summary,
                "chroma": chroma,
                "dur_hist": log_duration_bins(wd, 12),
                "ioi_hist": log_duration_bins(wioi, 12),
                "pitch_seq": pitch_seq,
                "ioi_seq": ioi_seq,
                "chord": chord,
            }

        head_windows = {}
        tail_windows = {}
        for n in [4, 8, 12, 16, 24]:
            head_windows[f"n{n}"] = make_window(np.arange(0, min(n, len(notes))))
            tail_windows[f"n{n}"] = make_window(np.arange(max(0, len(notes) - n), len(notes)))
        rel = (starts - starts.min()) / span
        for frac in [0.20, 0.33, 0.50]:
            head_windows[f"f{int(frac * 100)}"] = make_window(np.where(rel <= frac)[0])
            tail_windows[f"f{int(frac * 100)}"] = make_window(np.where(rel >= 1.0 - frac)[0])

        bundle = {
            "base": np.asarray(feats, dtype=np.float32),
            "first_notes": first_notes,
            "last_notes": last_notes,
            "head_windows": head_windows,
            "tail_windows": tail_windows,
            "pcs": pitch_classes,
            "transitions": transitions,
            "first_chord": first_chord,
            "last_chord": last_chord,
            "last_pitch": float(pitches[-1]),
            "first_pitch": float(pitches[0]),
            "last_vel": float(velocities[-1]),
            "first_vel": float(velocities[0]),
            "last_dur": float(durations[-1]),
            "first_dur": float(durations[0]),
            "last_tempo": last_tempo,
            "end_melody_dir": end_mel_dir,
            "start_melody_dir": start_mel_dir,
            "end_vel_mean": end_vel_mean,
            "start_vel_mean": start_vel_mean,
        }
        self.segment_cache[rel_path] = bundle
        return bundle

    def boundary_features(self, source, target):
        last_notes = source["last_notes"]
        first_notes = target["first_notes"]
        aligned_diffs = []
        aligned_signed = []
        for i in range(len(last_notes)):
            diff = last_notes[i] - first_notes[i]
            aligned_diffs.extend(np.abs(diff).tolist())
            aligned_signed.extend(diff.tolist())

        pair_pitch_diffs = []
        pair_duration_diffs = []
        pair_velocity_diffs = []
        pair_gap_diffs = []
        for left in last_notes:
            for right in first_notes:
                pair_pitch_diffs.append(abs(left[0] - right[0]))
                pair_duration_diffs.append(abs(left[1] - right[1]))
                pair_gap_diffs.append(abs(left[2] - right[2]))
                pair_velocity_diffs.append(abs(left[3] - right[3]))

        chord_left = source["last_chord"]
        chord_right = target["first_chord"]
        if len(chord_left) and len(chord_right):
            left_pc = set((chord_left.astype(int) % 12).tolist())
            right_pc = set((chord_right.astype(int) % 12).tolist())
            chord_features = [
                abs(float(chord_left.mean()) - float(chord_right.mean())),
                abs(float(np.median(chord_left)) - float(np.median(chord_right))),
                len(left_pc & right_pc),
                len(left_pc | right_pc),
            ]
        else:
            chord_features = [0.0, 0.0, 0.0, 0.0]

        end_note = last_notes[-1]
        start_note = first_notes[0]
        transition_alignment = np.dot(source["transitions"].reshape(-1), target["transitions"].reshape(-1))

        directed_interval = source["last_pitch"] - target["first_pitch"]
        abs_interval = abs(directed_interval)
        vel_continuity = abs(source["last_vel"] - target["first_vel"]) / 127.0
        vel_mean_continuity = abs(source["end_vel_mean"] - target["start_vel_mean"]) / 127.0
        dur_continuity = abs(source["last_dur"] - target["first_dur"]) / max(1.0, source["last_dur"])
        mel_dir_match = source["end_melody_dir"] * target["start_melody_dir"]
        tempo_continuity = abs(source["last_tempo"] - target["last_tempo"]) / 120.0
        continuity_feats = [
            directed_interval / 12.0,
            abs_interval / 12.0,
            1.0 if abs_interval <= 2 else 0.0,
            1.0 if abs_interval >= 7 else 0.0,
            1.0 if abs_interval == 6 else 0.0,
            vel_continuity,
            vel_mean_continuity,
            dur_continuity,
            mel_dir_match / 100.0,
            tempo_continuity,
        ]

        return np.array(
            aligned_diffs
            + aligned_signed
            + summarize(pair_pitch_diffs)
            + summarize(pair_duration_diffs)
            + summarize(pair_gap_diffs)
            + summarize(pair_velocity_diffs)
            + chord_features
            + [
                np.linalg.norm(source["pcs"] - target["pcs"]),
                float(np.dot(source["pcs"], target["pcs"])),
                transition_alignment,
                abs(end_note[0] - start_note[0]),
                end_note[0] - start_note[0],
                abs(end_note[1] - start_note[1]),
                abs(end_note[2] - start_note[2]),
                abs(end_note[3] - start_note[3]),
            ]
            + list(harmonic_continuity_features(source["last_chord"], target["first_chord"]))
            + list(harmonic_continuity_features(target["last_chord"], source["first_chord"]))
            + continuity_feats,
            dtype=np.float32,
        )

    def window_similarity_features(self, source_window, target_window):
        chroma_a = source_window["chroma"]
        chroma_b = target_window["chroma"]
        dur_a = source_window["dur_hist"]
        dur_b = target_window["dur_hist"]
        ioi_a = source_window["ioi_hist"]
        ioi_b = target_window["ioi_hist"]
        seq_a = source_window["pitch_seq"]
        seq_b = target_window["pitch_seq"]
        ioi_seq_a = source_window["ioi_seq"]
        ioi_seq_b = target_window["ioi_seq"]

        chroma_xcorr = [float(np.dot(chroma_a, np.roll(chroma_b, shift))) for shift in range(12)]
        best_shift = int(np.argmax(chroma_xcorr)) if chroma_xcorr else 0
        best_chroma = float(max(chroma_xcorr)) if chroma_xcorr else 0.0
        zero_chroma = float(chroma_xcorr[0]) if chroma_xcorr else 0.0
        summary_diff = source_window["summary"] - target_window["summary"]

        return np.concatenate(
            [
                np.abs(summary_diff),
                summary_diff,
                np.array(
                    [
                        float(np.linalg.norm(chroma_a - chroma_b)),
                        float(np.dot(chroma_a, chroma_b)),
                        zero_chroma,
                        best_chroma,
                        float(best_shift),
                        float(chroma_xcorr[5]) if chroma_xcorr else 0.0,
                        float(chroma_xcorr[7]) if chroma_xcorr else 0.0,
                        float(np.linalg.norm(dur_a - dur_b)),
                        float(np.dot(dur_a, dur_b)),
                        float(np.linalg.norm(ioi_a - ioi_b)),
                        float(np.dot(ioi_a, ioi_b)),
                        safe_corr(seq_a, seq_b),
                        float(np.abs(seq_a - seq_b).mean()),
                        safe_corr(ioi_seq_a, ioi_seq_b),
                        float(np.abs(ioi_seq_a - ioi_seq_b).mean()),
                    ],
                    dtype=np.float32,
                ),
                harmonic_continuity_features(source_window["chord"], target_window["chord"]),
            ]
        ).astype(np.float32)

    def multi_window_seam_features(self, source, target):
        feats = []
        for key in ["n4", "n8", "n12", "n16", "n24", "f20", "f33", "f50"]:
            feats.extend(self.window_similarity_features(source["tail_windows"][key], target["head_windows"][key]).tolist())
        return np.asarray(feats, dtype=np.float32)

    def pair_features(self, path_a, path_b):
        segment_a = self.segment_features(path_a)
        segment_b = self.segment_features(path_b)
        feat_a = segment_a["base"]
        feat_b = segment_b["base"]
        boundary_ab = self.boundary_features(segment_a, segment_b)
        boundary_ba = self.boundary_features(segment_b, segment_a)
        seam_ab = self.multi_window_seam_features(segment_a, segment_b)
        seam_ba = self.multi_window_seam_features(segment_b, segment_a)
        return np.concatenate(
            [
                feat_a,
                feat_b,
                feat_a - feat_b,
                feat_b - feat_a,
                np.abs(feat_a - feat_b),
                np.maximum(feat_a, feat_b),
                np.minimum(feat_a, feat_b),
                boundary_ab,
                boundary_ba,
                boundary_ab - boundary_ba,
                np.abs(boundary_ab - boundary_ba),
                seam_ab,
                seam_ba,
                seam_ab - seam_ba,
                np.abs(seam_ab - seam_ba),
            ]
        ).astype(np.float32)

    def build_matrix(self, pairs):
        return np.stack([self.pair_features(a, b) for a, b in pairs])

    def augment_reverse(self, pairs, labels):
        reversed_pairs = [(b, a) for a, b in pairs]
        reversed_labels = 1 - labels
        return list(pairs) + reversed_pairs, np.concatenate([labels, reversed_labels])

    def _blend_probs(self, probs, rev_probs, weights, alpha):
        forward = sum(weights[i] * probs[i] for i in range(len(weights)))
        reverse = sum(weights[i] * rev_probs[i] for i in range(len(weights)))
        consistency = 0.5 * (forward + (1.0 - reverse))
        return (1.0 - alpha) * forward + alpha * consistency

    def tune_weights(self, valid_probs, reverse_probs, y_valid):
        best_acc = -1.0
        best_weights = None
        best_threshold = 0.5
        best_alpha = 0.0

        for wa in [0.15, 0.2, 0.25]:
            for wb in [0.1, 0.15, 0.2]:
                for we in [0.2, 0.25, 0.3]:
                    for wr in [0.15, 0.2, 0.25]:
                        wl = 1.0 - (wa + wb + we + wr)
                        if wl < 0.05 or wl > 0.3:
                            continue
                        weights = np.array([wa, wb, we, wr, wl], dtype=np.float32)
                        for alpha in [0.0, 0.1, 0.2, 0.3]:
                            blended = self._blend_probs(valid_probs, reverse_probs, weights, alpha)
                            for threshold in [0.48, 0.5, 0.52, 0.55]:
                                preds = blended >= threshold
                                acc = accuracy_score(y_valid, preds)
                                if acc > best_acc:
                                    best_acc = acc
                                    best_weights = weights
                                    best_threshold = threshold
                                    best_alpha = alpha

        self.ensemble_weights = best_weights
        self.decision_threshold = best_threshold
        self.consistency_alpha = best_alpha
        print("Selected validation accuracy:", best_acc)
        print("Selected weights:", best_weights.tolist())
        print("Selected threshold:", best_threshold)
        print("Selected consistency alpha:", best_alpha)

    def train(self, train_json_path):
        labels = read_python_literal(train_json_path)
        pairs = list(labels)
        y = np.array([1 if labels[pair] else 0 for pair in pairs], dtype=np.int32)
        print(f"Extracting features for {len(pairs)} pairs...")
        _ = self.build_matrix(pairs)
        joblib.dump(self.segment_cache, self.feature_cache_path)

        train_pairs, valid_pairs, y_train, y_valid = train_test_split(
            pairs, y, test_size=0.15, random_state=42, stratify=y
        )
        train_pairs_aug, y_train_aug = self.augment_reverse(train_pairs, y_train)
        print(f"Training on {len(train_pairs_aug)} augmented pairs; validating on {len(valid_pairs)} original pairs...")
        X_train = self.build_matrix(train_pairs_aug)
        X_valid = self.build_matrix(valid_pairs)
        X_valid_rev = self.build_matrix([(b, a) for a, b in valid_pairs])

        valid_probs = []
        reverse_probs = []
        for idx, (name, model) in enumerate(self.models, start=1):
            print(f"Training {name} ({idx}/{len(self.models)})...")
            fitted = clone(model).fit(X_train, y_train_aug)
            valid_probs.append(fitted.predict_proba(X_valid)[:, 1])
            reverse_probs.append(fitted.predict_proba(X_valid_rev)[:, 1])

        self.tune_weights(valid_probs, reverse_probs, y_valid)

        full_pairs_aug, full_y_aug = self.augment_reverse(pairs, y)
        X_full = self.build_matrix(full_pairs_aug)
        self.fitted = []
        for idx, (name, model) in enumerate(self.models, start=1):
            print(f"Refitting {name} on full augmented data ({idx}/{len(self.models)})...")
            fitted = clone(model).fit(X_full, full_y_aug)
            self.fitted.append((name, fitted))
        joblib.dump(self.segment_cache, self.feature_cache_path)

    def predict(self, path, outpath):
        entries = read_python_literal(path)
        pairs = list(entries)
        print(f"Predicting {len(pairs)} pairs...")
        X = self.build_matrix(pairs)
        X_rev = self.build_matrix([(b, a) for a, b in pairs])
        joblib.dump(self.segment_cache, self.feature_cache_path)
        probs = [model.predict_proba(X)[:, 1] for _, model in self.fitted]
        rev_probs = [model.predict_proba(X_rev)[:, 1] for _, model in self.fitted]
        blended = self._blend_probs(probs, rev_probs, self.ensemble_weights, self.consistency_alpha)
        predictions = {pair: bool(prob >= self.decision_threshold) for pair, prob in zip(pairs, blended)}
        write_submission_predictions(predictions, outpath)
        print(f"Wrote {outpath}")

        outpath = Path(outpath)
        base_preds = np.array([predictions[pair] for pair in pairs], dtype=bool)
        for offset in [-0.06, -0.04, -0.02, 0.02, 0.04, 0.06]:
            threshold = float(np.clip(self.decision_threshold + offset, 0.35, 0.65))
            variant_preds = blended >= threshold
            variant = {pair: bool(pred) for pair, pred in zip(pairs, variant_preds)}
            variant_path = outpath.with_name(f"predictions2_v12_thr_{threshold:.2f}.json")
            write_submission_predictions(variant, variant_path)
            print(f"Wrote {variant_path} ({int((variant_preds != base_preds).sum())} predictions differ from primary)")


# =========================
# Task 3 exact standalone code
# =========================

"""
Task 3 - Music Tagging v3: MuQ added as 4th model + learned blend weights
Models: MERT-95M, MERT-330M, CLAP, MuQ-large-msd-iter
New:
  - MuQ (trained on MSD itself — most domain-matched model possible)
  - Learned per-model blend weights via held-out val mAP
  - Weighted ensemble instead of simple average
  - MuQ uses same 24kHz SR as MERT, trivial to integrate

Install: !pip install librosa transformers torch joblib numpy scikit-learn muq
Run on GPU T4.
"""

import ast
import gc
import random
import warnings
from pathlib import Path

import joblib
import librosa
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoProcessor, ClapModel, ClapProcessor

warnings.filterwarnings("ignore")

ROOT      = Path("/content/student_files")
CACHE_DIR = Path("/content/.cache_task3_v3")
CACHE_DIR.mkdir(exist_ok=True)

TAGS = ["rock", "oldies", "jazz", "pop", "dance", "blues", "punk",
        "chill", "electronic", "country"]

TASK3_SR       = 24000
TASK3_DURATION = 10
TASK3_BATCH    = 32
TASK3_EPOCHS   = 35
TASK3_LR       = 1e-3
TASK3_PATIENCE = 7
N_SINGLE       = 5   # MLPs per single model
N_CONCAT       = 5   # MLPs on concatenated embeddings
SEED           = 42
CLAP_SR        = 48000


def read_python_literal(path):
    with open(path) as f:
        return ast.literal_eval(f.read())


def write_submission_predictions(predictions, outpath):
    serializable = {
        (k[2:] if isinstance(k, str) and k.startswith("./") else k): v
        for k, v in predictions.items()
    }
    with open(outpath, "w") as f:
        f.write(repr(serializable) + "\n")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# ── embedding extraction ──────────────────────────────────────────────────────

def extract_mert_embeddings(audio_paths, root, processor, model, device, cache_path, sr=TASK3_SR):
    cache = joblib.load(cache_path) if Path(cache_path).exists() else {}
    missing = [p for p in audio_paths if p not in cache]
    if missing:
        print(f"  extracting {len(missing)} MERT embeddings...")
        model.eval()
        target_len = sr * TASK3_DURATION
        for idx, rel_path in enumerate(missing, 1):
            try:
                waveform, _ = librosa.load(str(Path(root) / rel_path), sr=sr, mono=True)
                if len(waveform) < target_len:
                    waveform = np.pad(waveform, (0, target_len - len(waveform)))
                else:
                    waveform = waveform[:target_len]
                inputs = processor(waveform, sampling_rate=sr, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = model(**inputs, output_hidden_states=True)
                last4 = torch.stack(outputs.hidden_states[-4:], dim=0)
                emb = last4.mean(dim=0).mean(dim=1).squeeze(0).cpu().numpy()
                cache[rel_path] = emb.astype(np.float32)
            except Exception as e:
                print(f"  warning: {rel_path}: {e}")
                cache[rel_path] = np.zeros(model.config.hidden_size, dtype=np.float32)
            if idx % 100 == 0 or idx == len(missing):
                print(f"  {idx}/{len(missing)}")
                joblib.dump(cache, cache_path)
    else:
        print(f"  all {len(audio_paths)} cached")
    return np.stack([cache[p] for p in audio_paths]).astype(np.float32)


def extract_clap_embeddings(audio_paths, root, processor, model, device, cache_path):
    cache = joblib.load(cache_path) if Path(cache_path).exists() else {}
    missing = [p for p in audio_paths if p not in cache]
    if missing:
        print(f"  extracting {len(missing)} CLAP embeddings...")
        model.eval()
        target_len = CLAP_SR * TASK3_DURATION
        for idx, rel_path in enumerate(missing, 1):
            try:
                waveform, _ = librosa.load(str(Path(root) / rel_path), sr=CLAP_SR, mono=True)
                if len(waveform) < target_len:
                    waveform = np.pad(waveform, (0, target_len - len(waveform)))
                else:
                    waveform = waveform[:target_len]
                inputs = processor(audio=waveform, sampling_rate=CLAP_SR, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    out = model.get_audio_features(**inputs)
                emb = out.pooler_output if hasattr(out, 'pooler_output') else out
                cache[rel_path] = emb.squeeze(0).cpu().numpy().astype(np.float32)
            except Exception as e:
                print(f"  warning: {rel_path}: {e}")
                cache[rel_path] = np.zeros(512, dtype=np.float32)
            if idx % 100 == 0 or idx == len(missing):
                print(f"  {idx}/{len(missing)}")
                joblib.dump(cache, cache_path)
    else:
        print(f"  all {len(audio_paths)} cached")
    return np.stack([cache[p] for p in audio_paths]).astype(np.float32)


def extract_muq_embeddings(audio_paths, root, model, device, cache_path, sr=24000):
    """
    MuQ uses 24kHz — same as MERT, so no resampling needed.
    We average the last 4 hidden states over the time dimension,
    identical strategy to MERT extraction above.
    """
    cache = joblib.load(cache_path) if Path(cache_path).exists() else {}
    missing = [p for p in audio_paths if p not in cache]
    if missing:
        print(f"  extracting {len(missing)} MuQ embeddings...")
        model.eval()
        target_len = sr * TASK3_DURATION
        for idx, rel_path in enumerate(missing, 1):
            try:
                waveform, _ = librosa.load(str(Path(root) / rel_path), sr=sr, mono=True)
                if len(waveform) < target_len:
                    waveform = np.pad(waveform, (0, target_len - len(waveform)))
                else:
                    waveform = waveform[:target_len]
                wavs = torch.tensor(waveform).unsqueeze(0).to(device)
                with torch.no_grad():
                    output = model(wavs, output_hidden_states=True)
                # Average last 4 hidden states over time dimension
                last4 = torch.stack(output.hidden_states[-4:], dim=0)
                emb = last4.mean(dim=0).mean(dim=1).squeeze(0).cpu().numpy()
                cache[rel_path] = emb.astype(np.float32)
            except Exception as e:
                print(f"  warning: {rel_path}: {e}")
                # MuQ-large hidden size is 1024
                cache[rel_path] = np.zeros(1024, dtype=np.float32)
            if idx % 100 == 0 or idx == len(missing):
                print(f"  {idx}/{len(missing)}")
                joblib.dump(cache, cache_path)
    else:
        print(f"  all {len(audio_paths)} cached")
    return np.stack([cache[p] for p in audio_paths]).astype(np.float32)


# ── MLP + focal loss ──────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    def __init__(self, pos_weight, gamma=2.0):
        super().__init__()
        self.pos_weight = pos_weight
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction='none')
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal = (1 - pt) ** self.gamma * bce
        return focal.mean()


class EmbeddingDataset(Dataset):
    def __init__(self, X, y, augment=False):
        self.X, self.y, self.augment = X, y, augment

    def __len__(self): return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].copy()
        if self.augment and random.random() < 0.3:
            x += np.random.normal(0, 0.01, x.shape).astype(np.float32)
        return torch.tensor(x), torch.tensor(self.y[idx], dtype=torch.float32)


class EmbeddingMLP(nn.Module):
    def __init__(self, input_dim, n_classes=len(TAGS)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, n_classes),
        )

    def forward(self, x): return self.net(x)


def train_and_predict_mlps(X_train, y_train, X_test, device, n_models, seed_offset=0):
    """Returns (test_preds, val_maps) where val_maps[i] is the best val mAP of MLP i."""
    all_test_preds = []
    val_maps = []
    for idx in range(n_models):
        seed = SEED + seed_offset + idx * 19
        set_seed(seed)
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.15, random_state=seed)

        train_loader = DataLoader(EmbeddingDataset(X_tr, y_tr, True),
                                  TASK3_BATCH, shuffle=True, num_workers=2)
        val_loader   = DataLoader(EmbeddingDataset(X_val, y_val, False),
                                  TASK3_BATCH, shuffle=False, num_workers=2)

        model = EmbeddingMLP(X_train.shape[1]).to(device)
        pos = y_tr.sum(0); neg = len(y_tr) - pos
        pos_weight = torch.tensor(np.clip(neg / np.maximum(pos, 1), 1.0, 12.0),
                                  dtype=torch.float32, device=device)
        criterion = FocalLoss(pos_weight)
        optimizer = torch.optim.AdamW(model.parameters(), lr=TASK3_LR, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=TASK3_EPOCHS, eta_min=1e-5)

        best_map, best_state, patience_count = -1.0, None, 0
        for epoch in range(1, TASK3_EPOCHS + 1):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad(set_to_none=True)
                criterion(model(xb), yb).backward()
                optimizer.step()
            scheduler.step()

            model.eval()
            preds, tgts = [], []
            with torch.no_grad():
                for xb, yb in val_loader:
                    preds.append(torch.sigmoid(model(xb.to(device))).cpu().numpy())
                    tgts.append(yb.numpy())
            val_map = average_precision_score(
                np.concatenate(tgts), np.concatenate(preds), average="macro")
            marker = " ✓" if val_map > best_map else ""
            print(f"    MLP {idx+1} ep {epoch:2d}  mAP={val_map:.4f}{marker}")
            if val_map > best_map:
                best_map = val_map
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= TASK3_PATIENCE:
                    print("    early stop")
                    break

        model.load_state_dict(best_state)
        print(f"    best mAP: {best_map:.4f}")
        val_maps.append(best_map)

        test_loader = DataLoader(EmbeddingDataset(X_test, np.zeros((len(X_test), len(TAGS))), False),
                                 TASK3_BATCH, shuffle=False, num_workers=2)
        test_preds = []
        model.eval()
        with torch.no_grad():
            for xb, _ in test_loader:
                test_preds.append(torch.sigmoid(model(xb.to(device))).cpu().numpy())
        all_test_preds.append(np.concatenate(test_preds))

    return np.mean(all_test_preds, axis=0), val_maps


def softmax_weights(scores):
    """Convert val mAP scores to softmax blend weights (higher mAP = more weight)."""
    scores = np.array(scores, dtype=np.float64)
    # temperature=0.05 sharpens the weighting toward best-performing model
    scores = scores / 0.05
    scores = scores - scores.max()
    weights = np.exp(scores)
    return weights / weights.sum()


# ── main ──────────────────────────────────────────────────────────────────────

def run():
    device = get_device()
    print(f"Device: {device}")
    set_seed(SEED)

    task3_dir   = ROOT / "task3_audio_classification"
    labels      = read_python_literal(task3_dir / "train.json")
    test_data   = read_python_literal(task3_dir / "test.json")
    train_paths = list(labels)
    test_paths  = list(test_data)

    y_train = np.array(
        [[1 if tag in labels[p] else 0 for tag in TAGS] for p in train_paths],
        dtype=np.float32)

    all_test_preds = []
    all_val_maps   = []
    emb_dict = {}

    # ── MERT-95M ──────────────────────────────────────────────────────────────
    print("\n" + "="*50 + "\nMERT-95M\n" + "="*50)
    proc = AutoProcessor.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)
    mert = AutoModel.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True).to(device)
    mert.eval()
    X_tr = extract_mert_embeddings(train_paths, task3_dir, proc, mert, device,
                                    CACHE_DIR / "mert_95M_train.joblib")
    X_te = extract_mert_embeddings(test_paths,  task3_dir, proc, mert, device,
                                    CACHE_DIR / "mert_95M_test.joblib")
    del proc, mert; gc.collect(); torch.cuda.empty_cache() if device.type=="cuda" else None
    emb_dict['95M'] = (X_tr, X_te)
    print(f"\nMLPs on MERT-95M (dim={X_tr.shape[1]})...")
    preds_95, maps_95 = train_and_predict_mlps(X_tr, y_train, X_te, device, N_SINGLE, seed_offset=0)
    all_test_preds.append(preds_95)
    all_val_maps.append(np.mean(maps_95))
    print(f"  MERT-95M ensemble val mAP: {np.mean(maps_95):.4f}")

    # ── MERT-330M ─────────────────────────────────────────────────────────────
    print("\n" + "="*50 + "\nMERT-330M\n" + "="*50)
    proc = AutoProcessor.from_pretrained("m-a-p/MERT-v1-330M", trust_remote_code=True)
    mert = AutoModel.from_pretrained("m-a-p/MERT-v1-330M", trust_remote_code=True).to(device)
    mert.eval()
    X_tr = extract_mert_embeddings(train_paths, task3_dir, proc, mert, device,
                                    CACHE_DIR / "mert_330M_train.joblib")
    X_te = extract_mert_embeddings(test_paths,  task3_dir, proc, mert, device,
                                    CACHE_DIR / "mert_330M_test.joblib")
    del proc, mert; gc.collect(); torch.cuda.empty_cache() if device.type=="cuda" else None
    emb_dict['330M'] = (X_tr, X_te)
    print(f"\nMLPs on MERT-330M (dim={X_tr.shape[1]})...")
    preds_330, maps_330 = train_and_predict_mlps(X_tr, y_train, X_te, device, N_SINGLE, seed_offset=100)
    all_test_preds.append(preds_330)
    all_val_maps.append(np.mean(maps_330))
    print(f"  MERT-330M ensemble val mAP: {np.mean(maps_330):.4f}")

    # ── CLAP ──────────────────────────────────────────────────────────────────
    print("\n" + "="*50 + "\nCLAP\n" + "="*50)
    clap_proc  = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
    clap_model = ClapModel.from_pretrained("laion/clap-htsat-unfused").to(device)
    clap_model.eval()
    X_tr_c = extract_clap_embeddings(train_paths, task3_dir, clap_proc, clap_model, device,
                                      CACHE_DIR / "clap_train.joblib")
    X_te_c = extract_clap_embeddings(test_paths,  task3_dir, clap_proc, clap_model, device,
                                      CACHE_DIR / "clap_test.joblib")
    del clap_proc, clap_model; gc.collect(); torch.cuda.empty_cache() if device.type=="cuda" else None
    emb_dict['clap'] = (X_tr_c, X_te_c)
    print(f"\nMLPs on CLAP (dim={X_tr_c.shape[1]})...")
    preds_clap, maps_clap = train_and_predict_mlps(X_tr_c, y_train, X_te_c, device, N_SINGLE, seed_offset=200)
    all_test_preds.append(preds_clap)
    all_val_maps.append(np.mean(maps_clap))
    print(f"  CLAP ensemble val mAP: {np.mean(maps_clap):.4f}")

    # ── MuQ (NEW — trained on MSD, most domain-matched model) ─────────────────
    print("\n" + "="*50 + "\nMuQ-large-msd-iter (new)\n" + "="*50)
    try:
        from muq import MuQ
        muq_model = MuQ.from_pretrained("OpenMuQ/MuQ-large-msd-iter").to(device)
        muq_model.eval()
        X_tr_muq = extract_muq_embeddings(train_paths, task3_dir, muq_model, device,
                                           CACHE_DIR / "muq_train.joblib")
        X_te_muq = extract_muq_embeddings(test_paths,  task3_dir, muq_model, device,
                                           CACHE_DIR / "muq_test.joblib")
        del muq_model; gc.collect(); torch.cuda.empty_cache() if device.type=="cuda" else None
        emb_dict['muq'] = (X_tr_muq, X_te_muq)
        print(f"\nMLPs on MuQ (dim={X_tr_muq.shape[1]})...")
        preds_muq, maps_muq = train_and_predict_mlps(X_tr_muq, y_train, X_te_muq, device, N_SINGLE, seed_offset=300)
        all_test_preds.append(preds_muq)
        all_val_maps.append(np.mean(maps_muq))
        print(f"  MuQ ensemble val mAP: {np.mean(maps_muq):.4f}")
        muq_available = True
    except Exception as e:
        print(f"  MuQ failed ({e}), skipping. Run: !pip install muq")
        muq_available = False

    # ── Concatenated embeddings ────────────────────────────────────────────────
    print("\n" + "="*50 + "\nConcatenated embeddings\n" + "="*50)
    concat_keys = ['95M', '330M', 'clap'] + (['muq'] if muq_available else [])
    X_tr_cat = np.concatenate([emb_dict[k][0] for k in concat_keys], axis=1)
    X_te_cat = np.concatenate([emb_dict[k][1] for k in concat_keys], axis=1)
    print(f"Concatenated dim: {X_tr_cat.shape[1]} ({' + '.join(concat_keys)})")
    preds_cat, maps_cat = train_and_predict_mlps(X_tr_cat, y_train, X_te_cat, device, N_CONCAT,
                                                  seed_offset=400)
    all_test_preds.append(preds_cat)
    all_val_maps.append(np.mean(maps_cat))
    print(f"  Concat ensemble val mAP: {np.mean(maps_cat):.4f}")

    # ── Learned blend weights ─────────────────────────────────────────────────
    print("\n" + "="*50 + "\nBlend weights\n" + "="*50)
    model_names = ['MERT-95M', 'MERT-330M', 'CLAP'] + (['MuQ'] if muq_available else []) + ['Concat']
    weights = softmax_weights(all_val_maps)
    for name, w, m in zip(model_names, weights, all_val_maps):
        print(f"  {name:12s}  val_mAP={m:.4f}  weight={w:.4f}")

    final_preds = sum(w * p for w, p in zip(weights, all_test_preds))

    predictions = {
        clip_path: {tag: float(final_preds[i, j]) for j, tag in enumerate(TAGS)}
        for i, clip_path in enumerate(test_paths)
    }
    write_submission_predictions(predictions, "/content/predictions3.json")
    print("\nSaved predictions3.json")
    print(f"Models used: {', '.join(model_names)}")
    print(f"Weighted blend weights: {dict(zip(model_names, [f'{w:.3f}' for w in weights]))}")


if __name__ == "__main__":
    # Install MuQ if not present:
    # !pip install muq
    run()
