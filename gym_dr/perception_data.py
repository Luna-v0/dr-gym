"""Data plumbing for the supervised perception net (W-perception) — shared by the
training notebook (`experiments/perception_cnn_v1_training.ipynb`) and the HPO
(`experiments/perception_hpo.py`), and reusable by v2.

The dataset is **129k per-episode shards** (~43 GB) in a by-track tree
``mlruns/**/perception_out/train/<track>/*.npz`` (keys ``frames (T,120,160) uint8``,
``targets (T,11) f32``, ``features``, ``diag``, ``meta``). Too big to ``np.concatenate``
whole, so this module offers two loaders:

* :class:`ShardFrameDataset` — the **streaming** loader (index once, LRU-cache shards),
  for full-scale training with ``DataLoader(num_workers>0)``.
* :func:`load_frames` + :func:`make_windows` + :func:`gather` — a **windowed in-RAM**
  path that stores each frame once and forms stack-N consecutive-frame windows within
  shard boundaries. The selected frames (~5 GB uint8 for the full TRAIN bucket) fit in
  GPU memory, so training needs no per-item disk I/O — this is what the notebook/HPO use.

Split discipline is **by base-track** (the project's locked
``camera_cnn_dataset._split_tracks``): ``_cw/_ccw/_mirrored`` variants follow their base
into one split (no variant leakage); reInvent/Oval are reserved physical. NB: the val/
test/variant frames live in the SAME tree under their own track dirs (labeled
``phase=train`` but bucketed by track NAME), so the canonical held-out TRACKS are
evaluable directly.
"""
from __future__ import annotations

import glob
import hashlib
import os
import pickle
import random
import re
import time
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

TRAIN_GLOB = "mlruns/**/perception_out/train/**/*.npz"
_VARIANT = re.compile(r"_(cw|ccw|mirrored)$")
_RESERVED = re.compile(r"(reinvent|Oval)", re.IGNORECASE)

# Frozen canonical by-track split — the exact output of
# ``experiments/camera_cnn_dataset._split_tracks(seed=42)`` (unique bases 70/15/15; the
# ``_cw/_ccw/_mirrored`` variants held out; reInvent/Oval reserved). Embedded here (rather
# than importing camera_cnn_dataset) so the split is identical on any machine regardless of
# that module's working-tree version — e.g. a remote checkout at an older commit.
TRAIN_TRACKS = (
    "2022_april_open", "2022_april_pro", "2022_august_pro", "2022_june_open", "2022_may_open",
    "2022_may_pro", "2022_october_pro", "2022_september_open", "2022_summit_speedway", "AWS_track",
    "Albert", "AmericasGeneratedInclStart", "Aragon", "Austin", "Canada_Training", "China_track",
    "FS_June2020", "July_2020", "LGSWide", "Monaco", "Monaco_building", "New_York_Track",
    "Singapore", "Straight_track", "Tokyo_Training_track", "Vegas_track", "Virtual_May19_Train_track",
    "arctic_open", "arctic_pro", "caecer_gp", "caecer_loop", "dubai_pro", "hamption_open",
    "hamption_pro", "jyllandsringen_open", "jyllandsringen_pro", "morgan_open", "morgan_pro",
    "penbay_open", "red_star_open", "red_star_pro", "thunder_hill_open", "thunder_hill_pro")
VAL_TRACKS = (
    "2022_august_open", "2022_september_pro", "Belille", "H_track", "Mexico_track",
    "Singapore_building", "Spain_track_f1", "dubai_open", "penbay_pro")
TEST_TRACKS = (
    "2022_july_open", "2022_july_pro", "2022_june_pro", "2022_march_open", "2022_march_pro",
    "2022_october_open", "2022_summit_speedway_mini", "Bowtie_track", "Singapore_f1", "Spain_track")
VARIANT_TRACKS = (
    "2022_april_open_ccw", "2022_april_open_cw", "2022_april_pro_ccw", "2022_april_pro_cw",
    "2022_august_open_ccw", "2022_august_open_cw", "2022_august_pro_ccw", "2022_august_pro_cw",
    "2022_july_pro_ccw", "2022_july_pro_cw", "2022_june_open_ccw", "2022_june_open_cw",
    "2022_june_pro_ccw", "2022_june_pro_cw", "2022_march_open_ccw", "2022_march_open_cw",
    "2022_march_pro_ccw", "2022_march_pro_cw", "2022_may_open_ccw", "2022_may_open_cw",
    "2022_may_pro_ccw", "2022_may_pro_cw", "2022_october_open_ccw", "2022_october_open_cw",
    "2022_october_pro_ccw", "2022_october_pro_cw", "2022_september_open_ccw", "2022_september_open_cw",
    "2022_september_pro_ccw", "2022_september_pro_cw", "2022_summit_speedway_ccw",
    "2022_summit_speedway_cw", "arctic_open_ccw", "arctic_open_cw", "arctic_pro_ccw", "arctic_pro_cw",
    "dubai_open_ccw", "dubai_open_cw", "jyllandsringen_open_ccw", "jyllandsringen_open_cw",
    "jyllandsringen_pro_ccw", "jyllandsringen_pro_cw", "penbay_open_ccw", "penbay_open_cw",
    "penbay_pro_ccw", "penbay_pro_cw", "red_star_pro_ccw", "red_star_pro_cw",
    "thunder_hill_pro_ccw", "thunder_hill_pro_cw")


def track_of(path: str) -> str:
    """``.../perception_out/train/<track>/epXXXX_carY.npz`` -> ``<track>``."""
    return re.sub(r".*/train/([^/]+)/.*", r"\1", path)


def canonical_split(train_glob: str = TRAIN_GLOB) -> Tuple[set, set, set, set]:
    """Return the frozen ``(TRAIN, VAL, TEST, VARIANT)`` track-name sets (see the embedded
    constants above). Deterministic and machine-independent — no import of the experiments
    package, so it can't silently mis-split on a checkout where camera_cnn_dataset differs."""
    return set(TRAIN_TRACKS), set(VAL_TRACKS), set(TEST_TRACKS), set(VARIANT_TRACKS)


def bucket_paths(train_glob: str = TRAIN_GLOB, verbose: bool = True) -> Dict[str, List[str]]:
    """Group every shard path into TRAIN/VAL/TEST/VARIANT/PHYSICAL by the canonical split."""
    TR, VA, TE, VAR = canonical_split(train_glob)
    buckets: Dict[str, List[str]] = defaultdict(list)
    for p in glob.glob(train_glob, recursive=True):
        t = track_of(p)
        b = ("PHYSICAL" if _RESERVED.search(t) else
             "VARIANT" if (t in VAR or _VARIANT.search(t)) else
             "VAL" if t in VA else "TEST" if t in TE else
             "TRAIN" if t in TR else "OTHER")
        buckets[b].append(p)
    if verbose:
        for b in ["TRAIN", "VAL", "TEST", "VARIANT", "PHYSICAL", "OTHER"]:
            ps = buckets.get(b, [])
            print(f"  {b:9s} shards={len(ps):7d} tracks={len({track_of(p) for p in ps})}")
    return dict(buckets)


def cap_per_track(paths: Sequence[str], cap: Optional[int], seed: int = 0) -> List[str]:
    """Deterministically keep at most ``cap`` shards per track (``None`` = keep all).
    Used to subsample the TRAIN bucket for fast HPO trials."""
    if cap is None:
        return sorted(paths)
    by: Dict[str, List[str]] = defaultdict(list)
    for p in paths:
        by[track_of(p)].append(p)
    rng = random.Random(seed)
    out: List[str] = []
    for t in sorted(by):
        ps = sorted(by[t])
        rng.shuffle(ps)
        out += ps[:cap]
    return out


# --------------------------------------------------------------------------- #
# Windowed in-RAM loader (store frames once; build stack-N windows on the fly)
# --------------------------------------------------------------------------- #
def load_frames(paths: Sequence[str], n_threads: int = 8, tag: str = ""
                ) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int, str]]]:
    """Threaded read of every shard's ``frames``/``targets``. Returns ``F (Nf,120,160)
    uint8``, ``Yf (Nf,11) f32`` (per-frame targets), and ``bounds`` = list of
    ``(start, length, track)`` so each shard's frames stay contiguous (for windowing)."""
    paths = sorted(paths)

    def _read(p):
        d = np.load(p, allow_pickle=False)
        return d["frames"], d["targets"].astype(np.float32), track_of(p)

    Fs, Ys, bounds, off, t0 = [], [], [], 0, time.time()
    with ThreadPoolExecutor(max_workers=n_threads) as ex:
        for fr, tg, trk in ex.map(_read, paths):
            Fs.append(fr)
            Ys.append(tg)
            bounds.append((off, fr.shape[0], trk))
            off += fr.shape[0]
    F = np.concatenate(Fs, 0)
    Yf = np.concatenate(Ys, 0)
    if tag:
        print(f"  [{tag}] F {F.shape} ({F.nbytes/1e9:.2f} GB), {len(bounds)} shards, "
              f"{len(F)} frames, {time.time()-t0:.0f}s", flush=True)
    return F, Yf, bounds


def make_windows(bounds: List[Tuple[int, int, str]], stack: int
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """For each shard, window starts ``s`` (into F) with ``[s, s+stack)`` inside one
    shard. Returns ``starts`` and ``tgt_idx = starts + stack - 1`` (latest frame's row)."""
    starts = []
    for off, n, _ in bounds:
        if n >= stack:
            starts.append(off + np.arange(0, n - stack + 1, dtype=np.int64))
    starts = np.concatenate(starts) if starts else np.zeros(0, np.int64)
    return starts, starts + (stack - 1)


def gather(F_t, starts_t, idx, stack):
    """Build a ``(B, stack, 120, 160)`` float batch (0..255) from the resident uint8
    frame tensor ``F_t`` and window-start tensor ``starts_t`` at indices ``idx``. Each
    window's ``stack`` rows are consecutive within one shard (guaranteed by make_windows)."""
    import torch
    rows = starts_t[idx][:, None] + torch.arange(stack, device=F_t.device)[None, :]
    return F_t[rows].float()


# --------------------------------------------------------------------------- #
# Streaming loader (the spec's §5 deliverable) — for DataLoader(num_workers>0)
# --------------------------------------------------------------------------- #
class ShardFrameDataset:
    """Streams ``(stacked_frames, target)`` from ``perception_out/<track>/*.npz``.

    Builds a ``[(shard_path, t), ...]`` index once (``t in [stack-1, T)``), pickled to a
    cache keyed by the path set + stack; LRU-caches open shards. ``__getitem__`` returns
    ``frames[t-stack+1 : t+1]`` (float, 0..255) and ``targets[t]``. A plain map-style
    dataset (``__len__`` + ``__getitem__``) — ``torch.utils.data.DataLoader`` accepts it
    directly, no ``Dataset`` subclass needed, so this module stays torch-free on import.
    """

    def __init__(self, paths: Sequence[str], stack: int = 1,
                 cache_dir: str = "tmp/perc_index", lru: int = 48) -> None:
        self.paths = sorted(paths)
        self.stack = stack
        self._lru = lru
        self._cache: "OrderedDict" = OrderedDict()
        sig = hashlib.md5(("|".join(self.paths) + f"|{stack}").encode()).hexdigest()[:16]
        idx_path = os.path.join(cache_dir, f"index_{sig}.pkl")
        if os.path.exists(idx_path):
            with open(idx_path, "rb") as fh:
                self.index = pickle.load(fh)
        else:
            self.index = []
            for p in self.paths:
                T = int(np.load(p)["targets"].shape[0])
                self.index += [(p, t) for t in range(stack - 1, T)]
            os.makedirs(cache_dir, exist_ok=True)
            with open(idx_path, "wb") as fh:
                pickle.dump(self.index, fh)

    def _shard(self, p):
        if p in self._cache:
            self._cache.move_to_end(p)
            return self._cache[p]
        d = np.load(p)
        arr = (d["frames"], d["targets"].astype(np.float32))
        self._cache[p] = arr
        if len(self._cache) > self._lru:
            self._cache.popitem(last=False)
        return arr

    def __len__(self):
        return len(self.index)

    def __getitem__(self, k):
        import torch
        p, t = self.index[k]
        fr, tg = self._shard(p)
        x = fr[t - self.stack + 1: t + 1].astype(np.float32)   # (stack,120,160), 0..255
        return torch.from_numpy(x), torch.from_numpy(tg[t])
