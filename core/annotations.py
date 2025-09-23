# core/annotations.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple, Dict, Any
import numpy as np
import csv
import math

ANNOT_DTYPE = np.dtype([
    ("start_s", "f8"),
    ("end_s",   "f8"),
    ("label",   "U64"),
    ("chan",    "U32"),   # optional: channel name/id or "" if global
])

@dataclass
class Annotations:
    """
    Container for time-interval annotations aligned to recording start (t=0).
    Data is a sorted (by start_s) numpy structured array with dtype=ANNOT_DTYPE.
    """
    data: np.ndarray  # sorted by start_s

    @staticmethod
    def empty() -> "Annotations":
        return Annotations(np.zeros(0, dtype=ANNOT_DTYPE))

    @property
    def size(self) -> int:
        return int(self.data.size)

    def between(self, t0: float, t1: float) -> np.ndarray:
        """
        Return a view of events overlapping [t0, t1].
        Overlap condition: start < t1 and end > t0.
        Uses binary search on start_s for speed.
        """
        if self.size == 0 or t1 <= t0:
            return self.data[:0]
        starts = self.data["start_s"]
        # left bound: first event with start_s >= (t0 - max_span_guess)
        # but we don't know span; do standard lower_bound at t0, then scan backwards a little.
        i = np.searchsorted(starts, t0, side="left")
        # move left while prior events might overlap (start_s < t0 but end_s > t0)
        j = max(0, i - 64)  # small safety back-scan window
        subset = self.data[j:np.searchsorted(starts, t1, side="right")]
        mask = (subset["start_s"] < t1) & (subset["end_s"] > t0)
        return subset[mask]

    def types(self) -> np.ndarray:
        return np.unique(self.data["label"])

# -------- Loaders --------

def from_edfplus(edf_reader, time_zero_s: float = 0.0) -> Annotations:
    """
    Convert pyEDFlib EDF+ annotations into our schema.
    pyEDFlib.readAnnotations() returns (onsets, durations, descriptions).
    - onsets are relative to recording start (may be floats).
    - durations may be NaN; treat as 0 for instants.
    """
    try:
        onsets, durations, texts = edf_reader.readAnnotations()
    except Exception:
        return Annotations.empty()

    onsets = np.asarray(onsets, dtype=float)
    durations = np.asarray(durations, dtype=float)
    labels = np.asarray(texts, dtype=str)

    # Replace NaN durations with 0 (instantaneous markers)
    durations = np.where(np.isnan(durations), 0.0, durations)
    end_s = onsets + durations

    arr = np.zeros(onsets.size, dtype=ANNOT_DTYPE)
    arr["start_s"] = onsets - float(time_zero_s)
    arr["end_s"]   = end_s   - float(time_zero_s)
    arr["label"]   = labels
    arr["chan"]    = ""      # EDF+ annotations are usually global

    # sort by start
    order = np.argsort(arr["start_s"], kind="mergesort")
    return Annotations(arr[order])


def from_csv(
    path: str,
    mapping: Dict[str, Any],
    *,
    delimiter: str = ",",
    has_header: bool = True,
) -> Annotations:
    """
    Load annotations from a CSV using a mapping dict.

    Required mapping keys:
      - 'start': column name or index (0-based) for start time
      - 'end' OR 'duration': (choose one) column for end time or duration
      - 'unit': one of {'s','ms'} for the time columns
    Optional:
      - 'label': column for text label (default 'event')
      - 'chan': column for channel tag (default '')
      - 'offset_s': float seconds to subtract (e.g., align to EDF start)

    Example mapping:
      {
        "start": "start_time",
        "end": "end_time",
        "unit": "s",
        "label": "event_type",
        "chan": "signal",
        "offset_s": 0.0
      }
    """
    def get(row, key):
        ref = mapping.get(key)
        if ref is None: return None
        if isinstance(ref, int):
            return row[ref]
        return row[ref]

    conv = 1.0 if mapping.get("unit", "s") == "s" else 1e-3
    offset = float(mapping.get("offset_s", 0.0))

    records = []
    with open(path, "r", newline="") as f:
        reader = csv.reader(f, delimiter=delimiter)
        header = None
        if has_header:
            header = next(reader)
        for row in reader:
            row_dict = row if header is None else dict(zip(header, row))
            s_raw = float(get(row_dict, "start"))
            e_ref = mapping.get("end")
            d_ref = mapping.get("duration")
            if e_ref is not None:
                e_raw = float(get(row_dict, "end"))
                start_s = s_raw * conv - offset
                end_s   = e_raw * conv - offset
            elif d_ref is not None:
                dur_raw = float(get(row_dict, "duration"))
                start_s = s_raw * conv - offset
                end_s   = start_s + dur_raw * conv
            else:
                raise ValueError("mapping must include 'end' or 'duration'")

            lab = str(get(row_dict, "label") or "event")
            ch  = str(get(row_dict, "chan") or "")

            # guard: ensure end >= start
            if end_s < start_s:
                start_s, end_s = end_s, start_s

            records.append((start_s, end_s, lab, ch))

    if not records:
        return Annotations.empty()

    arr = np.array(records, dtype=ANNOT_DTYPE)
    order = np.argsort(arr["start_s"], kind="mergesort")
    return Annotations(arr[order])

# -------- Utilities --------

def relabel(ann: Annotations, mapping: Dict[str, str]) -> Annotations:
    """
    Map labels (e.g., {'Obstructive apnea':'OA', 'Central apnea':'CA'}).
    """
    if ann.size == 0: return ann
    labels = ann.data["label"].copy()
    # vectorized remap via dict
    uniq = np.unique(labels)
    repl = {u: mapping.get(u, u) for u in uniq}
    vmap = np.vectorize(lambda x: repl.get(x, x))
    labels = vmap(labels)
    new = ann.data.copy()
    new["label"] = labels
    return Annotations(new)

def filter_types(ann: Annotations, keep: Iterable[str]) -> Annotations:
    """
    Keep only events with label in keep.
    """
    if ann.size == 0: return ann
    keep = set(keep)
    mask = np.array([lbl in keep for lbl in ann.data["label"]], dtype=bool)
    return Annotations(ann.data[mask])

def shift(ann: Annotations, delta_s: float) -> Annotations:
    """
    Shift all events by delta seconds (positive shifts right).
    """
    if ann.size == 0: return ann
    new = ann.data.copy()
    new["start_s"] += delta_s
    new["end_s"]   += delta_s
    order = np.argsort(new["start_s"], kind="mergesort")
    return Annotations(new[order])