# core/annotations.py
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Iterable, Optional, Tuple, Dict, Any
import csv
import math
import re

import numpy as np

ANNOT_DTYPE = np.dtype([
    ("start_s", "f8"),
    ("end_s",   "f8"),
    ("label",   "U64"),
    ("chan",    "U32"),   # optional: channel name/id or "" if global
])

STAGE_CHANNEL = "stage"
POSITION_CHANNEL = "position"

@dataclass
class Annotations:
    """
    Container for time-interval annotations aligned to recording start (t=0).
    Data is a sorted (by start_s) numpy structured array with dtype=ANNOT_DTYPE.
    """
    data: np.ndarray  # sorted by start_s
    meta: Dict[str, Any] = field(default_factory=dict)
    attrs: Optional[list[Dict[str, Any]]] = None

    @staticmethod
    def empty() -> "Annotations":
        return Annotations(np.zeros(0, dtype=ANNOT_DTYPE), {}, None)

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
        left = int(np.searchsorted(starts, t0, side="left"))

        # Determine earliest candidate index by checking if any earlier
        # events extend past t0. Older code only scanned a fixed window
        # (64 items) which missed long-running events when many short
        # events followed them. Instead, examine the whole prefix in a
        # vectorised manner and jump directly to the earliest overlap.
        if left > 0:
            prefix = self.data[:left]
            mask_prefix = prefix["end_s"] > t0
            if mask_prefix.any():
                first_overlap = int(np.argmax(mask_prefix))
                start_idx = first_overlap
            else:
                start_idx = left
        else:
            start_idx = 0

        end_idx = int(np.searchsorted(starts, t1, side="right"))
        subset = self.data[start_idx:end_idx]
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
    arr["chan"]    = ""

    position_canon = {
        "Supine",
        "Prone",
        "Left Lateral",
        "Right Lateral",
        "Upright",
        "Mixed",
        "Other",
        "Unknown",
    }

    for idx, raw_label in enumerate(labels):
        stage_label = _normalise_stage_label(raw_label)
        if stage_label:
            arr["label"][idx] = stage_label
            arr["chan"][idx] = STAGE_CHANNEL
            continue
        position_label = _normalise_position_label(raw_label)
        if position_label in position_canon:
            arr["label"][idx] = position_label
            arr["chan"][idx] = POSITION_CHANNEL
            continue
        lowered = str(raw_label or "").strip().lower()
        if lowered.startswith("position"):
            arr["label"][idx] = _normalise_position_label(raw_label)
            arr["chan"][idx] = POSITION_CHANNEL
        else:
            arr["label"][idx] = _normalise_label(raw_label)

    # sort by start
    order = np.argsort(arr["start_s"], kind="mergesort")
    return Annotations(arr[order], {"source": "edf+"}, None)


_FLOAT_PATTERN = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _extract_first_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if math.isnan(value):
            return None
        return float(value)
    text = str(value).strip()
    if not text or text == "-":
        return None
    match = _FLOAT_PATTERN.search(text)
    if not match:
        return None
    try:
        return float(match.group())
    except ValueError:
        return None


def _normalise_label(label: Optional[str]) -> str:
    if label is None:
        return "event"
    return str(label).strip() or "event"


def _normalise_channel(channel: Optional[str]) -> str:
    if channel is None:
        return ""
    return str(channel).strip()


_STAGE_ALIAS_MAP = {
    "w": "Wake",
    "wake": "Wake",
    "awake": "Wake",
    "n": "N1",
    "n1": "N1",
    "s1": "N1",
    "stage1": "N1",
    "1": "N1",
    "nrem1": "N1",
    "nrem01": "N1",
    "n2": "N2",
    "s2": "N2",
    "stage2": "N2",
    "2": "N2",
    "nrem2": "N2",
    "nrem02": "N2",
    "light": "N2",
    "n3": "N3",
    "s3": "N3",
    "stage3": "N3",
    "3": "N3",
    "n4": "N3",
    "stage4": "N3",
    "4": "N3",
    "deep": "N3",
    "sws": "N3",
    "rem": "REM",
    "r": "REM",
    "stage5": "REM",
    "5": "REM",
}


def _normalise_stage_label(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    lowered = text.lower()
    lowered = lowered.replace("non-rem", "nrem")
    lowered = lowered.replace("non rem", "nrem")
    lowered = lowered.replace("rapid eye movement", "rem")
    lowered = lowered.replace("rapid-eye-movement", "rem")
    lowered = lowered.replace("sleep stage", "")
    lowered = lowered.replace("slp stage", "")
    lowered = lowered.replace("stage ", "")
    lowered = lowered.replace("stage:", "")
    lowered = lowered.replace("stage-", "")
    lowered = lowered.replace("sleepstate", "")
    lowered = lowered.replace("sleepstage", "")
    lowered = lowered.replace("sleep", "")
    lowered = lowered.replace("nrem", "n")
    lowered = lowered.replace("light sleep", "light")
    lowered = lowered.replace("deep sleep", "deep")
    normalized = re.sub(r"[^a-z0-9]+", "", lowered)
    if not normalized:
        return None
    # strip leading zeros (e.g. 01 -> 1)
    normalized = normalized.lstrip("0")
    label = _STAGE_ALIAS_MAP.get(normalized)
    if label:
        return label
    return _STAGE_ALIAS_MAP.get(normalized.lower())


@dataclass
class CsvEventMapping:
    label_field: str = "Event type"
    epoch_field: Optional[str] = "Epoch"
    epoch_length_s: float = 30.0
    epoch_base: float = 1.0
    time_field: Optional[str] = "Time"
    time_format: str = "%I:%M:%S %p"
    duration_field: Optional[str] = "Duration"
    end_field: Optional[str] = None
    unit: str = "s"
    offset_s: float = 0.0
    channel_field: Optional[str] = None
    default_channel: Optional[str] = None
    channel_map: Dict[str, str] = field(default_factory=dict)
    validation_field: Optional[str] = "Validation"
    attrs_fields: Tuple[str, ...] = ()  # extra columns to include in metadata per event


def _parse_epoch(row_value: Any, mapping: CsvEventMapping) -> Optional[float]:
    val = _extract_first_float(row_value)
    if val is None:
        return None
    return (val - mapping.epoch_base) * mapping.epoch_length_s


def _parse_clock_time(
    value: Any,
    *,
    mapping: CsvEventMapping,
    start_dt: Optional[datetime],
    last_dt: Optional[datetime],
) -> Tuple[Optional[float], Optional[datetime]]:
    if value is None or start_dt is None:
        return None, last_dt
    text = str(value).strip()
    if not text:
        return None, last_dt
    try:
        parsed_time = datetime.strptime(text, mapping.time_format).time()
    except ValueError:
        return None, last_dt

    base_date = start_dt.date()
    candidate = datetime.combine(base_date, parsed_time)

    # handle rollovers (times past midnight)
    if last_dt is not None and candidate <= last_dt:
        candidate += timedelta(days=1)
    elif candidate < start_dt:
        candidate += timedelta(days=1)

    seconds = (candidate - start_dt).total_seconds()
    return seconds, candidate


def from_csv_events(
    path: str | Path,
    mapping: CsvEventMapping,
    *,
    start_dt: Optional[datetime] = None,
) -> Annotations:
    path = Path(path)
    records: list[tuple[float, float, str, str]] = []
    attr_records: list[Dict[str, Any]] = []
    validation_counter: Dict[str, int] = {}

    unit_conv = 1.0 if mapping.unit == "s" else 1e-3
    offset = float(mapping.offset_s)
    last_clock_dt: Optional[datetime] = None

    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = _normalise_label(row.get(mapping.label_field))
            chan = mapping.channel_map.get(label, _normalise_channel(row.get(mapping.channel_field)))
            if not chan:
                chan = mapping.default_channel or ""

            start_s = None
            if mapping.epoch_field and row.get(mapping.epoch_field) not in (None, ""):
                start_s = _parse_epoch(row.get(mapping.epoch_field), mapping)

            if start_s is None and mapping.time_field:
                start_s, last_clock_dt = _parse_clock_time(
                    row.get(mapping.time_field),
                    mapping=mapping,
                    start_dt=start_dt,
                    last_dt=last_clock_dt,
                )

            if start_s is None:
                # Skip records we cannot time-align yet; future improvements may log warning.
                continue

            start_s = start_s * unit_conv - offset

            duration_s = None
            if mapping.end_field:
                end_val = _extract_first_float(row.get(mapping.end_field))
                if end_val is not None:
                    duration_s = (end_val - (start_s + offset) / unit_conv) * unit_conv
            if duration_s is None and mapping.duration_field:
                dur_val = _extract_first_float(row.get(mapping.duration_field))
                duration_s = (dur_val or 0.0) * unit_conv
            if duration_s is None:
                duration_s = 0.0

            if duration_s < 0:
                duration_s = 0.0

            end_s = start_s + duration_s

            records.append((start_s, end_s, label, chan))

            if mapping.validation_field:
                val = str(row.get(mapping.validation_field, "")).strip()
                if val:
                    validation_counter[val] = validation_counter.get(val, 0) + 1

            attr_payload = {key: row.get(key) for key in mapping.attrs_fields}
            attr_payload["raw"] = row
            attr_records.append(attr_payload)

    if not records:
        return Annotations.empty()

    arr = np.array(records, dtype=ANNOT_DTYPE)
    order = np.argsort(arr["start_s"], kind="mergesort")
    attrs_sorted = [attr_records[i] for i in order] if attr_records else None
    meta = {
        "source": str(path),
        "type": "csv_events",
        "mapping": asdict(mapping),
        "validation_counts": validation_counter,
    }
    return Annotations(arr[order], meta, attrs_sorted)


def from_csv_stages(
    path: str | Path,
    *,
    epoch_length_s: float = 30.0,
    stage_map: Optional[Dict[str, str]] = None,
    offset_s: float = 0.0,
) -> Annotations:
    path = Path(path)
    if stage_map is None:
        stage_map = {
            "11": "Wake",
            "12": "N1",
            "13": "N2",
            "14": "REM",
        }

    records: list[tuple[float, float, str, str]] = []
    with path.open("r", newline="") as f:
        reader = csv.reader(f)
        for idx, row in enumerate(reader):
            if not row:
                continue
            code = row[0].strip()
            numeric = _extract_first_float(code)
            if numeric is not None:
                numeric_key = str(int(numeric))
                label = stage_map.get(code, stage_map.get(numeric_key, code))
            else:
                label = stage_map.get(code, code)
            start_s = idx * epoch_length_s + offset_s
            end_s = start_s + epoch_length_s
            records.append((start_s, end_s, label, STAGE_CHANNEL))

    if not records:
        return Annotations.empty()

    arr = np.array(records, dtype=ANNOT_DTYPE)
    meta = {
        "source": str(path),
        "type": "csv_stages",
        "epoch_length_s": epoch_length_s,
        "stage_map": stage_map,
    }
    return Annotations(arr, meta, None)


def _match_column(fieldnames: Iterable[str], *candidates: str) -> Optional[str]:
    """Return the first column name matching any of the candidate labels."""
    normalized = {}
    for name in fieldnames:
        if not name:
            continue
        key = re.sub(r"\s+", "", name).strip().lower()
        normalized[key] = name

    for candidate in candidates:
        if not candidate:
            continue
        key = re.sub(r"\s+", "", candidate).strip().lower()
        if key in normalized:
            return normalized[key]

    lowered = {name.strip().lower(): name for name in fieldnames if name}
    for candidate in candidates:
        if not candidate:
            continue
        key = candidate.strip().lower()
        if key in lowered:
            return lowered[key]

    for name in fieldnames:
        lowered_name = name.strip().lower()
        for candidate in candidates:
            if not candidate:
                continue
            candidate_lower = candidate.strip().lower()
            if candidate_lower in lowered_name:
                return name
    return None


def _normalise_position_label(value: Any) -> str:
    if value is None:
        return "Unknown"
    text = str(value).strip()
    if not text:
        return "Unknown"

    cleaned = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
    if not cleaned:
        return "Unknown"

    tokens = cleaned.split()
    token_set = set(tokens)

    unknown_markers = {"unknown", "unk", "na", "n/a", "none", "null"}
    if token_set & unknown_markers:
        return "Unknown"

    if "upright" in token_set or token_set & {"sitting", "sit", "standing", "stand", "up", "vertical"}:
        return "Upright"

    if token_set & {"supine", "sup", "back"}:
        return "Supine"

    if token_set & {"prone", "stomach", "abdomen", "ventral"}:
        return "Prone"

    if "left" in token_set and "right" not in token_set:
        return "Left lateral"

    if "right" in token_set and "left" not in token_set:
        return "Right lateral"

    if "lateral" in token_set:
        if "left" in cleaned:
            return "Left lateral"
        if "right" in cleaned:
            return "Right lateral"

    if token_set & {"other", "misc", "varied"}:
        return "Other"
    if "mixed" in token_set:
        return "Mixed"

    return text.strip().title()


def from_csv_positions(
    path: str | Path,
    *,
    epoch_length_s: float = 30.0,
    epoch_base: float = 1.0,
    label_field: str = "Position",
    epoch_field: str = "Epoch",
    offset_s: float = 0.0,
    channel: str = POSITION_CHANNEL,
) -> Annotations:
    """Parse body position CSV exported alongside the recording."""

    path = Path(path)
    records: list[tuple[float, float, str, str]] = []
    counts: Dict[str, int] = {}

    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        label_key = _match_column(fieldnames, label_field, "BodyPosition", "Body Position", "Posture")
        epoch_key = _match_column(fieldnames, epoch_field, "Epoch", "Epoch#", "EpochIndex", "Index")

        def _append_record(idx: float, raw_label: Any):
            norm = _normalise_position_label(raw_label)
            start_s = float(idx) * float(epoch_length_s) + float(offset_s)
            end_s = start_s + float(epoch_length_s)
            records.append((start_s, end_s, norm, channel))
            counts[norm] = counts.get(norm, 0) + 1

        if label_key:
            for row_idx, row in enumerate(reader):
                raw_label = row.get(label_key)
                if raw_label is None and not epoch_key:
                    continue
                if epoch_key:
                    epoch_val = _extract_first_float(row.get(epoch_key))
                    if epoch_val is not None:
                        idx = epoch_val - float(epoch_base)
                    else:
                        idx = row_idx
                else:
                    idx = row_idx
                if idx < 0:
                    idx = 0
                _append_record(idx, raw_label)
        else:
            f.seek(0)
            simple_reader = csv.reader(f)
            for row_idx, row in enumerate(simple_reader):
                if not row:
                    continue
                if row_idx == 0 and len(row) > 1:
                    header = " ".join(cell.strip().lower() for cell in row)
                    if "position" in header or "posture" in header:
                        continue
                raw_label = row[0]
                _append_record(float(row_idx), raw_label)

    if not records:
        return Annotations.empty()

    arr = np.array(records, dtype=ANNOT_DTYPE)
    order = np.argsort(arr["start_s"], kind="mergesort")
    meta = {
        "source": str(path),
        "type": "csv_positions",
        "epoch_length_s": float(epoch_length_s),
        "epoch_base": float(epoch_base),
        "offset_s": float(offset_s),
        "channel": channel,
        "label_counts": counts,
    }
    return Annotations(arr[order], meta, None)


def discover_annotation_files(edf_path: str | Path) -> Dict[str, Path]:
    """
    Locate companion CSV files that ship alongside an EDF recording.

    Historically we relied on an exact `<stem>STAGE.csv` / `<stem>POSITION.csv`
    naming convention. Field systems, however, export slightly different
    suffixes (e.g. `_Stages.csv`, `-position.csv`, mixed case extensions).
    We now scan sibling CSVs that share the EDF stem (ignoring punctuation)
    and pick the closest match for each role.
    """

    base = Path(edf_path)
    parent = base.parent
    stem = base.stem
    stem_normalized = re.sub(r"[^a-z0-9]", "", stem.lower())

    found: Dict[str, Path] = {}

    primary_events = base.with_suffix(".csv")
    if primary_events.exists():
        found["events"] = primary_events

    def _score_stage(name: str) -> int:
        if name.endswith("stage"):
            return 3
        if name.endswith("stages"):
            return 2
        if "stage" in name:
            return 1
        return 0

    def _score_position(name: str) -> int:
        if name.endswith("position"):
            return 3
        if "bodyposition" in name or "body_pos" in name:
            return 2
        if "position" in name or "posture" in name:
            return 1
        return 0

    def _prefer(existing: tuple[int, Path] | None, score: int, candidate: Path):
        if score <= 0:
            return existing
        if existing is None:
            return (score, candidate)
        prev_score, prev_path = existing
        if score > prev_score:
            return (score, candidate)
        if score == prev_score and len(candidate.name) < len(prev_path.name):
            return (score, candidate)
        return existing

    stage_choice: tuple[int, Path] | None = None
    position_choice: tuple[int, Path] | None = None
    events_choice: tuple[int, Path] | None = None

    for csv_path in parent.glob(f"{stem}*"):
        if not csv_path.is_file():
            continue
        if csv_path.suffix.lower() != ".csv":
            continue
        if csv_path == primary_events:
            continue
        name = csv_path.stem
        name_lower = name.lower()
        normalized = re.sub(r"[^a-z0-9]", "", name_lower)
        if stem_normalized and not normalized.startswith(stem_normalized):
            continue

        stage_choice = _prefer(stage_choice, _score_stage(normalized), csv_path)
        position_choice = _prefer(position_choice, _score_position(normalized), csv_path)
        if "event" in name_lower or "annotation" in name_lower:
            events_choice = _prefer(events_choice, 1, csv_path)

    if events_choice and "events" not in found:
        found["events"] = events_choice[1]
    if stage_choice:
        found["stages"] = stage_choice[1]
    if position_choice:
        found["positions"] = position_choice[1]

    return found


class AnnotationIndex:
    """Aggregated view over multiple `Annotations` collections."""

    def __init__(self, annotations: Iterable[Annotations]):
        ann_list = [ann for ann in annotations if ann.size]
        arrays: list[np.ndarray] = []
        attrs: list[Dict[str, Any]] = []
        self.sources: list[Dict[str, Any]] = []
        for ann in ann_list:
            arrays.append(ann.data)
            if ann.attrs is None:
                attrs.extend({} for _ in range(ann.size))
            else:
                attrs.extend(ann.attrs)
            self.sources.append(ann.meta)

        if arrays:
            stacked = np.concatenate(arrays)
            order = np.argsort(stacked["start_s"], kind="mergesort")
            self.data = stacked[order]
            self.attrs = [attrs[i] for i in order]
            self._indices = np.arange(stacked.shape[0])[order]
        else:
            self.data = np.zeros(0, dtype=ANNOT_DTYPE)
            self.attrs = []
            self._indices = np.zeros(0, dtype=int)

        self.channel_set = {str(c) for c in self.data["chan"]} if self.data.size else set()
        self.label_set = {str(l) for l in self.data["label"]} if self.data.size else set()

    def is_empty(self) -> bool:
        return self.data.size == 0

    def channels(self) -> set[str]:
        return set(self.channel_set)

    def labels(self) -> set[str]:
        return set(self.label_set)

    def between(
        self,
        t0: float,
        t1: float,
        *,
        channels: Optional[Iterable[str]] = None,
        labels: Optional[Iterable[str]] = None,
        with_attrs: bool = False,
        return_indices: bool = False,
    ):
        if self.data.size == 0 or t1 <= t0:
            empty = self.data[:0]
            if with_attrs and return_indices:
                return empty, [], np.zeros(0, dtype=int)
            if with_attrs:
                return empty, []
            if return_indices:
                return empty, np.zeros(0, dtype=int)
            return empty

        starts = self.data["start_s"]
        left = np.searchsorted(starts, t0, side="left")
        right = np.searchsorted(starts, t1, side="right")
        if right <= left:
            empty = self.data[:0]
            if with_attrs and return_indices:
                return empty, [], np.zeros(0, dtype=int)
            if with_attrs:
                return empty, []
            if return_indices:
                return empty, np.zeros(0, dtype=int)
            return empty

        idx = np.arange(left, right)
        subset = self.data[idx]
        mask = (subset["start_s"] < t1) & (subset["end_s"] > t0)

        if channels is not None:
            channel_set = set(channels)
            mask &= np.array([chan in channel_set for chan in subset["chan"]], dtype=bool)

        if labels is not None:
            label_set = set(labels)
            mask &= np.array([lbl in label_set for lbl in subset["label"]], dtype=bool)

        idx = idx[mask]
        view = self.data[idx]
        indices = self._indices[idx]
        if with_attrs:
            attrs = [self.attrs[i] for i in idx]
            if return_indices:
                return view, attrs, indices
            return view, attrs
        if return_indices:
            return view, indices
        return view

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
    attrs = None if ann.attrs is None else [dict(a) for a in ann.attrs]
    return Annotations(new, dict(ann.meta), attrs)

def filter_types(ann: Annotations, keep: Iterable[str]) -> Annotations:
    """
    Keep only events with label in keep.
    """
    if ann.size == 0: return ann
    keep = set(keep)
    mask = np.array([lbl in keep for lbl in ann.data["label"]], dtype=bool)
    attrs = None
    if ann.attrs is not None:
        attrs = [ann.attrs[i] for i in np.nonzero(mask)[0]]
    return Annotations(ann.data[mask], dict(ann.meta), attrs)

def shift(ann: Annotations, delta_s: float) -> Annotations:
    """
    Shift all events by delta seconds (positive shifts right).
    """
    if ann.size == 0: return ann
    new = ann.data.copy()
    new["start_s"] += delta_s
    new["end_s"]   += delta_s
    order = np.argsort(new["start_s"], kind="mergesort")
    meta = dict(ann.meta)
    meta["shifted_by"] = meta.get("shifted_by", 0.0) + delta_s
    attrs = None if ann.attrs is None else [ann.attrs[i] for i in order]
    return Annotations(new[order], meta, attrs)
