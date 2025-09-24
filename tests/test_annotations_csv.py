from datetime import datetime
from pathlib import Path

import numpy as np

from core.annotations import (
    Annotations,
    AnnotationIndex,
    CsvEventMapping,
    discover_annotation_files,
    from_csv_events,
    from_csv_stages,
)


def test_from_csv_events_appples(tmp_path: Path):
    csv_content = """Event type,Stage,Time,Epoch,Duration,Validation\n"""
    csv_content += "\n".join(
        [
            "Hypopnea,N2,11:18:34 PM,34,22,*",
            "Leg Movement,N1,11:41:31 PM,80,5 (43),-",
            "pt reading,W,11:09:34 PM,16,-,*",
        ]
    )
    csv_path = tmp_path / "study.csv"
    csv_path.write_text(csv_content)

    mapping = CsvEventMapping(
        label_field="Event type",
        epoch_field="Epoch",
        duration_field="Duration",
        channel_map={"Leg Movement": "Leg_LAT"},
        validation_field="Validation",
    )

    anns = from_csv_events(csv_path, mapping)

    assert isinstance(anns, Annotations)
    assert anns.size == 3

    starts = dict(zip(anns.data["label"], anns.data["start_s"]))
    # epoch is 1-based, 30 s each
    assert np.isclose(starts["Hypopnea"], (34 - 1) * 30.0)
    # Duration parsed from "5 (43)" should yield 5 seconds
    durations = dict(zip(anns.data["label"], anns.data["end_s"] - anns.data["start_s"]))
    assert np.isclose(durations["Leg Movement"], 5.0)
    # Missing duration becomes 0
    assert np.isclose(durations["pt reading"], 0.0)
    # Channel mapped via channel_map
    leg_idx = np.where(anns.data["label"] == "Leg Movement")[0][0]
    assert anns.data["chan"][leg_idx] == "Leg_LAT"
    assert anns.meta["validation_counts"]["*"] == 2


def test_annotation_index_between(tmp_path: Path):
    csv_path = tmp_path / "study.csv"
    csv_path.write_text(
        "Event type,Stage,Time,Epoch,Duration,Validation\n"
        "Hypopnea,N2,11:18:34 PM,34,22,*\n"
        "Leg Movement,N1,11:41:31 PM,80,5,-\n"
    )
    mapping = CsvEventMapping(label_field="Event type", epoch_field="Epoch", duration_field="Duration")
    events = from_csv_events(csv_path, mapping)

    stage_path = tmp_path / "studySTAGE.csv"
    stage_path.write_text("11\n12\n")
    stages = from_csv_stages(stage_path, epoch_length_s=30.0)

    index = AnnotationIndex([events, stages])
    view = index.between(0.0, 3000.0)
    assert view.size == events.size + stages.size
    stage_view = index.between(0.0, 100.0, channels=["stage"])
    assert stage_view.size == 2


def test_from_csv_stages(tmp_path: Path):
    stage_path = tmp_path / "studySTAGE.csv"
    stage_path.write_text("11\n12\n13\n14\n")

    anns = from_csv_stages(stage_path, epoch_length_s=30.0)
    assert anns.size == 4
    assert anns.data["label"].tolist() == ["Wake", "N1", "N2", "REM"]
    assert np.isclose(anns.data["start_s"][2], 60.0)
    assert anns.data["chan"][0] == "stage"


def test_discover_annotation_files(tmp_path: Path):
    edf_path = tmp_path / "study.edf"
    edf_path.write_text("dummy")
    events = tmp_path / "study.csv"
    stages = tmp_path / "studySTAGE.csv"
    events.write_text("a\n")
    stages.write_text("11\n")

    found = discover_annotation_files(edf_path)
    assert found["events"] == events
    assert found["stages"] == stages
