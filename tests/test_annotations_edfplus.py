import numpy as np

from core.annotations import (
    POSITION_CHANNEL,
    STAGE_CHANNEL,
    Annotations,
    from_edfplus,
)


class StubReader:
    def __init__(self, onsets, durations, labels):
        self._onsets = onsets
        self._durations = durations
        self._labels = labels

    def readAnnotations(self):
        return self._onsets, self._durations, self._labels


def test_from_edfplus_classifies_stage_and_position():
    reader = StubReader(
        [0.0, 30.0, 60.0, 90.0],
        [30.0, 30.0, 0.0, 10.0],
        [
            "Sleep stage N2",
            "Position-supine",
            "REM",
            "Apnea event",
        ],
    )

    ann = from_edfplus(reader)
    assert isinstance(ann, Annotations)
    assert ann.size == 4

    channels = ann.data["chan"].tolist()
    labels = ann.data["label"].tolist()

    assert channels.count(STAGE_CHANNEL) == 2
    assert labels[0] == "N2"
    assert labels[2] == "REM"
    assert channels[1] == POSITION_CHANNEL
    assert labels[1] == "Supine"
    assert channels[3] == ""
    assert labels[3] == "Apnea event"


def test_from_edfplus_handles_minimal_labels():
    reader = StubReader(
        [0.0, 15.0],
        [30.0, 30.0],
        ["Supine", "Lights On"],
    )
    ann = from_edfplus(reader)
    assert ann.size == 2

    assert ann.data["chan"][0] == POSITION_CHANNEL
    assert ann.data["label"][0] == "Supine"
    assert ann.data["chan"][1] == ""
    assert ann.data["label"][1] == "Lights On"
