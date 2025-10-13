from pathlib import Path

from config import ViewerConfig


def test_viewer_config_hidden_annotation_channels_default(tmp_path: Path):
    ini_path = tmp_path / "missing.ini"
    cfg = ViewerConfig.load(ini_path)
    assert set(cfg.hidden_annotation_channels) == {"stage", "position"}


def test_viewer_config_hidden_annotation_channels_parse(tmp_path: Path):
    ini_path = tmp_path / "config.ini"
    ini_path.write_text(
        """
[ui]
hidden_annotation_channels = stage,custom
""".strip()
    )

    cfg = ViewerConfig.load(ini_path)
    assert set(cfg.hidden_annotation_channels) == {"stage", "custom"}


def test_viewer_config_hidden_annotation_channels_save(tmp_path: Path):
    ini_path = tmp_path / "config.ini"
    cfg = ViewerConfig.load(ini_path)
    cfg.hidden_annotation_channels = ("stage", "custom")
    cfg.save()

    written = ini_path.read_text()
    assert "hidden_annotation_channels = stage,custom" in written
