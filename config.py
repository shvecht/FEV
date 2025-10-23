from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from core.prefetch import PrefetchConfig


@dataclass
class ViewerConfig:
    prefetch_tile_s: float = 5.0
    prefetch_max_tiles: int | None = 64
    prefetch_max_mb: float | None = 16.0
    prefetch_collapsed: bool = False
    controls_collapsed: bool = False
    int16_cache_enabled: bool = False
    int16_cache_max_mb: float = 512.0
    int16_cache_memmap: bool = False
    lod_envelope_ratio: float = 2.0
    hidden_channels: tuple[int, ...] = ()
    hidden_annotation_channels: tuple[str, ...] = ("stage", "position")
    annotation_focus_only: bool = False
    theme: str = "Midnight"
    canvas_backend: str = "pyqtgraph"
    lod_enabled: bool = True
    lod_min_bin_multiple: float = 2.0
    lod_min_view_duration_s: float = 240.0
    ini_path: Path | None = None

    @classmethod
    def load(cls, ini_path: str | Path | None = None) -> "ViewerConfig":
        cfg = cls()
        path = Path(ini_path or "config.ini")
        if path.exists():
            import configparser

            parser = configparser.ConfigParser()
            parser.read(path)
            section = parser["prefetch"] if "prefetch" in parser else None
            if section:
                cfg.prefetch_tile_s = section.getfloat("tile_s", fallback=cfg.prefetch_tile_s)
                cfg.prefetch_max_tiles = section.getint("max_tiles", fallback=cfg.prefetch_max_tiles)
                cfg.prefetch_max_mb = section.getfloat("max_mb", fallback=cfg.prefetch_max_mb)

            ui_section = parser["ui"] if "ui" in parser else None
            if ui_section:
                cfg.prefetch_collapsed = ui_section.getboolean(
                    "prefetch_collapsed", fallback=cfg.prefetch_collapsed
                )
                cfg.controls_collapsed = ui_section.getboolean(
                    "controls_collapsed", fallback=cfg.controls_collapsed
                )
                cfg.annotation_focus_only = ui_section.getboolean(
                    "annotation_focus_only", fallback=cfg.annotation_focus_only
                )
                cfg.theme = ui_section.get("theme", fallback=cfg.theme)
                cfg.canvas_backend = ui_section.get(
                    "canvas_backend", fallback=cfg.canvas_backend
                )
                hidden_raw = ui_section.get("hidden_channels", fallback="")
                if hidden_raw:
                    indices: set[int] = set()
                    for part in hidden_raw.split(","):
                        part = part.strip()
                        if not part:
                            continue
                        try:
                            value = int(part)
                        except ValueError:
                            continue
                        if value >= 0:
                            indices.add(value)
                    if indices:
                        cfg.hidden_channels = tuple(sorted(indices))

                hidden_ann_raw = ui_section.get("hidden_annotation_channels", fallback="")
                if hidden_ann_raw:
                    names = [part.strip() for part in hidden_ann_raw.split(",") if part.strip()]
                    if names:
                        normalized = tuple(part.lower() for part in names)
                        default_normalized = tuple(
                            part.lower() for part in cfg.hidden_annotation_channels
                        )
                        if normalized != default_normalized:
                            # Preserve unique entries while maintaining relative order
                            seen: set[str] = set()
                            ordered = []
                            for name in names:
                                if name not in seen:
                                    seen.add(name)
                                    ordered.append(name)
                            cfg.hidden_annotation_channels = tuple(ordered)
                    else:
                        cfg.hidden_annotation_channels = ()

            cache_section = parser["cache"] if "cache" in parser else None
            if cache_section:
                cfg.int16_cache_enabled = cache_section.getboolean(
                    "enabled", fallback=cfg.int16_cache_enabled
                )
                cfg.int16_cache_max_mb = cache_section.getfloat(
                    "max_mb", fallback=cfg.int16_cache_max_mb
                )
                cfg.int16_cache_memmap = cache_section.getboolean(
                    "memmap", fallback=cfg.int16_cache_memmap
                )
            lod_section = parser["lod"] if "lod" in parser else None
            if lod_section:
                cfg.lod_enabled = lod_section.getboolean(
                    "enabled", fallback=cfg.lod_enabled
                )
                cfg.lod_min_bin_multiple = lod_section.getfloat(
                    "min_bin_multiple", fallback=cfg.lod_min_bin_multiple
                )
                cfg.lod_min_view_duration_s = lod_section.getfloat(
                    "min_view_duration_s", fallback=cfg.lod_min_view_duration_s
                )
                cfg.lod_envelope_ratio = lod_section.getfloat(
                    "envelope_ratio", fallback=cfg.lod_envelope_ratio
                )
        cfg.ini_path = path
        return cfg

    def prefetch_config(self) -> PrefetchConfig:
        return PrefetchConfig(
            tile_duration=self.prefetch_tile_s,
            max_tiles=self.prefetch_max_tiles or 1,
            max_bytes=(self.prefetch_max_mb * 1024 * 1024) if self.prefetch_max_mb else None,
        )

    def save(self) -> None:
        if self.ini_path is None:
            return
        import configparser

        parser = configparser.ConfigParser()
        parser["prefetch"] = {
            "tile_s": f"{self.prefetch_tile_s:.3f}",
            "max_tiles": str(self.prefetch_max_tiles or 0),
            "max_mb": f"{self.prefetch_max_mb or 0}",
        }
        hidden_serialized = ",".join(str(idx) for idx in sorted(set(self.hidden_channels)))
        hidden_ann_serialized = ",".join(
            name for name in dict.fromkeys(self.hidden_annotation_channels)
        )
        parser["ui"] = {
            "prefetch_collapsed": "true" if self.prefetch_collapsed else "false",
            "controls_collapsed": "true" if self.controls_collapsed else "false",
            "hidden_channels": hidden_serialized,
            "hidden_annotation_channels": hidden_ann_serialized,
            "annotation_focus_only": "true" if self.annotation_focus_only else "false",
            "theme": self.theme,
            "canvas_backend": self.canvas_backend,
        }
        parser["cache"] = {
            "enabled": "true" if self.int16_cache_enabled else "false",
            "max_mb": f"{self.int16_cache_max_mb:.3f}",
            "memmap": "true" if self.int16_cache_memmap else "false",
        }
        parser["lod"] = {
            "enabled": "true" if self.lod_enabled else "false",
            "min_bin_multiple": f"{self.lod_min_bin_multiple:.3f}",
            "min_view_duration_s": f"{self.lod_min_view_duration_s:.3f}",
            "envelope_ratio": f"{self.lod_envelope_ratio:.3f}",
        }
        with self.ini_path.open("w") as fh:
            parser.write(fh)
