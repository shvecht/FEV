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
    int16_cache_enabled: bool = False
    int16_cache_max_mb: float = 512.0
    int16_cache_memmap: bool = False
    hidden_channels: tuple[int, ...] = ()
    annotation_focus_only: bool = False
    theme: str = "Midnight"
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
                cfg.annotation_focus_only = ui_section.getboolean(
                    "annotation_focus_only", fallback=cfg.annotation_focus_only
                )
                cfg.theme = ui_section.get("theme", fallback=cfg.theme)
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
        parser["ui"] = {
            "prefetch_collapsed": "true" if self.prefetch_collapsed else "false",
            "hidden_channels": hidden_serialized,
            "annotation_focus_only": "true" if self.annotation_focus_only else "false",
            "theme": self.theme,
        }
        parser["cache"] = {
            "enabled": "true" if self.int16_cache_enabled else "false",
            "max_mb": f"{self.int16_cache_max_mb:.3f}",
            "memmap": "true" if self.int16_cache_memmap else "false",
        }
        with self.ini_path.open("w") as fh:
            parser.write(fh)
