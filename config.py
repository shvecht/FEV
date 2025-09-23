from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from core.prefetch import PrefetchConfig


@dataclass
class ViewerConfig:
    prefetch_tile_s: float = 5.0
    prefetch_max_tiles: int | None = 64
    prefetch_max_mb: float | None = 16.0
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
        with self.ini_path.open("w") as fh:
            parser.write(fh)
