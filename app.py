# app.py
import sys
from PySide6 import QtWidgets, QtGui
from config import ViewerConfig
from core.edf_loader import EdfLoader
from ui.main_window import MainWindow


def _select_file_dialog(parent=None):
    dlg = QtWidgets.QFileDialog(parent)
    dlg.setWindowTitle("Select EDF file")
    dlg.setFileMode(QtWidgets.QFileDialog.ExistingFile)
    dlg.setNameFilters([
        "EDF Files (*.edf *.EDF)",
        "All Files (*)",
    ])
    if dlg.exec() == QtWidgets.QDialog.Accepted:
        files = dlg.selectedFiles()
        return files[0] if files else None
    return None


def main(
    path=None,
    *,
    config_path: str | None = None,
    prefetch_tile: float | None = None,
    prefetch_max_tiles: int | None = None,
    prefetch_max_mb: float | None = None,
):
    cfg = ViewerConfig.load(config_path)
    if prefetch_tile is not None:
        cfg.prefetch_tile_s = prefetch_tile
    if prefetch_max_tiles is not None and prefetch_max_tiles > 0:
        cfg.prefetch_max_tiles = prefetch_max_tiles
    if prefetch_max_mb is not None:
        cfg.prefetch_max_mb = prefetch_max_mb if prefetch_max_mb > 0 else None
    app = QtWidgets.QApplication(sys.argv)
    app.setWindowIcon(QtGui.QIcon("icon.png"))

    if not path:
        path = _select_file_dialog()
        if not path:
            return

    loader = EdfLoader(path)
    w = MainWindow(loader, config=cfg)
    w.resize(1200, 700)
    w.show()
    app.exec()
    loader.close()

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("edf_path", nargs="?")
    p.add_argument("--config")
    p.add_argument("--prefetch-tile", type=float)
    p.add_argument("--prefetch-max-tiles", type=int)
    p.add_argument("--prefetch-max-mb", type=float)
    args = p.parse_args()
    main(
        args.edf_path,
        config_path=args.config,
        prefetch_tile=args.prefetch_tile,
        prefetch_max_tiles=args.prefetch_max_tiles,
        prefetch_max_mb=args.prefetch_max_mb,
    )
