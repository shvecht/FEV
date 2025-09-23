# app.py
import sys
from PySide6 import QtWidgets
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


def main(path=None):
    app = QtWidgets.QApplication(sys.argv)

    if not path:
        path = _select_file_dialog()
        if not path:
            return

    loader = EdfLoader(path)
    w = MainWindow(loader)
    w.resize(1200, 700)
    w.show()
    app.exec()
    loader.close()

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("edf_path", nargs="?")
    args = p.parse_args()
    main(args.edf_path)
