# app.py
import sys
from PySide6 import QtWidgets
from core.edf_loader import EdfLoader
from ui.main_window import MainWindow

def main(path):
    loader = EdfLoader(path)
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow(loader)
    w.resize(1200, 700); w.show()
    app.exec()
    loader.close()

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("edf_path")
    args = p.parse_args()
    main(args.edf_path)