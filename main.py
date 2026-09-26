"""Chạy ứng dụng: python main.py"""

import sys

from PyQt6.QtWidgets import QApplication

from guf.app import MainWindow
from guf.config import default_data_dir


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("GUF Profile Manager")
    window = MainWindow(default_data_dir())
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
