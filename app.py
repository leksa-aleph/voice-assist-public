"""
Voice Assist - живые субтитры русской речи для слабослышащих.

Точка входа: логирование, Qt, главное окно, оверлей и трей.
Интерфейс - в пакете ui, распознавание - engine.py, Discord-бот - discord_bot.py.
"""
from __future__ import annotations

import os
import platform
import sys

if platform.system() == 'Linux':
    # на Wayland прозрачный оверлей поверх окон не работает
    os.environ.setdefault('QT_QPA_PLATFORM', 'xcb')

# BLAS в numpy по умолчанию заводит потоки на все ядра и держит их
# в холостом ожидании - ровно та беда, из-за которой старая версия
# грузила процессор. Наши матрицы маленькие, им хватает одного потока.
# Действует, только если задано до первого import numpy.
for _var in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_var, '1')

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QSystemTrayIcon

import overlay
import version
from logging_setup import setup_logging
from ui import theme


def main():
    if '--selftest' in sys.argv:
        import selftest
        sys.exit(selftest.run())
    log_file = setup_logging()
    print(f'Лог: {log_file}')

    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(sys.argv)
    app.setApplicationName('voice-transcriber')
    app.setApplicationDisplayName('Voice Assist')
    app.setApplicationVersion(version.TAG)
    app.setQuitOnLastWindowClosed(False)
    theme.init_fonts()
    app.setFont(theme.font(13))

    # окно импортируем после QApplication: виджетам нужен экран
    from ui.window import MainWindow, Tray

    window = MainWindow()
    window.set_overlay(overlay.OverlayWindow())

    if QSystemTrayIcon.isSystemTrayAvailable():
        Tray(window, app)
    else:
        print('Системный трей недоступен.', file=sys.stderr)

    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
