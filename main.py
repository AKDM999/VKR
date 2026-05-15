#!/usr/bin/env python3
import sys
import os

# Добавляем текущую директорию в путь для импорта
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# ВАЖНО: Сначала импортируем torch, чтобы инициализировать его DLL до PyQt
try:
    import torch
    print(f"PyTorch версия: {torch.__version__}")
    print(f"CUDA доступна: {torch.cuda.is_available()}")
except ImportError:
    print("PyTorch не установлен. Установите: pip install torch")
    # Не прерываем выполнение, так как YOLO плагин может и не использоваться

from PyQt5.QtWidgets import QApplication
from ui.main_window import MainWindow, ThemeManager

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # Применяем темную тему по умолчанию
    ThemeManager.apply_theme(app, "dark")
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()