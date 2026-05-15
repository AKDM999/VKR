# plugins/__init__.py
import sys
import os

# Добавляем родительскую директорию в путь
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from plugins.base_plugin import BaseDetectionPlugin, DetectionResult
from plugins.dummy_plugin import DummyPlugin
from plugins.yolo_plugin import YOLOPlugin

__all__ = ['BaseDetectionPlugin', 'DetectionResult', 'DummyPlugin', 'YOLOPlugin']