import cv2
import numpy as np
from typing import List, Optional, Tuple
import sys
import os

# Добавляем родительскую директорию для импорта base_plugin
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from plugins.base_plugin import BaseDetectionPlugin, DetectionResult


class DummyPlugin(BaseDetectionPlugin):
    """Плагин-заглушка для демонстрации (эмулирует обнаружение)"""
    
    def __init__(self):
        self._loaded = False
        self._name = "Dummy Detector (Haar-like cascade emulation)"
        
    def load_model(self, model_path: Optional[str] = None) -> bool:
        # В реальном плагине здесь была бы загрузка модели
        print(f"Загрузка модели-заглушки...")
        self._loaded = True
        return True
    
    def detect(self, image: np.ndarray) -> List[DetectionResult]:
        if not self._loaded:
            return []
        
        results = []
        h, w = image.shape[:2]
        
        # Эмуляция обнаружения: находим яркие области
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        blur = cv2.GaussianBlur(gray, (21, 21), 0)
        _, thresh = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, cw, ch = cv2.boundingRect(contour)
            area = cw * ch
            if area > 1000 and area < (w * h * 0.8):  # Фильтр по площади
                results.append(DetectionResult(
                    bbox=(x, y, cw, ch),
                    label="Bright Object",
                    confidence=0.7
                ))
        
        return results
    
    def get_name(self) -> str:
        return self._name
    
    def is_loaded(self) -> bool:
        return self._loaded