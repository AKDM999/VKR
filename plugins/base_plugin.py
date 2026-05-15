from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict
import numpy as np

class DetectionResult:
    """Класс для хранения результата обнаружения"""
    def __init__(self, bbox: Tuple[int, int, int, int], label: str, confidence: float, track_id: int = -1):
        self.bbox = bbox  # (x, y, width, height)
        self.label = label
        self.confidence = confidence
        self.track_id = track_id  # ID объекта для отслеживания

class BaseDetectionPlugin(ABC):
    """Базовый абстрактный класс для всех плагинов обнаружения"""
    
    @abstractmethod
    def load_model(self, model_path: Optional[str] = None) -> bool:
        """Загрузка модели"""
        pass
    
    @abstractmethod
    def detect(self, image: np.ndarray) -> List[DetectionResult]:
        """Обнаружение объектов на изображении"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Возвращает имя плагина"""
        pass
    
    @abstractmethod
    def is_loaded(self) -> bool:
        """Проверка, загружена ли модель"""
        pass
    
    def get_required_size(self) -> Tuple[int, int]:
        """Возвращает требуемый размер входного изображения (ширина, высота)"""
        return (640, 640)
    
    def enable_tracking(self, enable: bool = True):
        """Включить/выключить отслеживание объектов"""
        pass