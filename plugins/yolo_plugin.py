import cv2
import numpy as np
from typing import List, Optional, Tuple, Union
import os
import sys

# Добавляем родительскую директорию для импорта base_plugin
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from plugins.base_plugin import BaseDetectionPlugin, DetectionResult


class YOLOPlugin(BaseDetectionPlugin):
    """Плагин для моделей YOLO (поддерживает YOLOv5, YOLOv8, YOLOv9, YOLOv10, YOLOv11)"""
    
    def __init__(self):
        self.model = None
        self._loaded = False
        self._model_type = None  # 'ultralytics', 'onnx', 'opencv'
        self._classes = []
        self._input_size = (640, 640)
        self._confidence_threshold = 0.5
        
    def load_model(self, model_path: Optional[str] = None) -> bool:
        """
        Загрузка модели YOLO.
        Поддерживаемые форматы:
        - .pt (PyTorch) - для ultralytics YOLO
        - .onnx (ONNX) - для ONNX Runtime
        - .weights/.cfg (Darknet) - для OpenCV DNN
        """
        if not model_path or not os.path.exists(model_path):
            print(f"Файл модели не найден: {model_path}")
            return False
            
        try:
            # Пробуем загрузить как ultralytics YOLO (PyTorch)
            if model_path.endswith('.pt'):
                return self._load_ultralytics(model_path)
            
            # Пробуем загрузить как ONNX модель
            elif model_path.endswith('.onnx'):
                return self._load_onnx(model_path)
            
            # Пробуем загрузить как Darknet модель для OpenCV
            elif model_path.endswith(('.weights', '.cfg')):
                return self._load_darknet(model_path)
            
            else:
                print(f"Неподдерживаемый формат модели: {model_path}")
                return False
                
        except Exception as e:
            print(f"Ошибка загрузки модели: {e}")
            return False
    
    def _load_ultralytics(self, model_path: str) -> bool:
        """Загрузка модели через ultralytics"""
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            self._model_type = 'ultralytics'
            
            # Получаем имена классов
            if hasattr(self.model, 'names'):
                self._classes = self.model.names
            else:
                self._classes = {i: f"class_{i}" for i in range(80)}
                
            print(f"YOLO модель загружена: {model_path}")
            print(f"Доступные классы: {len(self._classes)}")
            self._loaded = True
            return True
            
        except ImportError:
            print("Не установлена библиотека ultralytics. Установите: pip install ultralytics")
            return False
        except Exception as e:
            print(f"Ошибка загрузки ultralytics модели: {e}")
            return False
    
    def _load_onnx(self, model_path: str) -> bool:
        """Загрузка ONNX модели"""
        try:
            import onnxruntime as ort
            
            # Создаем сессию ONNX Runtime
            providers = ['CPUExecutionProvider']
            self.model = ort.InferenceSession(model_path, providers=providers)
            self._model_type = 'onnx'
            
            # Стандартные классы COCO для YOLO
            self._classes = self._get_coco_classes()
            
            # Определяем входные параметры
            input_details = self.model.get_inputs()[0]
            input_shape = input_details.shape
            if len(input_shape) == 4:
                self._input_size = (input_shape[2], input_shape[3])
                
            print(f"ONNX модель загружена: {model_path}")
            print(f"Входной размер: {self._input_size}")
            self._loaded = True
            return True
            
        except ImportError:
            print("Не установлена библиотека onnxruntime. Установите: pip install onnxruntime")
            return False
        except Exception as e:
            print(f"Ошибка загрузки ONNX модели: {e}")
            return False
    
    def _load_darknet(self, model_path: str) -> bool:
        """Загрузка Darknet модели через OpenCV DNN"""
        try:
            # Для .weights файла нужен соответствующий .cfg файл
            if model_path.endswith('.weights'):
                cfg_path = model_path.replace('.weights', '.cfg')
                if not os.path.exists(cfg_path):
                    print(f"Файл конфигурации не найден: {cfg_path}")
                    return False
            else:
                cfg_path = model_path
                weights_path = model_path.replace('.cfg', '.weights')
                if not os.path.exists(weights_path):
                    print(f"Файл весов не найден: {weights_path}")
                    return False
                model_path = weights_path
                
            # Загружаем сеть
            self.model = cv2.dnn.readNetFromDarknet(cfg_path, model_path)
            self._model_type = 'opencv'
            
            # Стандартные классы COCO
            self._classes = self._get_coco_classes()
            
            print(f"Darknet модель загружена: {model_path}")
            self._loaded = True
            return True
            
        except Exception as e:
            print(f"Ошибка загрузки Darknet модели: {e}")
            return False
    
    def _get_coco_classes(self) -> dict:
        """Возвращает стандартные классы COCO dataset"""
        return {
            0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
            5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
            10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
            14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
            20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
            25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
            30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
            35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket',
            39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon',
            45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli',
            51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair',
            57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet',
            62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone',
            68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator',
            73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear',
            78: 'hair drier', 79: 'toothbrush'
        }
    
    def detect(self, image: np.ndarray) -> List[DetectionResult]:
        """
        Обнаружение объектов на изображении
        """
        if not self._loaded or self.model is None:
            return []
        
        if self._model_type == 'ultralytics':
            return self._detect_ultralytics(image)
        elif self._model_type == 'onnx':
            return self._detect_onnx(image)
        elif self._model_type == 'opencv':
            return self._detect_opencv(image)
        else:
            return []
    
    def _detect_ultralytics(self, image: np.ndarray) -> List[DetectionResult]:
        """Обнаружение через ultralytics YOLO"""
        results = []
        
        try:
            # Выполняем инференс
            predictions = self.model(image, verbose=False)
            
            for pred in predictions:
                boxes = pred.boxes
                if boxes is not None:
                    for box in boxes:
                        # Получаем координаты
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # Фильтрация по порогу уверенности
                        if confidence < self._confidence_threshold:
                            continue
                        
                        # Получаем имя класса
                        label = self._classes.get(class_id, f"class_{class_id}")
                        
                        # Конвертируем в формат (x, y, width, height)
                        x = int(x1)
                        y = int(y1)
                        width = int(x2 - x1)
                        height = int(y2 - y1)
                        
                        results.append(DetectionResult(
                            bbox=(x, y, width, height),
                            label=label,
                            confidence=confidence
                        ))
        except Exception as e:
            print(f"Ошибка при детекции YOLO: {e}")
        
        return results
    
    def _detect_onnx(self, image: np.ndarray) -> List[DetectionResult]:
        """Обнаружение через ONNX Runtime"""
        results = []
        
        try:
            # Предобработка изображения
            input_tensor = self._preprocess_for_onnx(image)
            
            # Выполняем инференс
            input_name = self.model.get_inputs()[0].name
            outputs = self.model.run(None, {input_name: input_tensor})
            
            # Постобработка
            detections = self._postprocess_onnx(outputs[0], image.shape)
            
            for det in detections:
                x, y, w, h, conf, class_id = det
                label = self._classes.get(int(class_id), f"class_{int(class_id)}")
                results.append(DetectionResult(
                    bbox=(int(x), int(y), int(w), int(h)),
                    label=label,
                    confidence=float(conf)
                ))
        except Exception as e:
            print(f"Ошибка при детекции ONNX: {e}")
        
        return results
    
    def _detect_opencv(self, image: np.ndarray) -> List[DetectionResult]:
        """Обнаружение через OpenCV DNN"""
        results = []
        
        try:
            # Получаем выходные имена слоев
            output_layers = self.model.getUnconnectedOutLayersNames()
            
            # Предобработка изображения
            blob = cv2.dnn.blobFromImage(image, 1/255.0, self._input_size, swapRB=True, crop=False)
            self.model.setInput(blob)
            
            # Выполняем инференс
            outputs = self.model.forward(output_layers)
            
            # Постобработка
            detections = self._postprocess_opencv(outputs, image.shape)
            
            for det in detections:
                x, y, w, h, conf, class_id = det
                label = self._classes.get(int(class_id), f"class_{int(class_id)}")
                results.append(DetectionResult(
                    bbox=(int(x), int(y), int(w), int(h)),
                    label=label,
                    confidence=float(conf)
                ))
        except Exception as e:
            print(f"Ошибка при детекции OpenCV: {e}")
        
        return results
    
    def _preprocess_for_onnx(self, image: np.ndarray) -> np.ndarray:
        """Предобработка изображения для ONNX модели"""
        # Изменяем размер
        resized = cv2.resize(image, self._input_size)
        
        # Нормализуем и меняем формат (H,W,C) -> (1,C,H,W)
        input_tensor = resized.astype(np.float32) / 255.0
        input_tensor = np.transpose(input_tensor, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)
        
        return input_tensor
    
    def _postprocess_onnx(self, output: np.ndarray, image_shape: Tuple[int, int]) -> List[Tuple]:
        """Постобработка выходов ONNX модели"""
        detections = []
        
        # Для YOLOv5/v8 в формате ONNX
        if len(output.shape) == 3:
            # output shape: (1, num_detections, 6) - x1,y1,x2,y2,conf,class
            output = output[0]
            for det in output:
                x1, y1, x2, y2, conf, class_id = det
                if conf > self._confidence_threshold:
                    # Конвертируем в (x, y, w, h)
                    w = x2 - x1
                    h = y2 - y1
                    # Масштабируем обратно к оригинальному размеру
                    scale_x = image_shape[1] / self._input_size[0]
                    scale_y = image_shape[0] / self._input_size[1]
                    detections.append((
                        x1 * scale_x, y1 * scale_y,
                        w * scale_x, h * scale_y,
                        conf, class_id
                    ))
        
        return detections
    
    def _postprocess_opencv(self, outputs: List[np.ndarray], image_shape: Tuple[int, int]) -> List[Tuple]:
        """Постобработка выходов OpenCV DNN модели"""
        detections = []
        
        height, width = image_shape[:2]
        
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > self._confidence_threshold:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    detections.append((x, y, w, h, float(confidence), int(class_id)))
        
        return detections
    
    def set_confidence_threshold(self, threshold: float):
        """Установка порога уверенности"""
        self._confidence_threshold = threshold
    
    def get_name(self) -> str:
        return "YOLO Detector (Ultralytics/ONNX/OpenCV)"
    
    def is_loaded(self) -> bool:
        return self._loaded
    
    def get_required_size(self) -> Tuple[int, int]:
        return self._input_size
    
    def get_available_classes(self) -> List[str]:
        """Возвращает список доступных классов"""
        if self._classes:
            # Если _classes это словарь, возвращаем значения
            if isinstance(self._classes, dict):
                return list(self._classes.values())
            # Если это список, возвращаем как есть
            elif isinstance(self._classes, list):
                return self._classes
        return []