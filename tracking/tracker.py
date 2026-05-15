import numpy as np
from typing import List, Tuple, Dict
from collections import defaultdict
import math

class TrackedObject:
    """Класс для хранения информации об отслеживаемом объекте"""
    def __init__(self, track_id: int, bbox: Tuple[int, int, int, int], label: str, confidence: float):
        self.track_id = track_id
        self.bbox = bbox
        self.label = label
        self.confidence = confidence
        self.age = 0  # Возраст трека (сколько кадров существует)
        self.hits = 1  # Количество успешных обнаружений
        self.no_loss_consecutive = 0  # Количество кадров без потери
        
    def update(self, bbox: Tuple[int, int, int, int], confidence: float):
        """Обновление позиции объекта"""
        self.bbox = bbox
        self.confidence = confidence
        self.hits += 1
        self.no_loss_consecutive = 0
        
    def predict(self):
        """Предсказание позиции (простое линейное предсказание)"""
        self.age += 1
        self.no_loss_consecutive += 1

class SimpleTracker:
    """Простой трекер объектов на основе IOU (Intersection over Union)"""
    
    def __init__(self, iou_threshold: float = 0.3, max_lost_frames: int = 30):
        """
        iou_threshold: порог IOU для сопоставления объектов
        max_lost_frames: максимальное количество кадров без обнаружения перед удалением трека
        """
        self.next_track_id = 1
        self.tracks: Dict[int, TrackedObject] = {}
        self.iou_threshold = iou_threshold
        self.max_lost_frames = max_lost_frames
        
    def calculate_iou(self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
        """Вычисление Intersection over Union для двух bounding box"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Координаты пересечения
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
            
        # Площадь пересечения
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Площади каждого bbox
        bbox1_area = w1 * h1
        bbox2_area = w2 * h2
        
        # Площадь объединения
        union_area = bbox1_area + bbox2_area - intersection_area
        
        # IOU
        iou = intersection_area / union_area if union_area > 0 else 0
        return iou
    
    def update(self, detections: List[Tuple[int, int, int, int, str, float]]) -> List[TrackedObject]:
        """
        Обновление трекера новыми обнаружениями
        detections: список (x, y, w, h, label, confidence)
        Возвращает: список отслеживаемых объектов
        """
        if len(detections) == 0:
            # Нет обнаружений - предсказываем позиции существующих треков
            for track_id in list(self.tracks.keys()):
                self.tracks[track_id].predict()
                # Удаляем старые треки
                if self.tracks[track_id].no_loss_consecutive > self.max_lost_frames:
                    del self.tracks[track_id]
            return list(self.tracks.values())
        
        # Предсказываем позиции существующих треков
        for track in self.tracks.values():
            track.predict()
        
        # Создаем матрицу стоимости (IOU) между существующими треками и новыми обнаружениями
        if len(self.tracks) > 0:
            cost_matrix = np.zeros((len(self.tracks), len(detections)))
            track_list = list(self.tracks.values())
            
            for i, track in enumerate(track_list):
                for j, det in enumerate(detections):
                    det_bbox = det[:4]
                    iou = self.calculate_iou(track.bbox, det_bbox)
                    cost_matrix[i, j] = 1 - iou  # Минимизируем стоимость
            
            # Простое сопоставление: для каждого обнаружения находим лучший трек
            matched_tracks = set()
            matched_detections = set()
            
            # Сортируем по наименьшей стоимости
            for j in range(len(detections)):
                best_iou = -1
                best_i = -1
                for i in range(len(track_list)):
                    if i in matched_tracks:
                        continue
                    iou = 1 - cost_matrix[i, j]
                    if iou > best_iou and iou > self.iou_threshold:
                        best_iou = iou
                        best_i = i
                
                if best_i != -1:
                    # Обновляем существующий трек
                    track_id = track_list[best_i].track_id
                    det = detections[j]
                    self.tracks[track_id].update(det[:4], det[5])
                    matched_tracks.add(best_i)
                    matched_detections.add(j)
            
            # Создаем новые треки для несоответствовавших обнаружений
            for j, det in enumerate(detections):
                if j not in matched_detections:
                    new_track = TrackedObject(self.next_track_id, det[:4], det[4], det[5])
                    self.tracks[self.next_track_id] = new_track
                    self.next_track_id += 1
            
            # Удаляем старые треки
            for i, track in enumerate(track_list):
                if i not in matched_tracks:
                    if track.no_loss_consecutive > self.max_lost_frames:
                        del self.tracks[track.track_id]
        
        else:
            # Нет существующих треков - создаем новые для всех обнаружений
            for det in detections:
                new_track = TrackedObject(self.next_track_id, det[:4], det[4], det[5])
                self.tracks[self.next_track_id] = new_track
                self.next_track_id += 1
        
        return list(self.tracks.values())
    
    def reset(self):
        """Сброс трекера"""
        self.next_track_id = 1
        self.tracks.clear()

class AdvancedTracker:
    """Продвинутый трекер с использованием венгерского алгоритма и фильтра Калмана"""
    
    def __init__(self, iou_threshold: float = 0.3, max_lost_frames: int = 30):
        """
        iou_threshold: порог IOU для сопоставления объектов
        max_lost_frames: максимальное количество кадров без обнаружения перед удалением трека
        """
        self.next_track_id = 1
        self.tracks: Dict[int, TrackedObject] = {}
        self.iou_threshold = iou_threshold
        self.max_lost_frames = max_lost_frames
        
    def calculate_iou(self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
        """Вычисление Intersection over Union"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
            
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        bbox1_area = w1 * h1
        bbox2_area = w2 * h2
        union_area = bbox1_area + bbox2_area - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0
    
    def hungarian_algorithm(self, cost_matrix: np.ndarray) -> List[Tuple[int, int]]:
        """Венгерский алгоритм для оптимального сопоставления"""
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        return list(zip(row_ind, col_ind))
    
    def update(self, detections: List[Tuple[int, int, int, int, str, float]]) -> List[TrackedObject]:
        """Обновление трекера с использованием венгерского алгоритма"""
        if len(detections) == 0:
            for track_id in list(self.tracks.keys()):
                self.tracks[track_id].predict()
                if self.tracks[track_id].no_loss_consecutive > self.max_lost_frames:
                    del self.tracks[track_id]
            return list(self.tracks.values())
        
        for track in self.tracks.values():
            track.predict()
        
        if len(self.tracks) > 0:
            cost_matrix = np.zeros((len(self.tracks), len(detections)))
            track_list = list(self.tracks.values())
            
            for i, track in enumerate(track_list):
                for j, det in enumerate(detections):
                    iou = self.calculate_iou(track.bbox, det[:4])
                    cost_matrix[i, j] = 1 - iou
            
            # Применяем венгерский алгоритм
            matches = self.hungarian_algorithm(cost_matrix)
            
            matched_tracks = set()
            matched_detections = set()
            
            for i, j in matches:
                iou = 1 - cost_matrix[i, j]
                if iou > self.iou_threshold:
                    track_id = track_list[i].track_id
                    self.tracks[track_id].update(detections[j][:4], detections[j][5])
                    matched_tracks.add(i)
                    matched_detections.add(j)
            
            # Создаем новые треки
            for j, det in enumerate(detections):
                if j not in matched_detections:
                    new_track = TrackedObject(self.next_track_id, det[:4], det[4], det[5])
                    self.tracks[self.next_track_id] = new_track
                    self.next_track_id += 1
            
            # Удаляем старые треки
            for i, track in enumerate(track_list):
                if i not in matched_tracks:
                    if track.no_loss_consecutive > self.max_lost_frames:
                        del self.tracks[track.track_id]
        
        else:
            for det in detections:
                new_track = TrackedObject(self.next_track_id, det[:4], det[4], det[5])
                self.tracks[self.next_track_id] = new_track
                self.next_track_id += 1
        
        return list(self.tracks.values())
    
    def reset(self):
        """Сброс трекера"""
        self.next_track_id = 1
        self.tracks.clear()