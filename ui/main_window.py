import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from typing import Optional, List, Dict, Tuple
import importlib.util
from pathlib import Path
import time
from collections import defaultdict

# Добавляем родительскую директорию в путь для импорта
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from plugins.base_plugin import BaseDetectionPlugin, DetectionResult
from tracking.tracker import SimpleTracker, AdvancedTracker

class VideoThread(QThread):
    """Поток для обработки видео с трекингом"""
    frame_processed = pyqtSignal(np.ndarray, list, float)
    error_occurred = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.running = False
        self.paused = False
        self.cap = None
        self.source_type = None
        self.source_path = 0
        self.plugin: Optional[BaseDetectionPlugin] = None
        self.resize_size = (640, 640)
        self.mutex = QMutex()
        
        # Для подсчета FPS
        self.frame_count = 0
        self.fps = 0
        self.last_time = time.time()
        
        # Трекер
        self.tracker = None
        self.use_tracking = True
        self.tracker_type = "simple"
        
    def set_source(self, source_type: str, path=None):
        self.source_type = source_type
        self.source_path = path if path else 0
        
    def set_plugin(self, plugin: BaseDetectionPlugin):
        self.plugin = plugin
        
    def enable_tracking(self, enable: bool, tracker_type: str = "simple"):
        """Включение/выключение трекинга"""
        self.use_tracking = enable
        self.tracker_type = tracker_type
        if enable:
            if tracker_type == "simple":
                self.tracker = SimpleTracker(iou_threshold=0.3, max_lost_frames=30)
            else:
                self.tracker = AdvancedTracker(iou_threshold=0.3, max_lost_frames=30)
        else:
            self.tracker = None
            
    def reset_tracker(self):
        """Сброс трекера"""
        if self.tracker:
            self.tracker.reset()
        
    def pause(self):
        self.mutex.lock()
        self.paused = True
        self.mutex.unlock()
        
    def resume(self):
        self.mutex.lock()
        self.paused = False
        self.mutex.unlock()
        
    def stop(self):
        self.running = False
        self.mutex.lock()
        self.paused = False
        self.mutex.unlock()
        self.wait()
    
    def resize_with_padding(self, image: np.ndarray, target_size: Tuple[int, int] = (640, 640)):
        h, w = image.shape[:2]
        target_w, target_h = target_size
        
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        resized = cv2.resize(image, (new_w, new_h))
        
        if len(image.shape) == 3:
            padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        else:
            padded = np.zeros((target_h, target_w), dtype=np.uint8)
        
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2
        padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
        
        return padded, (x_offset, y_offset, new_w, new_h, scale)
        
    def run(self):
        self.running = True
        self.frame_count = 0
        self.last_time = time.time()
        
        if self.source_type == 'camera':
            self.cap = cv2.VideoCapture(self.source_path)
        elif self.source_type == 'video':
            self.cap = cv2.VideoCapture(self.source_path)
        else:
            self.error_occurred.emit("Неизвестный тип источника")
            return
            
        if not self.cap.isOpened():
            self.error_occurred.emit("Не удалось открыть источник видео")
            return
            
        while self.running:
            self.mutex.lock()
            paused = self.paused
            self.mutex.unlock()
            
            if paused:
                self.msleep(100)
                continue
                
            ret, frame = self.cap.read()
            if not ret:
                if self.source_type == 'video':
                    self.error_occurred.emit("Видео закончилось")
                    break
                else:
                    self.msleep(10)
                    continue
            
            # Подсчет FPS
            self.frame_count += 1
            current_time = time.time()
            if current_time - self.last_time >= 1.0:
                self.fps = self.frame_count
                self.frame_count = 0
                self.last_time = current_time
            
            # Сохраняем оригинальный размер
            original_frame = frame.copy()
            h_orig, w_orig = original_frame.shape[:2]
            
            # Изменяем размер для детекции
            processed_frame, (x_offset, y_offset, new_w, new_h, scale) = self.resize_with_padding(original_frame, self.resize_size)
            
            # Обнаружение объектов
            detections = []
            if self.plugin and self.plugin.is_loaded():
                raw_detections = self.plugin.detect(processed_frame)
                
                # Конвертируем в формат для трекера и корректируем координаты
                detections_for_tracker = []
                for det in raw_detections:
                    x, y, w, h = det.bbox
                    # Проверяем, находится ли детекция в полезной области
                    if (x >= x_offset and x <= x_offset + new_w and 
                        y >= y_offset and y <= y_offset + new_h):
                        # Пересчитываем координаты в оригинальное пространство
                        x_orig = int((x - x_offset) / scale)
                        y_orig = int((y - y_offset) / scale)
                        w_orig_box = int(w / scale)
                        h_orig_box = int(h / scale)
                        
                        # Ограничиваем границы
                        x_orig = max(0, min(x_orig, w_orig))
                        y_orig = max(0, min(y_orig, h_orig))
                        w_orig_box = min(w_orig_box, w_orig - x_orig)
                        h_orig_box = min(h_orig_box, h_orig - y_orig)
                        
                        if w_orig_box > 0 and h_orig_box > 0:
                            detections_for_tracker.append(
                                (x_orig, y_orig, w_orig_box, h_orig_box, det.label, det.confidence)
                            )
                
                # Применяем трекинг или используем обычные детекции
                if self.use_tracking and self.tracker and len(detections_for_tracker) > 0:
                    tracked_objects = self.tracker.update(detections_for_tracker)
                    # Конвертируем обратно в DetectionResult с ID треков
                    for obj in tracked_objects:
                        detections.append(DetectionResult(
                            bbox=obj.bbox,
                            label=obj.label,
                            confidence=obj.confidence,
                            track_id=obj.track_id
                        ))
                else:
                    # Без трекинга - создаем временные ID
                    for i, det in enumerate(detections_for_tracker):
                        detections.append(DetectionResult(
                            bbox=det[:4],
                            label=det[4],
                            confidence=det[5],
                            track_id=-1
                        ))
                    # Если трекинг включен, но нет детекций, все равно обновляем трекер
                    if self.use_tracking and self.tracker:
                        self.tracker.update([])
            
            # Передаем кадр с детекциями
            self.frame_processed.emit(original_frame, detections, self.fps)
            self.msleep(1)
            
        if self.cap:
            self.cap.release()

class ImageNavigationWidget(QWidget):
    """Виджет для навигации по изображениям"""
    next_clicked = pyqtSignal()
    prev_clicked = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.prev_btn = QPushButton("◀ Предыдущее")
        self.prev_btn.clicked.connect(self.prev_clicked)
        self.prev_btn.setEnabled(False)
        
        self.next_btn = QPushButton("Следующее ▶")
        self.next_btn.clicked.connect(self.next_clicked)
        self.next_btn.setEnabled(False)
        
        self.info_label = QLabel("Изображение: 0/0")
        
        # Добавляем кнопку для открытия папки/изображения
        self.open_btn = QPushButton("📁 Открыть изображение/папку")
        self.open_btn.clicked.connect(self.open_clicked)
        
        layout.addWidget(self.prev_btn)
        layout.addWidget(self.next_btn)
        layout.addWidget(self.info_label)
        layout.addStretch()
        layout.addWidget(self.open_btn)
        
        self.setLayout(layout)
        
    def open_clicked(self):
        """Сигнал для открытия файла/папки"""
        # Этот сигнал будет обрабатываться главным окном
        pass
        
    def set_navigation_enabled(self, enabled: bool):
        self.prev_btn.setEnabled(enabled)
        self.next_btn.setEnabled(enabled)
        
    def update_info(self, current: int, total: int):
        self.info_label.setText(f"Изображение: {current + 1}/{total}")
        self.prev_btn.setEnabled(current > 0)
        self.next_btn.setEnabled(current < total - 1)

class VideoControlWidget(QWidget):
    """Виджет управления видео"""
    play_pause_clicked = pyqtSignal()
    stop_clicked = pyqtSignal()
    tracking_toggled = pyqtSignal(bool)
    tracker_type_changed = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.is_playing = False
        self.use_tracking = True
        self.init_ui()
        
    def init_ui(self):
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.play_pause_btn = QPushButton("▶ Старт")
        self.play_pause_btn.clicked.connect(self.on_play_pause_clicked)
        
        self.stop_btn = QPushButton("⏹ Сброс")
        self.stop_btn.clicked.connect(self.stop_clicked)
        
        # Чекбокс для включения/выключения трекинга
        self.tracking_checkbox = QCheckBox("Отслеживание объектов")
        self.tracking_checkbox.setChecked(True)
        self.tracking_checkbox.toggled.connect(self.on_tracking_toggled)
        
        # Выбор типа трекера
        self.tracker_type_combo = QComboBox()
        self.tracker_type_combo.addItems(["Simple (IOU)", "Advanced (Hungarian)"])
        self.tracker_type_combo.currentTextChanged.connect(self.on_tracker_type_changed)
        self.tracker_type_combo.setEnabled(True)
        
        # Метка для отображения FPS
        self.fps_label = QLabel("FPS: 0")
        self.fps_label.setStyleSheet("font-weight: bold; color: #00ff00; background-color: rgba(0,0,0,0.7); padding: 2px 5px; border-radius: 3px;")
        
        layout.addWidget(self.play_pause_btn)
        layout.addWidget(self.stop_btn)
        layout.addWidget(QLabel("  |  "))
        layout.addWidget(self.tracking_checkbox)
        layout.addWidget(QLabel("Тип трекера:"))
        layout.addWidget(self.tracker_type_combo)
        layout.addStretch()
        layout.addWidget(self.fps_label)
        
        self.setLayout(layout)
    
    def on_play_pause_clicked(self):
        """Обработчик нажатия кнопки Старт/Пауза"""
        self.play_pause_clicked.emit()
    
    def set_playing(self, playing: bool):
        """Установить состояние воспроизведения"""
        self.is_playing = playing
        if playing:
            self.play_pause_btn.setText("⏸ Пауза")
            self.play_pause_btn.setToolTip("Пауза")
        else:
            self.play_pause_btn.setText("▶ Старт")
            self.play_pause_btn.setToolTip("Старт")
    
    def is_playing_state(self) -> bool:
        """Вернуть текущее состояние"""
        return self.is_playing
    
    def on_tracking_toggled(self, checked: bool):
        self.use_tracking = checked
        self.tracker_type_combo.setEnabled(checked)
        self.tracking_toggled.emit(checked)
        
    def on_tracker_type_changed(self, text: str):
        tracker_type = "simple" if "Simple" in text else "advanced"
        self.tracker_type_changed.emit(tracker_type)
    
    def update_fps(self, fps: float):
        self.fps_label.setText(f"FPS: {fps:.1f}")
        
    def reset(self):
        """Сброс кнопки в состояние остановлено"""
        self.set_playing(False)
        self.update_fps(0)

class ThemeManager:
    """Менеджер для управления темами оформления"""
    
    # Темная тема (Dark Theme)
    DARK_THEME = """
        QMainWindow {
            background-color: #2b2b2b;
        }
        QWidget {
            background-color: #2b2b2b;
            color: #ffffff;
            font-family: 'Segoe UI', 'Arial', sans-serif;
            font-size: 10pt;
        }
        QLabel {
            color: #ffffff;
            background-color: transparent;
        }
        QPushButton {
            background-color: #3c3c3c;
            border: 1px solid #555555;
            border-radius: 3px;
            padding: 5px 10px;
            color: #ffffff;
        }
        QPushButton:hover {
            background-color: #4a4a4a;
            border-color: #777777;
        }
        QPushButton:pressed {
            background-color: #2a2a2a;
        }
        QPushButton:disabled {
            background-color: #2b2b2b;
            color: #666666;
            border-color: #3a3a3a;
        }
        QComboBox {
            background-color: #3c3c3c;
            border: 1px solid #555555;
            border-radius: 3px;
            padding: 3px;
            color: #ffffff;
        }
        QComboBox:hover {
            border-color: #777777;
        }
        QComboBox::drop-down {
            border: none;
        }
        QComboBox::down-arrow {
            image: none;
            border-left: 5px solid transparent;
            border-right: 5px solid transparent;
            border-top: 5px solid #ffffff;
            margin-right: 5px;
        }
        QSlider::groove:horizontal {
            border: 1px solid #555555;
            height: 6px;
            background: #3c3c3c;
            border-radius: 3px;
        }
        QSlider::handle:horizontal {
            background: #5a5a5a;
            border: 1px solid #777777;
            width: 14px;
            height: 14px;
            margin: -5px 0;
            border-radius: 7px;
        }
        QSlider::handle:horizontal:hover {
            background: #6a6a6a;
        }
        QTextEdit {
            background-color: #3c3c3c;
            border: 1px solid #555555;
            border-radius: 3px;
            color: #ffffff;
        }
        QListWidget {
            background-color: #3c3c3c;
            border: 1px solid #555555;
            border-radius: 3px;
            color: #ffffff;
            outline: none;
        }
        QListWidget::item:selected {
            background-color: #4a6e8a;
        }
        QListWidget::item:hover {
            background-color: #3a5a6a;
        }
        QToolBar {
            background-color: #3c3c3c;
            border: none;
            border-bottom: 1px solid #555555;
            spacing: 5px;
            padding: 3px;
        }
        QToolBar QToolButton {
            background-color: transparent;
            border: none;
            padding: 5px;
        }
        QToolBar QToolButton:hover {
            background-color: #4a4a4a;
            border-radius: 3px;
        }
        QStatusBar {
            background-color: #3c3c3c;
            color: #ffffff;
            border-top: 1px solid #555555;
        }
        QCheckBox {
            color: #ffffff;
            spacing: 5px;
        }
        QCheckBox::indicator {
            width: 16px;
            height: 16px;
            border-radius: 3px;
            background-color: #3c3c3c;
            border: 1px solid #555555;
        }
        QCheckBox::indicator:checked {
            background-color: #4a6e8a;
            border-color: #6a8eaa;
        }
        QCheckBox::indicator:hover {
            border-color: #777777;
        }
        QScrollBar:vertical {
            background-color: #2b2b2b;
            width: 12px;
            border-radius: 6px;
        }
        QScrollBar::handle:vertical {
            background-color: #5a5a5a;
            border-radius: 6px;
            min-height: 20px;
        }
        QScrollBar::handle:vertical:hover {
            background-color: #6a6a6a;
        }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            border: none;
            background: none;
        }
        QScrollBar:horizontal {
            background-color: #2b2b2b;
            height: 12px;
            border-radius: 6px;
        }
        QScrollBar::handle:horizontal {
            background-color: #5a5a5a;
            border-radius: 6px;
            min-width: 20px;
        }
        QScrollBar::handle:horizontal:hover {
            background-color: #6a6a6a;
        }
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
            border: none;
            background: none;
        }
        QSplitter::handle {
            background-color: #555555;
        }
        QSplitter::handle:hover {
            background-color: #777777;
        }
        QMenuBar {
            background-color: #3c3c3c;
            color: #ffffff;
            border-bottom: 1px solid #555555;
        }
        QMenuBar::item:selected {
            background-color: #4a4a4a;
        }
        QMenu {
            background-color: #3c3c3c;
            color: #ffffff;
            border: 1px solid #555555;
        }
        QMenu::item:selected {
            background-color: #4a6e8a;
        }
        QGroupBox {
            border: 1px solid #555555;
            border-radius: 5px;
            margin-top: 10px;
            padding-top: 10px;
            font-weight: bold;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px 0 5px;
        }
    """
    
    # Светлая тема (Light Theme)
    LIGHT_THEME = """
        QMainWindow {
            background-color: #f0f0f0;
        }
        QWidget {
            background-color: #f0f0f0;
            color: #2b2b2b;
            font-family: 'Segoe UI', 'Arial', sans-serif;
            font-size: 10pt;
        }
        QLabel {
            color: #2b2b2b;
            background-color: transparent;
        }
        QPushButton {
            background-color: #ffffff;
            border: 1px solid #cccccc;
            border-radius: 3px;
            padding: 5px 10px;
            color: #2b2b2b;
        }
        QPushButton:hover {
            background-color: #e0e0e0;
            border-color: #aaaaaa;
        }
        QPushButton:pressed {
            background-color: #d0d0d0;
        }
        QPushButton:disabled {
            background-color: #f0f0f0;
            color: #999999;
            border-color: #dddddd;
        }
        QComboBox {
            background-color: #ffffff;
            border: 1px solid #cccccc;
            border-radius: 3px;
            padding: 3px;
            color: #2b2b2b;
        }
        QComboBox:hover {
            border-color: #aaaaaa;
        }
        QComboBox::drop-down {
            border: none;
        }
        QComboBox::down-arrow {
            image: none;
            border-left: 5px solid transparent;
            border-right: 5px solid transparent;
            border-top: 5px solid #2b2b2b;
            margin-right: 5px;
        }
        QSlider::groove:horizontal {
            border: 1px solid #cccccc;
            height: 6px;
            background: #e0e0e0;
            border-radius: 3px;
        }
        QSlider::handle:horizontal {
            background: #aaaaaa;
            border: 1px solid #888888;
            width: 14px;
            height: 14px;
            margin: -5px 0;
            border-radius: 7px;
        }
        QSlider::handle:horizontal:hover {
            background: #999999;
        }
        QTextEdit {
            background-color: #ffffff;
            border: 1px solid #cccccc;
            border-radius: 3px;
            color: #2b2b2b;
        }
        QListWidget {
            background-color: #ffffff;
            border: 1px solid #cccccc;
            border-radius: 3px;
            color: #2b2b2b;
            outline: none;
        }
        QListWidget::item:selected {
            background-color: #4a6e8a;
            color: #ffffff;
        }
        QListWidget::item:hover {
            background-color: #e0e8f0;
        }
        QToolBar {
            background-color: #ffffff;
            border: none;
            border-bottom: 1px solid #cccccc;
            spacing: 5px;
            padding: 3px;
        }
        QToolBar QToolButton {
            background-color: transparent;
            border: none;
            padding: 5px;
        }
        QToolBar QToolButton:hover {
            background-color: #e0e0e0;
            border-radius: 3px;
        }
        QStatusBar {
            background-color: #ffffff;
            color: #2b2b2b;
            border-top: 1px solid #cccccc;
        }
        QCheckBox {
            color: #2b2b2b;
            spacing: 5px;
        }
        QCheckBox::indicator {
            width: 16px;
            height: 16px;
            border-radius: 3px;
            background-color: #ffffff;
            border: 1px solid #cccccc;
        }
        QCheckBox::indicator:checked {
            background-color: #4a6e8a;
            border-color: #6a8eaa;
        }
        QCheckBox::indicator:hover {
            border-color: #aaaaaa;
        }
        QScrollBar:vertical {
            background-color: #f0f0f0;
            width: 12px;
            border-radius: 6px;
        }
        QScrollBar::handle:vertical {
            background-color: #c0c0c0;
            border-radius: 6px;
            min-height: 20px;
        }
        QScrollBar::handle:vertical:hover {
            background-color: #a0a0a0;
        }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            border: none;
            background: none;
        }
        QScrollBar:horizontal {
            background-color: #f0f0f0;
            height: 12px;
            border-radius: 6px;
        }
        QScrollBar::handle:horizontal {
            background-color: #c0c0c0;
            border-radius: 6px;
            min-width: 20px;
        }
        QScrollBar::handle:horizontal:hover {
            background-color: #a0a0a0;
        }
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
            border: none;
            background: none;
        }
        QSplitter::handle {
            background-color: #cccccc;
        }
        QSplitter::handle:hover {
            background-color: #aaaaaa;
        }
        QMenuBar {
            background-color: #ffffff;
            color: #2b2b2b;
            border-bottom: 1px solid #cccccc;
        }
        QMenuBar::item:selected {
            background-color: #e0e0e0;
        }
        QMenu {
            background-color: #ffffff;
            color: #2b2b2b;
            border: 1px solid #cccccc;
        }
        QMenu::item:selected {
            background-color: #4a6e8a;
            color: #ffffff;
        }
        QGroupBox {
            border: 1px solid #cccccc;
            border-radius: 5px;
            margin-top: 10px;
            padding-top: 10px;
            font-weight: bold;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px 0 5px;
        }
    """
    
    @staticmethod
    def apply_theme(app: QApplication, theme_name: str = "dark"):
        """Применить тему к приложению"""
        if theme_name == "dark":
            app.setStyleSheet(ThemeManager.DARK_THEME)
        elif theme_name == "light":
            app.setStyleSheet(ThemeManager.LIGHT_THEME)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Система обнаружения и отслеживания объектов")
        self.setGeometry(100, 100, 1200, 800)
        
        self.current_plugin: Optional[BaseDetectionPlugin] = None
        self.plugins = []
        self.video_thread = None
        self.is_processing_video = False
        
        # Переменные для видео
        self.video_file_path = None
        self.video_source_type = None
        self.video_is_loaded = False  # Флаг, загружено ли видео
        
        # Переменные для навигации по изображениям
        self.current_image_list = []
        self.current_image_index = 0
        self.current_image = None
        self.current_folder = None
        
        # Текущая тема
        self.current_theme = "dark"
        
        self.init_ui()
        self.load_plugins()
        
    def init_ui(self):
        """Инициализация пользовательского интерфейса"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Создаем меню
        self.create_menu()
        
        # Панель инструментов
        toolbar = self.create_toolbar()
        main_layout.addWidget(toolbar)
        
        # Виджеты управления
        self.image_nav_widget = ImageNavigationWidget()
        self.image_nav_widget.next_clicked.connect(self.next_image)
        self.image_nav_widget.prev_clicked.connect(self.prev_image)
        # Подключаем кнопку открытия
        self.image_nav_widget.open_btn.clicked.disconnect()
        self.image_nav_widget.open_btn.clicked.connect(self.open_images)
        self.image_nav_widget.setVisible(False)
        main_layout.addWidget(self.image_nav_widget)
        
        self.video_control_widget = VideoControlWidget()
        self.video_control_widget.play_pause_clicked.connect(self.toggle_video_play_pause)
        self.video_control_widget.stop_clicked.connect(self.reset_video)
        self.video_control_widget.tracking_toggled.connect(self.on_tracking_toggled)
        self.video_control_widget.tracker_type_changed.connect(self.on_tracker_type_changed)
        self.video_control_widget.setVisible(False)
        main_layout.addWidget(self.video_control_widget)
        
        # Основная область
        splitter = QSplitter(Qt.Horizontal)
        
        control_panel = self.create_control_panel()
        splitter.addWidget(control_panel)
        
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(640, 640)
        self.image_label.setStyleSheet("border: 1px solid gray; background-color: #2b2b2b;")
        self.image_label.setScaledContents(False)
        splitter.addWidget(self.image_label)
        
        splitter.setSizes([300, 900])
        main_layout.addWidget(splitter)
        
        # Статус бар
        self.status_label = QLabel("Готов")
        self.statusBar().addWidget(self.status_label)
        
    def create_menu(self):
        """Создание главного меню"""
        menubar = self.menuBar()
        
        # Меню "Файл"
        file_menu = menubar.addMenu("Файл")
        
        open_action = QAction("Открыть изображения...", self)
        open_action.triggered.connect(self.open_images)
        file_menu.addAction(open_action)
        
        open_video_action = QAction("Открыть видео...", self)
        open_video_action.triggered.connect(self.open_video)
        file_menu.addAction(open_video_action)
        
        open_camera_action = QAction("Веб-камера", self)
        open_camera_action.triggered.connect(self.start_camera)
        file_menu.addAction(open_camera_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Выход", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Меню "Вид"
        view_menu = menubar.addMenu("Вид")
        
        # Действие для переключения темы
        self.theme_action = QAction("Темная тема", self)
        self.theme_action.setCheckable(True)
        self.theme_action.setChecked(True)
        self.theme_action.triggered.connect(self.toggle_theme)
        view_menu.addAction(self.theme_action)
        
        view_menu.addSeparator()
        
        # Действие для полноэкранного режима
        fullscreen_action = QAction("Полноэкранный режим (F11)", self)
        fullscreen_action.setShortcut(QKeySequence("F11"))
        fullscreen_action.triggered.connect(self.toggle_fullscreen)
        view_menu.addAction(fullscreen_action)
        
        # Меню "Справка"
        help_menu = menubar.addMenu("Справка")
        
        about_action = QAction("О программе", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def find_images_in_folder(self, file_path: str) -> List[str]:
        """
        Находит все изображения в папке, содержащей указанный файл
        Возвращает отсортированный список путей к изображениям
        """
        # Нормализуем путь для корректной работы на разных ОС
        file_path = os.path.normpath(file_path)
        folder_path = os.path.dirname(file_path)
        
        # Поддерживаемые расширения изображений
        images_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')
        image_files = []
        
        # Сканируем папку
        try:
            for file in sorted(os.listdir(folder_path)):
                if file.lower().endswith(images_extensions):
                    # Нормализуем путь для каждого файла
                    full_path = os.path.normpath(os.path.join(folder_path, file))
                    image_files.append(full_path)
        except Exception as e:
            print(f"Ошибка при сканировании папки {folder_path}: {e}")
            return []
        
        return image_files

    def open_images(self):
        """Открытие изображений (одиночное или из папки)"""
        # Диалог выбора файла
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Выбрать изображение", "", 
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.webp);;All files (*.*)"
        )
        
        if not file_path:
            return
        
        # Нормализуем путь к выбранному файлу
        file_path = os.path.normpath(file_path)
        print(f"Выбран файл: {file_path}")
        
        self.stop_processing()
        
        # Ищем все изображения в папке
        all_images = self.find_images_in_folder(file_path)
        
        if not all_images:
            QMessageBox.warning(self, "Ошибка", "Не удалось найти изображения в папке")
            return
        
        print(f"Найдено изображений в папке: {len(all_images)}")
        print(f"Первые 5 изображений: {all_images[:5]}")
        
        # Устанавливаем список изображений
        self.current_image_list = all_images
        
        # Находим индекс выбранного изображения
        try:
            # Пытаемся найти выбранный файл в списке
            self.current_image_index = self.current_image_list.index(file_path)
            print(f"Найден индекс {self.current_image_index} для файла {os.path.basename(file_path)}")
        except ValueError:
            # Если файл не найден, пробуем сравнить только имена файлов
            print(f"Точное совпадение не найдено, пробуем сравнить по имени файла...")
            found = False
            file_basename = os.path.basename(file_path).lower()
            
            for i, img_path in enumerate(self.current_image_list):
                if os.path.basename(img_path).lower() == file_basename:
                    self.current_image_index = i
                    found = True
                    print(f"Найдено совпадение по имени: индекс {i}")
                    break
            
            if not found:
                # Если все еще не найден, показываем первое изображение
                self.current_image_index = 0
                print(f"Предупреждение: файл {file_path} не найден в списке изображений. Показываем первое изображение.")
                QMessageBox.information(
                    self, 
                    "Информация", 
                    f"Выбранный файл '{os.path.basename(file_path)}' не найден в списке изображений.\n"
                    f"Будет показано первое изображение в папке."
                )
        
        # Загружаем выбранное изображение
        self.load_current_image()
        
        # Показываем информацию о количестве найденных изображений
        folder_name = os.path.basename(os.path.dirname(file_path))
        file_name = os.path.basename(file_path)
        
        if len(all_images) > 1:
            self.status_label.setText(
                f"Загружено {len(self.current_image_list)} изображений из папки '{folder_name}'. "
                f"Текущее: {file_name} ({self.current_image_index + 1}/{len(all_images)})"
            )
        else:
            self.status_label.setText(f"Загружено изображение: {file_name} из папки '{folder_name}'")
    
    def load_current_image(self):
        """Загрузка текущего изображения"""
        if not self.current_image_list:
            return
            
        file_path = self.current_image_list[self.current_image_index]
        original_image = cv2.imread(file_path)
        
        if original_image is not None:
            self.current_image = original_image
            resized_for_detection, (x_offset, y_offset, new_w, new_h, scale) = self.resize_with_padding(original_image, (640, 640))
            
            detections = []
            if self.current_plugin and self.current_plugin.is_loaded():
                raw_detections = self.current_plugin.detect(resized_for_detection)
                
                h_orig, w_orig = original_image.shape[:2]
                
                for det in raw_detections:
                    x, y, w, h = det.bbox
                    if (x >= x_offset and x <= x_offset + new_w and 
                        y >= y_offset and y <= y_offset + new_h):
                        x_orig = int((x - x_offset) / scale)
                        y_orig = int((y - y_offset) / scale)
                        w_orig_box = int(w / scale)
                        h_orig_box = int(h / scale)
                        
                        x_orig = max(0, min(x_orig, w_orig))
                        y_orig = max(0, min(y_orig, h_orig))
                        w_orig_box = min(w_orig_box, w_orig - x_orig)
                        h_orig_box = min(h_orig_box, h_orig - y_orig)
                        
                        if w_orig_box > 0 and h_orig_box > 0:
                            detections.append(DetectionResult(
                                bbox=(x_orig, y_orig, w_orig_box, h_orig_box),
                                label=det.label,
                                confidence=det.confidence,
                                track_id=-1
                            ))
            
            self.display_image(original_image, detections)
            
            # Показываем виджет навигации
            self.image_nav_widget.setVisible(True)
            self.video_control_widget.setVisible(False)
            
            # Обновляем информацию о навигации
            total_images = len(self.current_image_list)
            self.image_nav_widget.update_info(self.current_image_index, total_images)
            
            # Формируем информативное сообщение
            file_name = os.path.basename(file_path)
            folder_name = os.path.basename(os.path.dirname(file_path))
            size_info = f"({original_image.shape[1]}x{original_image.shape[0]})"
            
            if total_images > 1:
                self.status_label.setText(
                    f"[{self.current_image_index + 1}/{total_images}] {file_name} {size_info} | Папка: {folder_name}"
                )
            else:
                self.status_label.setText(f"Изображение: {file_name} {size_info} | Папка: {folder_name}")
        else:
            QMessageBox.warning(self, "Ошибка", f"Не удалось загрузить изображение: {file_path}")
            # Если изображение не загрузилось, удаляем его из списка
            self.current_image_list.pop(self.current_image_index)
            if self.current_image_list:
                # Переходим к следующему изображению
                if self.current_image_index >= len(self.current_image_list):
                    self.current_image_index = len(self.current_image_list) - 1
                self.load_current_image()
            else:
                # Нет больше изображений
                self.image_nav_widget.setVisible(False)
                self.status_label.setText("Нет доступных изображений")
    
    def next_image(self):
        """Следующее изображение"""
        if self.current_image_index < len(self.current_image_list) - 1:
            self.current_image_index += 1
            self.load_current_image()
            # Обновляем информацию о позиции в статус-баре
            file_name = os.path.basename(self.current_image_list[self.current_image_index])
            self.status_label.setText(
                f"[{self.current_image_index + 1}/{len(self.current_image_list)}] {file_name}"
            )
            
    def prev_image(self):
        """Предыдущее изображение"""
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.load_current_image()
            # Обновляем информацию о позиции в статус-баре
            file_name = os.path.basename(self.current_image_list[self.current_image_index])
            self.status_label.setText(
                f"[{self.current_image_index + 1}/{len(self.current_image_list)}] {file_name}"
            )
    
    def toggle_theme(self):
        """Переключение между темной и светлой темой"""
        if self.theme_action.isChecked():
            self.theme_action.setText("Темная тема")
            self.current_theme = "dark"
            ThemeManager.apply_theme(QApplication.instance(), "dark")
        else:
            self.theme_action.setText("Светлая тема")
            self.current_theme = "light"
            ThemeManager.apply_theme(QApplication.instance(), "light")
            
        # Обновляем стиль QLabel для изображения (должен оставаться темным для контраста)
        if self.current_theme == "dark":
            self.image_label.setStyleSheet("border: 1px solid gray; background-color: #2b2b2b;")
        else:
            self.image_label.setStyleSheet("border: 1px solid gray; background-color: #f0f0f0;")
            
        self.status_label.setText(f"Переключено на {self.current_theme} тему")
    
    def toggle_fullscreen(self):
        """Переключение полноэкранного режима"""
        if self.isFullScreen():
            self.showNormal()
            self.status_label.setText("Полноэкранный режим выключен")
        else:
            self.showFullScreen()
            self.status_label.setText("Полноэкранный режим включен")
    
    def show_about(self):
        """Показать информацию о программе"""
        about_text = """
        <h2>Система обнаружения и отслеживания объектов</h2>
        <p>Версия: 2.0</p>
        <p>Программа предназначена для обнаружения и отслеживания объектов 
        в реальном времени с использованием плагинной архитектуры.</p>
        
        <h3>Возможности:</h3>
        <ul>
        <li>Обнаружение объектов на изображениях, видео и с веб-камеры</li>
        <li>Автоматическое определение всех изображений в папке</li>
        <li>Навигация между изображениями в папке</li>
        <li>Отслеживание объектов с уникальными ID</li>
        <li>Поддержка плагинов (YOLO, Dummy и др.)</li>
        <li>Сохранение пропорций изображений</li>
        <li>Отображение FPS в реальном времени</li>
        <li>Темная и светлая темы оформления</li>
        </ul>
        
        <h3>Управление:</h3>
        <ul>
        <li>F11 - полноэкранный режим</li>
        <li>Esc - выход из полноэкранного режима</li>
        </ul>
        
        <p>© 2026</p>
        """
        
        QMessageBox.about(self, "О программе", about_text)
        
    def create_toolbar(self):
        """Создание панели инструментов"""
        toolbar = QToolBar()
        toolbar.setMovable(False)
        
        # Объединенная кнопка для открытия изображений
        open_images_action = QAction("📷 Открыть изображения", self)
        open_images_action.triggered.connect(self.open_images)
        toolbar.addAction(open_images_action)
        
        open_video_action = QAction("🎬 Открыть видео", self)
        open_video_action.triggered.connect(self.open_video)
        toolbar.addAction(open_video_action)
        
        camera_action = QAction("📹 Веб-камера", self)
        camera_action.triggered.connect(self.start_camera)
        toolbar.addAction(camera_action)
        
        toolbar.addSeparator()
        
        stop_action = QAction("⏹ Стоп", self)
        stop_action.triggered.connect(self.stop_processing)
        toolbar.addAction(stop_action)
        
        return toolbar
        
    def create_control_panel(self):
        """Создание панели управления"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        layout.addWidget(QLabel("Выберите плагин обнаружения:"))
        self.plugin_combo = QComboBox()
        self.plugin_combo.currentIndexChanged.connect(self.on_plugin_changed)
        layout.addWidget(self.plugin_combo)
        
        self.load_model_btn = QPushButton("Загрузить модель")
        self.load_model_btn.clicked.connect(self.load_model_for_plugin)
        layout.addWidget(self.load_model_btn)
        
        layout.addWidget(QLabel("Настройки:"))
        
        layout.addWidget(QLabel("Порог уверенности:"))
        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setRange(0, 100)
        self.confidence_slider.setValue(50)
        self.confidence_slider.valueChanged.connect(self.update_confidence)
        layout.addWidget(self.confidence_slider)
        self.confidence_label = QLabel("0.50")
        layout.addWidget(self.confidence_label)
        
        layout.addWidget(QLabel("Информация о плагине:"))
        self.plugin_info = QTextEdit()
        self.plugin_info.setReadOnly(True)
        self.plugin_info.setMaximumHeight(150)
        layout.addWidget(self.plugin_info)
        
        layout.addWidget(QLabel("Доступные классы:"))
        self.classes_list = QListWidget()
        self.classes_list.setMaximumHeight(200)
        layout.addWidget(self.classes_list)
        
        layout.addStretch()
        
        return panel
    
    def on_tracking_toggled(self, enabled: bool):
        """Включение/выключение трекинга"""
        if self.video_thread:
            tracker_type = "simple" if self.video_control_widget.tracker_type_combo.currentText() == "Simple (IOU)" else "advanced"
            self.video_thread.enable_tracking(enabled, tracker_type)
            if enabled:
                self.status_label.setText("Отслеживание объектов включено")
            else:
                self.status_label.setText("Отслеживание объектов выключено")
    
    def on_tracker_type_changed(self, tracker_type: str):
        """Смена типа трекера"""
        if self.video_thread and self.video_control_widget.use_tracking:
            self.video_thread.enable_tracking(True, tracker_type)
            self.status_label.setText(f"Тип трекера: {tracker_type}")
    
    def load_plugins(self):
        """Загрузка плагинов"""
        try:
            from plugins.dummy_plugin import DummyPlugin
            dummy_plugin = DummyPlugin()
            self.plugins.append(dummy_plugin)
            self.plugin_combo.addItem(dummy_plugin.get_name())
            print(f"Загружен плагин: {dummy_plugin.get_name()}")
        except Exception as e:
            print(f"Ошибка загрузки DummyPlugin: {e}")
        
        try:
            from plugins.yolo_plugin import YOLOPlugin
            yolo_plugin = YOLOPlugin()
            self.plugins.append(yolo_plugin)
            self.plugin_combo.addItem(yolo_plugin.get_name())
            print(f"Загружен плагин: {yolo_plugin.get_name()}")
        except Exception as e:
            print(f"Ошибка загрузки YOLOPlugin: {e}")
        
        if self.plugins:
            self.current_plugin = self.plugins[0]
            self.update_plugin_info()
    
    def on_plugin_changed(self, index):
        if 0 <= index < len(self.plugins):
            self.current_plugin = self.plugins[index]
            self.update_plugin_info()
            
            self.classes_list.clear()
            if hasattr(self.current_plugin, 'get_available_classes'):
                classes = self.current_plugin.get_available_classes()
                for cls in classes[:50]:
                    self.classes_list.addItem(cls)
    
    def update_plugin_info(self):
        if self.current_plugin:
            info = f"Имя: {self.current_plugin.get_name()}\n"
            info += f"Загружен: {'Да' if self.current_plugin.is_loaded() else 'Нет'}\n"
            info += f"Размер входа: {self.current_plugin.get_required_size()}"
            self.plugin_info.setText(info)
    
    def load_model_for_plugin(self):
        if not self.current_plugin:
            QMessageBox.warning(self, "Ошибка", "Нет выбранного плагина")
            return
        
        if "Dummy" in self.current_plugin.get_name():
            self.current_plugin.load_model()
            self.update_plugin_info()
            self.status_label.setText("Модель-заглушка загружена")
            return
        
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Выберите файл модели", "", 
            "Model files (*.pt *.onnx *.weights *.pth);;All files (*.*)"
        )
        
        if file_path:
            self.status_label.setText("Загрузка модели...")
            QApplication.processEvents()
            
            success = self.current_plugin.load_model(file_path)
            
            if success:
                self.update_plugin_info()
                self.status_label.setText(f"Модель загружена из {os.path.basename(file_path)}")
                
                if hasattr(self.current_plugin, 'set_confidence_threshold'):
                    confidence = self.confidence_slider.value() / 100.0
                    self.current_plugin.set_confidence_threshold(confidence)
            else:
                QMessageBox.warning(self, "Ошибка", "Не удалось загрузить модель")
                self.status_label.setText("Ошибка загрузки модели")
    
    def update_confidence(self, value):
        confidence = value / 100.0
        self.confidence_label.setText(f"{confidence:.2f}")
        
        if hasattr(self.current_plugin, 'set_confidence_threshold'):
            self.current_plugin.set_confidence_threshold(confidence)
    
    def resize_with_padding(self, image: np.ndarray, target_size: Tuple[int, int] = (640, 640)):
        h, w = image.shape[:2]
        target_w, target_h = target_size
        
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        resized = cv2.resize(image, (new_w, new_h))
        
        if len(image.shape) == 3:
            padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        else:
            padded = np.zeros((target_h, target_w), dtype=np.uint8)
        
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2
        padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
        
        return padded, (x_offset, y_offset, new_w, new_h, scale)

    def draw_detections(self, image: np.ndarray, detections: List[DetectionResult], 
                       padding_info: Tuple = None) -> np.ndarray:
        """Отрисовка результатов с ID треков"""
        img_copy = image.copy()
        
        # Разные цвета для разных ID
        colors = {}
        
        for det in detections:
            x, y, w, h = det.bbox
            
            if padding_info:
                x_offset, y_offset, new_w, new_h, scale = padding_info
                x = int(x * scale + x_offset)
                y = int(y * scale + y_offset)
                w = int(w * scale)
                h = int(h * scale)
            
            x = max(0, min(x, img_copy.shape[1] - 1))
            y = max(0, min(y, img_copy.shape[0] - 1))
            w = min(w, img_copy.shape[1] - x)
            h = min(h, img_copy.shape[0] - y)
            
            # Генерируем цвет для ID
            if det.track_id >= 0:
                if det.track_id not in colors:
                    # Генерация уникального цвета
                    np.random.seed(det.track_id)
                    colors[det.track_id] = tuple(np.random.randint(0, 255, 3).tolist())
                color = colors[det.track_id]
            else:
                color = (0, 255, 0)  # Зеленый для объектов без ID
            
            # Рисуем прямоугольник
            cv2.rectangle(img_copy, (x, y), (x + w, y + h), color, 2)
            
            # Подпись с ID
            if det.track_id >= 0:
                label = f"ID:{det.track_id} {det.label} ({det.confidence:.2f})"
            else:
                label = f"{det.label} ({det.confidence:.2f})"
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            (label_w, label_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
            
            label_y = y - 5
            if label_y - label_h < 0:
                label_y = y + h + label_h + 5
            
            cv2.rectangle(img_copy, (x, label_y - label_h - 5), 
                         (x + label_w, label_y), color, -1)
            cv2.putText(img_copy, label, (x, label_y - 5), 
                       font, font_scale, (0, 0, 0), thickness)
            
        return img_copy

    def display_image(self, image: np.ndarray, detections: List[DetectionResult] = None,
                     original_size: Tuple[int, int] = None, fps: float = None):
        padded_image, padding_info = self.resize_with_padding(image, (640, 640))
        
        if detections:
            padded_image = self.draw_detections(padded_image, detections, padding_info)
        
        if fps is not None:
            fps_text = f"FPS: {fps:.1f}"
            cv2.putText(padded_image, fps_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if len(padded_image.shape) == 3:
            padded_image = cv2.cvtColor(padded_image, cv2.COLOR_BGR2RGB)
        
        height, width = padded_image.shape[:2]
        
        label_width = self.image_label.width() - 10
        label_height = self.image_label.height() - 10
        
        if label_width > 0 and label_height > 0:
            scale_w = label_width / width
            scale_h = label_height / height
            scale = min(scale_w, scale_h, 1.0)
            
            if scale < 1.0:
                new_width = int(width * scale)
                new_height = int(height * scale)
                padded_image = cv2.resize(padded_image, (new_width, new_height))
                width, height = new_width, new_height
        
        bytes_per_line = 3 * width
        q_img = QImage(padded_image.data, width, height, 
                      bytes_per_line, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(q_img))
    
    def open_video(self):
        """Открытие видеофайла (без автоматического запуска)"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Выбрать видео", "", "Video files (*.mp4 *.avi *.mov *.mkv)"
        )
        if file_path:
            # Сохраняем путь к видео
            self.video_file_path = file_path
            self.video_source_type = 'video'
            self.video_is_loaded = True
            
            # Останавливаем предыдущий поток, если он есть
            if self.video_thread:
                self.video_thread.stop()
                self.video_thread = None
            
            self.is_processing_video = False
            
            # Показываем информацию о загруженном видео
            self.status_label.setText(f"Видео загружено: {os.path.basename(file_path)}. Нажмите ▶ Старт для воспроизведения")
            
            # Показываем виджет управления видео
            self.image_nav_widget.setVisible(False)
            self.video_control_widget.setVisible(True)
            self.video_control_widget.set_playing(False)  # Состояние "Старт"
            self.video_control_widget.reset()
            
            # Очищаем область отображения
            self.image_label.clear()
            self.image_label.setText("📹 Видео загружено\n\nНажмите ▶ Старт для начала воспроизведения")

    def start_camera(self):
        """Подготовка веб-камеры (без автоматического запуска)"""
        self.video_source_type = 'camera'
        self.video_file_path = 0
        self.video_is_loaded = True
        
        # Останавливаем предыдущий поток
        if self.video_thread:
            self.video_thread.stop()
            self.video_thread = None
        
        self.is_processing_video = False
        
        # Показываем информацию
        self.status_label.setText("Веб-камера готова. Нажмите ▶ Старт для запуска")
        
        # Показываем виджет управления видео
        self.image_nav_widget.setVisible(False)
        self.video_control_widget.setVisible(True)
        self.video_control_widget.set_playing(False)
        self.video_control_widget.reset()
        
        # Очищаем область отображения
        self.image_label.clear()
        self.image_label.setText("📹 Веб-камера готова\n\nНажмите ▶ Старт для запуска")

    def start_video_processing(self):
        """Запуск обработки видео/камеры (вызывается только по кнопке Старт)"""
        if not self.current_plugin or not self.current_plugin.is_loaded():
            QMessageBox.warning(self, "Ошибка", "Сначала загрузите модель")
            return
        
        if not self.video_is_loaded:
            QMessageBox.warning(self, "Ошибка", "Сначала откройте видео или веб-камеру")
            return
        
        # Останавливаем предыдущий поток, если он есть
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()
            self.video_thread = None
        
        # Создаем новый поток
        self.video_thread = VideoThread()
        self.video_thread.set_source(self.video_source_type, self.video_file_path)
        self.video_thread.set_plugin(self.current_plugin)
        
        # Включаем трекинг
        tracker_type = "simple" if self.video_control_widget.tracker_type_combo.currentText() == "Simple (IOU)" else "advanced"
        self.video_thread.enable_tracking(True, tracker_type)
        
        # Подключаем сигналы
        self.video_thread.frame_processed.connect(self.on_video_frame)
        self.video_thread.error_occurred.connect(self.on_video_error)
        
        # Запускаем поток
        self.video_thread.start()
        
        self.is_processing_video = True
        
        # ВАЖНО: Меняем состояние кнопки на "Пауза"
        self.video_control_widget.set_playing(True)
        
        source_name = "видео" if self.video_source_type == 'video' else "веб-камеры"
        self.status_label.setText(f"Запущена обработка {source_name} с отслеживанием объектов")

    def resume_video(self):
        """Возобновить воспроизведение видео"""
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.resume()
            self.status_label.setText("Воспроизведение возобновлено")
            # ВАЖНО: Меняем состояние кнопки на "Пауза"
            self.video_control_widget.set_playing(True)

    def pause_video(self):
        """Поставить видео на паузу"""
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.pause()
            self.status_label.setText("Воспроизведение на паузе")
            # ВАЖНО: Меняем состояние кнопки на "Старт"
            self.video_control_widget.set_playing(False)

    def reset_video(self):
        """Сброс видео (остановка и подготовка к перезапуску)"""
        if self.video_thread:
            # Останавливаем текущий поток
            self.video_thread.stop()
            self.video_thread = None
        
        self.is_processing_video = False
        
        # Сбрасываем состояние кнопки на "Старт"
        self.video_control_widget.set_playing(False)
        
        # Показываем сообщение в зависимости от типа источника
        if self.video_source_type == 'video' and self.video_file_path:
            self.status_label.setText(f"Видео сброшено. Нажмите ▶ Старт для воспроизведения с начала")
            self.image_label.clear()
            self.image_label.setText("📹 Видео сброшено\n\nНажмите ▶ Старт для воспроизведения с начала")
        elif self.video_source_type == 'camera':
            self.status_label.setText(f"Веб-камера готова. Нажмите ▶ Старт для запуска")
            self.image_label.clear()
            self.image_label.setText("📹 Веб-камера готова\n\nНажмите ▶ Старт для запуска")

    def toggle_video_play_pause(self):
        """Переключение паузы/воспроизведения видео"""
        if not self.video_is_loaded:
            # Если ничего не загружено, предлагаем открыть видео
            self.open_video()
            return
        
        if not self.video_control_widget.is_playing_state():
            # Кнопка "Старт" - запускаем воспроизведение
            if not self.is_processing_video:
                # Если видео не запущено, запускаем
                self.start_video_processing()
            else:
                # Если видео на паузе, возобновляем
                self.resume_video()
        else:
            # Кнопка "Пауза" - ставим на паузу
            if self.is_processing_video:
                self.pause_video()

    def on_video_frame(self, frame: np.ndarray, detections: List[DetectionResult], fps: float):
        """Обработка кадра из видео"""
        confidence_threshold = self.confidence_slider.value() / 100.0
        filtered_detections = [d for d in detections if d.confidence >= confidence_threshold]
        self.display_image(frame, filtered_detections, fps=fps)
        self.video_control_widget.update_fps(fps)

    def on_video_error(self, error_msg: str):
        """Обработка ошибки видео"""
        if error_msg == "Видео закончилось":
            QMessageBox.information(self, "Информация", "Воспроизведение видео завершено")
            # Сбрасываем состояние для возможности повторного запуска
            self.reset_video()
        else:
            QMessageBox.warning(self, "Ошибка", error_msg)
            self.stop_processing()

    def stop_processing(self):
        """Остановка обработки видео/камеры"""
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()
            self.video_thread = None
        
        self.is_processing_video = False
        
        # Сбрасываем состояние кнопки на "Старт"
        if self.video_control_widget.isVisible():
            self.video_control_widget.set_playing(False)
        
        self.status_label.setText("Обработка остановлена")