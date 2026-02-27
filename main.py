import sys
import time
from pathlib import Path

import mss
import numpy as np
import torch
import cv2
from PyQt6.QtCore import QThread, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)
from ultralytics import YOLO

MODEL_FILE = Path("yolov8n.pt")
DEFAULT_CONFIDENCE = 0.35
TARGET_FPS = 30
IDLE_FPS = 10


class DetectorThread(QThread):
    detections_ready = pyqtSignal(list)
    runtime_error = pyqtSignal(str)

    def __init__(self, model, monitor, class_map):
        super().__init__()
        self.model = model
        self.monitor = monitor
        self.class_map = class_map
        self.active_class_ids = []
        self.confidence = DEFAULT_CONFIDENCE
        self.running = False
        self.target_fps = TARGET_FPS
        self.idle_fps = IDLE_FPS

    def set_active_classes(self, class_ids):
        self.active_class_ids = list(class_ids)

    def set_confidence(self, confidence):
        self.confidence = max(0.01, min(0.99, float(confidence)))

    def stop(self):
        self.running = False

    def sleep_to_cap_rate(self, started_at, idle=False):
        fps = self.idle_fps if idle else self.target_fps
        if fps <= 0:
            return
        target_delta = 1.0 / fps
        remaining = target_delta - (time.perf_counter() - started_at)
        if remaining > 0:
            time.sleep(remaining)

    def run(self):
        self.running = True
        try:
            with mss.mss() as sct:
                while self.running:
                    started_at = time.perf_counter()
                    try:
                        raw_frame = np.array(sct.grab(self.monitor))
                    except Exception as exc:
                        self.runtime_error.emit(f"Falha ao capturar a tela: {exc}")
                        break

                    frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGRA2BGR)
                    active_class_ids = list(self.active_class_ids)
                    if not active_class_ids:
                        self.detections_ready.emit([])
                        self.sleep_to_cap_rate(started_at, idle=True)
                        continue

                    try:
                        result = self.model(
                            frame,
                            conf=self.confidence,
                            classes=active_class_ids,
                            verbose=False,
                        )[0]
                    except Exception as exc:
                        self.runtime_error.emit(f"Falha em interferências do yolo: {exc}")
                        break

                    boxes = []
                    if result.boxes is not None:
                        for box in result.boxes:
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            cls_id = int(box.cls[0].item())
                            conf = float(box.conf[0].item())
                            boxes.append(
                                {
                                    "x1": int(x1),
                                    "y1": int(y1),
                                    "x2": int(x2),
                                    "y2": int(y2),
                                    "label": self.class_map.get(cls_id, str(cls_id)),
                                    "conf": conf,
                                }
                            )

                    self.detections_ready.emit(boxes)
                    self.sleep_to_cap_rate(started_at)
        except Exception as exc:
            self.runtime_error.emit(f"Erro no loop: {exc}")
        finally:
            self.running = False


class OverlayWindow(QWidget):
    def __init__(self, monitor):
        super().__init__()
        self.monitor = monitor
        self.detections = []
        self.setWindowTitle("Detecção com IA - Python")
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setGeometry(
            self.monitor["left"],
            self.monitor["top"],
            self.monitor["width"],
            self.monitor["height"],
        )

    def update_detections(self, detections):
        self.detections = detections
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(QColor(0, 255, 90), 2))
        painter.setFont(QFont("Segoe UI", 11))

        for det in self.detections:
            x1 = det["x1"]
            y1 = det["y1"]
            x2 = det["x2"]
            y2 = det["y2"]
            w = max(1, x2 - x1)
            h = max(1, y2 - y1)
            painter.drawRect(x1, y1, w, h)

            text = f"{det['label']} {det['conf']:.2f}"
            text_y = max(14, y1 - 8)
            text_y = min(self.height() - 4, text_y)
            painter.drawText(x1 + 4, text_y, text)


class ControlPanel(QWidget):
    start_clicked = pyqtSignal()
    stop_clicked = pyqtSignal()
    classes_changed = pyqtSignal(list)
    confidence_changed = pyqtSignal(float)

    def __init__(self, class_map):
        super().__init__()
        self.class_map = class_map
        self.class_ids = sorted(self.class_map.keys())
        self.boxes = {}
        self.setWindowTitle("Detecção como IA")
        self.setFixedSize(420, 680)
        self.build_ui()

    def build_ui(self):
        root = QVBoxLayout()

        title = QLabel("Seleção de classes do modelo:")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        root.addWidget(title)

        quick = QHBoxLayout()
        all_btn = QPushButton("Todos")
        none_btn = QPushButton("Nenhum")
        person_btn = QPushButton("Pessoa")
        animals_btn = QPushButton("Animais")
        vehicles_btn = QPushButton("Veiculos")

        all_btn.clicked.connect(self.select_all)
        none_btn.clicked.connect(self.select_none)
        person_btn.clicked.connect(self.select_people)
        animals_btn.clicked.connect(self.select_animals)
        vehicles_btn.clicked.connect(self.select_vehicles)

        quick.addWidget(all_btn)
        quick.addWidget(none_btn)
        quick.addWidget(person_btn)
        quick.addWidget(animals_btn)
        quick.addWidget(vehicles_btn)
        root.addLayout(quick)

        class_box = QGroupBox("Classes do YOLO:")
        class_layout = QGridLayout()

        row = 0
        col = 0
        for cls_id in self.class_ids:
            cb = QCheckBox(self.class_map[cls_id])
            cb.stateChanged.connect(self.emit_classes)
            self.boxes[cls_id] = cb
            class_layout.addWidget(cb, row, col)
            col += 1
            if col == 2:
                col = 0
                row += 1

        class_box.setLayout(class_layout)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout()
        scroll_layout.addWidget(class_box)
        scroll_content.setLayout(scroll_layout)
        scroll.setWidget(scroll_content)
        root.addWidget(scroll)

        conf_title = QLabel("Confiança da IA: 0.35")
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setMinimum(10)
        slider.setMaximum(90)
        slider.setValue(35)

        def on_slide(value):
            conf = value / 100.0
            conf_title.setText(f"Confiança da IA: {conf:.2f}")
            self.confidence_changed.emit(conf)

        slider.valueChanged.connect(on_slide)
        root.addWidget(conf_title)
        root.addWidget(slider)

        controls = QHBoxLayout()
        start_btn = QPushButton("Iniciar")
        stop_btn = QPushButton("Parar")
        start_btn.clicked.connect(self.start_clicked.emit)
        stop_btn.clicked.connect(self.stop_clicked.emit)
        controls.addWidget(start_btn)
        controls.addWidget(stop_btn)
        root.addLayout(controls)

        info = QLabel("Ao pressionar iniciar, a deteccao inicia automaticamente.")
        info.setWordWrap(True)
        root.addWidget(info)

        self.setLayout(root)
        self.select_people()

    def select_all(self):
        for cb in self.boxes.values():
            cb.blockSignals(True)
            cb.setChecked(True)
            cb.blockSignals(False)
        self.emit_classes()

    def select_none(self):
        for cb in self.boxes.values():
            cb.blockSignals(True)
            cb.setChecked(False)
            cb.blockSignals(False)
        self.emit_classes()

    def select_people(self):
        self.select_by_labels({"person"})

    def select_animals(self):
        self.select_by_labels(
            {
                "bird",
                "cat",
                "dog",
                "horse",
                "sheep",
                "cow",
                "elephant",
                "bear",
                "zebra",
                "giraffe",
            }
        )

    def select_vehicles(self):
        self.select_by_labels(
            {
                "bicycle",
                "car",
                "motorcycle",
                "airplane",
                "bus",
                "train",
                "truck",
                "boat",
            }
        )

    def select_by_labels(self, labels):
        for cls_id, cb in self.boxes.items():
            cb.blockSignals(True)
            cb.setChecked(self.class_map[cls_id] in labels)
            cb.blockSignals(False)
        self.emit_classes()

    def emit_classes(self):
        selected = [cls_id for cls_id, cb in self.boxes.items() if cb.isChecked()]
        self.classes_changed.emit(selected)


class App:
    def __init__(self):
        self.qt = QApplication(sys.argv)
        self.model = self.load_model()
        self.class_map = {int(i): name for i, name in self.model.names.items()}
        self.monitor = self.load_monitor()

        self.overlay = OverlayWindow(self.monitor)
        self.panel = ControlPanel(self.class_map)
        self.detector = DetectorThread(self.model, self.monitor, self.class_map)

        self.panel.classes_changed.connect(self.detector.set_active_classes)
        self.panel.confidence_changed.connect(self.detector.set_confidence)
        self.panel.start_clicked.connect(self.start)
        self.panel.stop_clicked.connect(self.stop)
        self.detector.detections_ready.connect(self.overlay.update_detections)
        self.detector.runtime_error.connect(self.on_detector_error)

    def load_model(self):
        if not MODEL_FILE.exists():
            raise RuntimeError(f"Arquivo do modelo não foi encontrado: {MODEL_FILE.resolve()}")

        try:
            return YOLO(str(MODEL_FILE))
        except Exception as exc:
            raise RuntimeError(f"Erro ao carregar {MODEL_FILE.name}: {exc}") from exc

    def load_monitor(self):
        try:
            with mss.mss() as sct:
                if len(sct.monitors) < 2:
                    raise RuntimeError("Nenhum monitor disponivel para captura de tela.")
                return dict(sct.monitors[1])
        except Exception as exc:
            raise RuntimeError(f"Não foi possível iniciar a captura de tlea: {exc}") from exc

    def on_detector_error(self, message):
        self.stop()
        QMessageBox.critical(self.panel, "Erro no detector", message)

    def start(self):
        if self.detector.isRunning():
            return
        self.overlay.show()
        self.detector.start()

    def stop(self):
        if self.detector.isRunning():
            self.detector.stop()
            self.detector.wait()
        self.overlay.hide()

    def run(self):
        self.panel.show()
        code = self.qt.exec()
        self.stop()
        return code


def main():
    try:
        app = App()
    except Exception as exc:
        if QApplication.instance() is None:
            _ = QApplication(sys.argv)
        QMessageBox.critical(None, "Falha ao iniciar", str(exc))
        return 1

    return app.run()


if __name__ == "__main__":
    sys.exit(main())
