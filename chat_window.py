import sys
import cv2
import threading
import queue
import tempfile
import os
import time
import imageio
from PyQt6 import QtWidgets, QtGui, QtCore
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from frontend import Ui_MainWindow  # your .ui -> py converted file


class VideoThread(QThread):
    frame_ready = pyqtSignal(QtGui.QImage)

    def __init__(self, source, buffer_seconds=15, analyze_interval=10, parent=None):
        super().__init__(parent)
        self.source = source
        self.buffer_seconds = buffer_seconds
        self.analyze_interval = analyze_interval
        self.running = False

    def run(self):
        self.running = True
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            print(f"Failed to open source: {self.source}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 30.0
        frame_delay = 1.0 / fps

        last_time = time.time()
        while self.running:
            ok, frame = cap.read()
            if not ok:
                break

            # Convert BGR → RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            self.frame_ready.emit(qimg)

            # Sleep to maintain playback rate
            elapsed = time.time() - last_time
            sleep_time = max(0, frame_delay - elapsed)
            time.sleep(sleep_time)
            last_time = time.time()

        cap.release()

    def stop(self):
        self.running = False
        self.wait()


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.video_thread = None

        # Directly load test.mp4 from the same folder
        video_path = os.path.join(os.path.dirname(__file__), "test.mp4")
        if os.path.exists(video_path):
            self.start_video(video_path)
        else:
            print("❌ test.mp4 not found in current directory.")

    def start_video(self, source):
        if self.video_thread:
            self.video_thread.stop()

        self.video_thread = VideoThread(source)
        self.video_thread.frame_ready.connect(self.update_frame)
        self.video_thread.start()

    @QtCore.pyqtSlot(QtGui.QImage)
    def update_frame(self, image):
        pix = QPixmap.fromImage(image)
        self.video_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        self.video_label.setPixmap(pix)

    def closeEvent(self, event):
        if self.video_thread:
            self.video_thread.stop()
        event.accept()


if __name__ == "__main__":
    import qdarkstyle
    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt6())

    window = MainWindow()
    window.show()

    sys.exit(app.exec())
