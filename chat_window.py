from collections import deque
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
from typing import Deque, List, Tuple
from dataclasses import dataclass
from PyQt6.QtCore import QThreadPool, QRunnable, QObject
from video_player import AnalyzerThread
from agent.agent_baby_safety import get_baby_safety_answer

@dataclass
class Frame:
    ts: float  # presentation timestamp (seconds)
    img: any   # numpy array (H, W, 3) BGR


class WorkerSignals(QObject):
    frame_ready = pyqtSignal(QtGui.QImage)
    text_ready = pyqtSignal(str)



class RollingBuffer:
    def __init__(self, seconds: int, fps: float):
        self.seconds = max(1, int(seconds))
        self.fps = fps if fps and fps > 0 else 30.0
        self.maxlen = int(self.seconds * self.fps)
        self._dq: Deque[Frame] = deque(maxlen=self.maxlen)
        self._lock = threading.Lock()

    def update_fps(self, fps: float):
        if fps and fps > 0 and abs(fps - self.fps) > 1e-3:
            with self._lock:
                self.fps = fps
                self.maxlen = int(self.seconds * self.fps)
                old = list(self._dq)
                self._dq = deque(old[-self.maxlen:], maxlen=self.maxlen)

    def push(self, frame: Frame):
        with self._lock:
            self._dq.append(frame)

    def snapshot(self) -> Tuple[List[Frame], float]:
        with self._lock:
            frames = list(self._dq)
            fps = self.fps
        return frames, fps

def dump_current_buffer(buf: RollingBuffer) -> str:
    frames, fps = buf.snapshot()
    if not frames:
        return ""
    tmpdir = tempfile.mkdtemp(prefix="vidseg_")
    out_path = os.path.join(tmpdir, "segment.mp4")
    try:
        write_segment_to_mp4(frames, fps, out_path)
        return out_path
    except Exception as e:
        print(f"Failed to write segment: {e}")
        return ""


def write_segment_to_mp4(frames: List[Frame], fps: float, out_path: str) -> str:
    if not frames:
        raise ValueError("No frames to write")
    h, w = frames[0].img.shape[:2]
    writer = imageio.get_writer(out_path, fps=fps, codec="libx264", quality=7)
    try:
        for fr in frames:
            # Convert BGR (OpenCV) -> RGB (imageio expects RGB)
            rgb = cv2.cvtColor(fr.img, cv2.COLOR_BGR2RGB)
            writer.append_data(rgb)
    finally:
        writer.close()
    return out_path

class VideoThread(QRunnable):
    # frame_ready = pyqtSignal(QtGui.QImage)

    def __init__(self, source, buffer_seconds=15, analyze_interval=10, window=None, parent=None):
        super().__init__()
        self.source = source
        self.window = window
        self.buffer_seconds = buffer_seconds
        self.analyze_interval = analyze_interval
        self.running = False
        self.signals = WorkerSignals()


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
        buf = RollingBuffer(seconds=4, fps=fps)
        # Queues for analysis work
        task_q: "queue.Queue[str]" = queue.Queue()
        results_q: "queue.Queue[str]" = queue.Queue()
        analyzer = AnalyzerThread(task_q, results_q)
        analyzer.start()

        # Timing for periodic analysis
        last_analyze = time.time()

        # Timing for playback speed control
        frame_delay = 1.0 / fps  # seconds per frame

        last_time = time.time()
        while self.running:
            ok, frame = cap.read()
            if not ok:
                break
            
            ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            buf.push(Frame(ts=ts, img=frame))
            # Convert BGR → RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            self.signals.frame_ready.emit(qimg)

            # Periodic analysis
            now = time.time()
            if now - last_analyze >= 5:
                last_analyze = now
                print("Submitting segment for analysis...")
                segment_path = dump_current_buffer(buf)
                if segment_path:
                    task_q.put(segment_path)

            # Sleep to maintain playback rate
            elapsed = time.time() - last_time
            sleep_time = max(0, frame_delay - elapsed)
            time.sleep(sleep_time)
            last_time = time.time()

            # Drain any analyzer results without blocking
            try:
                while True:
                    msg = results_q.get_nowait()
                    self.signals.text_ready.emit(msg)
            except queue.Empty:
                pass

        cap.release()

    def stop(self):
        self.running = False
        self.wait()


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.video_thread = None

        self.thread = QThreadPool()

        # Directly load test.mp4 from the same folder
        video_path = os.path.join(os.path.dirname(__file__), "craddling.mp4")
        if os.path.exists(video_path):
            self.start_video(video_path)
        else:
            print("❌ craddling.mp4 not found in current directory.")

    def start_video(self, source):
        if self.video_thread:
            self.video_thread.stop()

        self.video_thread = VideoThread(source, window=self)
        self.video_thread.signals.frame_ready.connect(self.update_frame)
        self.video_thread.signals.text_ready.connect(self.update_text)
        self.thread.start(self.video_thread)
        
    @QtCore.pyqtSlot(str)
    def update_text(self, text):
        self.model_repsonse_plainTextEdit.setPlainText(text)

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
