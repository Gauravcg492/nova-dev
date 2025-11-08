#!/usr/bin/env python3
"""
Simple Python video player that keeps a rolling 10–20s buffer and periodically
sends the latest chunk to an LLM for analysis.

Features
- Plays local files or network streams (e.g., rtsp/http) via OpenCV.
- Maintains a rolling buffer sized by seconds (default 15s).
- Every N seconds (or on hotkey), writes the buffered clip to disk and
  sends it to an LLM analysis function (stub provided; easy to plug in).
- Background worker thread so playback stays smooth.

Controls
- Press 'a' to force an immediate analysis of the current buffer.
- Press 'q' to quit.

Dependencies
    pip install opencv-python imageio imageio-ffmpeg

Usage
    python stream_player_llm.py --source path_or_url --buffer-seconds 15 --analyze-interval 10

Notes
- On some platforms, mp4 codecs can be finicky with cv2.VideoWriter; imageio-ffmpeg
  is used here for robust mp4 writing.
- If the FPS is 0/unknown, we fall back to 30 FPS.
- The LLM call is a stub. Replace `call_llm_on_video_segment` with your provider's SDK.
"""

import argparse
import base64
import os
import queue
import signal
import sys
import tempfile
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Tuple
from agent.run_agent_pipeline import analyze_video_and_answer
import cv2
import imageio

# ------------------------------
# Data classes and helpers
# ------------------------------

@dataclass
class Frame:
    ts: float  # presentation timestamp (seconds)
    img: any   # numpy array (H, W, 3) BGR


def safe_makedirs(path: str):
    os.makedirs(path, exist_ok=True)


# ------------------------------
# Rolling buffer for last N seconds
# ------------------------------

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


# ------------------------------
# Video writing with imageio-ffmpeg
# ------------------------------

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


# ------------------------------
# LLM call stub (replace with your SDK)
# ------------------------------

def call_llm_on_video_segment(segment_path: str, sample_every_nth_frame: int = 10) -> str:
    """Example stub that extracts a few frames, base64-encodes them, and
    prints how you'd send them to an LLM. Replace with your provider code.

    For providers that accept video files directly, simply stream the file
    rather than frames. This function returns a mock response string.
    """
    # cap = cv2.VideoCapture(segment_path)
    # if not cap.isOpened():
    #     return "[LLM] Could not open segment for sampling."

    # sampled = []
    # idx = 0
    # while True:
    #     ok, frame = cap.read()
    #     if not ok:
    #         break
    #     if idx % sample_every_nth_frame == 0:
    #         # JPEG encode for base64 packaging
    #         okj, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    #         if okj:
    #             b64 = base64.b64encode(buf).decode('ascii')
    #             sampled.append(b64)
    #     idx += 1
    # cap.release()
    
    result = analyze_video_and_answer(segment_path)

    # --- Replace this with your actual LLM call ---
    # Example (OpenAI-style pseudocode):
    # from openai import OpenAI
    # client = OpenAI()
    # resp = client.chat.completions.create(
    #   model="gpt-4o",  # a vision-capable model
    #   messages=[
    #     {"role": "system", "content": "You are a video assistant."},
    #     {"role": "user", "content": [
    #         {"type": "text", "text": "Summarize what's happening in these frames."},
    #     ] + [
    #         {"type": "image_url", "image_url": f"data:image/jpeg;base64,{b}"} for b in sampled
    #     ]}
    #   ]
    # )
    # return resp.choices[0].message.content

    # For now, just report how many frames we sampled.
    return result


# ------------------------------
# Background analyzer thread
# ------------------------------

class AnalyzerThread(threading.Thread):
    def __init__(self, task_q: "queue.Queue[str]", results_q: "queue.Queue[str]"):
        super().__init__(daemon=True)
        self.task_q = task_q
        self.results_q = results_q
        self._stop_event = threading.Event()

    def run(self):
        while not self._stop_event.is_set():
            try:
                print("Analyzer waiting for segment...")
                segment_path = self.task_q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                print(f"Analyzer processing segment: {segment_path}")
                result = call_llm_on_video_segment(segment_path)
                self.results_q.put(result)
            except Exception as e:
                self.results_q.put(f"[Analyzer error] {e}")
            finally:
                # Best-effort cleanup of temp segment files
                try:
                    os.remove(segment_path)
                except Exception:
                    pass

    def stop(self):
        self._stop_event.set()


# ------------------------------
# Main player loop
# ------------------------------

def player(source: str, buffer_seconds: int, analyze_interval: int):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Failed to open source: {source}")
        return 1

    # Initial FPS (may be 0 for streams); fallback to 30
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or fps != fps:  # NaN guard
        fps = 30.0

    buf = RollingBuffer(seconds=buffer_seconds, fps=fps)

    # Queues for analysis work
    task_q: "queue.Queue[str]" = queue.Queue()
    results_q: "queue.Queue[str]" = queue.Queue()
    analyzer = AnalyzerThread(task_q, results_q)
    analyzer.start()

    # Timing for periodic analysis
    last_analyze = time.time()

    # Timing for playback speed control
    frame_delay = 1.0 / fps  # seconds per frame
    last_frame_time = time.time()

    # Graceful Ctrl+C handling
    stopping = False

    def handle_sigint(sig, frame):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGINT, handle_sigint)

    window_name = "Video (press 'a' to analyze, 'q' to quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while not stopping:
        ok, frame = cap.read()
        if not ok:
            # For live streams, short sleep & retry; for files, break
            if source.startswith("rtsp:") or source.startswith("http"):
                time.sleep(0.02)
                continue
            else:
                break

        # If FPS changes mid-stream, update buffer and frame delay
        curr_fps = cap.get(cv2.CAP_PROP_FPS)
        if curr_fps and curr_fps > 0:
            buf.update_fps(curr_fps)
            frame_delay = 1.0 / curr_fps

        ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        buf.push(Frame(ts=ts, img=frame))

        # Show frame
        cv2.imshow(window_name, frame)

        # Calculate appropriate wait time to maintain playback speed
        elapsed = time.time() - last_frame_time
        wait_time = max(1, int((frame_delay - elapsed) * 1000))  # convert to milliseconds, min 1ms

        key = cv2.waitKey(wait_time) & 0xFF
        last_frame_time = time.time()

        if key == ord('q'):
            break
        elif key == ord('a'):
            # Force immediate analysis
            segment_path = dump_current_buffer(buf)
            if segment_path:
                task_q.put(segment_path)

        # Periodic analysis
        now = time.time()
        if analyze_interval > 0 and now - last_analyze >= analyze_interval:
            last_analyze = now
            segment_path = dump_current_buffer(buf)
            if segment_path:
                task_q.put(segment_path)

        # Drain any analyzer results without blocking
        try:
            while True:
                msg = results_q.get_nowait()
                print(msg)
        except queue.Empty:
            pass

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    analyzer.stop()
    return 0


# ------------------------------
# Buffer -> temp mp4 utility
# ------------------------------

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


# ------------------------------
# CLI
# ------------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Simple player with rolling buffer -> LLM")
    ap.add_argument("--source", required=True, help="Video source (file path, rtsp/http URL)")
    ap.add_argument("--buffer-seconds", type=int, default=15, help="Rolling buffer length in seconds (10-20s recommended)")
    ap.add_argument("--analyze-interval", type=int, default=10, help="Seconds between automatic analyses (0 to disable)")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.buffer_seconds < 10 or args.buffer_seconds > 20:
        print(f"[warn] buffer-seconds={args.buffer_seconds} — typical range is 10–20s.")
    sys.exit(player(args.source, args.buffer_seconds, args.analyze_interval))
