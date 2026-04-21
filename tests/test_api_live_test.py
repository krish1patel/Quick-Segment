import sys
import requests
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading

API_URL = "http://localhost:8000"

PALETTE = [
    (255, 50,  50),
    (50,  200, 50),
    (50,  100, 255),
    (255, 200, 0),
    (200, 0,   255),
]

# ── State ─────────────────────────────────────────────────────────────────────

current_query = "person"
latest_results = []
latest_matches = []
results_lock = threading.Lock()
frame_lock = threading.Lock()
latest_frame = None


# ── Inference thread ──────────────────────────────────────────────────────────

def inference_worker():
    global latest_results, latest_matches
    while True:
        with frame_lock:
            frame = latest_frame.copy() if latest_frame is not None else None

        if frame is None:
            continue

        query = current_query.strip()
        if not query:
            continue

        _, encoded = cv2.imencode(".jpg", frame)
        img_bytes = encoded.tobytes()

        try:
            response = requests.post(
                f"{API_URL}/segment",
                params={"query": query, "threshold": 0.5, "conf": .3},
                files={"file": ("frame.jpg", img_bytes, "image/jpeg")},
                timeout=5,
            )
            if response.ok:
                data = response.json()
                with results_lock:
                    latest_results = data.get("detections", [])
                    latest_matches = data.get("matches", [])
        except requests.exceptions.RequestException:
            pass


# ── Drawing ───────────────────────────────────────────────────────────────────

def draw_results(frame: np.ndarray, results: list[dict]) -> np.ndarray:
    overlay = frame.copy()

    for i, det in enumerate(results):
        color = PALETTE[i % len(PALETTE)]
        bgr = (color[2], color[1], color[0])

        contour = det.get("mask", [])
        if contour:
            pts = np.array(contour, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(overlay, [pts], bgr)
            cv2.polylines(frame, [pts], isClosed=True, color=bgr, thickness=2)

        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)

        label = f"{det['class_name']} {det['confidence']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), bgr, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
    return frame


# ── GUI ───────────────────────────────────────────────────────────────────────

def build_gui(cap):
    global current_query

    root = tk.Tk()
    root.title("QuickSegment — Live")
    root.configure(bg="#1a1a1a")

    # ── Query bar ──────────────────────────────────────────────────
    bar = tk.Frame(root, bg="#1a1a1a")
    bar.pack(fill=tk.X, padx=10, pady=(10, 4))

    tk.Label(bar, text="Query:", fg="white", bg="#1a1a1a",
             font=("Helvetica", 13)).pack(side=tk.LEFT, padx=(0, 6))

    query_var = tk.StringVar(value=current_query)
    entry = ttk.Entry(bar, textvariable=query_var, font=("Helvetica", 13), width=30)
    entry.pack(side=tk.LEFT)

    def on_query_change(*_):
        global current_query
        current_query = query_var.get()

    query_var.trace_add("write", on_query_change)

    # ── Matches panel ──────────────────────────────────────────────
    matches_frame = tk.Frame(root, bg="#1a1a1a")
    matches_frame.pack(fill=tk.X, padx=10, pady=(0, 6))

    matches_label = tk.Label(
        matches_frame,
        text="Active classes: —",
        fg="#aaaaaa",
        bg="#1a1a1a",
        font=("Helvetica", 11),
        anchor="w",
        wraplength=640,
        justify=tk.LEFT,
    )
    matches_label.pack(fill=tk.X)

    # ── Video canvas ───────────────────────────────────────────────
    canvas = tk.Label(root, bg="#1a1a1a")
    canvas.pack(padx=10, pady=(0, 10))

    def update_frame():
        global latest_frame
        ret, frame = cap.read()
        if ret:
            with frame_lock:
                latest_frame = frame.copy()

            with results_lock:
                results = list(latest_results)
                matches = list(latest_matches)

            # Update matches label
            if matches:
                # Group by dataset, show class names as pills
                by_dataset: dict[str, list[str]] = {}
                for m in matches:
                    by_dataset.setdefault(m["dataset"], []).append(m["class_name"])
                parts = [
                    f"[{ds}]  " + "  ·  ".join(classes)
                    for ds, classes in by_dataset.items()
                ]
                matches_label.config(
                    text="Active classes:  " + "    ".join(parts),
                    fg="#66ccff"
                )
            else:
                matches_label.config(text="Active classes: — (no matches)", fg="#888888")

            annotated = draw_results(frame, results)
            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(Image.fromarray(rgb))
            canvas.configure(image=img)
            canvas.image = img

        root.after(33, update_frame)

    update_frame()
    return root


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam")
        sys.exit(1)

    worker = threading.Thread(target=inference_worker, daemon=True)
    worker.start()

    root = build_gui(cap)

    try:
        root.mainloop()
    finally:
        cap.release()


if __name__ == "__main__":
    main()