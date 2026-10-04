"""
video_detect.py
----------------
Handles video input by breaking it into individual frames (images), then
reusing the SAME detection logic from detect.py on each one.

This is the key idea behind "video AI": there's no separate video model.
A video is just many images shown quickly in sequence, so we sample some
of those images and run our normal image model on each.

We don't process EVERY frame — a 10-second video at 30fps is 300 frames,
and running the model 300 times would be slow and mostly redundant (the
banana barely changes between consecutive frames). Instead we sample a
handful of evenly-spaced frames across the video's duration.
"""

import cv2
import numpy as np
import tempfile
import os
from collections import Counter

import detect  # reuses the already-loaded model and run_detection()


def extract_sample_frames(video_path: str, max_frames: int = 12):
    """
    Pulls out evenly-spaced frames from a video file.

    Args:
        video_path: path to the video file on disk
        max_frames: how many frames to sample across the whole video

    Returns:
        A list of (timestamp_seconds, frame_image) tuples, where frame_image
        is a numpy array (BGR, OpenCV's format) ready to feed into detection.
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError("Could not open video file. Is it a valid video format?")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30  # fall back to 30 if metadata is missing
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        raise RuntimeError("Video appears to have no frames.")

    # Pick `max_frames` frame INDEXES evenly spaced from start to end
    sample_count = min(max_frames, total_frames)
    frame_indexes = np.linspace(0, total_frames - 1, sample_count, dtype=int)

    sampled = []
    for frame_index in frame_indexes:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        success, frame = cap.read()
        if success:
            timestamp = frame_index / fps
            sampled.append((round(float(timestamp), 2), frame))

    cap.release()
    return sampled


def run_video_detection(video_bytes: bytes, max_frames: int = 12, confidence_threshold: float = 0.4):
    """
    Runs detection across sampled frames of an uploaded video.

    Args:
        video_bytes: raw bytes of the uploaded video file
        max_frames: how many frames to sample and analyze
        confidence_threshold: passed through to the underlying image model

    Returns:
        {
            "frame_count_analyzed": int,
            "banana_found_in_any_frame": bool,
            "most_common_label": str | None,
            "frame_results": [
                {
                    "timestamp": 1.25,
                    "detections": [...]   # same shape as image detections
                },
                ...
            ],
            "best_frame_annotated_image": str  # base64 PNG of the most confident frame
        }
    """
    # OpenCV needs an actual file path to read a video, not raw bytes in memory,
    # so we write the upload to a temporary file first.
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
        tmp_file.write(video_bytes)
        tmp_path = tmp_file.name

    try:
        sampled_frames = extract_sample_frames(tmp_path, max_frames=max_frames)
    finally:
        os.remove(tmp_path)  # always clean up the temp file, even if extraction fails

    frame_results = []
    all_labels = []
    best_confidence = -1
    best_frame_annotated_base64 = None

    for timestamp, frame_bgr in sampled_frames:
        # detect.run_detection() expects encoded image bytes (like an upload),
        # so we re-encode this raw frame array back into PNG bytes first.
        success, buffer = cv2.imencode(".png", frame_bgr)
        if not success:
            continue
        frame_image_bytes = buffer.tobytes()

        detections, annotated_base64 = detect.run_detection(
            frame_image_bytes, confidence_threshold=confidence_threshold
        )

        frame_results.append({
            "timestamp": timestamp,
            "detections": detections,
        })

        for d in detections:
            all_labels.append(d["label"])
            if d["confidence"] > best_confidence:
                best_confidence = d["confidence"]
                best_frame_annotated_base64 = annotated_base64

    most_common_label = None
    if all_labels:
        most_common_label = Counter(all_labels).most_common(1)[0][0]

    return {
        "frame_count_analyzed": len(sampled_frames),
        "banana_found_in_any_frame": len(all_labels) > 0,
        "most_common_label": most_common_label,
        "frame_results": frame_results,
        "best_frame_annotated_image": best_frame_annotated_base64,
    }


 