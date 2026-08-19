import cv2
import json
from pathlib import Path
from ultralytics import YOLO


# ============================================================
# CONFIG
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CAMERA_DIR = PROJECT_ROOT / "data" / "cameras"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

TRACK_OUTPUT_DIR = OUTPUT_DIR / "tracks"    
VISUALIZATION_DIR = OUTPUT_DIR / "visualizations"

MODEL_NAME = "yolov8n.pt"

# Only detect people
PERSON_CLASS_ID = 0

# Process every Nth frame for saving crops
CROP_INTERVAL = 5

# Detection confidence
CONFIDENCE = 0.35

# ============================================================
# CREATE DIRECTORIES
# ============================================================

TRACK_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Load Model
# ============================================================

model = YOLO(MODEL_NAME)  # Load a pretrained YOLOv8n model

# ============================================================
# PROCESS ONE CAMERA
# ============================================================

def process_camera(video_path):
    camera_name = video_path.stem
    print()
    print("=" * 60)
    print(f"Processing: {camera_name}")
    print("=" * 60)

    
    camera_track_dir = TRACK_OUTPUT_DIR / camera_name
    camera_track_dir.mkdir(parents=True, exist_ok=True)

    visualization_path = (
        VISUALIZATION_DIR / f"{camera_name}_tracked.mp4"
    )

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"ERROR: Could not open {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"FPS: {fps}")
    print(f"Resolution: {width} x {height}")
    print(f"Frames: {total_frames}")

        # Output video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(visualization_path),
        fourcc,
        fps,
        (width, height)
    )

    frame_number = 0

    # Store information about each track
    tracks = {}

    while True:

        success, frame = cap.read()

        if not success:
            break

        frame_number += 1


        # ----------------------------------------------------
        # YOLO + ByteTrack
        # ----------------------------------------------------

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[PERSON_CLASS_ID],
            conf=CONFIDENCE,
            verbose=False
        )

        result = results[0]

        # ----------------------------------------------------
        # No detections
        # ----------------------------------------------------

        if result.boxes is None:
            writer.write(frame)
            continue

        boxes = result.boxes

        # Make sure IDs exist
        if boxes.id is None:
            writer.write(frame)
            continue

        xyxy = boxes.xyxy.cpu().numpy()
        track_ids = boxes.id.int().cpu().numpy()
        confidences = boxes.conf.cpu().numpy()

        # ----------------------------------------------------
        # Process each tracked person
        # ----------------------------------------------------

        for box, track_id, confidence in zip(
            xyxy,
            track_ids,
            confidences
        ):

            x1, y1, x2, y2 = map(int, box)

            track_id = int(track_id)

            # Safety checks
            x1 = max(0, x1)
            y1 = max(0, y1)

            x2 = min(width, x2)
            y2 = min(height, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            # ------------------------------------------------
            # Create track directory
            # ------------------------------------------------

            track_dir = camera_track_dir / f"track_{track_id:04d}"

            track_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            # ------------------------------------------------
            # Store metadata
            # ------------------------------------------------

            if track_id not in tracks:

                tracks[track_id] = {
                    "track_id": track_id,
                    "camera": camera_name,
                    "first_frame": frame_number,
                    "last_frame": frame_number,
                    "frames": [],
                    "boxes": [],
                    "confidences": []
                }

            tracks[track_id]["last_frame"] = frame_number

            tracks[track_id]["frames"].append(frame_number)

            tracks[track_id]["boxes"].append(
                [x1, y1, x2, y2]
            )

            tracks[track_id]["confidences"].append(
                float(confidence)
            )

            # ------------------------------------------------
            # Save crops every N frames
            # ------------------------------------------------

            if frame_number % CROP_INTERVAL == 0:

                crop = frame[y1:y2, x1:x2]

                if crop.size > 0:

                    crop_path = (
                        track_dir /
                        f"frame_{frame_number:06d}.jpg"
                    )

                    cv2.imwrite(
                        str(crop_path),
                        crop
                    )

            # ------------------------------------------------
            # Draw tracking box
            # ------------------------------------------------

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            label = (
                f"ID: {track_id} "
                f"{confidence:.2f}"
            )

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        # ----------------------------------------------------
        # Frame information
        # ----------------------------------------------------

        cv2.putText(
            frame,
            f"{camera_name} | Frame {frame_number}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2
        )

        writer.write(frame)

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if frame_number % 100 == 0:

            print(
                f"Processed "
                f"{frame_number}/{total_frames} frames"
            )

    cap.release()
    writer.release()
    # --------------------------------------------------------
    # Save track metadata
    # --------------------------------------------------------

    metadata_path = (
        OUTPUT_DIR /
        f"{camera_name}_tracks.json"
    )

    with open(metadata_path, "w") as f:

        json.dump(
            tracks,
            f,
            indent=4
        )

    print()
    print(f"Finished: {camera_name}")
    print(f"Tracks found: {len(tracks)}")
    print(f"Metadata: {metadata_path}")
    print(f"Video: {visualization_path}")


# ============================================================
# MAIN
# ============================================================

def main():

    videos = sorted(
        CAMERA_DIR.glob("*.mp4")
    )

    if not videos:

        print(
            "No videos found in "
            f"{CAMERA_DIR}"
        )
        return

    for video_path in videos:
        process_camera(video_path)


    print()
    print("=" * 60)
    print("ALL CAMERAS PROCESSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
