from __future__ import annotations

from io import BytesIO
from pathlib import Path

import cv2
import streamlit as st
from fastapi import FastAPI
from PIL import Image

app = FastAPI(title="Cross-Camera Object Tracking")


@app.get("/")
async def root() -> dict[str, str]:
	return {
		"message": "Cross-Camera Object Tracking is running.",
		"ui": "Use 'streamlit run app.py' to launch the Streamlit interface.",
	}


PROJECT_ROOT = Path(__file__).resolve().parent
CAMERA_DIR = PROJECT_ROOT / "data" / "cameras"
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"
TARGET_PATH = REFERENCE_DIR / "target.jpg"


def ensure_directories() -> None:
	REFERENCE_DIR.mkdir(parents=True, exist_ok=True)


def save_pil_image_as_target(image: Image.Image) -> None:
	rgb_image = image.convert("RGB")
	rgb_image.save(TARGET_PATH, format="JPEG", quality=95)


def load_frame(video_path: Path, frame_index: int) -> Image.Image | None:
	cap = cv2.VideoCapture(str(video_path))
	if not cap.isOpened():
		return None

	cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
	success, frame = cap.read()
	cap.release()

	if not success or frame is None:
		return None

	frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	return Image.fromarray(frame_rgb)


def get_video_metadata(video_path: Path) -> tuple[int, float]:
	cap = cv2.VideoCapture(str(video_path))
	if not cap.isOpened():
		return 0, 0.0

	total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
	cap.release()
	return total_frames, fps


def render_current_target() -> None:
	st.subheader("Current Target")
	if TARGET_PATH.exists():
		st.image(str(TARGET_PATH), caption=str(TARGET_PATH.relative_to(PROJECT_ROOT)), width=280)
	else:
		st.info("No target is set yet. Upload an image or choose a frame from a video.")


def target_from_upload() -> None:
	st.subheader("Option 1: Upload Target Image")
	uploaded_file = st.file_uploader(
		"Upload image",
		type=["jpg", "jpeg", "png", "webp"],
		help="The selected image will be saved as data/reference/target.jpg",
	)

	if uploaded_file is None:
		return

	image_bytes = BytesIO(uploaded_file.getvalue())
	image = Image.open(image_bytes)
	st.image(image, caption="Preview", width=300)

	if st.button("Set uploaded image as target", type="primary"):
		save_pil_image_as_target(image)
		st.success(f"Target set successfully: {TARGET_PATH}")
		st.rerun()


def target_from_video_frame() -> None:
	st.subheader("Option 2: Select Target From Video")

	videos = sorted(CAMERA_DIR.glob("*.mp4"))
	if not videos:
		st.warning(f"No .mp4 videos found in: {CAMERA_DIR}")
		return

	selected_video_name = st.selectbox(
		"Choose video",
		options=[path.name for path in videos],
	)
	selected_video = CAMERA_DIR / selected_video_name

	total_frames, fps = get_video_metadata(selected_video)
	if total_frames <= 0:
		st.error(f"Could not read frames from {selected_video}")
		return

	st.caption(f"Frames: {total_frames} | FPS: {fps:.2f}")

	frame_index = st.slider(
		"Pick frame index",
		min_value=0,
		max_value=max(total_frames - 1, 0),
		value=min(total_frames // 2, max(total_frames - 1, 0)),
	)

	frame_image = load_frame(selected_video, frame_index)
	if frame_image is None:
		st.error("Failed to load selected frame.")
		return

	st.image(frame_image, caption=f"{selected_video_name} | Frame {frame_index}", width=520)

	if st.button("Set selected frame as target", type="primary"):
		save_pil_image_as_target(frame_image)
		st.success(
			"Target set successfully from video frame: "
			f"{selected_video_name} (frame {frame_index})"
		)
		st.rerun()


def main() -> None:
	ensure_directories()

	st.set_page_config(page_title="Cross-Camera Target Setup", layout="centered")
	st.title("Cross-Camera Object Tracking")
	st.write("Set the target person image before running the tracking and matching pipeline.")

	render_current_target()
	st.divider()

	source = st.radio(
		"Choose how to set the target",
		options=["Upload target image", "Select frame from existing video"],
	)

	if source == "Upload target image":
		target_from_upload()
	else:
		target_from_video_frame()


if __name__ == "__main__":
	main()
