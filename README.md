# Cross-Camera Object Tracking

This project detects and tracks people in one or more camera videos, creates a
person re-identification embedding for each track, and ranks tracks against a
reference image.

The current pipeline is script-based. `app.py` and `requirements.txt` are
placeholders; use the commands below from the project root.

## Pipeline

1. **Track people:** YOLOv8n detects people and ByteTrack maintains identities
	 within each video. A crop is saved every fifth frame, together with the
	 bounding-box metadata and an annotated video.
2. **Create track embeddings:** OSNet extracts an embedding from up to 10
	 representative crops per track. The embeddings are averaged, normalized,
	 and saved as `.npy` files.
3. **Create a reference embedding:** OSNet embeds `data/reference/target.jpg`.
4. **Search matches:** cosine similarity ranks every track embedding against
	 the reference embedding.

## Repository Layout

```text
data/
	cameras/                 Input videos (`.mp4` files)
	reference/target.jpg    Person image to search for
src/
	track.py                Detection, tracking, crops, and metadata
	reid.py                 OSNet model and embedding extraction
	create_embeddings.py    Track-level embedding generation
	reference_embedding.py  Reference embedding generation
	match.py                Cosine-similarity search
agent/                    LangGraph/OpenAI verification prototype
outputs/
	tracks/                 Crops grouped by camera and track ID
	embeddings/             Reference and track embeddings
	visualizations/         Annotated output videos
	*_tracks.json           Per-camera track metadata
```

## Setup

Create and activate a virtual environment, then install the dependencies used
by the scripts:

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install opencv-python numpy scikit-learn pillow torch torchvision ultralytics torchreid
```

The first model run may download pretrained weights. The repository includes
`yolov8n.pt`; OSNet is configured with `pretrained=True` and may download its
weights through `torchreid`.

On Apple Silicon, the ReID code automatically uses the MPS device when
available, then CUDA, and otherwise CPU.

## Run the Working Pipeline

Run these commands from the repository root and in this order:

```bash
# Process every .mp4 file in data/cameras/
python src/track.py

# Create one normalized embedding for each saved track
python src/create_embeddings.py

# Embed data/reference/target.jpg
python src/reference_embedding.py

# Print the top 20 cosine-similarity matches
python src/match.py
```

To process another camera, add its `.mp4` file to `data/cameras/`. To search
for another person, replace `data/reference/target.jpg` and regenerate the
reference embedding before running the matcher.

## Outputs

For an input video named `test1.mp4`, tracking produces:

```text
outputs/tracks/test1/track_0001/frame_000005.jpg
outputs/test1_tracks.json
outputs/visualizations/test1_tracked.mp4
```

Embedding generation produces files such as:

```text
outputs/embeddings/test1_track_0001.npy
outputs/embeddings/reference.npy
```

The JSON metadata contains each track’s camera name, first and last frame,
frame numbers, bounding boxes, and detector confidences.

## Agent Prototype

`agent/graph.py` defines a LangGraph workflow that describes the reference
person, retrieves candidates, optionally verifies ambiguous candidates with a
vision-language model, and ranks results. This layer is experimental and is
not part of the working command-line pipeline yet.

Before using it, configure the OpenAI client with an environment variable or a
secret manager rather than placing an API key in source code. The current
prototype also contains import-time example code and requires the additional
`openai` and `langgraph` packages.

## Limitations

- ByteTrack IDs are local to each camera video; cross-camera identity comes
	from embedding similarity rather than shared tracker IDs.
- The matcher compares against all `.npy` files except `reference.npy` and
	returns the top 20 results by default.
- Detection confidence, crop interval, and the YOLO model are currently
	constants in `src/track.py`.
- `requirements.txt` and `app.py` do not yet define a packaged installation or
	web application.
