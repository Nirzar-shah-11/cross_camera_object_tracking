import numpy as np

from pathlib import Path

from reid import OSNetReID

ROOT_DIR = Path(__file__).resolve().parent.parent

# ============================================================
# CONFIG
# ============================================================

TRACK_DIR = ROOT_DIR /"outputs" /"tracks"

EMBEDDING_DIR = ROOT_DIR /"outputs" / "embeddings"

SAMPLES_PER_TRACK = 10


# ============================================================
# SETUP
# ============================================================

EMBEDDING_DIR.mkdir(
    parents=True,
    exist_ok=True
)

reid_model = OSNetReID()


# ============================================================
# SELECT REPRESENTATIVE IMAGES
# ============================================================

def select_images(track_dir):

    images = sorted(
        list(track_dir.glob("*.jpg"))
        + list(track_dir.glob("*.jpeg"))
        + list(track_dir.glob("*.png"))
    )

    if len(images) <= SAMPLES_PER_TRACK:
        return images

    # Uniform sampling
    indices = np.linspace(
        0,
        len(images) - 1,
        SAMPLES_PER_TRACK,
        dtype=int
    )

    return [
        images[i]
        for i in indices
    ]


# ============================================================
# PROCESS TRACK
# ============================================================

def process_track(camera_name, track_dir):

    images = select_images(track_dir)

    if not images:
        print(f"No images found: {track_dir}")
        return

    embeddings = []

    for image_path in images:
        try:
            embedding = reid_model.extract(image_path)
            embeddings.append(embedding)

        except Exception as e:
            print(f"Error processing " f"{image_path}: {e}")

    if not embeddings:
        return

    embeddings = np.array(
        embeddings
    )

    # --------------------------------------------------------
    # Average embedding
    # --------------------------------------------------------

    track_embedding = np.mean(
        embeddings,
        axis=0
    )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    track_embedding = (
        track_embedding /
        (
            np.linalg.norm(
                track_embedding
            ) + 1e-12
        )
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_path = (
        EMBEDDING_DIR /
        f"{camera_name}_{track_dir.name}.npy"
    )

    np.save(
        output_path,
        track_embedding
    )

    print(
        f"{camera_name} | "
        f"{track_dir.name} | "
        f"{len(images)} images | "
        f"saved"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    cameras = sorted(
        [
            x for x in TRACK_DIR.iterdir()
            if x.is_dir()
        ]
    )

    print(
        f"Found {len(cameras)} cameras"
    )

    for camera_dir in cameras:

        camera_name = camera_dir.name

        tracks = sorted(
            [
                x for x in camera_dir.iterdir()
                if x.is_dir()
            ]
        )

        print()
        print("=" * 60)
        print(camera_name)
        print("=" * 60)

        for track_dir in tracks:

            process_track(
                camera_name,
                track_dir
            )


if __name__ == "__main__":

    main()