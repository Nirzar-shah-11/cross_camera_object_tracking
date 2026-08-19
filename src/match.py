from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def find_candidate_image(embedding_file, tracks_dir=None):
    """Return a representative crop for the track represented by an embedding."""
    if tracks_dir is None:
        tracks_dir = Path(__file__).resolve().parents[1] / "outputs" / "tracks"

    try:
        camera_name, track_name = embedding_file.stem.rsplit("_track_", maxsplit=1)
    except ValueError:
        return None

    track_dir = tracks_dir / camera_name / f"track_{track_name}"
    images = sorted(
        path
        for pattern in ("*.jpg", "*.jpeg", "*.png")
        for path in track_dir.glob(pattern)
    )

    if not images:
        return None

    return str(images[len(images) // 2])


def search_candidates(reference_embedding, embedding_dir=None, top_k=20):
    """Return ranked candidate matches for a reference embedding."""
    if embedding_dir is None:
        embedding_dir = Path(__file__).resolve().parents[1] / "outputs" / "embeddings"

    reference_vector = np.asarray(reference_embedding, dtype=np.float32).reshape(1, -1)
    results = []

    for embedding_file in sorted(embedding_dir.glob("*.npy")):
        if embedding_file.name == "reference.npy":
            continue

        embedding = np.load(embedding_file).reshape(1, -1)
        similarity = cosine_similarity(reference_vector, embedding)[0][0]
        results.append({
            "file": embedding_file.name,
            "image": find_candidate_image(embedding_file),
            "similarity": float(similarity)
        })

    results.sort(key=lambda item: item["similarity"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    embedding_dir = Path(__file__).resolve().parents[1] / "outputs" / "embeddings"
    reference_file = embedding_dir / "reference.npy"

    if not reference_file.exists():
        raise FileNotFoundError(f"Reference embedding not found at {reference_file}")

    reference_embedding = np.load(reference_file).reshape(1, -1)
    results = search_candidates(reference_embedding, embedding_dir=embedding_dir, top_k=20)

    print()
    print("=" * 70)
    print("TOP MATCHES")
    print("=" * 70)

    for rank, result in enumerate(results, start=1):
        print(f"{rank:02d}. {result['file']:40s} {result['similarity']:.4f}")
