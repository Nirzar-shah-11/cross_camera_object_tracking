import numpy as np
from pathlib import Path
from reid import OSNetReID


ROOT_DIR = Path(__file__).resolve().parent.parent

REFERENCE_IMAGE = ROOT_DIR /"data" /"reference"/"target.jpg"

OUTPUT_FILE = ROOT_DIR /"outputs" /"embeddings" /"reference.npy"



model = OSNetReID()

embedding = model.extract(
    REFERENCE_IMAGE
)

np.save(
    OUTPUT_FILE,
    embedding
)

print()
print("Reference embedding created.")
print("Shape:", embedding.shape)
print("Saved:", OUTPUT_FILE)