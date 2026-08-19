import torch
import torchreid
import numpy as np

from PIL import Image
from torchvision import transforms


class OSNetReID:

    def __init__(self):

        # --------------------------------------------------
        # Device
        # --------------------------------------------------

        if torch.backends.mps.is_available():
            self.device = torch.device("mps")

        elif torch.cuda.is_available():
            self.device = torch.device("cuda")

        else:
            self.device = torch.device("cpu")

        print(f"Using device: {self.device}")

        # --------------------------------------------------
        # Load OSNet
        # --------------------------------------------------

        self.model = torchreid.models.build_model(
            name="osnet_x1_0",
            num_classes=1000,
            pretrained=True
        )

        self.model.eval()
        self.model.to(self.device)

        # --------------------------------------------------
        # Image preprocessing
        # --------------------------------------------------

        self.transform = transforms.Compose([
            transforms.Resize((256, 128)),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    # ======================================================
    # Extract embedding
    # ======================================================

    def extract(self, image_path):

        image = Image.open(image_path).convert("RGB")

        image = self.transform(image)

        image = image.unsqueeze(0)

        image = image.to(self.device)

        with torch.no_grad():

            embedding = self.model(image)

        embedding = embedding.detach().cpu().numpy()

        # L2 normalization
        embedding = embedding / (
            np.linalg.norm(
                embedding,
                axis=1,
                keepdims=True
            ) + 1e-12
        )

        return embedding[0]