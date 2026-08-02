from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


# ==================================================
# CONFIGURATION
# ==================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_SIZE = 266

# Use two images from the SAME site.
# The first image must be older than the second image.

EARLY_IMAGE_PATH = Path(
    r"C:\Users\Gaurawa Mihiranga\Downloads\2016_01.jpg"
)

RECENT_IMAGE_PATH = Path(
    r"C:\Users\Gaurawa Mihiranga\Downloads\2023_12.jpg"
)

# predict_two_frame.py is inside src, so find the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "two_frame_change_resnet18_best.pth"
)


# ==================================================
# TWO-FRAME MODEL
# ==================================================

class TwoFrameChangeResNet18(nn.Module):
    """
    Input:
        [B, 2, 3, H, W]

    Combined ResNet input:
        Early image       = 3 channels
        Recent image      = 3 channels
        Absolute change   = 3 channels

        Total             = 9 channels
    """

    def __init__(self, num_classes=2):
        super().__init__()

        # weights=None is correct here because all trained weights
        # will be loaded from the saved .pth file.
        self.backbone = models.resnet18(weights=None)

        old_conv = self.backbone.conv1

        self.backbone.conv1 = nn.Conv2d(
            in_channels=9,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )

        in_features = self.backbone.fc.in_features

        self.backbone.fc = nn.Linear(
            in_features,
            num_classes,
        )

    def forward(self, x):
        # x shape: [B, 2, 3, H, W]

        early = x[:, 0, :, :, :]
        recent = x[:, 1, :, :, :]

        absolute_difference = torch.abs(
            recent - early
        )

        combined = torch.cat(
            [
                early,
                recent,
                absolute_difference,
            ],
            dim=1,
        )

        # combined shape: [B, 9, H, W]
        logits = self.backbone(combined)

        return logits


# ==================================================
# CHECK FILES
# ==================================================

def validate_file(path, description):
    if not path.exists():
        raise FileNotFoundError(
            f"{description} was not found:\n{path}"
        )

    if not path.is_file():
        raise ValueError(
            f"{description} is not a file:\n{path}"
        )


validate_file(
    EARLY_IMAGE_PATH,
    "Early satellite image",
)

validate_file(
    RECENT_IMAGE_PATH,
    "Recent satellite image",
)

validate_file(
    MODEL_PATH,
    "Two-frame trained model",
)


# ==================================================
# LOAD TRAINED MODEL
# ==================================================

model = TwoFrameChangeResNet18(
    num_classes=2
).to(DEVICE)

state_dict = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True,
)

model.load_state_dict(state_dict)

model.eval()


# ==================================================
# IMAGE PREPROCESSING
# ==================================================

transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406,
        ],
        std=[
            0.229,
            0.224,
            0.225,
        ],
    ),
])


def load_image(image_path):
    image = Image.open(
        image_path
    ).convert("RGB")

    return transform(image)


early_image = load_image(
    EARLY_IMAGE_PATH
)

recent_image = load_image(
    RECENT_IMAGE_PATH
)


# Stack the two images.
# Current shape: [2, 3, H, W]

two_frames = torch.stack(
    [
        early_image,
        recent_image,
    ],
    dim=0,
)

# Add batch dimension.
# Final shape: [1, 2, 3, H, W]

two_frames = two_frames.unsqueeze(0)

two_frames = two_frames.to(DEVICE)


# ==================================================
# PREDICTION
# ==================================================

with torch.no_grad():
    logits = model(two_frames)

    probabilities = torch.softmax(
        logits,
        dim=1,
    )

    predicted_class = torch.argmax(
        probabilities,
        dim=1,
    ).item()


preserved_probability = (
    probabilities[0, 0].item()
)

looted_probability = (
    probabilities[0, 1].item()
)


if predicted_class == 1:
    prediction = "LOOTED"
else:
    prediction = "PRESERVED"


# ==================================================
# DISPLAY RESULT
# ==================================================

print("\n========================================")
print("TWO-FRAME CHANGE DETECTION RESULT")
print("========================================")

print(f"Device: {DEVICE}")

print(
    f"Early image:  {EARLY_IMAGE_PATH.name}"
)

print(
    f"Recent image: {RECENT_IMAGE_PATH.name}"
)

print("----------------------------------------")

print(
    f"Prediction: {prediction}"
)

print(
    "Preserved probability: "
    f"{preserved_probability:.4f} "
    f"({preserved_probability * 100:.2f}%)"
)

print(
    "Looted probability: "
    f"{looted_probability:.4f} "
    f"({looted_probability * 100:.2f}%)"
)

print("========================================\n")