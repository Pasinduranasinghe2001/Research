import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


IMAGE_PATH = r"C:\Users\Gaurawa Mihiranga\Downloads\2023_12.jpg"
MODEL_PATH = "models/single_frame_resnet18_best.pth"


IMAGE_SIZE = 266


# -------------------------------
# MODEL
# -------------------------------

class SingleFrameResNet18(nn.Module):

    def __init__(self):
        super().__init__()

        self.backbone = models.resnet18(
            weights=None
        )

        in_features = self.backbone.fc.in_features

        self.backbone.fc = nn.Linear(
            in_features,
            2
        )


    def forward(self,x):

        return self.backbone(x)



# -------------------------------
# LOAD MODEL
# -------------------------------

model = SingleFrameResNet18()

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model.to(DEVICE)

model.eval()



# -------------------------------
# IMAGE PREPROCESSING
# -------------------------------

transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE,IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )

])



# -------------------------------
# LOAD IMAGE
# -------------------------------

image = Image.open(
    IMAGE_PATH
).convert("RGB")


image = transform(image)


# add batch dimension

image = image.unsqueeze(0)


image = image.to(DEVICE)



# -------------------------------
# PREDICTION
# -------------------------------

with torch.no_grad():

    output = model(image)

    probability = torch.softmax(
        output,
        dim=1
    )


predicted_class = torch.argmax(
    probability,
    dim=1
).item()



looted_probability = probability[0][1].item()


preserved_probability = probability[0][0].item()



# -------------------------------
# RESULT
# -------------------------------


if predicted_class == 1:

    result = "LOOTED"

else:

    result = "PRESERVED"



print("======================")
print("Prediction:",result)

print(
    f"Preserved probability: {preserved_probability:.3f}"
)

print(
    f"Looted probability: {looted_probability:.3f}"
)

print("======================")