
import cv2
import torch
from torchvision import transforms


# ResNet18 input preprocessing
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def preprocess_frame(frame):
    # OpenCV uses BGR; convert to RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # Convert image into model input tensor
    tensor = transform(rgb_frame)

    # Add batch dimension
    tensor = tensor.unsqueeze(0)

    return tensor


if __name__ == "__main__":

    # Test with a sample image
    sample = torch.zeros(
        (224, 224, 3),
        dtype=torch.uint8
    ).numpy()

    processed = preprocess_frame(sample)

    print("VeriFrame AI - Preprocessing")
    print("Input shape:", sample.shape)
    print("Output shape:", processed.shape)
    print("Preprocessing successful!")
