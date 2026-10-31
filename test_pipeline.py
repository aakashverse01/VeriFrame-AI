
import numpy as np
import torch

from model import create_model
from preprocess import preprocess_frame


def test_pipeline():
    print("VeriFrame AI - AI Pipeline Test")
    print("-" * 35)

    # Initialize ResNet18 architecture
    model = create_model()
    model.eval()

    # Create a sample frame for testing
    sample_frame = np.zeros(
        (224, 224, 3),
        dtype=np.uint8
    )

    # Convert frame to model-ready tensor
    input_tensor = preprocess_frame(sample_frame)

    print("Input tensor:", input_tensor.shape)

    # Test forward pass
    with torch.no_grad():
        output = model(input_tensor)

    print("Model output shape:", output.shape)

    assert output.shape == (1, 2), (
        "Unexpected model output shape"
    )

    print("Forward pass successful!")
    print("Architecture and preprocessing connected.")
    print("Note: Model weights are not trained.")


if __name__ == "__main__":
    test_pipeline()
