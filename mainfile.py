import torch
import matplotlib.pyplot as plt
from torchvision.io import read_image
from skimage import io, transform
from torchvision.transforms import Compose, Resize, ToTensor
from torch.utils.data import DataLoader, Dataset
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset
import os
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader
from data_file import ImageMaskDataset  # Ensure this is set up properly
import matplotlib.pyplot as plt
label_colors = {
    'urban_land': (0, 255, 255),
    'agriculture_land': (255, 255, 0),
    'rangeland': (255, 0, 255),
    'forest_land': (0, 255, 0),
    'water': (0, 0, 255),
    'barren_land': (255, 255, 255),
    'unknown': (0, 0, 0)
}


def main():
    # Create a model
    model = models.segmentation.fcn_resnet50(pretrained=True)
    model.eval()  # Set the model to evaluation mode if not training

    # Dataset and DataLoader
    image_directory = '/Users/polfuentes/TUM_Makeathon/archive/train3/images'
    mask_directory = '/Users/polfuentes/TUM_Makeathon/archive/train3/masks'


    transform = Compose([
    ToTensor(),  # Converts numpy array to tensor
    Resize((256, 256))  # Resize the tensor
    ])

    # Assuming transform and dataset setup includes necessary transformations
    dataset = ImageMaskDataset(image_directory, mask_directory, transform= transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            output = model(images)['out']
            print(output)
            print(output.shape)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            _, predicted_masks = torch.max(probabilities, dim=1)
            predicted_mask = predicted_masks.squeeze(0)

            color_mask = class_to_rgb(predicted_mask, label_colors)
            print(color_mask)
             # Get the most likely class for each pixel

            # Visualization
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.imshow(images[0].cpu().permute(1, 2, 0))  # Assuming the image is a torch.Tensor
            plt.title('Original Image')
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(predicted_mask)
            plt.title('Segmentation Output')
            plt.axis('off')

            plt.show()

            # Optionally, break after the first batch for demonstration
            break


def class_to_rgb(prediction, label_colors):
    """Convert class index predictions to RGB color image for visualization."""
    color_mask = torch.zeros(prediction.shape + (3,), dtype=torch.uint8)
    for class_idx, color in enumerate(label_colors.values()):
        color_mask[prediction == class_idx] = torch.tensor(color, dtype=torch.uint8)
    return color_mask


if __name__ == "__main__":
    main()
