import matplotlib.pyplot as plt
from torchvision.io import read_image
from skimage import io, transform
from torchvision.transforms import Compose, Resize, ToTensor
from torch.utils.data import DataLoader, Dataset
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset
import os
import torch
from torchvision.transforms import Resize, ToTensor, Normalize, Compose


label_colors = {
    'urban_land': (0, 255, 255),
    'agriculture_land': (255, 255, 0),
    'rangeland': (255, 0, 255),
    'forest_land': (0, 255, 0),
    'water': (0, 0, 255),
    'barren_land': (255, 255, 255),
    'unknown': (0, 0, 0)
}


class ImageMaskDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, mask_transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.images = [img for img in os.listdir(image_dir) if img.endswith('_sat.jpg')]
        self.masks = [self._find_mask(img) for img in self.images]

    def _find_mask(self, image_filename):
        base = image_filename.split('_sat')[0]
        mask_filename = f"{base}_mask.png"
        return mask_filename

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        mask_name = self.masks[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = io.imread(img_path)
        mask = io.imread(mask_path)

        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask


"""""""""""
# Define transformations
transform = Compose([
    ToTensor(),  # Converts numpy array to tensor
    Resize((256, 256))  # Resize the tensor
])

# Dataset and DataLoader
image_directory = '/Users/polfuentes/TUM_Makeathon/archive/train3/images'
mask_directory = '/Users/polfuentes/TUM_Makeathon/archive/train3/masks'

dataset = ImageMaskDataset(image_directory, mask_directory, transform=transform, mask_transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)



def show_images(images, masks, num_images=3):
    fig, axs = plt.subplots(num_images, 2, figsize=(10, 10))
    for i, (image, mask) in enumerate(zip(images, masks)):
        if i >= num_images:
            break
        image = image.squeeze(0)
        mask = mask.squeeze(0)
        
        axs[i, 0].imshow(image.permute(1, 2, 0))
        axs[i, 0].axis('off')
        axs[i, 0].set_title('Image')

        axs[i, 1].imshow(mask.permute(1, 2, 0))  # Display the mask in color
        axs[i, 1].axis('off')
        axs[i, 1].set_title('predict')

    plt.show()




# Load and display a few images
images, masks = [], []
for image, mask in dataloader:
    images.append(image)
    masks.append(mask)
    if len(images) == 3:
        break


#show_images(images, masks, num_images=3)
"""""
