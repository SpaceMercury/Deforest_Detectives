import matplotlib.pyplot as plt
from torchvision.io import read_image
from torchvision.transforms import Compose, Resize, ToTensor
from torch.utils.data import DataLoader, Dataset
import os
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, ToTensor

from torchvision.transforms.functional import to_pil_image, to_tensor, rgb_to_grayscale

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

        image = read_image(img_path)
        mask = read_image(mask_path)

        if self.transform:
            print(f"Type before transform: {image.dtype}, {image.shape}")
            image = self.transform(image)
            print(f"Type after transform: {image.dtype}, {image.shape}")
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask


transform = Compose([
    Resize((256, 256)),
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

