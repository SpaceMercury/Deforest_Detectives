import torch
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader
from data_file import ImageMaskDataset  # Ensure this is set up properly
import matplotlib.pyplot as plt

def main():
    # Create a model
    model = models.segmentation.fcn_resnet50(pretrained=True)
    model.eval()  # Set the model to evaluation mode if not training

    # Dataset and DataLoader
    image_directory = '/Users/polfuentes/TUM_Makeathon/archive/train2/images'
    mask_directory = '/Users/polfuentes/TUM_Makeathon/archive/train2/masks'

    # Assuming transform and dataset setup includes necessary transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])
    dataset = ImageMaskDataset(image_directory, mask_directory, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            output = model(images)['out']
            _, predicted_masks = torch.max(output, 1)  # Get the most likely class for each pixel

            # Visualization
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.imshow(images[0].cpu().permute(1, 2, 0))  # Assuming the image is a torch.Tensor
            plt.title('Original Image')
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(predicted_masks[0].cpu(), cmap='gray')
            plt.title('Segmentation Output')
            plt.axis('off')

            plt.show()

            # Optionally, break after the first batch for demonstration
            break

if __name__ == "__main__":
    main()
