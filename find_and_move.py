import os
import shutil
import numpy as np
from PIL import Image
from torchvision import models, transforms
from torch import nn
from annoy import AnnoyIndex
images_folder = './all'
images = os.listdir(images_folder)

def search_and_move_similar_images(images_folder, threshold=0.1, move_threshold=0.85):
    # Initialize the ResNet18 model and Annoy index
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)
    model.fc = nn.Identity()
    model.eval()
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    annoy_index = AnnoyIndex(512, 'angular')
    annoy_index.load('paintings_index.ann')

    find_similarities_folder = 'find_similarities'
    target_images = [os.path.join(find_similarities_folder, file) for file in os.listdir(find_similarities_folder) if file.endswith(('.jpg', '.png'))]

    # Create the separate folders for each target image
    for i, target_image_path in enumerate(target_images):
        destination_folder = f'folder_{i+1}'  # Create a separate folder for each target image
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)

        # Load the target image and obtain its embedding
        target_img = Image.open(target_image_path)
        input_tensor = transform(target_img).unsqueeze(0)
        output_tensor = model(input_tensor)

        # Find similar images
        similar_image_indices = annoy_index.get_nns_by_vector(output_tensor[0], 24)  # Get 24 nearest neighbors

        # Move similar images to the separate folder based on similarity threshold
        for j in similar_image_indices:
            img_path = os.path.join(images_folder, images[j])
            similar_img = Image.open(img_path)
            input_tensor = transform(similar_img).unsqueeze(0)
            similar_output_tensor = model(input_tensor)

            # Calculate cosine similarity
            similarity = (output_tensor[0].detach().numpy() @ similar_output_tensor[0].detach().numpy()) / (np.linalg.norm(output_tensor[0].detach().numpy()) * np.linalg.norm(similar_output_tensor[0].detach().numpy()))

            if similarity > move_threshold:
                file_name = os.path.basename(img_path)
                shutil.move(img_path, os.path.join(destination_folder, file_name))

# Example usage:
images_folder = 'all'
search_and_move_similar_images(images_folder, threshold=0.1, move_threshold=0.85)
