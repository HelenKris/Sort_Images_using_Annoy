import os
import torch
from torchvision import models, transforms
from PIL import Image, ImageDraw
import torch.nn as nn
from annoy import AnnoyIndex

images_folder = './all'
images = os.listdir(images_folder)

weights = models.ResNet18_Weights.IMAGENET1K_V1
model = models.resnet18(weights=weights)
model.fc = nn.Identity()

model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

annoy_index = AnnoyIndex(512, 'angular')
annoy_index.load('paintings_index.ann')

image_grid = Image.new('RGB', (1000, 1000)) # Поставить размер входящего изображения

for i in range(len(images)):
    image = Image.open(os.path.join(images_folder,images[i]))
    input_tensor = transform(image).unsqueeze(0)

    if input_tensor.size()[1] == 3:
        output_tensor = model(input_tensor)
        # annoy_index.add_item(i, output_tensor[0])

    nns = annoy_index.get_nns_by_vector(output_tensor[0], 24 ) # Будет изображено 24 ближайших соседа к исходной фоторграфии +1 исходная
    image = image.resize((200,200))
    image_draw = ImageDraw.Draw(image)
    image_draw.rectangle([(0,0), (199,199)], outline = 'red', width = 8 )
    image_grid.paste(image, ((0,0)))

    for j in range(24):
        search_image = Image.open(os.path.join(images_folder,images[nns[j]]))
        search_image = search_image.resize((200,200))
        image_grid.paste(search_image, (200 * ((j+1)%5), 200 * ((j+1)//5)))

    image_grid.save(f'ImageDump/image_{i}.png')
