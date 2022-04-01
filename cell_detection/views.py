import io
import os
import json
import base64

import torch
from torch import nn
from torchvision import models
from torchvision import transforms 
from PIL import Image

from django.conf import settings
from django.shortcuts import render
from multiprocessing import context
from .forms import ImageUploadForm
 
FILE = 'models/detection_model_optimized.pt' 
model = models.resnet50(pretrained = True)
# model.fc = torch.nn.Linear(2048, 2, bias = True)

model.fc = nn.Sequential(
    nn.Linear(2048,1024),
    nn.ReLU(),
    nn.Linear(1024,2)
  )   
checkpoint = torch.load(FILE, map_location = 'cpu')
model.load_state_dict(checkpoint)
model.eval()


json_path = os.path.join(settings.STATIC_ROOT,"class_index.json")
json_mapping = json.load(open(json_path))

def transform_image(image_bytes):
    my_transforms =  transforms.Compose([transforms.RandomRotation(30),
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomVerticalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456,0.406],
                                                         [0.229, 0.224,0.225])])

            
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)


def predict(image_bytes):
    tensor = transform_image(image_bytes)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    return json_mapping[predicted_idx]

def index(request):
    image_uri = None
    predicted_label = None

    if request.method == 'POST':
         
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image = form.cleaned_data['image']
            image_bytes = image.file.read()
            
            encoded_img = base64.b64encode(image_bytes).decode('ascii')
            image_uri = 'data:%s;base64,%s' % ('image/jpeg', encoded_img)
 
            try:
                predicted_label = predict(image_bytes)

            except RuntimeError as re :
                print(re)
    else :
        
        form = ImageUploadForm()

    context = {
        'form': form,
        'image_uri': image_uri,
        'predicted_label': predicted_label,
    }

    return render(request,'cell_detection/index.html', context)



            

