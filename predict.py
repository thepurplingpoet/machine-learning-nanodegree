import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn
from torch import optim
from torchvision import datasets, models, transforms
import torch.nn.functional as F
import torch.utils.data
import pandas as pd
from collections import OrderedDict
from PIL import Image
import argparse
import json


parser = argparse.ArgumentParser (description = "Parser of prediction script")

parser.add_argument ('image_dir', help = 'Path of image', type = str)
parser.add_argument ('load_dir', help = 'Path of checkpoint', type = str)
parser.add_argument ('--top_k', help = 'Top K most likely classes', type = int)
parser.add_argument ('--category_names', help = 'JSON file which maps from categories to real', type = str)
parser.add_argument ('--gpu', help = "Toggle for GPU", action="store_true")


args = parser.parse_args ()
image_dir = args.image_dir
load_dir = args.load_dir
if args.top_k:
    topk = args.top_k
else:
    topk = 5
    
if args.gpu:
    device = 'cuda'
else:
    device = 'cpu'
    
def load_checkpoint(load_dir):
    
    checkpoint = torch.load(load_dir)
    model = models.vgg16(pretrained=True)         
    exec("model = models.{}(pretrained=True)".format(checkpoint['arch']))
        
    for param in model.parameters():
        param.requires_grad = False
    
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])

    
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    size = 224
    im = Image.open(image)
    width, height = im.size
    
    if height > width:
        height = int(max(height * size / width, 1))
        width = int(size)
    else:
        width = int(max(width * size / height, 1))
        height = int(size)
        
    resized_image = im.resize((width, height))
        
    x0 = (width - size) / 2
    y0 = (height - size) / 2
    x1 = x0 + size
    y1 = y0 + size
    
    cropped_image = im.crop((x0, y0, x1, y1))
    np_image = np.array(cropped_image) / 255.
    np_image -= np.array ([0.485, 0.456, 0.406]) 
    np_image /= np.array ([0.229, 0.224, 0.225])
    
    np_image = np_image.transpose ((2,0,1))
    
    return np_image

def predict(image_path, model, topk, device):
    
    image = process_image (image_path) 
 
    im = torch.from_numpy (image).type (torch.FloatTensor)
    model.to(device)
    im.to(device)
    im = im.unsqueeze (dim = 0) 
        
    with torch.no_grad ():
        output = model.forward (im)
    output_probability = torch.exp (output) 
    
    probability, indices = output_probability.topk (topk)
    probs = probability.numpy () 
    indices = indices.numpy () 
    
    probs = probs.tolist () [0] 
    indices = indices.tolist () [0]
    
    
    mapping = {
                val: key for key, val in
                model.class_to_idx.items()
                }
    
    classes = [mapping [item] for item in indices]
    classes = np.array (classes) 
    
    return probs, classes

model = load_checkpoint(load_dir)
probs, classes = predict (image_dir, model, topk, device)

if args.category_names:
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
else:
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
class_names = [cat_to_name [item] for item in classes]

for l in range (topk):
     print("Rank: {}/{}.. ".format(l+1, topk),
            "Flower name: {}.. ".format(class_names [l]),
            "Probability: {:.3f}..% ".format(probs [l]*100),
            )
