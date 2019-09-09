import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torchvision.models as models
from PIL import Image
import json
from collections import OrderedDict
import argparse

parser = argparse.ArgumentParser (description = "Argument parser for training")

parser.add_argument ('data_dir', help = 'Data directory', type = str)
parser.add_argument ('--save_dir', help = 'Directory in which model will be saved', type = str)
parser.add_argument ('--arch', help = 'Default is VGG13', type = str)
parser.add_argument ('--learning_rate', help = 'Learning rate, default is 0.001', type = float)
parser.add_argument ('--hidden_units', help = 'Hidden units in Classifier. Default is 2048', type = int)
parser.add_argument ('--epochs', help = 'Number of epochs', type = int)
parser.add_argument ('--gpu', help = "Toggle for GPU", action="store_true")

args = parser.parse_args()

if args.gpu:
    device = 'cuda'
else:
    device = 'cpu'

if args.learning_rate:
    learning_rate = args.learning_rate
else:
    learning_rate = 0.001
    
data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

validation_transforms = transforms.Compose([transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], 
                                                                 [0.229, 0.224, 0.225])])


test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])


# TODO: Load the datasets with ImageFolder
train_images = datasets.ImageFolder (train_dir, transform = train_transforms)
validation_images = datasets.ImageFolder (valid_dir, transform = validation_transforms)
test_images = datasets.ImageFolder (test_dir, transform = test_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
train_loader = torch.utils.data.DataLoader(train_images, batch_size = 72, shuffle = True)
validation_loader = torch.utils.data.DataLoader(validation_images, batch_size = 32,shuffle = True)
test_loader = torch.utils.data.DataLoader(test_images, batch_size = 32, shuffle = True)


with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

if args.arch: 
        exec("model = models.{}(pretrained=True)".format(args.arch))
        modelname = args.arch
else:
    model = models.vgg16(pretrained=True)
    modelname = "vgg16"
for param in model.parameters():
    param.requires_grad = False

hidden = 4096    
if args.hidden_units :
    hidden = args.hidden_units
    
classifier = nn.Sequential  (OrderedDict ([
                            ('fc1', nn.Linear (25088, hidden)),
                            ('relu1', nn.ReLU ()),
                            ('dropout1', nn.Dropout(p=0.5)),
                            ('fc2', nn.Linear(hidden, 102, bias=True)),
                            ('output', nn.LogSoftmax(dim=1))
                            ]))

def validation(model, loader, criterion):
    test_loss = 0
    accuracy = 0
    
    for inputs, labels in (loader):
        
        inputs, labels = inputs.to(device), labels.to(device)
        
        output = model.forward(inputs)
        test_loss += criterion(output, labels).item()
        
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return test_loss, accuracy
model.classifier = classifier
criterion = nn.NLLLoss ()
optimizer = optim.Adam (model.classifier.parameters (), lr = learning_rate)
model.to(device)

if args.epochs:
    epochs = args.epochs
else:
    epochs = 5
    
steps = 0
print_every = 20

for e in range (epochs): 
    running_loss = 0
    for inputs, labels in (train_loader):
        steps += 1
        
        inputs, labels = inputs.to(device), labels.to(device)
    
        optimizer.zero_grad () 
    
        outputs = model.forward (inputs) 
        loss = criterion (outputs, labels) 
        loss.backward () 
        optimizer.step ()  
    
        running_loss += loss.item () 
    
        if steps % print_every == 0:
            model.eval () 
            
            with torch.no_grad():
                valid_loss, accuracy = validation(model, validation_loader, criterion)
            
            print("Epoch: {}/{}.. ".format(e+1, epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                  "Valid Loss: {:.3f}.. ".format(valid_loss/len(validation_loader)),
                  "Valid Accuracy: {:.3f}%".format(accuracy/len(validation_loader)*100))
            
            running_loss = 0
            
            model.train()  
            
            
            
# TODO: Save the checkpoint 
model.class_to_idx = train_images.class_to_idx 
#creating dictionary 
checkpoint = {'classifier': model.classifier,
              'state_dict': model.state_dict (),
              'class_to_idx': model.class_to_idx,
              'arch':modelname
             }        

if args.save_dir:
    torch.save (checkpoint, args.save_dir + '/checkpoint.pth')
else:
    torch.save (checkpoint, 'checkpoint.pth')        