import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
from torchvision.datasets import ImageFolder
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
import argparse
from PIL import Image
from collections import OrderedDict
import time
from utils import save_checkpoint, load_checkpoint


def parse_args():
#create argument parser
    argumentparser = argparse.ArgumentParser(description='train.py')

#enter arguments for model with vgg13 pretrained model
    argumentparser.add_argument('data_dir', nargs='*', action="store", default="./flowers/")
    argumentparser.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth")
    argumentparser.add_argument('--gpu', dest="gpu", action="store", default="gpu")
    argumentparser.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.003)
    argumentparser.add_argument('--arch', dest="arch", action="store", default="vgg13", choices=['vgg13', 'densenet121'], type = str)
    argumentparser.add_argument('--dropout', dest = "dropout", action = "store", default = 0.2)
    argumentparser.add_argument('--epochs', dest="epochs", action="store", type=int, default=2)
    argumentparser.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=512)
    return argumentparser.parse_args()

#set parameters equal to defined parameters from above
#parser = argumentparser.parse_args()
#where = parser.data_dir
#path = parser.save_dir
#power = parser.gpu
#lr = parser.learning_rate
#structures = parser.arch
#dropout = parser.dropout
#epochs = parser.epochs
#hidden_layer = parser.hidden_units


#use GPU if availanle, otherwise use regular CPU
if torch.cuda.is_available():
    if power == 'gpu':
        device = 'cuda'
    else:
        device = 'cpu'
else:
        device = 'cpu'

 
#define train function with same principle as in jupiter notebook file and in class example
def train(model, criterion, optimizer, trainloader, validloader, epochs, gpu):
    steps = 0
    print_every = 15
    running_loss = 0
    for e in range(epochs):
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1 
            if torch.cuda.is_available() and power =='gpu':
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
            
            optimizer.zero_grad()
            
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                
                for ii, (inputs,labels) in enumerate(validloader):
                        optimizer.zero_grad()
                        if torch.cuda.is_available() and power =='gpu':
                            inputs, labels = inputs.to('cuda'), labels.to('cuda')
                        
                        
                        with torch.no_grad():    
                            logps = model.forward(inputs)
                            test_loss = criterion(logps,labels)
                            #batch_loss = criterion(logps, labels)
                    
                            #test_loss += batch_loss.item()
                            
                            #calcualte accuracy
                            ps = torch.exp(logps).data
                            top_p, top_class = ps.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += equals.type_as(torch.FloatTensor()).mean()

                #print out results for every nth given run
                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Training Loss: {:.4f}".format(running_loss/print_every),
                      "Test Loss {:.4f}".format(test_loss / len(validloader)),
                      "Accuracy: {:.4f}".format(accuracy /len(validloader)),
                     )

                #restet running loss to 0 before next iteration
                running_loss = 0

#define main function based on jupiter notebook running code and classroom exercises

def main(): 
    args = parse_args()
    
    data_dir = argumentparser.data_dir
    train_dir = data_dir + '/train'
    test_dir = data_dir + '/test'
    valid_dir = data_dir + '/valid'
    
    train_transforms = transforms.Compose([transforms.RandomRotation(30), 
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(), 
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])
    
    test_transforms = transforms.Compose([transforms.Resize(256), 
                                          transforms.CenterCrop(224), 
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])
    
    valid_transforms = transforms.Compose([transforms.Resize(256), 
                                           transforms.CenterCrop(224), 
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform = train_transforms)
    test_data = datasets.ImageFolder(test_dir, transform = test_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform = valid_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size = 32, shuffle = True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size = 32, shuffle = True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size = 32, shuffle = True)
    
   
    model = getattr(models, args.arch)(pretrained=True)
        
    for param in model.parameters():
        param.requires_grad = False
    
    #redfining the classifier as in the lecture example for each option
    if args.arch == "vgg13":
        hidden_layer = parser.hidden_units
        classifier = nn.Sequential(OrderedDict([
                                  ('fc1', nn.Linear(25088, hidden_layer)),
                                  ('drop', nn.Dropout(p=0.2)),
                                  ('relu', nn.ReLU()),
                                  ('fc2', nn.Linear(hidden_layer, 102)),
                                  ('output', nn.LogSoftmax(dim=1))]))
    
    else args.arch == "densenet121":
        hidden_layer = parser.hidden_units
        classifier = nn.Sequential(OrderedDict([
                                  ('fc1', nn.Linear(1024, hidden_layer)),
                                  ('drop', nn.Dropout(p=0.2)),
                                  ('relu', nn.ReLU()),
                                  ('fc2', nn.Linear(hidden_layer, 102)),
                                  ('output', nn.LogSoftmax(dim=1))]))
    

    # define model and save to checkpoint
    model.classifier = classifier
    criterion = nn.NLLLoss() 
    optimizer = optim.Adam(model.classifier.parameters(), lr=float(args.learning_rate))
    epochs = int(args.epochs)
    class_index = train_data.class_to_idx
    gpu = args.gpu 
    train(model, criterion, optimizer, trainloader, validloader, epochs, gpu)
    model.class_to_idx = class_index
    path = args.save_dir
    save_checkpoint(model, optimizer, args, classifier)


if __name__ == "__main__":
    main()
    
 