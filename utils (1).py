from torchvision import transforms, datasets
import torch
import argparse
import copy
import os
import json


def load_names(filename):
    with open(filename) as f:
        category_names = json.load(f)
    return category_names

#creating function to save the checkpoint as in classroom excercises 
def save_checkpoint(model, optimizer, args, classifier):   
    checkpoint = {'arch': args.arch, 
                  'model': model,
                  'learning_rate': args.learning_rate,
                  'hidden_units': args.hidden_units,
                  'classifier' : classifier,
                  'epochs': args.epochs,
                  'optimizer': optimizer.state_dict(),
                  'state_dict': model.state_dict(),
                  'class_to_idx': model.class_to_idx}
    
    #does not return anything but saving the checkpoint
    torch.save(checkpoint, 'checkpoint.pth') 
    

    #building load checkpoint function as in classroom excercise
def load_checkpoint(path):
    checkpoint = torch.load(path)
    model = checkpoint['model']
    model.classifier = checkpoint['classifier']
    learning_rate = checkpoint['learning_rate']
    epochs = checkpoint['epochs']
    optimizer = checkpoint['optimizer']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    #outputs the model when loaded back in
    return model
