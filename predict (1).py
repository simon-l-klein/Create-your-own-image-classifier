import numpy as np
from torchvision import transforms, models
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import argparse
from PIL import Image
import json
import os
import random
from utils import load_checkpoint, load_names


#define parse args for train file 
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', action='store', default='checkpoint.pth')
    parser.add_argument('--top_k', dest='top_k', default='3')
    parser.add_argument('--filepath', dest='filepath', default='flowers/test/1/image_06764.jpg') #from primrose folder 
    parser.add_argument('--category_names', dest='category_names', default='cat_to_name.json')
    parser.add_argument('--gpu', action='store', default='gpu')
    #returns parse_args for model to use
    return parser.parse_args()



#defining function to process images as in classroom excercise
def process_image(image):
    #load image
    img = Image.open(image) 
   
#define transform image to fit algorithm
    adjustments = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])])
    #transform image
    image = adjustments(img)
    return image #return image

#define finction to predict class for image
def predict(image_path, model, topk=3, gpu='gpu'):
    if gpu == 'gpu':
        model = model.cuda()
    else:
        model = model.cpu()
        
    image_torch = process_image(image_path)
    #create one dimensional tensor
    image_torch = image_torch.unsqueeze_(0)
    image_torch = image_torch.float()

    #set cuda if available
    if gpu == 'gpu':
        with torch.no_grad():
            output = model.forward(image_torch.cuda())
    else:
        with torch.no_grad():
            output=model.forward(image_torch)
    
    #calculate probability for class with softmax activation
    probability = F.softmax(output.data,dim=1) 
    
    #return top probability and top class
    probs = np.array(probability.topk(topk)[0][0])
    index_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_classes = [np.int(index_to_class[each]) for each in np.array(probability.topk(topk)[1][0])]
    
    #return the top class and probability
    return probs, top_classes


#define function to run the code
def main(): 
    #load back args, checkpoint and labels
    args = parse_args()
    gpu = args.gpu
    model = load_checkpoint(args.checkpoint)
    cat_to_name = load_names(args.category_names)
    img_path = args.filepath
    
    #load probabilites calculates
    probs, classes = predict(img_path, model, int(args.top_k), gpu)
    labels = [cat_to_name[str(index)] for index in classes]
    probability = probs
    print('File: ' + img_path)
    print(labels)
    print(probability)
    
    i=0 
    while i < len(labels):
        print("{} class returned at percentage {}".format(labels[i], probability[i]))
        #next iteration
        i += 1 

if __name__ == "__main__":
    main()