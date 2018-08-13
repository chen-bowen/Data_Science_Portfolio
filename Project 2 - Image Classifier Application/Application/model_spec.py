import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from workspace_utils import active_session
from PIL import Image
import json
import argparse
from collections import OrderedDict # use dict, but we have to keep the order
import matplotlib.pyplot as plt

# ================= Train Model Functions =====================
def train_model(hyperparameters, data_dir, save_dir, device):
    """Train the neural network, called in main function utilize the following helper functions,
    """
    
    model_init, resize_aspect = get_model(hyperparameters['architecture'])
    image_dataset = loadImageData(resize_aspect, data_dir) 
    model_spec = buildNeuralNetwork(model_init, hyperparameters, data_dir, device) 
    
    for e in range(hyperparameters['epochs']):
        model_spec['model'].train()
        running_loss = 0 # the loss for every batch
        
        for i, train_batch in enumerate(image_dataset['trainloader']): # minibatch training
            
            # send the inputs labels to the tensors that uses the specified devices
            inputs, labels = tuple(map(lambda x: x.to(device), train_batch))
            model_spec['optimizer'].zero_grad() # clear out previous gradients, avoids accumulations
            
            # Forward and backward passes
            try:
                predictions,_ = model_spec['model'].forward(inputs)
                
            except:
                predictions = model_spec['model'].forward(inputs)
                
            loss = model_spec['criterion'](predictions, labels)
            loss.backward()
            model_spec['optimizer'].step()
            # calculate the total loss for 1 epoch of training
            running_loss += loss.item()
            
            # print the loss every .. batches
            if i % hyperparameters['print_every'] == 0:
                model_spec['model'].eval() # set to evaluation mode
                train_accuracy = evaluate_performance(model_spec['model'], 
                                                      image_dataset['trainloader'], 
                                                      model_spec['criterion']) # see evaluate function below
                
                validate_accuracy = evaluate_performance(model_spec['model'], 
                                                         image_dataset['validloader'],
                                                         model_spec['criterion'])
                
                print("Epoch: {}/{}... :".format(e+1, hyperparameters['epochs']),
                      "Loss: {:.4f},".format(running_loss/hyperparameters['print_every']),
                      "Training Accuracy:{: .4f} %,".format(train_accuracy * 100),
                      "Validation Accuracy:{: .4f} %".format(validate_accuracy * 100)
                     )
                running_loss = 0
                model_spec['model'].train()
                
    saveModel(image_dataset, model_spec['model'], model_spec['classifier'], save_dir)
    return model_spec['model']

def get_model(architecture):
    # set model architecture
    if architecture == 'inception_v3':
        model_init = models.inception_v3(pretrained=True)
        model_init.arch = 'inception_v3'
        resize_aspect = [320, 299]

    elif architecture == 'densenet161':
        model_init = models.densenet161(pretrained=True)
        model_init.arch = 'densenet161'
        resize_aspect = [256, 224]
    
    elif architecture == 'vgg19':
        model_init = models.vgg19(pretrained=True)
        model_init.arch = 'vgg19'
        resize_aspect = [256, 224]
    
    return model_init, resize_aspect

def loadImageData(resize_aspect, data_dir):
    """Input: 
            resize_aspect - depends on the architecture
            data_dir - directory of all image data"""
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    # Define transforms for the training, validation, and testing sets, using data augumentations on training set,
    # Inception_v3 has input size 299x299
    
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(resize_aspect[1]),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])

    validation_transforms = transforms.Compose([transforms.Resize(resize_aspect[1]),
                                                transforms.CenterCrop(resize_aspect[1]),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406], 
                                                                    [0.229, 0.224, 0.225])])


    test_transforms = transforms.Compose([transforms.Resize(resize_aspect[1]),
                                          transforms.CenterCrop(resize_aspect[1]),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])

    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    validation_data = datasets.ImageFolder(valid_dir, transform=validation_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(validation_data, batch_size= 32)
    testloader = torch.utils.data.DataLoader(test_data, batch_size= 32)
    # label mapping
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    
    image_dataset = {'train': train_data, 'test': test_data, 'validate': validation_data, 
                     'trainloader':trainloader, 'validloader':validloader, 'testloader': testloader, 
                     'mapping': cat_to_name}
    return  image_dataset

def buildNeuralNetwork(model, hyperparameters,  data_dir, device = 'cuda'):
    """Builds the transfer learning network according to the given architecture
    """
    # turns off gradient
    for param in model.parameters():
        param.requires_grad = False
        
    # input units mapping:
    input_units = {'inception_v3': 2048, 'densenet161': 2208, 'vgg19': 25088}
        
   # rebuild last layer
    classifier = nn.Sequential(OrderedDict([
                                            ('fc1', nn.Linear(input_units[model.arch],
                                                              hyperparameters['hidden_units'])),
                                            ('relu1', nn.ReLU()),
                                            ('dropout1', nn.Dropout(hyperparameters['dropout_prob'])),
                                            ('fc2', nn.Linear(hyperparameters['hidden_units'],
                                                              102)),
                                            ('output', nn.LogSoftmax(dim=1))
                                           ]))
    # Attach the feedforward neural network, adjust for nameing conventions
    # Define criteria and loss
    criterion = nn.NLLLoss()
    if model.arch == 'inception_v3':
        model.fc = classifier
        optimizer = optim.Adam(model.fc.parameters(), lr = hyperparameters['learning_rate'])
        
    else:
        model.classifier = classifier
        optimizer = optim.Adam(model.classifier.parameters(), lr = hyperparameters['learning_rate'])
        
    # Important: Send model to use gpu cuda
    model = model.to(device)
    model_spec = {'model': model, 'criterion': criterion,
                  'optimizer': optimizer, 'classifier':classifier}
    
    return model_spec

def evaluate_performance(model, dataloader,criterion, device = 'cuda'):
     # Evaluate performance for all batches in an epoch
    performance = [evaluate_performance_batch(model, i, criterion) for i in iter(dataloader)]  
    correct, total = list(map(sum, zip(*performance)))
    return correct/total
    
def evaluate_performance_batch(model,batch, criterion, device = 'cuda'):
    """Evaluate performance for a single batch"""
    with torch.no_grad():
        images, labels = tuple(map(lambda x: x.to(device), batch))
        predictions = model.forward(images)
        _, predict = torch.max(predictions, 1)
        
        correct = (predict == labels).sum().item()
        total  = len(labels)
        
    return correct, total
        
def saveModel(image_dataset, model, classifier, save_dir):
    # Saves the pretrained model
    with active_session():
        check_point_file = save_dir + model.arch +  '_checkpoint.pth'
    model.class_to_idx =  image_dataset['train'].class_to_idx

    checkpoint_dict = {
        'architecture': 'inception_v3',
        'class_to_idx': model.class_to_idx, 
        'state_dict': model.state_dict(),
        'classifier': classifier
    }
    torch.save(checkpoint_dict, check_point_file)
    print("Model saved")
    return None

# ================= Predict Functions =====================

def predict(image_path, checkpoint_path, category_names, device, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # Implement the code to predict the class from an image file
    model, resize_aspect = load_model_checkpoint(checkpoint_path)
    model.eval()
    image = process_image(image_path, resize_aspect)
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)

    # use forward propagation to obtain the class probabilities
    image = torch.tensor(image, dtype= torch.float).unsqueeze(0).to(device)
    predict_prob_tensor = torch.exp(model.forward(image)) # convert log probabilities to real probabilities
    predict_prob = predict_prob_tensor.cpu().detach().numpy()[0] # change into numpy array
    
    # Find the correspoinding top k classes
    top_k_idx =  predict_prob.argsort()[-topk:][::-1]
    probs =  predict_prob[top_k_idx]
    classes = np.array(list(range(1, 102)))[top_k_idx]
    visualize_pred(image, model, probs, classes, cat_to_name, topk)
    
    return probs, classes

def load_model_checkpoint(path):
    """Load model checkpoint given path"""
    checkpoint = torch.load(path, map_location={'cuda:0': 'cpu'})
    model, resize_aspect = get_model(checkpoint['architecture'])
    if model.arch == 'inception_v3':
        model.fc = checkpoint['classifier']
    else: 
        model.classifier = checkpoint['classifier']
        
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model, resize_aspect
    
def process_image(image, resize_aspect):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # Process a PIL image for use in a PyTorch model
    im = Image.open(image)
    
    # resize image to 320 on the shortest side
    size = (resize_aspect[0], resize_aspect[0])
    im.thumbnail(size)
    
    # crop out 299 portion in the center
    width, height = im.size
    left = (width - resize_aspect[1])/2
    top = (height - resize_aspect[1])/2
    right = (width + resize_aspect[1])/2
    bottom = (height + resize_aspect[1])/2
    im = im.crop((left, top, right, bottom))
    
    # normalize image
    np_image = np.array(im)
    im_mean = np.array([0.485, 0.456, 0.406])
    im_sd = np.array([0.229, 0.224, 0.225])
    np_image = (np_image/255 - im_mean)/im_sd
    
    # transpose the image
    np_image = np_image.T
    return np_image

def imshow2(image, ax=None, title=None):
    """Returns the original image after preprocessing"""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
   
    #plt.suptitle(title)
    ax.imshow(image)

    return ax

# Display an image along with the top 5 classes
def visualize_pred(image, model, probs, classes, cat_to_name, topk):
    """ Visualize the top k probabilities an image is predicted as"""
    im = process_image(image) 
    flower_names = [cat_to_name[str(x)] for x in classes]
    
    # Build subplots above
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,10))
    # set axis settings top
    imshow2(im, ax =ax1)
    ax1.set_title(cat_to_name[image.split('/')[2]])
    # set axis settings bottom
    ax2.barh(np.arange(1, topk + 1), probs)
    ax2.set_yticks(np.arange(1, topk + 1))
    ax2.set_yticklabels(flower_names) 
    ax2.set_aspect(0.187)
    ax2.set_xlim(0,1)
    return None
    
#=================== get input args train / predict ======================
def get_input_args_train():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_directory', type=str, default = None,
                    help="data directory")
    parser.add_argument('--save_dir', type=str, default='checkpoints/',
                        help='save checkpoints to directory')
    parser.add_argument('--arch', type=str, default='inception_v3',
                        help='model architecture')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate, default 0.001')
    parser.add_argument('--hidden_units', type=int, default=500,
                        help='hidden units, default 500')
    parser.add_argument('--print_every', type=int, default=20,
                        help='print every iterations')    
    parser.add_argument('--dropout_prob', type=int, default=0.1,
                        help='print every iterations')   
    parser.add_argument('--epochs', type=int, default=15,
                        help='epochs, default 15')
    parser.add_argument('--gpu', action='store_true',
                        default= 'cuda', help='to cuda gpu')

    return parser.parse_args()

def get_input_args_predict():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('path_to_image', type=str, default=None,
                    help='image file to predict')

    parser.add_argument('checkpoint', type=str, default='checkpoints/inception_v3_checkpoint.pth',
                    help='path to checkpoint')

    parser.add_argument('--topk', type=int, default=5,
                        help='return top k most likely classes the image belongs to')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json',
                        help='class names mapping')
    parser.add_argument('--gpu', default='cuda',
                        help='use cuda')

    return parser.parse_args()
