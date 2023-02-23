import torch
import torch.nn as nn
from torchvision import transforms, datasets
import numpy as np
import os
import time
import copy
import matplotlib.pyplot as plt
from PIL import Image
from resnet_feature_extracter import Img2Vec
import pdb
from tqdm import tqdm
from torchvision import models
import json

#Antoencoder definition
class EncoderRNN(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, bidirectional):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, 
                            dropout=0.2, bidirectional=bidirectional)
        self.relu = nn.ReLU()

        # initialize weights
        #nn.init.xavier_uniform_(self.lstm.weight_ih_l0, gain=np.sqrt(2))
        #nn.init.xavier_uniform_(self.lstm.weight_hh_l0, gain=np.sqrt(2))
        nn.init.orthogonal_(self.lstm.weight_ih_l0, gain=np.sqrt(2))
        nn.init.orthogonal_(self.lstm.weight_hh_l0, gain=np.sqrt(2))

    def forward(self, x):
        # set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # forward propagate lstm
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        return out[:, -1, :].unsqueeze(1)


class DecoderRNN(nn.Module):

    def __init__(self, hidden_size, output_size, num_layers, bidirectional):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(hidden_size, output_size, num_layers, batch_first=True,
                            dropout=0.2, bidirectional=bidirectional)

        # initialize weights
        #nn.init.xavier_uniform_(self.lstm.weight_ih_l0, gain=np.sqrt(2))
        #nn.init.xavier_uniform_(self.lstm.weight_hh_l0, gain=np.sqrt(2))
        nn.init.orthogonal_(self.lstm.weight_ih_l0, gain=np.sqrt(2))
        nn.init.orthogonal_(self.lstm.weight_hh_l0, gain=np.sqrt(2))
        
    def forward(self, x):
        # set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.output_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.output_size).to(device)

        # forward propagate lstm
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        return out


class AutoEncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bidirectional=False):
        super(AutoEncoderRNN, self).__init__()
        self.encoder = EncoderRNN(input_size, hidden_size, num_layers, bidirectional)
        self.decoder = DecoderRNN(hidden_size, input_size, num_layers, bidirectional)

    def forward(self, x,sequence_length):
        encoded_x = self.encoder(x).expand(-1, sequence_length, -1)
        decoded_x = self.decoder(encoded_x)

        return encoded_x, decoded_x


#Data preparation
class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path
      
class CVDLPT_lstm_autoencoder_Class():

    def __init__(self,LSTM_model,Img2Vec):

       #Feature vector extractor
       self.extractor = Img2Vec()

       #LSTM model
       self.LSTM_model=LSTM_model

       return


    def readImages(self,data_dir,batch_size):
       self.data_dir = data_dir

       self.data_transforms = transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])

       self.image_datasets = {x: datasets.ImageFolder(os.path.join(self.data_dir, x), 
                                          transform=self.data_transforms) for x in ['Raw']}
       self.data_loaders = {x: torch.utils.data.DataLoader(self.image_datasets[x], 
                                               batch_size=batch_size, shuffle=False) for x in ['Raw']}
       self.dataset_sizes = {x: len(self.image_datasets[x]) for x in ['Raw']}

       return self.data_loaders


    def encode(self,sequence_length,input_size,hidden_size):
       for phase in ['Raw']:
         self.LSTM_model.eval()
         for counter, [inputs,k] in enumerate(self.data_loaders[phase]):
             if (counter+1)*sequence_length <= len(self.data_loaders[phase].dataset.samples):
                fv_filenameFirst=self.data_loaders[phase].dataset.samples[counter*sequence_length][0]
                fv_filenameLast=self.data_loaders[phase].dataset.samples[(counter+1)*sequence_length][0]

                if (len(k) == sequence_length):
                  inputs = self.extractor.get_vec(inputs)
                  
                  inputs = inputs.reshape(-1, sequence_length, input_size).to(device)

                  with torch.set_grad_enabled(False):
                    encoded_x, outputs = self.LSTM_model(inputs,sequence_length)

                    inv_idx = torch.arange(sequence_length - 1, -1, -1).long()

                    code=encoded_x[0].reshape(sequence_length*hidden_size).cpu().numpy()

       return code


'''
#----------------
#Hyper parameters
sequence_length = 32
input_size = 2048
hidden_size = 32
num_layers = 2
batch_size = sequence_length # set to the number of images of a seqence # 36
learning_rate = 0.01
'''

#Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

'''
## Set the model and load the model weights
LSTM_model = AutoEncoderRNN(input_size, hidden_size, num_layers)
LSTM_model.load_state_dict(torch.load('./lstm_autoencoder_model_SL32_HS32.pt'))
LSTM_model = LSTM_model.to(device)

## Read the latent space representation of the video of the Exercises
File_To_Read="Compressed_Seq_SL32_HS32.json"
with open(File_To_Read, 'r', encoding='utf-8') as content:
     VectorSet=json.load(content)

## Create an instantation of the autoencoder class
CVDLPT_AE=CVDLPT_lstm_autoencoder_Class(LSTM_model,Img2Vec)

## Read images from data_dir, images should be stored in a folder called Raw inside the data_dir folder
data_dir = '../Kinetics/kinetics-dataset/k400images'
data_loaders = CVDLPT_AE.readImages(data_dir,batch_size)

## Encode sequence of images and produce x_enc, must be 32 images only
x_enc=CVDLPT_AE.encode(sequence_length,input_size,hidden_size)

## Measure euclidean distance between x_enc and all latent vector representations of al training dataset and return
## the minimum euclidean distance the name of the Exercise
BestEucDist=1000
BestExercise=''
for x in VectorSet.keys():
   EucDist=np.linalg.norm(np.array(VectorSet[str(x)])-x_enc)
   if (EucDist <= BestEucDist):
         BestEucDist=EucDist
         BestExercise=x

print(f"Best Exercise = {BestExercise.split('_')[0]}")
print(f"Best Euclidean Distance = {BestEucDist}")

'''
