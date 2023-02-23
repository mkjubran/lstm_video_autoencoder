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
import CVDLPT_lstm_autoencoder_CLass as CVDLPT

#----------------
#Hyper parameters
sequence_length = 32
input_size = 2048
hidden_size = 32
num_layers = 2
batch_size = sequence_length # set to the number of images of a seqence # 36
learning_rate = 0.01

#Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

## Set the model and load the model weights
LSTM_model = CVDLPT.AutoEncoderRNN(input_size, hidden_size, num_layers)
LSTM_model.load_state_dict(torch.load('./lstm_autoencoder_model_SL32_HS32.pt'))
LSTM_model = LSTM_model.to(device)

## Read the latent space representation of the video of the Exercises
File_To_Read="Compressed_Seq_SL32_HS32.json"
with open(File_To_Read, 'r', encoding='utf-8') as content:
     VectorSet=json.load(content)

## Create an instantation of the autoencoder class
CVDLPT_AE=CVDLPT.CVDLPT_lstm_autoencoder_Class(LSTM_model,Img2Vec)

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

