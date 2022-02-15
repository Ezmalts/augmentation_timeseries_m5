import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset
from IPython.display import clear_output
from tqdm import tqdm


class Decoder(nn.Module):
    def __init__(self, seq_len, input_dim=64, n_features=1):
        super(Decoder, self).__init__()

        self.seq_len, self.input_dim = seq_len, input_dim
        self.hidden_dim, self.n_features =  input_dim, n_features
        
        self.rnn1 = nn.LSTM(
          input_size=n_features,
          hidden_size=input_dim,
          num_layers=3,
          batch_first=True,
          dropout = 0.3
        )
        
        
      
        self.output_layer = nn.Linear(self.hidden_dim, n_features)

    def forward(self, x, input_hidden, input_cell):
       
       
        x = x.reshape((-1, 1, self.n_features))
        #print("decode input",x.size())
             

        x, (hidden_n, cell_n) = self.rnn1(x,(input_hidden,input_cell))
        #print(x.shape)
        x = self.output_layer(x)
        #print(f'forward decoder: {x.shape}')
        return x, hidden_n, cell_n
    
    
    
class Encoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64):
        super(Encoder, self).__init__()

        self.seq_len, self.n_features = seq_len, n_features
        self.embedding_dim, self.hidden_dim = embedding_dim,  embedding_dim
        self.num_layers = 3
        self.rnn1 = nn.LSTM(
          input_size=n_features,
          hidden_size=self.hidden_dim,
          num_layers=3,
          batch_first=True,
          dropout = 0.3
        )
   
    def forward(self, x):
       
        x = x.reshape((-1, self.seq_len, self.n_features))
        #print(x.shape)
        
        h_1 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_dim))
         
        
        c_1 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_dim))
              
        x, (hidden, cell) = self.rnn1(x,(h_1, c_1))
        
        
        return hidden , cell 
    
    
    
class Seq2Seq(nn.Module):

    def __init__(self, seq_len, n_features, embedding_dim=64,output_length=1):
        super(Seq2Seq, self).__init__()

        self.encoder = Encoder(seq_len, n_features, embedding_dim)
        self.n_features = n_features
        self.output_length = output_length
        self.decoder = Decoder(seq_len, embedding_dim, n_features)
        

    def forward(self, x, prev_y):
        hidden,cell = self.encoder(x)
        
        #Prepare place holder for decoder output
        targets_ta = []
        #prev_output become the next input to the LSTM cell
        dec_input = prev_y
        

        
        #print(f'dec_input_init: {dec_input.shape}')
        #itearate over LSTM - according to the required output days
        for out_days in range(self.output_length) :
            
          
            prev_x,prev_hidden,prev_cell = self.decoder(dec_input,hidden,cell)
            hidden,cell = prev_hidden,prev_cell
            
            #print(prev_x)
            #prev_x = prev_x[:,:,0:1]
            #print("preve x shape is:",prev_x.size())
           
            dec_input = prev_x
            #print(f'dec_input: {dec_input.shape}')
            
            targets_ta.append(prev_x.reshape(-1, self.n_features))
           
            
        
        
        targets = torch.stack(targets_ta)
        
        targets = targets.reshape(-1, self.output_length, self.n_features)

        return targets
    
class DatasetTs(Dataset):
    def __init__(self, X, y = None):
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        assert index < len(self), 'index {} is out of bounds'.format(index)
        X = self.X[index]
        
        X = Variable(torch.tensor(X))

        
        if self.y is not None:
            return X, Variable(torch.tensor(self.y[index]))

        return X