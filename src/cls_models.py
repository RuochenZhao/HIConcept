import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class BiLSTM(nn.Module):
    def __init__(self, max_features = 9000, hidden_dim = 250, embedding_dim = 600):
        super().__init__()
        self.encoder = nn.Embedding(max_features, embedding_dim)
        # self.lstm = nn.LSTM(embedding_dim, hidden_dim,
        #                     num_layers=2, bidirectional=True)
        # self.linear = nn.Linear(hidden_dim * 2, 1)
        self.lstm = nn.LSTM(embedding_dim, 200,
                            num_layers=2, batch_first = True)
        self.dense1 = nn.Linear(200, hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, 1)
        self.drop = nn.Dropout(p=0.1)
        self.activation = nn.Sigmoid()

        nn.init.xavier_uniform_(self.dense1.weight)
        nn.init.xavier_uniform_(self.dense2.weight)
        self.dense1.bias.data.zero_()
        self.dense2.bias.data.zero_()
        self.init_weights()

    def init_weights(self):
        ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param.data for name, param in self.named_parameters() if 'bias' in name)
        self.encoder.weight.data.uniform_(-0.5, 0.5)
        for t in ih:
            nn.init.xavier_uniform(t)
        for t in hh:
            nn.init.orthogonal(t)
        for t in b:
            nn.init.constant(t, 0)

    def encode(self, src):
        output = self.encoder(src)
        output, _ = self.lstm(output)
        output = nn.functional.tanh(output[:, -1, :])
        output = F.relu(self.dense1(output))
        output = self.drop(output)
        return output

    def predict(self, output):
        output = self.dense2(output)
        output = self.activation(output)
        return output

    def forward(self, src):
        output = self.encode(src)
        output = self.predict(output)
        return output

class mean_layer(nn.Module):
    def __init__(self, axis):
        super().__init__()
        self.axis =axis
    def forward(self, input):
        output = torch.mean(input, axis = self.axis)
        return output

class permute_layer(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input):
        x_reshaped = input.permute(0, 2, 1)
        return x_reshaped


class CNN_cls(nn.Module):
    def __init__(self, width, height, channel):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(3, 64, 5),
                                        nn.ReLU(),
                                        nn.MaxPool2d(4, 4),
                                        nn.Conv2d(64, 64, 5),
                                        nn.ReLU(),
                                        nn.MaxPool2d(4, 4),
                                        nn.Conv2d(64, 64, 5),
                                        nn.ReLU(),
                                        nn.MaxPool2d(2, 2))

        self.classifier = nn.Sequential(nn.Flatten(), 
                                        nn.Linear(1024, 200), 
                                        nn.ReLU(),
                                        nn.Linear(200, 15), 
                                        nn.Sigmoid())
        
    def encode(self, x):
        x = self.encoder(x)
        return x

    def predict(self, x):
        x = self.classifier(x)
        return x

    def forward(self, src):
        output = self.encode(src)
        output = self.predict(output)
        return output, None

class CNN_NLP(nn.Module):
    """An 1D Convulational Neural Network for Sentence Classification."""
    def __init__(self, input_dim, embedding_dim, hidden_dim, short_sentences,
                 pretrained_embedding=None,
                 freeze_embedding=False,
                 vocab_size=None,
                 filter_sizes=[3, 4, 5],
                 num_filters=[100, 100, 100],
                 num_classes=1,
                 dropout=0.5):
        """
        The constructor for CNN_NLP class.

        Args:
            pretrained_embedding (torch.Tensor): Pretrained embeddings with
                shape (vocab_size, embed_dim)
            freeze_embedding (bool): Set to False to fine-tune pretraiend
                vectors. Default: False
            vocab_size (int): Need to be specified when not pretrained word
                embeddings are not used.
            embed_dim (int): Dimension of word vectors. Need to be specified
                when pretrained word embeddings are not used. Default: 300
            filter_sizes (List[int]): List of filter sizes. Default: [3, 4, 5]
            num_filters (List[int]): List of number of filters, has the same
                length as `filter_sizes`. Default: [100, 100, 100]
            n_classes (int): Number of classes. Default: 2
            dropout (float): Dropout rate. Default: 0.5
        """

        super(CNN_NLP, self).__init__()
        self.embed_dim = embedding_dim
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        nc = 10
        # Conv Network
        self.conv1 = nn.Conv1d(in_channels=self.embed_dim,out_channels=nc,kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=nc,out_channels=nc,kernel_size=4)
        self.conv3 = nn.Conv1d(in_channels=nc,out_channels=nc,kernel_size=5)
        # Fully-connected layer and Dropout
        if short_sentences:
            # self.classifier = nn.Sequential(nn.Flatten(start_dim = 1),
            #                 nn.Dropout(p=dropout),
            #                 nn.Linear(600, num_classes),
            #                 nn.Sigmoid())
            self.classifier = nn.Sequential(nn.Dropout(p=dropout),
                            nn.Linear(nc, num_classes),
                            nn.Sigmoid())
        else:
            self.classifier = nn.Sequential(nn.Dropout(p=dropout),
                                nn.Linear(nc, num_classes),
                                nn.Sigmoid())

    def encode(self, input_ids):
        """Perform a forward pass through the network.

        Args:
            input_ids (torch.Tensor): A tensor of token ids with shape
                (batch_size, max_sent_length)

        Returns:
            logits (torch.Tensor): Output logits with shape (batch_size,
                n_classes)
        """

        # Get embeddings from `input_ids`. Output shape: (b, max_len, embed_dim)
        x_embed = self.embedding(input_ids).float()

        # Permute `x_embed` to match input shape requirement of `nn.Conv1d`.
        # Output shape: (b, embed_dim, max_len)
        x_reshaped = x_embed.permute(0, 2, 1)

        # Apply CNN and ReLU. Output shape: (b, num_filters[i], L_out)
        x_conv1 = F.relu(self.conv1(x_reshaped))
        x_conv2 = F.relu(self.conv2(x_conv1))
        x_conv3 = F.relu(self.conv3(x_conv2))

        output = torch.mean(x_conv3, axis = -1) #bs, nc
        
        return output, x_conv3
    
    def forward(self, input_ids):
        output, _ = self.encode(input_ids)
        output = self.classifier(output)
        return output