import torch
from torch import nn, Tensor
import torch.nn.functional as F
import numpy as np
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer #1.8.1
from torch.utils.data import Dataset
import re

class BiLSTM(nn.Module):
    def __init__(self, max_features = 9000, hidden_dim = 250, embedding_dim = 600):
        super().__init__()
        self.encoder = nn.Embedding(max_features, embedding_dim)=
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

# -----------------------------------------------------------

class TransformerModel(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.00, max_len: int = 512, n_classes = 2):
        super().__init__()
        self.model_type = 'Transformer'
        self.encoder = nn.Embedding(ntoken, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model = d_model, nhead = nhead, dim_feedforward = d_hid, batch_first = True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.d_model = d_model
        if n_classes ==2:
            self.decoder = torch.nn.Linear(d_model, 1)  # 0=neg, 1=pos
        else:
            self.decoder = torch.nn.Linear(d_model, n_classes)
        self.max_len = max_len
        self.classifier = nn.Sequential(self.decoder, 
                                        nn.Sigmoid())

    def encode(self, src: Tensor, src_mask: Tensor) -> Tensor:
        src = self.encoder(src) 
        src = src.reshape(-1, self.max_len, self.d_model) #(seq, batch, feature).
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src) # bs, max_len, d_model
        meaned = torch.mean(output, axis = -2) #bs, d_model

    def predict(self, output):
        return self.classifier(output)

    def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        output, _ = self.encode(src, src_mask)
        output = self.predict(output)
        return output


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

# -----------------------------------------------------------

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class NewsDataset(Dataset):
  def __init__(self, tokenizer, x, y,  max_len=512):
    
    self.max_len = max_len
    self.tokenizer = tokenizer
    self.inputs = []
    self.targets = []
    self.input_sentences = x
    self.target_ids = []
    for i in y:
      if i == 0:
        self.target_ids.append('World')
      elif i == 1:
        self.target_ids.append('Sports')
      elif i == 2:
        self.target_ids.append('Business')
      elif i == 3:
        self.target_ids.append('Science')
      else:
        raise Exception(f'{i} not found in labels')
    self._build()
  
  def __len__(self):
    return len(self.inputs)
  
  def __getitem__(self, index):
    source_ids = self.inputs[index]["input_ids"].squeeze()
    target_ids = self.targets[index]["input_ids"].squeeze()

    src_mask    = self.inputs[index]["attention_mask"].squeeze()  # might need to squeeze
    target_mask = self.targets[index]["attention_mask"].squeeze()  # might need to squeeze

    return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask}
  
  def _build(self):
    self._buil_examples_from_files()

  
  def _buil_examples_from_files(self):
    REPLACE_NO_SPACE = re.compile("[.;:!\'?,\"()\[\]]")
    REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

    for (i, x) in enumerate(self.input_sentences):
      line = x.strip()
      line = REPLACE_NO_SPACE.sub("", line) 
      line = REPLACE_WITH_SPACE.sub("", line)
      line = line + ' </s>'
      sentiment = self.target_ids[i]
      target = sentiment + " </s>"

       # tokenize inputs
      tokenized_inputs = self.tokenizer.batch_encode_plus(
          [line], max_length=self.max_len, pad_to_max_length=True, return_tensors="pt"
      )
       # tokenize targets
      tokenized_targets = self.tokenizer.batch_encode_plus(
          [target], max_length=2, pad_to_max_length=True, return_tensors="pt"
      )

      self.inputs.append(tokenized_inputs)
      self.targets.append(tokenized_targets)