# imdb_transformer.py
# Transformer Architecture classification for IMDB
# PyTorch 1.10.0-CPU Anaconda3-2020.02  Python 3.7.6
# Windows 10/11

import numpy as np
# import torch as T
import math
import copy
import time
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from pathlib import Path
import logging
from transformers import BertTokenizer
from datasets import load_dataset
from helper import *

dataset = 'news'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_logger():
    global logger
    logger = logging.getLogger('root')
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt = '%m/%d/%Y %H:%M:%S',
        handlers=[
            logging.FileHandler(f"../log/dummy.log"),
            logging.StreamHandler()
        ]
    )
    return logger
    


# -----------------------------------------------------------

def main():
  logger = set_logger()
  # 0. get started
  logger.info("\nBegin PyTorch IMDB Transformer Architecture demo ")
  logger.info('device: {}'.format(device))
  torch.manual_seed(1)
  np.random.seed(1)

  # 1. load data 
  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

  def preprocess_function(examples):
    labels = [element["label"] for element in examples]
    text_batch = [element["text"] for element in examples]
    results = tokenizer(text_batch, padding='max_length', truncation=True, return_tensors='pt')
    results['labels'] = torch.from_numpy(np.array(labels))
    return results

  # fast = False
  # fast = 2000
  bat_size = 128
  epochs = 10

  # if fast:
  #   train_ds = load_dataset("imdb", split=f"train[:{fast}]")
  #   test_ds = load_dataset("imdb", split=f"test[:{int(0.2*fast)}]")
  # else:
  if dataset == 'imdb':
    train_ds = load_dataset("imdb", split="train")
    test_ds = load_dataset("imdb", split="test")
  elif dataset == 'news':
    train_ds = load_dataset('ag_news', split="train", cache_dir="../../hf_datasets/")
    test_ds = load_dataset('ag_news', split="test", cache_dir="../../hf_datasets/")

  train_ldr = torch.utils.data.DataLoader(dataset=train_ds, batch_size=bat_size, shuffle=True, collate_fn=preprocess_function)
  test_ldr = torch.utils.data.DataLoader(dataset=test_ds, batch_size=bat_size, shuffle=True, collate_fn=preprocess_function)
    
  n_train = len(train_ds)
  n_test = len(test_ds)
  logger.info("Num train = %d Num test = %d " % (n_train, n_test))

# -----------------------------------------------------------

  # 2. create network
  ntokens = 129892  # size of vocabulary
  emsize = 200  # embedding dimension
  d_hid = 200  # dimension of the feedforward network model in nn.TransformerEncoder
  nlayers = 6  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
  nhead = 2  # number of heads in nn.MultiheadAttention
  dropout = 0.2  # dropout probability
  if dataset == 'imdb':
    model = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout).to(device)
  elif dataset == 'news':
    model = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout, n_classes=4).to(device)

  # 3. train model

  # criterion = nn.NLLLoss()
  if dataset == 'imdb':
    criterion = nn.BCELoss()
  elif dataset == 'news':
    criterion = nn.CrossEntropyLoss()

  # lr = 5e-4  # learning rate
  # optimizer = torch.optim.SGD(model.parameters(), lr=lr)
  optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
  scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

# -----------------------------------------------------------

  def train(model: nn.Module) -> None:
      model.train()  # turn on train mode
      total_loss = 0.
      log_interval = 50
      start_time = time.time()
      src_mask = generate_square_subsequent_mask(bat_size).to(device)
      for i, batch in enumerate(train_ldr):
          b_input_ids = batch['input_ids']
          b_labels = batch['labels']
          seq_len = b_input_ids.size(0)
          if seq_len != bat_size:  # only on last batch
              src_mask = src_mask[:seq_len, :seq_len]
          # output = model(data, src_mask)  dtype=torch.float64
          output = model(b_input_ids.to(device), src_mask.to(device))
          
          if dataset == 'imdb':
            loss = criterion(output, b_labels.unsqueeze(1).to(device).to(torch.float)) 
          elif dataset == 'news':
            loss = criterion(output, b_labels.to(device).to(torch.long)) 

          optimizer.zero_grad()
          loss.backward()
          torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
          optimizer.step()

          total_loss += loss.item()
          if i % log_interval == 0 and i > 0:
              lr = scheduler.get_last_lr()[0]
              ms_per_batch = (time.time() - start_time) * 1000 / log_interval
              cur_loss = total_loss / log_interval
              ppl = math.exp(cur_loss)
              logger.info(f'| batch {i} out of {len(train_ldr)} batches | '
                    f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                    f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
              total_loss = 0
              start_time = time.time()
# -----------------------------------------------------------

  # 4. evaluate model
  def evaluate(model: nn.Module) -> float:
      model.eval()  # turn on evaluation mode
      total_loss = 0.
      acc = 0
      total = 0
      src_mask = generate_square_subsequent_mask(bat_size).to(device)
      for i, batch in enumerate(test_ldr):
        b_input_ids = batch['input_ids']
        b_labels = batch['labels']
        seq_len = b_input_ids.size(0)
        if seq_len != bat_size:  # only on last batch
          src_mask = src_mask[:seq_len, :seq_len]
        with torch.no_grad():
          output = model(b_input_ids.to(device), src_mask.to(device))
        # loss = criterion(output.view(-1, 2), b_labels.to(device))
        if dataset == 'imdb':
          loss = criterion(output, b_labels.unsqueeze(1).to(device).to(torch.float)) 
        elif dataset == 'news':
          loss = criterion(output.squeeze(), b_labels.long().to(device))
        total_loss += loss.item()
        if dataset == 'imdb':
          idx = torch.round(output.data).squeeze()
        elif dataset == 'news':
          idx = output.data.argmax(1).squeeze()
        try:
          acc += sum(idx.detach().cpu().numpy()==b_labels.numpy())
        except:
          logger.info(f'output: {output}')
          logger.info(f'idx.detach().cpu().numpy(): {idx.detach().cpu().numpy()}')
          logger.info(f'b_labels.numpy(): {b_labels.numpy()}')
          raise Exception('end')
        total += b_input_ids.size(0)
      logger.info(f'evaluation acc: {acc/total}')
      logger.info(f'total loss: {total_loss / (i-1)}')
      return total_loss / (total - 1)
      

  best_val_loss = float('inf')
  best_model = None

  for epoch in range(1, epochs + 1):
      epoch_start_time = time.time()
      # val_loss = evaluate(model)
      # raise Exception('end')
      train(model)
      val_loss = evaluate(model)
      val_ppl = math.exp(val_loss)
      elapsed = time.time() - epoch_start_time
      logger.info('-' * 89)
      logger.info(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
                f'valid loss {val_loss:5.2f} | valid ppl {val_ppl:8.2f}')
      logger.info('-' * 89)

      if val_loss < best_val_loss:
          best_val_loss = val_loss
          best_model = copy.deepcopy(model)

      scheduler.step()

  # 5. save model
  logger.info("\nSaving trained model state")
  save_folder = f'models/{dataset}/transformer/'
  Path(save_folder).mkdir(parents=True, exist_ok=True)
  fn = f"{save_folder}model.pt"

  torch.save(best_model.state_dict(), fn)

  logger.info("\nEnd PyTorch IMDB Transformer demo")

if __name__ == "__main__":
  main()