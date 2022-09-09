import numpy as np
from tensorflow.keras.datasets import imdb
import tensorflow_datasets as tfds
from utils import ids_to_sents, make_sliding_window
import pandas as pd
import torch
import os
from datasets import load_dataset
from tensorflow.keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertForSequenceClassification, BertConfig, BertTokenizer
from transformers import DistilBertForSequenceClassification, DistilBertConfig, DistilBertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from cls_models import *
# from tqdm import tqdm, trange
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_20newsgroups

def load_csv(x_train, y_train, x_val, y_val):
    train_save_dir = '../../DATASETS/imdb/train.csv'
    val_save_dir = '../../DATASETS/imdb/val.csv'
    try:
        train_df = pd.read_csv(train_save_dir)
        val_df = pd.read_csv(val_save_dir)
    except:
        #RECOVER TEXT
        index_offset = 3
        word_index = imdb.get_word_index(path="imdb_word_index.json")
        word_index = {k: (v + index_offset) for k,v in word_index.items()}
        word_index["<PAD>"] = 0
        word_index["<START>"] = 1
        word_index["<UNK>"] = 2
        word_index["<UNUSED>"] = 3
        index_to_word = { v: k for k, v in word_index.items()}

        train_df = ids_to_sents(x_train, y_train, index_to_word)
        val_df = ids_to_sents(x_val, y_val, index_to_word)
        train_df.to_csv(train_save_dir)
        val_df.to_csv(val_save_dir)
    return train_df, val_df

def load_data_text(args):
    MAX_LEN = 512
    if args.dataset == 'imdb':
        (x_train, y_train), (x_val, y_val) = imdb.load_data(num_words=args.max_features)
        x_train = np.concatenate([x_train,x_val[:12500]],axis=0)
        y_train = np.concatenate([y_train,y_val[:12500]],axis=0)
        x_val = x_val[12500:]
        y_val = y_val[12500:]
        if 'bert' in args.model_name:
            # load in sentences and ids
            train_df, val_df = load_csv(x_train, y_train, x_val, y_val)
            x_train = train_df.sentence.values
            x_train = ["[CLS] " + s for s in x_train]
            x_val = val_df.sentence.values
            x_val = ["[CLS] " + s for s in x_val]
        args.logger.info(f'len(x_train): {len(x_train)}')
        args.logger.info(f'len(x_val): {len(x_val)}')
    else:
        if args.dataset == 'news':
            data = load_dataset('ag_news', cache_dir=args.dataset_cache_dir)
            x_train = data['train']['text']
            y_train = np.array(data['train']['label'])
            x_val = data['test']['text']
            y_val = np.array(data['test']['label'])
        elif args.dataset == 'yahoo_answers':
            train = pd.read_csv(f'{args.dataset_cache_dir}yahoo_answers_csv/train.csv', header=None)
            test = pd.read_csv(f'{args.dataset_cache_dir}yahoo_answers_csv/test.csv', header=None)
            # print(train[0])
            # train['all'] = train.agg(lambda x: f"{x[1]}\n{x[2]}\n{x[3]}", axis=1)
            # x_train = train['all']
            x_train = train[3].to_numpy()
            y_train = np.array(train[0])
            # test['all'] = test.agg(lambda x: f"{x[1]}\n{x[2]}\n{x[3]}", axis=1)
            # x_val = test['all']
            x_val = test[3].to_numpy()
            y_val = np.array(test[0])
            # print('x_train[:10]: ', x_train[:10])
            # print('y_train[:10]: ', y_train[:10])
            # raise Exception('end')
        elif args.dataset == '20news':
            (x_train, y_train) = fetch_20newsgroups(data_home = args.dataset_cache_dir, subset = 'train', shuffle=True,
                                        remove=('headers', 'footers', 'quotes'), download_if_missing=True, 
                                        random_state = 0, return_X_y = True)
            (x_val, y_val) = fetch_20newsgroups(data_home = args.dataset_cache_dir, subset = 'test', shuffle=True,
                                        remove=('headers', 'footers', 'quotes'), download_if_missing=True, 
                                        random_state = 0, return_X_y = True)
        if 'bert' in args.model_name:
            x_train = ["[CLS] " + str(s) for s in x_train]
            x_val = ["[CLS] " + str(s) for s in x_val]
        # print('x_train[:10]: ', x_train[:10])
        # print('y_train[:10]: ', y_train[:10])
        # print('len(x_train): ', len(x_train))
        # print('len(x_val): ', len(x_val))
        args.logger.info(f'{np.unique(y_train)} unique labels')
    if args.fast:
        x_train = x_train[:args.fast]
        y_train = y_train[:args.fast]
        x_val = x_val[:int(args.fast*0.2)]
        y_val = y_val[:int(args.fast*0.2)]
    if 'bert' in args.model_name or args.dataset == 'news' or args.dataset == '20news': #when it's news, cnn model also needs tokenization
        if args.model_name=='cnn':
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        else:
            tokenizer = BertTokenizer.from_pretrained(f'{args.model_name}-base-uncased', do_lower_case=True)
        # tokenize, pad
        tokenized_train = [tokenizer.tokenize(s) for s in x_train]
        tokenized_val = [tokenizer.tokenize(s) for s in x_val]

        tokenized_train = [t[:(MAX_LEN - 1)] + ['SEP'] for t in tokenized_train]
        tokenized_val = [t[:(MAX_LEN - 1)] + ['SEP'] for t in tokenized_val]

        ids_train = [tokenizer.convert_tokens_to_ids(t) for t in tokenized_train]
        ids_train = np.array([np.pad(i, (0, MAX_LEN - len(i)),
                                    mode='constant') for i in ids_train])
        ids_val = [tokenizer.convert_tokens_to_ids(t) for t in tokenized_val]
        ids_val = np.array([np.pad(i, (0, MAX_LEN - len(i)),
                                    mode='constant') for i in ids_val])
        # LOAD INTO PYTORCH FORMAT
        args.logger.info(f'ids_train.shape: {ids_train.shape}')
        args.logger.info(f'y_train.shape: {y_train.shape}')
        x_train = ids_train
        x_val = ids_val

        if 'bert' in args.model_name:
            #attention masks
            amasks_train, amasks_val = [], []
            for seq in ids_train:
                seq_mask = [float(i > 0) for i in seq]
                amasks_train.append(seq_mask)
            for seq in ids_val:
                seq_mask = [float(i > 0) for i in seq]
                amasks_val.append(seq_mask)
            train_masks = torch.tensor(amasks_train)
            val_masks = torch.tensor(amasks_val)
            return tokenizer, (x_train, y_train), (x_val, y_val), (train_masks, val_masks)
        else:
            return tokenizer, (x_train, y_train), (x_val, y_val)
    else:
        # PAD SEQUENCES
        x_train = pad_sequences(x_train, maxlen=args.maxlen, padding='post')
        x_val = pad_sequences(x_val, maxlen=args.maxlen, padding='post')
        args.logger.info(f"{len(x_train)} Training sequences")
        args.logger.info(f"{len(x_val)} Validation sequences")
        if args.dataset == 'imdb':
            return None, (x_train, y_train), (x_val, y_val)
        else:
            return tokenizer, (x_train, y_train), (x_val, y_val)


def load_cls_model(args, device, data):
    args.logger.info(f'args.pretrained: {args.pretrained}')
    if args.pretrained == False and args.dataset not in ['news', 'yahoo_answers']:
        if 'bert' in args.model_name:
            tokenizer, (x_train, y_train), (x_val, y_val), (train_masks, val_masks) = data
            train_data = TensorDataset(torch.from_numpy(x_train), train_masks, torch.from_numpy(y_train.astype('int64')))
            valid_data = TensorDataset(torch.from_numpy(x_val), val_masks, torch.from_numpy(y_val.astype('int64')))
            train_loader = DataLoader(train_data, shuffle=True, batch_size=args.batch_size, drop_last=False)
            valid_loader = DataLoader(valid_data, shuffle=True, batch_size=args.batch_size, drop_last=False)
            if args.model_name == 'bert':  
                if args.without_pretraining:
                    config = BertConfig()
                else:
                    config = BertConfig.from_pretrained('bert-base-uncased')
                config.num_labels = len(np.unique(y_train))
                args.logger.info(f'config.num_labels: {config.num_labels}')
                model = BertForSequenceClassification(config)
            elif args.model_name == 'distilbert':
                if args.without_pretraining:
                    config = DistilBertConfig()
                    model = DistilBertForSequenceClassification(config)
                else:
                    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
        
        else:
            _, (x_train, y_train), (x_val, y_val) = data
            # LOAD INTO PYTORCH FORMAT
            args.logger.info(f'x_train.shape: {x_train.shape}')
            train_data = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
            valid_data = TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))
            train_loader = DataLoader(train_data, shuffle=True, batch_size=args.batch_size, drop_last=False)
            valid_loader = DataLoader(valid_data, shuffle=True, batch_size=args.batch_size, drop_last=False)
            if args.model_name == 'lstm':
                model = BiLSTM()
            elif args.model_name == 'cnn':
                if args.dataset == 'news':
                    model = CNN_NLP(args.max_features, args.embedding_dim, args.hidden_dim, args.short_sentences, num_classes = 4)
                elif args.dataset == '20news':
                    model = CNN_NLP(args.max_features, args.embedding_dim, args.hidden_dim, args.short_sentences, num_classes = 20)
                else:
                    model = CNN_NLP(args.max_features, args.embedding_dim, args.hidden_dim, args.short_sentences)
        # print(model)
        model.to(device)
        args.logger.info('{} parameters to train'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
        if 'bert' in args.model_name:
            criterion = nn.CrossEntropyLoss().to(device)
            if args.dataset == '20news':
                args.epochs = 40
                WEIGHT_DECAY = 0.01
                LR = 2e-5
                WARMUP_STEPS =int(0.1*len(train_loader))

                no_decay = ['bias', 'LayerNorm.weight']
                optimizer_grouped_parameters = [
                    {'params': [p for n, p in model.named_parameters()
                                if not any(nd in n for nd in no_decay)],
                    'weight_decay': WEIGHT_DECAY},
                    {'params': [p for n, p in model.named_parameters()
                                if any(nd in n for nd in no_decay)],
                    'weight_decay': 0.0}
                ]
                optimizer = AdamW(optimizer_grouped_parameters, lr=LR, eps=1e-8)
                scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS,
                                                            num_training_steps=-1)
            else:
                WEIGHT_DECAY = 0.01
                LR = args.lr
                WARMUP_STEPS = int(0.2 * len(train_loader))

                no_decay = ['bias', 'LayerNorm.weight']
                optimizer_grouped_parameters = [
                    {'params': [p for n, p in model.named_parameters()
                                if not any(nd in n for nd in no_decay)],
                    'weight_decay': WEIGHT_DECAY},
                    {'params': [p for n, p in model.named_parameters()
                                if any(nd in n for nd in no_decay)],
                    'weight_decay': 0.0}
                ]
                optimizer = AdamW(optimizer_grouped_parameters, lr=LR, eps=1e-8)
                scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=-1)

        else:
            if args.dataset == 'imdb':
                criterion = nn.BCELoss().to(device)
            else:
                criterion = nn.CrossEntropyLoss().to(device) #multi-class classification for news dataset
            optimizer = torch.optim.Adam(model.parameters())
        epoch_history = {
            "loss": [],
            "accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
        }
        for ep in range(args.epochs):
            model.train()
            batch_history = {
                    "loss": [],
                    "accuracy": []
                }
            for i, batch in enumerate(train_loader):
                if 'bert' in args.model_name:
                    b_input_ids, b_input_mask, b_labels = batch
                    b_input_ids = b_input_ids.to(device)
                    b_input_mask = b_input_mask.to(device)
                    b_labels = b_labels.to(device)
                    optimizer.zero_grad()
                    results = model(b_input_ids,
                        attention_mask=b_input_mask, labels=b_labels)
                    predictions = results['logits']
                    targets = b_labels
                    loss = results[0]
                    # loss = criterion(predictions.squeeze(), targets.float())
                else:
                    samples, targets = batch
                    samples = samples.to(device).long()
                    targets = targets.to(device)#100
                    model.zero_grad()
                    predictions = model(samples) #100, n_classes
                    loss = criterion(predictions.squeeze(), targets.float())
                if 'bert' in args.model_name:
                    predictions = F.softmax(predictions, dim = 1)
                    # print('predictions: ', predictions)
                    # print('predictions.shape: ', predictions.shape)
                    # print('predictions.max(1).indices: ', predictions.max(1).indices)
                    # print('targets: ', targets)
                    acc = (predictions.max(1).indices == targets).sum().item()
                else:
                    if args.dataset == 'news' or args.dataset == '20news':
                        # print('predictions: ', predictions)
                        # print('predictions.shape: ', predictions.shape)
                        # print('predictions.max(1).indices: ', predictions.max(1).indices)
                        # print('predictions.max(1).indices.shape: ', predictions.max(1).indices.shape)
                        # print('targets: ', targets)
                        # print('targets.shape: ', targets.shape)
                        acc = (predictions.max(1).indices == targets).sum().item()
                    else:
                        acc = (predictions.squeeze().round() == targets).sum().item()
                # print('acc: ', acc)
                acc = acc / args.batch_size
                # print("args.batch_size: ", args.batch_size)
                # print('acc: ', acc)
                # raise Exception('end')
                loss.backward()
                if 'bert' in args.model_name:
                    optimizer.step()
                    scheduler.step()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                    optimizer.step()

                batch_history["loss"].append(loss.item())
                batch_history["accuracy"].append(acc)
                
                # tbatch.set_postfix(loss=sum(batch_history["loss"]) / len(batch_history["loss"]),
                #                 acc=sum(batch_history["accuracy"]) / len(batch_history["accuracy"]))
            epoch_history["loss"].append(sum(batch_history["loss"]) / len(batch_history["loss"]))
            epoch_history["accuracy"].append(sum(batch_history["accuracy"]) / len(batch_history["accuracy"]))
            # args.logger.info("Validation...")
            model.eval()
            with torch.no_grad():
                val_losses = []
                val_acces = []                    
                # validation loop
                for i, batch in enumerate(valid_loader):
                    if 'bert' in args.model_name:
                        b_input_ids, b_input_mask, b_labels = batch
                        b_input_ids = b_input_ids.to(device)
                        b_input_mask = b_input_mask.to(device)
                        b_labels = b_labels.to(device)
                        optimizer.zero_grad()
                        results = model(b_input_ids,
                            attention_mask=b_input_mask, labels=b_labels)
                        predictions = results['logits']
                        targets = b_labels
                        loss = results[0]
                    else:
                        samples, targets = batch
                        samples = samples.to(device).long()
                        targets = targets.to(device)
                        model.zero_grad()
                        predictions = model(samples)
                        loss = criterion(predictions.squeeze(), targets.float())
                    val_losses.append(loss)
                    if 'bert' in args.model_name:
                        predictions = F.softmax(predictions, dim = 1)
                        acc = (predictions.max(1).indices == targets).sum().item()
                    else:
                        if args.dataset == 'news' or args.dataset == '20news':
                            acc = (predictions.max(1).indices == targets).sum().item()
                        else:
                            acc = (predictions.squeeze().round() == targets).sum().item()
                    acc = acc / args.batch_size
                    val_acces.append(acc)
            val_loss = sum(val_losses) / len(val_losses)
            val_accuracy = sum(val_acces) / len(val_acces)
            epoch_history["val_loss"].append(val_loss.item())
            epoch_history["val_accuracy"].append(val_accuracy)
            print_history = {}
            for key in epoch_history.keys():
                print_history[key] = epoch_history[key][-1]
            args.logger.info(f"epoch {ep} out of {args.epochs}: {print_history}")
        if 'bert' in args.model_name:
            model.save_pretrained(args.model_save_dir)
        else: 
            torch.save(model, args.model_save_dir)
        args.logger.info('MODEL SAVED AT EPOCH {}'.format(ep+1))

        fig, ax = plt.subplots(1, 2)
        metrics = ['loss', 'accuracy']
        for i in range(2):
            ax[i].plot(epoch_history[metrics[i]], label = 'train')
            ax[i].plot(epoch_history['val_' + metrics[i]], label = 'val')
            ax[i].set_title(metrics[i], fontsize='small')
            ax[i].legend()
        plt.savefig(args.save_dir +'/cls_model_training.png')
    else:
        if args.model_name == 'bert':
            if args.dataset == 'news' and (not args.without_pretraining):
                model = BertForSequenceClassification.from_pretrained('fabriceyhc/bert-base-uncased-ag_news')
            elif args.dataset == 'yahoo_answers' and (not args.without_pretraining):
                model = BertForSequenceClassification.from_pretrained('fabriceyhc/bert-base-uncased-yahoo_answers_topics')
            else:
                model = BertForSequenceClassification.from_pretrained(args.model_save_dir)
        elif args.model_name == 'distilbert':
            model = DistilBertForSequenceClassification.from_pretrained(args.model_save_dir)
        else:
            model = torch.load(args.model_save_dir)
        model = model.to(device)
        args.logger.info('model loaded')
    return model