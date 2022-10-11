from typing import overload
import pandas as pd
import numpy as np
import torch
import random
# from tensorflow.keras.datasets import imdb
from matplotlib import pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import cv2
from torch.utils.data import TensorDataset, DataLoader
from PIL import Image
from conceptshap import *
import os
import toy_helper_v2
from helper import *
from statsmodels.stats import weightstats as stests
import lib.dist as dist
import lib.utils as utils
import lib.datasets as dset
from lib.flows import FactorialNormalizingFlow
from elbo_decomposition import elbo_decomposition
from bcvae import *
import time
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
import transformers
import copy
stop_word_list = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", 
                  "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", 
                  "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", 
                  "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", 
                  "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", 
                  "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", 
                  "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", 
                  "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", 
                  "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor",  
                  "only", "own", "same", "so", "than", "very", "s", "t", "can", "will", "just", "don", "should", "now",
                  "one", "it's", "br", "<PAD>", "<START>", "<UNK>", "<UNUSED>" "would", "could", "also", "may", "many", "go", "another",
                  "want", "two", "actually", "every", "thing", "know", "made", "get", "something", "back", "though"]
def seed_everything(args):
    random.seed(args.seed)
    os.environ['PYTHONASSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def get_pca_concept(f_train, n_concept, logger):
    pca = PCA(n_components = n_concept)
    logger.info(f'f_train.shape: {f_train.shape}')
    if f_train.dim() == 4: #toy dataset
        f_train = f_train.flatten(start_dim = 1, end_dim = -1)
    elif f_train.dim()>2:
        f_train = f_train.flatten(start_dim = 0, end_dim = -2)
    f_train = f_train.numpy()
    logger.info(f'flattened shape: {f_train.shape}')
    logger.info('fitting...')
    logger.info(f'nan: {np.isnan(f_train).any()}')
    logger.info(f'isposinf: {np.isposinf(f_train).any()}')
    logger.info(f'isneginf: {np.isneginf(f_train).any()}')
    logger.info(f'isinf: {np.isinf(f_train).any()}')
    try:
        pca.fit(f_train.astype(float))
    except:
        pca.fit(f_train.astype(float))
    weight_pca = pca.components_.T #hidden_dim, n_concept
    logger.info(f'weight_pca shape: {weight_pca.shape}')
    return pca, weight_pca

def get_kmeans_concept(f_train, n_concept):
    print('original f_train.shape {}'.format(f_train.shape))
    if f_train.dim()>2:
        f_train = f_train.flatten(start_dim = 0, end_dim = -2)
    print('new f_train.shape {}'.format(f_train.shape))
    kmeans = KMeans(n_clusters=n_concept, random_state=0).fit(f_train)
    weight_cluster = kmeans.cluster_centers_.T
    return kmeans, weight_cluster

def get_ace_concept_stm(cluster_new, predict, f_train):
    """Calculates ACE/TCAV concepts."""
    concept_input = Input(shape=(f_train.shape[1]), name='concept_input')
    softmax_tcav = predict(concept_input)
    tcav_model = Model(inputs=concept_input, outputs=softmax_tcav)
    tcav_model.layers[-1].activation = None
    tcav_model.layers[-1].trainable = False
    tcav_model.layers[-2].trainable = False
    tcav_model.compile(
        loss='mean_squared_error',
        optimizer=SGD(lr=0.0),
        metrics=['binary_accuracy'])
    tcav_model.summary()

    n_cluster = cluster_new.shape[0]
    n_percluster = cluster_new.shape[1]
    print(cluster_new.shape)
    weight_ace = np.zeros((64, n_cluster))
    tcav_list_rand = np.zeros((15, 300))
    tcav_list_ace = np.zeros((15, n_cluster))
    for i in range(n_cluster):
        y = np.zeros((n_cluster * n_percluster))
        y[i * n_percluster:(i + 1) * n_percluster] = 1
        clf = LogisticRegression(
            random_state=0,
            solver='lbfgs',
            max_iter=10000,
            C=10.0,
            multi_class='ovr').fit(cluster_new.reshape((-1, 64)), y)
        weight_ace[:, i] = clf.coef_

    weight_rand = np.zeros((64, 300))
    for i in range(300):
        y = np.random.randint(2, size=n_cluster * n_percluster)
        clf = LogisticRegression(
            random_state=0,
            solver='lbfgs',
            max_iter=10000,
            C=10.0,
            multi_class='ovr').fit(cluster_new.reshape((-1, 64)), y)
        weight_rand[:, i] = clf.coef_

    sig_list = np.zeros(n_cluster)

    for j in range(15):
        grads = (
            K.gradients(target_category_loss(softmax_tcav, j, 15),
                        concept_input)[0])
        gradient_function = K.function([tcav_model.input], [grads])
        grads_val = np.mean(gradient_function([f_train])[0],axis=(1,2))

        grad_rand = np.matmul(grads_val, weight_rand)
        grad_ace = np.matmul(grads_val, weight_ace)
        tcav_list_rand[j, :] = np.sum(grad_rand > 0.000, axis=(0))
        tcav_list_ace[j, :] = np.sum(grad_ace > 0.000, axis=(0))
        mean = np.mean(tcav_list_rand[j, :])
        std = np.std(tcav_list_rand[j, :])
        sig_list += (tcav_list_ace[j, :] > mean + std * 1.0).astype(int)
        sig_list += (tcav_list_ace[j, :] < mean - std * 1.0).astype(int)
    top_k_index = np.array(sig_list).argsort()[-1 * 15:][::-1]
    print(sig_list)
    print(top_k_index)
    return weight_ace[:, top_k_index]


def recover_text(sample, index_to_word):
    #don't do pad
    return ' '.join([index_to_word[i] if i != 0 else '' for i in sample])

def recover_word_list(sample, index_to_word):
    return [index_to_word[i] for i in sample]

def ids_to_sents(xs, ys, index_to_word):
    sentences = []
    for x in xs:
        sentences.append(recover_text(x, index_to_word))
    df = pd.DataFrame(list(zip(xs, sentences, ys)), columns =['ids', 'sentence', 'label'])
    return df

def make_sliding_window(x, y, save_dir):
    windows = []
    labels = []

    for i in range(len(x)):
        x_new = x[i]
        label = y[i]
        for j in range(10, len(x_new)):
            sliding_window = x_new[j-10:j]
            windows.append(np.array(sliding_window))
            labels.append(label)

    windows = np.stack(windows)
    labels = np.array(labels)

    # np.save(save_dir[0], windows)
    # np.save(save_dir[1], labels)

    return windows, labels
    
def run_model(_model, loader, device, args):
  ce_loss = nn.CrossEntropyLoss()

  all_losses = []
  src_mask = generate_square_subsequent_mask(args.batch_size).to(device)
  for batch in loader:
    b_input_ids, b_input_mask, b_labels = batch
    b_input_ids = b_input_ids.to(device)
    b_input_mask = b_input_mask.to(device)
    b_labels = b_labels.to(device)
    with torch.no_grad():
        if args.model_name == 'transformer':
            seq_len = b_input_ids.size(0)
            if seq_len != args.batch_size:  # only on last batch
                src_mask = src_mask[:seq_len, :seq_len]
            results = _model(b_input_ids.to(device), src_mask.to(device))
            loss = 0
            all_losses.append(loss)
        else:
            results = _model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
            predictions = results['logits']
            loss = results[0]
            all_losses.append(loss.item())
  print("inference loss:", np.mean(np.array(all_losses)))

# to help with bert inference
def get_sentence_activation(model, loader, device, args):
    global extracted_activations
    extracted_activations = []
    def extract_activation_hook(model, input, output):
        if args.model_name in ['imdb', 'transformer']:
            print('output.shape: ', output.shape) #batch_size, 512, 768
            extracted_activations.append(output[0].detach().cpu().numpy())
        elif args.model_name in ['bert']:
            print('output.shape: ', output.shape) #batch_size, 512, 768
            extracted_activations.append(output.detach().cpu().squeeze().numpy())
        else:
            extracted_activations.append(output[0].detach().cpu().numpy())
            print('output.shape: ', output.shape) #batch_size, 512, 768
            print('extracted_activations[-1].shape: ', extracted_activations[-1].shape) #batch_size, 512, 768
            raise Exception('end')

    def add_activation_hook(model, layer_idx):
        # if we're here, it can only be the last layer in bert
        all_modules_list = list(model.modules())
        module = all_modules_list[-2]
        print('module: ', module)
        module.register_forward_hook(extract_activation_hook)

    if not args.added:
        add_activation_hook(model, layer_idx=args.layer_idx)
        args.added = True

    print("running inference..")
    run_model(model, loader, device, args) # run the whole model
    res = np.concatenate(extracted_activations, axis=0)
    print('res.shape: ', res.shape)
    return res

def get_t5_output(lm_output, args, logits = True):
    if logits:
        lm_output = lm_output.argmax(-1) #1465 is positive, 2841 is negative
    y_pred = []
    if args.dataset == 'imdb':
        for i in lm_output:
            if i == 1465:
                y_pred.append(1)
            elif i == 2841:
                y_pred.append(0)
            else:
                y_pred.append(-1)
    else:
        # ids_0 World:  [1150, 1]
        # ids_1 Sports:  [5716, 1]
        # ids_2 Business:  [1769, 1]
        # ids_3 Science:  [2854, 1]
        for i in lm_output:
            if i == 1150:
                y_pred.append(0)
            elif i == 5716:
                y_pred.append(1)
            elif i == 1769:
                y_pred.append(2)
            elif i == 2854:
                y_pred.append(3)
            else:
                y_pred.append(-1)
    return torch.from_numpy(np.array(y_pred))

def get_t5_targets(y_train, args):
    args.logger.info(f'y_train: {y_train}')
    if args.dataset == 'imdb':
        y_train[y_train==1] = 1465
        y_train[y_train==0] = 2841
    elif args.dataset == 'news':
        y_train[y_train==0] = 1150
        y_train[y_train==1] = 5716
        y_train[y_train==2] = 1769
        y_train[y_train==3] = 2854
    args.logger.info(f'y_train: {y_train}')
    return y_train

def get_t5_activation(model, tokenizer, loader, device, args):
    extracted_activations = []
    y_pred = []
    # all_losses = []
    for batch in loader:
        if args.dataset == 'imdb':
            b_input_ids, b_input_mask, b_labels = batch
            b_input_ids = b_input_ids.to(device)
            b_labels = b_labels.to(device)
            b_labels = b_labels.unsqueeze(1)
            with torch.no_grad():
                results = model(input_ids=b_input_ids, labels=b_labels)
        else:
            lm_labels = batch["target_ids"]
            lm_labels[lm_labels[:, :] == tokenizer.pad_token_id] = -100
            results = model(
                input_ids=batch["source_ids"].to(device),
                attention_mask=batch["source_mask"].to(device),
                labels=lm_labels.to(device),
                decoder_attention_mask=batch['target_mask'].to(device)
            )
        loss = results.loss
        hidden = results.t5_hidden
        extracted_activations.append(hidden.detach().cpu().squeeze().numpy())
        lm_logits = model.lm_head(hidden).cpu().detach().squeeze()
        lm_output = lm_logits.argmax(-1)
        y_pred.append(lm_output)
        if lm_output.shape[0]!=hidden.shape[0]:
            args.logger.info(f'lm_output.shape: {lm_output.shape}')
            args.logger.info(f'hidden.shape: {hidden.shape}')
            raise Exception
    res = torch.from_numpy(np.concatenate(extracted_activations, axis=0))
    y_pred = torch.from_numpy(np.concatenate(y_pred, axis=0))
    if args.dataset == 'news':
        res = res[:, 0, :]
        y_pred = y_pred[:, 0]
    print('res.shape: ', res.shape)
    print(f'y_pred.shape: {y_pred.shape}') #size, 1
    return res, y_pred

def inference(data, model, tokenizer, classifier, device, args):
    if args.do_inference: 
        if args.model_name =='t5' and args.dataset == 'news':
            train_data, valid_data, tokenizer = data
            train_loader = DataLoader(train_data, shuffle=False, batch_size=args.batch_size, drop_last=False)
            valid_loader = DataLoader(valid_data, shuffle=False, batch_size=args.batch_size, drop_last=False)
        elif args.model_name in ['bert', 'distilbert', 't5', 'transformer']:
            tokenizer, (x_train, y_train), (x_val, y_val), (train_masks, val_masks) = data
            args.logger.info(f'x_train.shape: {x_train.shape}')
            if args.model_name == 't5': #from class indices to tokenizer ids
                train_data = TensorDataset(torch.from_numpy(x_train), train_masks, torch.from_numpy(get_t5_targets(y_train, args).astype('int64')))
                valid_data = TensorDataset(torch.from_numpy(x_val), val_masks, torch.from_numpy(get_t5_targets(y_val, args).astype('int64')))
            else:
                train_data = TensorDataset(torch.from_numpy(x_train), train_masks, torch.from_numpy(y_train.astype('int64')))
                valid_data = TensorDataset(torch.from_numpy(x_val), val_masks, torch.from_numpy(y_val.astype('int64')))
            train_loader = DataLoader(train_data, shuffle=False, batch_size=args.batch_size, drop_last=False)
            valid_loader = DataLoader(valid_data, shuffle=False, batch_size=args.batch_size, drop_last=False)
        else:
            tokenizer, (x_train, y_train), (x_val, y_val) = data
            args.logger.info(f'x_train.shape: {x_train.shape}')
            train_data = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
            valid_data = TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))
            #do not shuffle to have the correct inferred data
            train_loader = DataLoader(train_data, shuffle=False, batch_size=args.batch_size, drop_last=False)
            valid_loader = DataLoader(valid_data, shuffle=False, batch_size=args.batch_size, drop_last=False)
        model.eval()
        f_train = []
        y_pred_train = []
        f_val = []
        y_pred_val = []
        if args.model_name in ['bert', 'distilbert']: #with hook
            f_train = get_sentence_activation(model, train_loader, device, args)
            y_pred_train = classifier(torch.from_numpy(f_train).to(device)).cpu().detach().numpy()
            args.logger.info(f'y_pred_train[:10]: {y_pred_train[:10]}')
            f_val = get_sentence_activation(model, valid_loader, device, args)
            y_pred_val = classifier(torch.from_numpy(f_val).to(device)).cpu().detach().numpy()
        elif args.model_name == 't5':
            f_train, y_pred_train = get_t5_activation(model, tokenizer, train_loader, device, args)
            f_val, y_pred_val = get_t5_activation(model, tokenizer, valid_loader, device, args)
        else: #transformer
            src_mask = generate_square_subsequent_mask(args.batch_size).to(device)
            with torch.no_grad():
                for i, batch in enumerate(train_loader):
                    if args.model_name == 'transformer':
                        (samples, targets, masks) = batch
                        seq_len = samples.size(0)
                        if seq_len != args.batch_size:  # only on last batch
                            src_mask = src_mask[:seq_len, :seq_len]
                        predictions, _ = model.encode(samples.to(device).long(), src_mask.to(device))
                        targets = classifier(predictions)
                    else:
                        (samples, targets) = batch
                        if args.model_name == 'cnn':
                            if args.dataset != 'toy':
                                meaned, predictions = model.encode(samples.to(device).float())
                                targets = classifier(meaned)
                            else:
                                predictions = model.encode(samples.to(device).float())
                                targets = classifier(predictions)
                        else:
                            predictions = model.encode(samples.to(device).long())
                            targets = classifier(predictions)
                    f_train.append(predictions.cpu().detach())
                    y_pred_train.append(targets.cpu().detach())
                for i, batch in enumerate(valid_loader):
                    if args.model_name == 'transformer':
                        (samples, targets, masks) = batch
                        seq_len = samples.size(0)
                        if seq_len != args.batch_size:  # only on last batch
                            src_mask = src_mask[:seq_len, :seq_len]
                        predictions, _ = model.encode(samples.to(device).long(), src_mask.to(device))
                        targets = classifier(predictions)
                    else:
                        (samples, targets) = batch
                        if args.model_name == 'cnn':
                            if args.dataset != 'toy':
                                meaned, predictions = model.encode(samples.to(device).float())
                                targets = classifier(meaned)
                            else:
                                predictions = model.encode(samples.to(device).float())
                                targets = classifier(predictions)
                        else:
                            predictions = model.encode(samples.to(device).long())
                            targets = classifier(predictions)
                    f_val.append(predictions.cpu().detach())
                    y_pred_val.append(targets.cpu().detach())
            f_train = torch.cat(f_train, 0)
            y_pred_train = torch.cat(y_pred_train, 0)
            f_val = torch.cat(f_val, 0)
            y_pred_val = torch.cat(y_pred_val, 0)
        with open(args.save_dir + f'train_embeddings_{args.layer_idx}.npy', 'wb') as f:
            np.save(f, f_train.numpy())
        np.save(args.save_dir + 'pred_train.npy', y_pred_train.numpy())
        with open(args.save_dir + f'val_embeddings_{args.layer_idx}.npy', 'wb') as f:
            np.save(f, f_val.numpy())
        np.save(args.save_dir + 'pred_val.npy', y_pred_val.numpy())
        print('finished inference!')
    else:
        if not args.divide_bert:
            f_train = torch.from_numpy(np.load(args.save_dir + f'train_embeddings_{args.layer_idx}.npy'))
            f_val = torch.from_numpy(np.load(args.save_dir + f'val_embeddings_{args.layer_idx}.npy'))
            print('f_train.shape: ', f_train.shape) #32000， 64
            print('f_val.shape: ', f_val.shape) #7936， 64
        else:
            f_train = False
            f_val = False
        y_pred_train = torch.from_numpy(np.load(args.save_dir + 'pred_train.npy'))
        y_pred_val = torch.from_numpy(np.load(args.save_dir + 'pred_val.npy'))
        print('inference results loaded!')
        args.logger.info(f'args.save_dir: {args.save_dir}')
    return f_train, y_pred_train, f_val, y_pred_val

def concept_analysis(f_train, x_train, topic_vector, save_file, model_name = None):
    if model_name == 'cnn':
        f_train = f_train.swapaxes(1, 2)
    # 3-D data
    # concepts: (n_concepts, dim)
    # train_embeddings: (n_embeddings, x, dim)
    # train_data: df => (n_sentences, label)
    c_idx = 0
    write = ''
    # word-index dictionary
    index_offset = 3
    word_index = imdb.get_word_index(path="imdb_word_index.json")
    word_index = {k: (v + index_offset) for k,v in word_index.items()}
    word_index["<PAD>"] = 0
    word_index["<START>"] = 1
    word_index["<UNK>"] = 2
    word_index["<UNUSED>"] = 3
    index_to_word = { v: k for k, v in word_index.items()}
        
    for concept in topic_vector.T:
        c_idx+=1
        
        if len(f_train.shape)==3:
            # f_train: (size, nc, w)
            # concept: (w)
            distance = torch.norm(f_train - concept, dim=-1) #(n_embeddings, x)
            v, i = distance.flatten().topk(150, largest=False)
            knn = np.array(np.unravel_index(i.numpy(), distance.shape))[0] #before [0]: 2-d: [x_indices], [y_indices]
        else:
            # f_train: (size, w)
            # concepts: (w)
            distance = torch.norm(f_train - concept, dim = -1) # (size)
            knn = distance.topk(150, largest=False).indices
            print('knn.shape: ', knn.shape) # (150)

        words = []
        sentences = []
        for j in range(150):
            idx = knn[j]
            x = x_train[idx]
            # recover words
            new_words = recover_word_list(x, index_to_word)
            words += new_words
            # print top 10 closest sentences
            if j <= 10:
                sentences.append(recover_text(x, index_to_word))
                

        cx = Counter(words)
        #filter stopwords
        for word in stop_word_list:
            if word in cx:
                del cx[word]
        most_occur = list(cx.most_common(25))
        print("Concept " + str(c_idx) + " most common words: \n")
        write += "Concept " + str(c_idx) + " most common words: \n"
        print(str(most_occur) + '\n')
        write += str(most_occur) + '\n'
        for s in sentences:
            print(s)
            write += s + '\n'
        print("\n\n")
        write += '\n\n'
    #save the results to a file
    text_file = open(save_file, "w")
    n = text_file.write(write)
    text_file.close()

def make_PDP(f_train, classifier, topic_vector, save_dir, device, method, model, rge = [-1, 1]):
    if model == 'cnn':
        f_train = f_train.swapaxes(1, 2) # from size, n_concept, maxlen  -> size, maxlen, n_concept
    fig, axs = plt.subplots(1)
    plt_i = 0
    print(topic_vector.T.shape) # 4, 100
    for t in topic_vector:
        xs = []
        ys = []
        for i in np.linspace(rge[0], rge[1], 100):
            xs.append(i)
            new_f_train = f_train + t*i
            if model == 'cnn':
                new_f_train = new_f_train.swapaxes(1, 2) # from size, maxlen, n_concept -> size, n_concept, maxlen  
                # do average
                new_f_train = torch.mean(new_f_train, axis = -1) #bs, nc
            new_res = classifier(new_f_train.to(device)).cpu().detach().numpy()
            ys.append(np.mean(new_res))
        plt.plot(xs, ys, label = plt_i)
        plt_i += 1
    plt.xlabel('concept intensity')
    plt.legend()
    plt.ylabel('average prediction')
    plt.title('{} concept PDP'.format(method))
    plt.savefig(save_dir)
    

# GRADCAM analysis
def returnCAM(feature_conv, weight_softmax, concepts, max_len = 400):
    bz, nc, h, w = feature_conv.shape 
    size_upsample = (400, bz)
    output_cam = []
    for idx in concepts:
        cam = weight_softmax[idx].dot(feature_conv.reshape((bz, nc, h*w))) 
        cam = cam.reshape(bz, h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)[:, 0, :]
        resized = cv2.resize(cam_img, size_upsample)
        output_cam.append(resized)
    return output_cam

def produce_heatmaps_concept(figname, model, samples, features_blobs, topic_vec, method = 1):
    ws = 50 #window_size
    n = 5 #how many top examples to show
    nc = topic_vec.T.shape[0] #how many concepts
    _, _, height, width = samples.shape
    print('height: {}, width: {}'.format(height, width))
    all_cams = returnCAM(features_blobs, topic_vec.T, range(nc), size_upsample = [height, width])
    print(all_cams.shape) #100, 5, 224, 224
    # displaying image
    fig, ax = plt.subplots(nc, n, figsize = (2*nc, 2*n))
    for c in range(nc):
        cams = all_cams[:, c, :, :]
        min_cams = cams.min(-1).min(-1)
        min_idx = min_cams.argsort(axis=None)[:n]
        i = 0
        for min_i in min_idx:
            img = samples[min_i].cpu().detach().numpy().transpose(1,2,0)*255
            new_cams = cams[min_i]
            idx = np.unravel_index(np.argmin(new_cams, axis=None), new_cams.shape)
            
            heatmap = cv2.applyColorMap(new_cams, cv2.COLORMAP_JET)
            bld = 0.2
            heatmap = (heatmap * bld + img*(1-bld)).astype(int)
            a1 = idx[1]
            a2 = idx[0]
            if method == 0:
                ax[c, i].imshow(heatmap.astype(int))
                ax[c, i].plot(a1, a2, 'x', color = 'black', markersize = 10)
            elif method == 1:
                a11 = (max(0, a1-ws), min(height, a1+ws))
                a22 = (max(0, a2-ws), min(width, a2+ws))
                heatmap = cv2.resize(img[a11[0]:a11[1], a22[0]:a22[1]].astype(float), (height, width))
                ax[c, i].imshow(heatmap.astype(int))
            i+=1
    plt.savefig(figname)

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x = None, input_ids = None, position_ids=None,
            token_type_ids=None,
            inputs_embeds=None,
            past_key_values_length=None):
        if x == None:
            return inputs_embeds
        return x

def divide_bert_model(model, args):
    encoder = copy.deepcopy(model)
    encoder.bert.encoder.layer = encoder.bert.encoder.layer[:args.layer_idx]
    encoder.bert.pooler = Identity()
    encoder.dropout = Identity()
    encoder.classifier = Identity()
    classifier = copy.deepcopy(model)
    classifier.bert.embeddings = Identity()
    classifier.bert.encoder.layer = encoder.bert.encoder.layer[args.layer_idx:]
    return encoder, classifier

def extract_activation_hook(model, input, output):
    try:
        extracted_activations.append(output.detach().cpu().numpy())
    except:
        extracted_activations.append(output[0].detach().cpu().numpy())

def add_activation_hook(model, layer_idx, args):
    global extracted_activations
    extracted_activations = []
    bertlayers = [i for i in list(model.modules()) if isinstance(i, transformers.models.bert.modeling_bert.BertLayer)]
    module = bertlayers[layer_idx]
    module.register_forward_hook(extract_activation_hook)
    args.added = True

def get_bert_intermediate_output(model, samples, targets, masks, device):
    global extracted_activations
    extracted_activations = []
    targets = model(samples.to(device), attention_mask=masks.to(device), labels=targets.to(device)).logits
    samples = torch.from_numpy(np.concatenate(extracted_activations, axis=0)).to(device)
    return samples, targets

def load_topic_model(classifier, f_train, y_pred_train, y_train, f_val, y_pred_val, y_val, model, device, args, toy = True, use_cuda = True, train_masks = None, val_masks = None):
    start = time.time()
    if args.overall_method == 'kmeans':
        model, topic_vector = get_kmeans_concept(f_train, args.n_concept)
        end = time.time()
        args.logger.info(f'training time elapsed: {end - start}')
        return model, topic_vector
    if args.overall_method == 'pca':
        args.logger.info('starting')
        model, topic_vector = get_pca_concept(f_train, args.n_concept, args.logger)
        end = time.time()
        args.logger.info(f'training time elapsed: {end - start}')
        return model, topic_vector
    if args.overall_method == 'BCVAE':
        args.logger.info('starting BCVAE')
         # use normal as priors now
        prior_dist = dist.Normal()
        q_dist = dist.Normal()
        use_cuda = (device == torch.device('cuda'))
        if toy == True:
            hidden_dim = 64
        elif args.model_name == 'cnn':
            hidden_dim = f_train.shape[-2] * f_train.shape[-1]
        else:
            hidden_dim = f_train.shape[-1]
        if args.train_topic_model:
            args.logger.info('training')
            flattened = False
            if toy == True:
                f_train = f_train.swapaxes(1, 3).flatten(start_dim = 0, end_dim = -2) # size, 64, 4, 4 -> size, 4, 4, 64 -> size, 16, 64
            elif args.model_name == 'cnn':
                f_train = f_train.flatten(start_dim = 1, end_dim = -1)
            train_loader = DataLoader(dataset = f_train, batch_size = args.batch_size, shuffle = True, drop_last = False)
            # initialize model
            vae = VAE(z_dim=args.n_concept, use_cuda=use_cuda, prior_dist=prior_dist, q_dist=q_dist,
                include_mutinfo=True, tcvae=True, conv=False, mss=False, hidden_dim = hidden_dim)
            # setup the optimizer
            optimizer = optim.Adam(vae.parameters(), lr=3e-4)
            train_elbo = []
            # training loop
            dataset_size = len(train_loader.dataset)
            num_iterations = len(train_loader) * args.epochs
            iteration = 0
            # initialize loss accumulator
            elbo_running_mean = utils.RunningAverageMeter()
            while iteration < num_iterations:
                for i, x in enumerate(train_loader):
                    iteration += 1
                    batch_time = time.time()
                    vae.train()
                    anneal_kl(vae, iteration)
                    optimizer.zero_grad()
                    # transfer to GPU
                    x = x.cuda()
                    # wrap the mini-batch in a PyTorch Variable
                    x = Variable(x)
                    # do ELBO gradient and accumulate loss
                    obj, elbo = vae.elbo(x, dataset_size)
                    if utils.isnan(obj).any():
                        raise ValueError('NaN spotted in objective.')
                        # print('NaN spotted in objective.')
                    obj.mean().mul(-1).backward()
                    elbo_running_mean.update(elbo.mean().cpu().detach().item())
                    optimizer.step()

                    # report training diagnostics
                    if iteration % 1000 == 0:
                        train_elbo.append(elbo_running_mean.avg)
                        args.logger.info('[iteration %03d] time: %.2f \tbeta %.2f \tlambda %.2f training ELBO: %.4f (%.4f)'.format(
                            iteration, time.time() - batch_time, vae.beta, vae.lamb,
                            elbo_running_mean.val, elbo_running_mean.avg))

                        vae.eval()
                        torch.save(vae.state_dict(), f'{args.graph_save_folder}topic_model_{args.overall_method}_{args.layer_idx}.pth')
                        
            # plot training elbo
            plt.plot(train_elbo)
            plt.title('training elbo')
            plt.savefig(f'{args.graph_save_folder}training_{args.overall_method}_{args.layer_idx}.png')
            # Report statistics after training
            vae.eval()
            torch.save(vae.state_dict(), f'{args.graph_save_folder}topic_model_{args.overall_method}_{args.layer_idx}.pth')
            dataset_loader = DataLoader(train_loader.dataset, batch_size=1000, num_workers=1, shuffle=False)
            logpx, dependence, information, dimwise_kl, analytical_cond_kl, marginal_entropies, joint_entropy = \
                elbo_decomposition(vae, dataset_loader, hidden_dim)
            torch.save({
                'logpx': logpx,
                'dependence': dependence,
                'information': information,
                'dimwise_kl': dimwise_kl,
                'analytical_cond_kl': analytical_cond_kl,
                'marginal_entropies': marginal_entropies,
                'joint_entropy': joint_entropy
                }, args.graph_save_folder + 'elbo_decomposition.pth')
            topic_vec = vae.get_topic_word_dist()
            end = time.time()
            args.logger.info(f'training time elapsed: {end - start}')
            return vae, topic_vec.squeeze().transpose(1, 0)
        else:
            args.logger.info('loading')
            vae = VAE(z_dim=args.n_concept, use_cuda=use_cuda, prior_dist=prior_dist, q_dist=q_dist,
                include_mutinfo=True, tcvae=True, conv=False, mss=False, hidden_dim = hidden_dim)
            vae.load_state_dict(torch.load(f'{args.graph_save_folder}topic_model_{args.overall_method}_{args.layer_idx}.pth'))    
            topic_vec = vae.get_topic_word_dist()
            end = time.time()
            args.logger.info(f'training time elapsed: {end - start}')
            return vae, topic_vec.squeeze().transpose(1, 0)
    else:
        args.logger.info('loading topic model')
        if args.overall_method == 'conceptshap':
            args.reg_0 = 0 #flip loss
            args.ae_loss_reg = 0 #ae reconstruction
        if args.train_topic_model:
            if args.overall_method == 'conceptshap':
                args.logger.info('using ground truth as conceptshap')
                if args.dataset == 'imdb' and args.model_name == 't5':
                    # lm logits on #1465 is positive, 2841 is negative
                    y_train[y_train==1] = 1465
                    y_train[y_train==0] = 2841
                    y_val[y_val==1] = 1465
                    y_val[y_val==0] = 2841
                train_data = TensorDataset(f_train, y_train)
                valid_data = TensorDataset(f_val, y_val)
            elif not args.divide_bert:
                train_data = TensorDataset(f_train, y_pred_train)
                valid_data = TensorDataset(f_val, y_pred_val)
            else: #divIde bert
                train_data = TensorDataset(f_train, y_train, train_masks)
                valid_data = TensorDataset(f_val, y_val, val_masks)
                encoder, classifier = divide_bert_model(model, args)
                encoder = encoder.to(device)
                classifier = classifier.to(device)
                encoder.eval()
                classifier.eval()
                f_train = torch.rand(10, 768)
                add_activation_hook(model, args.layer_idx, args)
            train_loader = DataLoader(train_data, shuffle=True, batch_size=args.batch_size, drop_last=False)
            valid_loader = DataLoader(valid_data, shuffle=True, batch_size=args.batch_size, drop_last=False)

            #loss criterion
            if args.model_name == 't5':
                criterion = nn.CrossEntropyLoss(ignore_index=-100)
            elif args.dataset in ['imdb', 'toy']:
                criterion = nn.BCELoss().to(device)
            else:
                criterion = nn.CrossEntropyLoss().to(device)
            args.logger.info(f'f_train.shape: {f_train.shape}')
            args.logger.info(f'f_train.dim(): {f_train.dim()}')
            if f_train.dim() >2:
                args.logger.info('3d model..')
                topic_model = topic_model_toy(criterion, classifier, f_train, args.n_concept, args.thres, device, args)
            else:
                args.logger.info('2d model..')
                topic_model = topic_model_main(criterion, classifier, f_train, args.n_concept, args.thres, device, args)
            
            # Only optimize the new parameters
            total_number_params = sum([np.prod(p.size()) for p in topic_model.parameters()])
            args.logger.info('{} total parameters'.format(total_number_params))
            trainable_params = filter(lambda p: p.requires_grad, topic_model.parameters())
            non_trainable_params = filter(lambda p: not p.requires_grad, topic_model.parameters())
            trainable_number_params = sum([np.prod(p.size()) for p in trainable_params])
            args.logger.info('{} parameters to train'.format(trainable_number_params))
            args.logger.info('non_trainable_params {}'.format(non_trainable_params))
            topic_model.to(device)

            # Adam optimizer
            main_items = ["ae_loss", "flip_loss", "pred_loss", "concept_sim", "concept_far", "total_loss", "accuracy"]
            epoch_history = {}
            for item in main_items:
                epoch_history[item] = []
                epoch_history['val_'+item] = []
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, topic_model.parameters()), lr = args.lr)
            
            for ep in range(args.epochs):
                batch_history = {}
                for item in main_items:
                    batch_history[item] = []
                    batch_history['val_'+item] = []
                topic_model.train()
                if args.overall_method == 'two_stage':
                    if ep < args.shap_epochs or args.reg_0 ==0: #flip loss ablation
                        causal = False
                    else:
                        causal = True
                        if args.freeze:
                            args.logger.info('freezing')
                            #in two-stage training, freeze cc other weights except for the topic vector
                            for param in topic_model.parameters():
                                param.requires_grad = False
                            topic_model.topic_vector.requires_grad = True
                            trainable_params = filter(lambda p: p.requires_grad, topic_model.parameters())
                            trainable_number_params = sum([np.prod(p.size()) for p in trainable_params])
                        # Adam optimizer
                        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, topic_model.parameters()))
                else:
                    causal = False #all other models
                for i, batch in enumerate(train_loader):
                    if args.divide_bert:
                        (samples, targets, masks) = batch
                        samples, targets = get_bert_intermediate_output(model, samples, targets, masks, device)
                        samples = samples.flatten(start_dim = 0, end_dim = 1)
                        targets = targets.argmax(-1)
                    else:
                        (samples, targets) = batch
                        samples = samples.to(device).float()
                        targets = targets.to(device)
                    optimizer.zero_grad()
                    samples.requires_grad = True
                    predictions, flip_loss, concept_sim, concept_far, _, ae_loss = topic_model(samples, causal, targets)
                    if 'bert' in args.model_name:
                        predictions = F.softmax(predictions, dim = 1)
                        if args.dataset=='imdb':
                            try:
                                predictions = predictions[:, 1]
                            except:
                                args.logger.info(f'predictions.shape: {predictions.shape}')
                                args.logger.info(f'targets.shape: {targets.shape}')
                                raise Exception('end')
                    if args.model_name == 't5':
                        try:
                            pred_loss = criterion(predictions.view(-1, predictions.size(-1)).float(), targets.float())
                        except:
                            pred_loss = criterion(predictions.view(-1, predictions.size(-1)).float(), targets.long())
                    else:
                        try:
                            pred_loss = criterion(predictions.squeeze(), targets.squeeze().float().round().float()) #here is wrong!
                        except:
                            pred_loss = criterion(predictions.squeeze(), targets.squeeze().float().round().long()) #here is wrong!
                            
                    loss = args.ae_loss_reg * ae_loss + args.reg_0 * flip_loss + args.reg_1 * pred_loss + args.reg_2 * concept_sim + args.reg_3 * concept_far
                    
                    if args.model_name =='t5':
                        size = (predictions.size(0))
                        acc = (predictions.argmax(-1).squeeze() == targets.squeeze()).sum().item()
                    elif args.dataset!= 'imdb':
                        if toy == True:
                            size = (predictions.size(0)*predictions.size(1))
                            predictions = predictions.round()
                            acc = sum(sum(predictions.detach().cpu().numpy()==targets.detach().cpu().numpy()))
                        else:
                            size = (predictions.size(0))
                            predictions = predictions.max(1).indices
                            acc = sum(predictions.detach().cpu().numpy()==targets.detach().cpu().numpy())
                    else:
                        try:
                            size = (predictions.size(0))
                            acc = (predictions.float().round().squeeze() == targets.float().round().squeeze()).sum().item()
                        except:
                            args.logger.info(f'predictions: {predictions}')
                            args.logger.info(f'predictions.shape: {predictions.shape}')
                            args.logger.info(f'targets: {targets}')
                            args.logger.info(f'targets.shape: {targets.shape}')
                    acc = acc/size
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(topic_model.parameters(), 5)
                    optimizer.step()
                    
                    if causal:
                        batch_history["flip_loss"].append(args.reg_0 * flip_loss.item())
                    else:
                        batch_history["flip_loss"].append(args.reg_0 * flip_loss)

                    batch_history["pred_loss"].append(args.reg_1 * pred_loss.item())
                    batch_history["concept_sim"].append(args.reg_2 * concept_sim.item())
                    batch_history["ae_loss"].append(args.ae_loss_reg * ae_loss.item())
                    batch_history["concept_far"].append(args.reg_3 * concept_far.item())
                    batch_history["total_loss"].append(loss.item())
                    batch_history["accuracy"].append(acc)
                        
                for item in main_items:
                    epoch_history[item].append(np.mean(batch_history[item]))
                topic_model.eval()
                with torch.no_grad():        
                    # validation loop
                    for i, batch in enumerate(valid_loader):
                        if args.divide_bert:
                            (samples, targets, masks) = batch
                            samples, targets = get_bert_intermediate_output(model, samples, targets, masks, device)
                            samples = samples.flatten(start_dim = 0, end_dim = 1)
                            targets = targets.argmax(-1)
                        else:
                            (samples, targets) = batch
                            samples = samples.to(device).float()
                            targets = targets.to(device)
                        predictions, flip_loss, concept_sim, concept_far, topic_prob_n, ae_loss = topic_model(samples, causal, targets)
                        if 'bert' in args.model_name:
                            predictions = F.softmax(predictions, dim = 1)
                            if args.dataset=='imdb':
                                predictions = predictions[:, 1]
                        if args.model_name == 't5':
                            pred_loss = criterion(predictions.view(-1, predictions.size(-1)), targets.long())
                        else:
                            try:
                                pred_loss = criterion(predictions.squeeze(), targets.squeeze().float())
                            except:
                                pred_loss = criterion(predictions.squeeze(), targets.squeeze().long())
                        loss = args.ae_loss_reg * ae_loss + args.reg_0 * flip_loss + args.reg_1 * pred_loss + args.reg_2 * concept_sim + args.reg_3 * concept_far
                        
                        if args.model_name =='t5':
                            size = (predictions.size(0))
                            acc = (predictions.argmax(-1).squeeze() == targets.squeeze()).sum().item()
                        elif args.dataset!= 'imdb':
                            if toy == True:
                                size = (predictions.size(0)*predictions.size(1))
                                predictions = predictions.round()
                                acc = sum(sum(predictions.detach().cpu().numpy()==targets.detach().cpu().numpy()))
                            else:
                                size = (predictions.size(0))
                                predictions = predictions.max(1).indices
                                acc = sum(predictions.detach().cpu().numpy()==targets.detach().cpu().numpy())
                        else:
                            size = (predictions.size(0))
                            acc = (predictions.float().round().squeeze() == targets.float().round().squeeze()).sum().item()
                        acc = acc/size
                        if causal:
                            batch_history["val_flip_loss"].append(args.reg_0 * flip_loss.item())
                        else:
                            batch_history["val_flip_loss"].append(args.reg_0 * flip_loss)

                        batch_history["val_pred_loss"].append(args.reg_1 * pred_loss.item())
                        batch_history["val_concept_sim"].append(args.reg_2 * concept_sim.item())
                        batch_history["val_ae_loss"].append(args.ae_loss_reg * ae_loss.item())
                        batch_history["val_concept_far"].append(args.reg_3 * concept_far.item())
                        batch_history["val_total_loss"].append(loss.item())
                        batch_history["val_accuracy"].append(acc)

                #save this epoch's validation accuracy for evaluation
                val_accuracy = np.mean(batch_history["val_accuracy"])

                for item in main_items:
                    if item == 'accuracy':
                        epoch_history["val_accuracy"].append(val_accuracy)
                    else:
                        epoch_history[f"val_{item}"].append(np.mean(batch_history[f"val_{item}"]))
                if args.early_stopping:
                    if val_accuracy<=0.59: # can change here
                        args.logger.info(f'EARLY STOPPING AT EPOCH {ep}')
                        break
                print_history = {}
                for key in epoch_history.keys():
                    print_history[key] = epoch_history[key][-1]
                args.logger.info(f'epoch {ep} out of {args.epochs}: {print_history}')
            args.logger.info('STOPPED')
            if args.n_concept==10:
                torch.save(topic_model, f'{args.graph_save_folder}{args.overall_method}_layer_{args.layer_idx}_{args.ae_loss_reg}_{args.reg_0}_{args.reg_1}_{args.reg_2}_{args.reg_3}.pkl')
            else:
                torch.save(topic_model, f'{args.graph_save_folder}{args.overall_method}_{args.n_concept}_concept_layer_{args.layer_idx}_{args.ae_loss_reg}_{args.reg_0}_{args.reg_1}_{args.reg_2}_{args.reg_3}.pkl')
            
            args.logger.info('MODEL SAVED AT EPOCH {}'.format(ep+1))
            topic_vec = topic_model.topic_vector.cpu().detach().numpy()
            
            fig, axs = plt.subplots(4, 2)
            metrics = ['ae_loss', 'pred_loss', 'flip_loss', 'concept_sim', 'concept_far', 'total_loss', 'accuracy']

            # PLOTTING
            for i in range(len(metrics)):
                met = metrics[i]
                axs[i//2, i%2].plot(epoch_history[met], label='train')
                axs[i//2, i%2].plot(epoch_history['val_' + met], label = 'val')
                axs[i//2, i%2].legend()
                axs[i//2, i%2].set_title(met, fontsize='small')
            plt.savefig(args.graph_save_folder + f'{args.overall_method}_layer_{args.layer_idx}_{args.ae_loss_reg}_{args.reg_0}_{args.reg_1}_{args.reg_2}_{args.reg_3}.png')
        else:
            if args.n_concept==10:
                topic_model = torch.load(args.graph_save_folder + f'{args.overall_method}_layer_{args.layer_idx}_{args.ae_loss_reg}_{args.reg_0}_{args.reg_1}_{args.reg_2}_{args.reg_3}.pkl')
            else:
                topic_model = torch.load(f'{args.graph_save_folder}{args.overall_method}_{args.n_concept}_concept_layer_{args.layer_idx}_{args.ae_loss_reg}_{args.reg_0}_{args.reg_1}_{args.reg_2}_{args.reg_3}.pkl')
            topic_vec = topic_model.topic_vector.cpu().detach().numpy()
            args.logger.info('topic model loaded')
    end = time.time()
    args.logger.info(f'training time elapsed: {end - start}')
    return topic_model, topic_vec

def run_model_to_get_pred(args, f_val, toy, topic_model, model, classifier, device, perturb = -1, targets = None, masks = None):
    if args.overall_method == 'pca':
        flattened = False
        if f_val.dim() == 4: #toy dataset
            flattened = True
            original_shape = f_val.shape
            f_val = f_val.flatten(start_dim = 1, end_dim = -1)
        elif f_val.dim()>2:
            flattened = True
            original_shape = f_val.shape
            f_val = f_val.flatten(start_dim = 0, end_dim = -2)
        y_pred = topic_model.transform(f_val)
        if perturb >=0: 
            topic_model = copy.deepcopy(topic_model)
            topic_model.components_[:, perturb] = 0
        y_pred = torch.from_numpy(topic_model.inverse_transform(y_pred)).float().to(device)
        if flattened:
            y_pred = y_pred.view(original_shape)
            if args.dataset!='toy':
                y_pred = torch.mean(y_pred, axis = -1) #cnn models
        y_pred = classifier(y_pred)
    elif args.overall_method == 'kmeans':
        flattened = False
        if f_val.dim()>2:
            flattened = True
            original_shape = f_val.shape
            f_val = f_val.flatten(start_dim = 0, end_dim = -2)
        y_pred = topic_model.transform(f_val) #size, n_concept #distance
        y_pred = torch.from_numpy(1/(1+y_pred)) # similarity
        if perturb >= 0:
            topic_model = copy.deepcopy(topic_model)
            topic_model.cluster_centers_[perturb, :] = 0 #assign similarity to a concept to be zero
        sums = y_pred.sum(axis = 1)# make sure they sum up to one / 20
        y_pred = torch.div(y_pred.T, sums).T
        y_pred = torch.matmul(y_pred, torch.from_numpy(topic_model.cluster_centers_)).float().to(device) #size, hidden_dim
        if flattened:
            y_pred = y_pred.view(original_shape)
            if args.dataset!='toy' and args.model_name == 'cnn':
                y_pred = torch.mean(y_pred, axis = -1) #cnn models
        y_pred = classifier(y_pred)
    elif args.overall_method == 'BCVAE':
        shape = f_val.shape
        if toy == True:
            f = f_val.swapaxes(1, 3).flatten(start_dim = 0, end_dim = -2) # size, 64, 4, 4 -> size, 4, 4, 64 -> size, 16, 64
        elif args.model_name == 'cnn':
            f = f_val.flatten(start_dim = 1, end_dim = -1)
        else:
            f = f_val
        z, params = topic_model.encode(f.to(device))
        if perturb >=0:
            z[:, perturb] = 0
        f_reconstructed, params = topic_model.decode(z)
        if toy:
            f_reconstructed = f_reconstructed.view(shape[0], shape[-2], shape[-1], -1).swapaxes(1,3)
        elif args.model_name == 'cnn':
            f_reconstructed = f_reconstructed.view(shape)
        valid_data = TensorDataset(f_reconstructed)
        valid_loader = DataLoader(valid_data, shuffle=False, batch_size=args.batch_size, drop_last=False)
        y_pred = []
        for [f] in valid_loader:
            if args.dataset!='toy' and args.model_name == 'cnn':
                f = torch.mean(f, axis = -1) #cnn models
            y = classifier(f)
            y_pred.append(y.cpu().detach())
        y_pred = torch.cat(y_pred, axis = 0).squeeze(1)
    else: #concept models
        if args.divide_bert:
            args.logger.info(f'f_val.shape: {f_val.shape}')
            args.logger.info(f'targets.unsqueeze(1).shape: {torch.from_numpy(targets).unsqueeze(1).shape}')
            args.logger.info(f'masks.shape: {masks.shape}')
            valid_data = TensorDataset(torch.from_numpy(f_val), torch.from_numpy(targets).unsqueeze(1), torch.from_numpy(masks))
        else:
            valid_data = TensorDataset(f_val)
        valid_loader = DataLoader(valid_data, shuffle=False, batch_size=args.batch_size, drop_last=False)
        y_pred_old = []
        for batch in valid_loader:
            if args.divide_bert:
                samples, targets, masks = batch
                if perturb >= 0:
                    # get intermediate output
                    f, _ = get_bert_intermediate_output(model, samples, targets, masks, device)
                    f = f.flatten(start_dim = 0, end_dim = 1)
                    # pass into topic_model
                    y, _, _, _, _, _ = topic_model(f.type(torch.FloatTensor).to(device), 'conceptshap', 0, perturb = perturb)
                else:
                    try:
                        targets = targets.squeeze().argmax(-1)
                        y = model(samples.to(device), attention_mask=masks.to(device), labels=targets.to(device)).logits
                    except:
                        args.logger.info(f'samples.shape: {samples.shape}')
                        args.logger.info(f'masks.shape: {masks.shape}')
                        args.logger.info(f'targets.shape: {targets.shape}')
                        raise Exception('end')
            else:
                f = batch[0]
                y, _, _, _, _, _ = topic_model(f.type(torch.FloatTensor).to(device), 'conceptshap', 0, perturb = perturb)
            y_pred_old.append(y.cpu().detach())
        y_pred = torch.cat(y_pred_old, axis = 0) 
    if 'bert' in args.model_name:
        if args.dataset=='imdb':    
            y_pred = F.softmax(y_pred, dim = 1)[:, 1]
        else: #news
            y_pred = F.softmax(y_pred, dim = 1)
    return y_pred

def eval_causal_effect_model(topic_model, model, classifier, device, f_val, y_val, topic_vec, args, last_valid = True, toy = True, model_name = False, val_masks = None):
    with torch.no_grad():
        if args.overall_method in ['pca', 'kmeans']:
            y_pred = run_model_to_get_pred(args, f_val, toy, topic_model, model, classifier, device).cpu().detach()
        else:
            if args.load_cs:
                topic_vector = topic_model.topic_vector
                topic_model = torch.load(args.save_dir + 'conceptshap/topic_model_conceptshap.pkl')
                topic_model.topic_vector = topic_vector
            topic_model = topic_model.to(device)
            y_pred = run_model_to_get_pred(args, f_val, toy, topic_model, model, classifier, device, targets = y_val, masks = val_masks).cpu().detach()
    if args.model_name == 't5':
        new_y_pred = y_pred.argmax(-1)
        new_y_pred = get_t5_output(list(new_y_pred.numpy()), args, logits = False)
        original_acc = sum(new_y_pred.numpy() == y_val)
    elif args.dataset!='imdb' and args.dataset!='toy':
        args.logger.info(f'y_val.shape: {y_val.shape}')
        args.logger.info(f'y_pred.shape: {y_pred.shape}')
        args.logger.info(f'y_val[:10]: {y_val[:10]}')
        new_y_pred = y_pred.max(1).indices
        args.logger.info(f'new_y_pred[:10]: {new_y_pred[:10]}')
        args.logger.info(f'new_y_pred.shape: {new_y_pred.shape}')
        try:
            original_acc = sum(new_y_pred.numpy() == y_val)
        except:
            y_val = y_val.squeeze().argmax(-1)
            original_acc = sum(new_y_pred.numpy() == y_val)
    else:
        args.logger.info(f'y_pred[:10]: {y_pred[:10]}')
        new_y_pred = y_pred
        original_acc = (new_y_pred.cpu().detach().numpy().squeeze().round() == y_val.squeeze().round()).sum().item()
    args.logger.info('1original accuracy: {} \n'.format(original_acc))
    if toy == True:
        original_acc = original_acc / (new_y_pred.shape[0]*new_y_pred.shape[1])
    else:
        original_acc = original_acc / new_y_pred.shape[0]
    args.logger.info('2original accuracy: {} \n'.format(original_acc))
    write = ''
    write += str(args)
    write += '\n'
    write += 'original accuracy: {} \n'.format(original_acc)
    overall_effects = []
    overall_accs_change = []
    y_pred_grouped = None
    for i in range(topic_vec.shape[1]):
        if args.overall_method in ['pca', 'kmeans']:
            y_pred_perturbed = run_model_to_get_pred(args, f_val, toy, topic_model, model, classifier, device, perturb = i).cpu().detach()
        else:
            topic_model = topic_model.to(device)
            y_pred_perturbed = run_model_to_get_pred(args, f_val, toy, topic_model, model, classifier, device, perturb = i, targets = y_val, masks = val_masks).cpu().detach()
        if args.model_name == 't5':
            if args.dataset == 'imdb':
                li_indices = [1465, 2841]
            else:
                li_indices = [1150, 5716, 1769, 2854]
            indices = torch.tensor(li_indices)
            other_indices = torch.tensor([i for i in range(y_pred.shape[-1]) if i not in li_indices])
            if y_pred_grouped == None:
                softmaxed = F.softmax(y_pred, dim=-1)
                y_pred_other_dim = torch.index_select(softmaxed, 1, other_indices).sum(-1).unsqueeze(1)
                y_pred_main_dims = torch.index_select(softmaxed, 1, indices)
                y_pred_grouped = torch.cat([y_pred_main_dims, y_pred_other_dim], 1)
                y_pred_grouped = F.softmax(y_pred_grouped, dim = -1)
            softmaxed = F.softmax(y_pred_perturbed, dim=-1)
            y_pred_perturbed_other_dim = torch.index_select(softmaxed, 1, other_indices).sum(-1).unsqueeze(1)
            y_pred_perturbed_main_dims = torch.index_select(softmaxed, 1, indices)
            y_pred_grouped_perturbed = torch.cat([y_pred_perturbed_main_dims, y_pred_perturbed_other_dim], 1)
            y_pred_grouped_perturbed = F.softmax(y_pred_grouped_perturbed, dim = -1)
            effect = y_pred_grouped - y_pred_grouped_perturbed
            effect = torch.mean(torch.abs(effect)).item()
        else:
            effect = torch.mean(torch.abs(y_pred - y_pred_perturbed)).item()
        if args.model_name == 't5':
            y_pred_perturbed = y_pred_perturbed.argmax(-1)
            y_pred_perturbed = get_t5_output(list(y_pred_perturbed.numpy()), args, logits = False)
            new_acc = sum(y_pred_perturbed.numpy() == y_val)
        elif args.dataset not in ['imdb', 'toy']:
            y_pred_perturbed = y_pred_perturbed.max(1).indices
            new_acc = sum(y_pred_perturbed.numpy() == y_val)
        else:
            new_acc = (y_pred_perturbed.cpu().detach().numpy().squeeze().round() == y_val.squeeze().round()).sum().item()
        if toy == True:
            new_acc = new_acc / (y_pred_perturbed.shape[0]*y_pred_perturbed.shape[1])
        else:
            new_acc = new_acc / y_pred_perturbed.shape[0]
        overall_accs_change.append(new_acc - original_acc)
        y_pred_perturbed = y_pred_perturbed.cpu().detach().numpy()
        
        write += 'Concept {}: {}'.format(i, effect)
        write += '; Accuracy change: {} \n'.format(overall_accs_change[-1])
        overall_effects.append(effect)
    overall_accs_change = np.abs(overall_accs_change)
    write += 'Average effect: {}; standard deviation: {}\n'.format(np.mean(overall_effects), np.std(overall_effects))
    write += 'Average accuracy change: {}; standard deviation: {}'.format(np.mean(overall_accs_change), np.std(overall_accs_change))
    if args.one_correlated_dimension and last_valid:
        write += '\nWithout last dimension:\n'
        write += 'Average effect: {}; standard deviation: {}\n'.format(np.mean(overall_effects[:-1]), np.std(overall_effects[:-1]))
        write += 'Average accuracy change: {}; standard deviation: {}'.format(np.mean(overall_accs_change[:-1]), np.std(overall_accs_change[:-1]))

    return write, np.array(overall_effects), overall_accs_change

def postprocess(topic_model, model, classifier, device, f_val, y_val, topic_vec, args, toy = True, model_name = False, val_masks = None):
    skipped = False
    last_valid = True
    if args.postprocess:
        write, overall_effects, overall_accs_change = eval_causal_effect_model(topic_model, model, classifier, device, f_val, y_val, topic_vec, args, toy, model_name, val_masks = val_masks)
        # filter out small effect concepts
        valid_indices = torch.from_numpy(np.where(overall_effects>1e-5)[0]).to(device)
        args.logger.info(f'valid_indices: {valid_indices}')
        if (args.n_concept - 1) not in valid_indices:
            last_valid = False
            args.logger.info('last dimension nonvalid')
        if len(valid_indices) != topic_model.n_concept and len(valid_indices)!=0:
            # change topic_model accordingly
            topic_model.n_concept = len(valid_indices)
            args.logger.info(f'topic_vector.shape: {topic_model.topic_vector.shape}')
            original_topic_vector = topic_model.topic_vector
            topic_model.topic_vector = nn.Parameter(torch.index_select(topic_model.topic_vector, -1, valid_indices))
            args.logger.info(f'topic_vector.shape: {topic_model.topic_vector.shape}')
            args.logger.info(f'rec_vector_1.shape: {topic_model.rec_vector_1.shape}')
            topic_model.rec_vector_1 = nn.Parameter(torch.index_select(topic_model.rec_vector_1, 0, valid_indices))
            args.logger.info(f'rec_vector_1.shape: {topic_model.rec_vector_1.shape}')
            topic_vec = topic_model.topic_vector
            if args.n_concept-1 not in list(valid_indices.detach().cpu().numpy()):
                args.one_correlated_dimension = False #not calculate without
            # save the new topic model
            torch.save(topic_model, args.graph_save_folder + 'topic_model_' + args.overall_method + '.pkl')
        else:
            skipped = True
    if args.eval_causal_effect:
        if not skipped:
            # do new evaluation
            write, overall_effects, overall_accs_change = eval_causal_effect_model(topic_model, model, classifier, device, f_val, y_val, topic_vec, args, toy = toy, model_name = model_name, last_valid = last_valid, val_masks = val_masks)
        args.logger.info(write)
    return topic_model

# PERTURB
def perturbation_analysis(x_val, y_val, concept, model, batch_size, device, cov, p, save_folder, shapes_to_test = range(15)):
    valid_data = TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))
    valid_loader = DataLoader(valid_data, shuffle=True, batch_size=args.batch_size, drop_last=False)
    accs = []
    preds = []
    with torch.no_grad():
        for i, (samples, targets) in enumerate(valid_loader):
            samples = samples.to(device=device, dtype=torch.float)
            targets = targets.to(device=device, dtype=torch.float)
            predictions, _ = model(samples)
            preds.append(predictions.cpu().detach())
            acc = (predictions.round().squeeze() == targets).sum().item()
            acc = acc / (predictions.size(0)*predictions.size(1))
            accs.append(acc)
    preds = torch.cat(preds, axis = 0)
    preds_mean = preds.mean(axis = 1).numpy()

    df1_dict = {'shape_to_exclude': ['none'], 'accuracy_without': [np.mean(accs)], 'mean_pred_change': [0]}
    df2_dict = {'shape_to_exclude': ['none']}

    preds_mean_hor = preds.mean(axis = 0).numpy()
    for i in range(preds.shape[1]):
        df2_dict['class_{}'.format(i)] = [preds_mean_hor[i]]

    avg_preds_changes = []
    for m in shapes_to_test:
        #mask out the shape
        n1 = x_val.shape[0]
        concept_wo = concept[-n1:, :]
        concept_wo[:, m] = concept_wo[:, m]*0
        x_wo, y_wo = toy_helper_v2.create_dataset(n1, cov, p = p, return_directly = True, concept = concept_wo)
        x_wo = x_wo.swapaxes(1, 3)
        valid_data_wo = TensorDataset(torch.from_numpy(x_wo), torch.from_numpy(y_wo))
        valid_loader_wo = DataLoader(valid_data_wo, shuffle=True, batch_size=args.batch_size, drop_last=False)
        accs_wo = []
        preds_wo = []
        with torch.no_grad():
            for i, (samples, targets) in enumerate(valid_loader_wo):
                samples = samples.to(device=device, dtype=torch.float)
                targets = targets.to(device=device, dtype=torch.float)
                predictions, _ = model(samples)
                preds_wo.append(predictions.cpu().detach())
                acc = (predictions.round().squeeze() == targets).sum().item()
                acc = acc / (predictions.size(0)*predictions.size(1))
                accs_wo.append(acc)
        preds_wo = torch.cat(preds_wo, axis = 0)
        preds_mean_wo_new = preds_wo.mean(axis = 1).numpy()

        avg_preds_changes.append(np.abs(preds_mean_wo_new - preds_mean))
        #first csv: accuracy change, pred change
        df1_dict['shape_to_exclude'].append(m)
        df1_dict['accuracy_without'].append(np.mean(accs_wo))
        df1_dict['mean_pred_change'].append(np.mean(np.abs(preds_mean - preds_mean_wo_new)))
        #second csv: mean pred for each class
        df2_dict['shape_to_exclude'].append(m)

        preds_mean_wo = preds_wo.mean(axis = 0).numpy()
        for i in range(preds_mean_wo.shape[0]):
            df2_dict['class_{}'.format(i)].append(preds_mean_wo[i])

    avg_pred_change = np.mean(df1_dict['mean_pred_change'][1:])
    df1_dict['p_value'] = [None]
    for i, m in enumerate(shapes_to_test):
        # Null Hypothesis: mean is the same
        # p<0.05: reject null hypothesis, mean is not the same
        ztest, pval = stests.ztest(avg_preds_changes[i], x2 = None, value = avg_pred_change)
        df1_dict['p_value'].append(pval)

    df1 = pd.DataFrame.from_dict(df1_dict)
    df2 = pd.DataFrame.from_dict(df2_dict)
    df1_file = save_folder + 'perturbation_analysis_1.csv'
    df2_file = save_folder + 'perturbation_analysis_2.csv'
    #save results
    df1.to_csv(df1_file)
    df2.to_csv(df2_file)

    num = sum([p<0.05 for p in df1_dict['p_value'][1:]])
    args.logger.info('{} dependent concepts'.format(num))
    return num