import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
from BERT_explainability.modules.layers_ours import NormLayer
import nltk
from nltk.corpus import stopwords
from BERT_explainability.modules.BERT.ExplanationGenerator import Generator
from BERT_explainability.modules.BERT.BertForSequenceClassification import BertForSequenceClassification
from BERT_explainability.modules.BERT.ExplanationGenerator import Generator
from BERT_explainability.modules.layers_ours import *
import torch
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from matplotlib import pyplot as plt
import os
from utils import returnCAM, get_bert_intermediate_output, add_activation_hook, extract_activation_hook
from torch.utils.data import TensorDataset, DataLoader

def contains_letters(z):
    return (z.isupper() or z.islower())

def only_letters(z):
    return z.isalpha()

def gradcam_analysis(model, tokenizer, topic_model, f_train, x_train, max_len, save_file, args):
    stop_word_list = stopwords.words('english') + ['[CLS]', '[UNK]', '[PAD]', "<PAD>", "<START>", "<UNK>", "<UNUSED>"]
    if tokenizer == None:
        # word-index dictionary
        index_offset = 3
        word_index = imdb.get_word_index(path="imdb_word_index.json")
        word_index = {k: (v + index_offset) for k,v in word_index.items()}
        word_index["<PAD>"] = 0
        word_index["<START>"] = 1
        word_index["<UNK>"] = 2
        word_index["<UNUSED>"] = 3
        index_to_word = { v: k for k, v in word_index.items()}
    nc, n_concepts = topic_model.topic_vector.shape
    print('nc = {} and n_concepts = {}'.format(nc, n_concepts))
    #replace the original classifier to a classifier to topics, trained in the topic model
    topic_classifier = nn.Linear(nc, n_concepts, bias = False)
    with torch.no_grad():
        topic_vector_n = F.normalize(topic_model.topic_vector, dim = 0, p=2)
        topic_classifier.weight = nn.Parameter(topic_vector_n.T, requires_grad = True)
    model.classifier = topic_classifier
    f_train = f_train.squeeze().unsqueeze(-2)
    print('f_train.shape: ', f_train.shape)
    params = list(model.parameters()) 
    weight_softmax = np.squeeze(params[-1].data.cpu().numpy()) #use the last one because bias=False in our case
    print('weight_softmax.shape: ', weight_softmax.shape) #4, 100

    cams = returnCAM(f_train, weight_softmax, range(n_concepts), max_len)
    
    write = ''

    for c_idx in range(len(cams)):
        sub_cams = torch.from_numpy(cams[c_idx])
        print('sub_cams.shape: ', sub_cams.shape) # size, 400
        #gather the words with the highest cams
        tops = 500
        v, i = sub_cams.flatten().topk(tops, largest = True) #largest
        indices = np.array(np.unravel_index(i.numpy(), sub_cams.shape))
        words = []
        for i in range(tops):
            try:
                x = x_train[indices[0][i]][indices[1][i]]
            except:
                print('x_train.shape: ', x_train.shape)
                print('indices[0][i]: ', indices[0][i])
                print('indices[1][i]: ', indices[1][i])
                raise Exception('end')
            if tokenizer == None:
                w = index_to_word[x]
            else:
                w = tokenizer.convert_ids_to_tokens([x])[0]
            if w!='' and w not in stop_word_list and only_letters(w):
                words.append(w)
        try:
            cx = Counter(words)
        except:
            print(words)
            raise Exception('end')
        if args.visualize_wordcloud:
            if not os.path.exists(f'{args.graph_save_folder}wordclouds_{args.layer_idx}/'):
                os.makedirs(f'{args.graph_save_folder}wordclouds_{args.layer_idx}/')
            wordcloud = WordCloud(background_color='white').generate(' '.join(words))
            # Display the generated image:
            plt.axis("off")
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.savefig(f'{args.graph_save_folder}wordclouds_{args.layer_idx}/topic_{c_idx}.png')
        most_occur = cx.most_common(50)
        most_occur = [mo[0] for mo in most_occur]
        print("Concept " + str(c_idx) + " most common words: \n")
        write += "Concept " + str(c_idx) + " most common words: \n"
        print(str(most_occur).replace('\'', '').replace('[', '').replace(']', '') + '\n')
        write += str(most_occur).replace('\'', '').replace('[', '').replace(']', '') + '\n'
        print("\n\n")
        write += '\n\n'
    #save the results to a file
    text_file = open(save_file, "w")
    n = text_file.write(write)
    text_file.close()
    print('SAVED!')

def get_exp(i, x_val, y_val, val_mask, t, tokenizer, device, model, explanations):
    x=x_val[i]
    y=y_val[i]
    input_ids = torch.from_numpy(x).unsqueeze(0).to(device)
    attention_mask = val_mask[i].unsqueeze(0).to(device)
    # true class is positive - 1
    true_class = y
    output = model(input_ids=input_ids, attention_mask=attention_mask)[0]
    expl = explanations.generate_LRP(input_ids=input_ids, attention_mask=attention_mask, start_layer=0, index = t)[0]
    # normalize scores
    expl = (expl - expl.min()) / (expl.max() - expl.min())
    tokens = tokenizer.convert_ids_to_tokens(input_ids.flatten())
    return expl, tokens

def bert_topics(model, tokenizer, topic_model, f_train, x_train, y_train, train_masks, args, save_file, device, overall_model = None):
    # change the model to be enabled with relprop
    if not args.divide_bert:
        model = BertForSequenceClassification.from_pretrained(args.model_save_dir).to(device)
    model.eval()
    # write the topic model to a sequential classifier
    linearlayer1 = Linear(768, args.n_concept, bias = False)
    if args.overall_method == 'pca':
        topic_vector_n = torch.from_numpy(topic_model.components_.T)
    else:
        topic_vector_n = F.normalize(topic_model.topic_vector, dim = 0, p=2).transpose(1,0)
    linearlayer1.weight = nn.Parameter(topic_vector_n)

    try:
        thres = topic_model.thres
    except:
        thres = 0.3
    
    print('thres: ', thres)

    linearlayer2 = Linear(args.n_concept, args.n_concept, bias = True)
    linearlayer2.weight = nn.Parameter(torch.eye(args.n_concept), requires_grad=False)
    linearlayer2.bias = nn.Parameter(torch.ones(args.n_concept)*(-thres), requires_grad = False)

    linearlayer3 = Linear(topic_model.rec_vector_1.shape[0], topic_model.rec_vector_1.shape[1], bias = False)
    linearlayer3.weight = nn.Parameter(topic_model.rec_vector_1.transpose(1,0))
    new_cut_classifier = Sequential(NormLayer(),
                            linearlayer1,
                                linearlayer2,
                                ReLU(),
                            NormLayer()).to(device)
    model.classifier = new_cut_classifier
    # initialize the explanations generator
    if not args.divide_bert:
        explanations = Generator(model)

    topic_model.to(device)
    y_train = torch.from_numpy(y_train).to(device)
    if not args.divide_bert:
        f_train = f_train.to(device)
        print(f'f_train.shape: {f_train.shape}')
        with torch.no_grad():
            _, _, _, _, topic_prob_n, _ = topic_model(f_train, 'conceptshap', y_train)
    else:
        train_data = TensorDataset(torch.from_numpy(x_train), y_train, train_masks)
        train_loader = DataLoader(train_data, shuffle=True, batch_size=args.batch_size, drop_last=True)
        topic_prob_n = []
        for batch in train_loader:
            (samples, targets, masks) = batch
            print('1samples.shape: ', samples.shape)
            samples, targets = get_bert_intermediate_output(overall_model, samples, targets, masks, device)
            print('2samples.shape: ', samples.shape)
            samples = samples.flatten(start_dim = 0, end_dim = 1)
            print('3samples.shape: ', samples.shape)
            # pass samples to topic model
            _, _, _, _, topic_prob, _ = topic_model(samples, 'conceptshap', targets)
            topic_prob_n.append(topic_prob.detach().cpu())
        topic_prob_n = torch.cat(topic_prob_n, dim = 0)
        print('topic_prob_n.shape: ', topic_prob_n.shape)#size*512, 10
    # filter stopwords
    sw = stopwords.words('english') + ['[CLS]', '[UNK]', '[PAD]']
    write = ''
    all_words = []
    for x in x_train:
        all_words += list(tokenizer.convert_ids_to_tokens(x.flatten())) #size*512
    print('len(all_words): ', len(all_words))
    for i in range(topic_prob_n.shape[1]):
        args.logger.info(f'Doing topic {i}!')
        write += f'Topic {i} \n'
        # instead of most common ones, use the most highlighted ones
        words = []
        topic_prob = topic_prob_n[:, i].cpu().detach().numpy()
        print(f'topic_prob.shape: {topic_prob.shape}')
        topk = topic_prob.argsort()[::-1][:args.topk]
        if args.divide_bert:
            words += [all_words[tk] for tk in topk if all_words[tk] not in sw and only_letters(all_words[tk])]
        else:
            for j in topk:
                prob = topic_prob[j]
                # only take ones with probabilities greater than 0.7
                if args.visualize=='most_common':
                    tokens = tokenizer.convert_ids_to_tokens(x_train[j].flatten())
                    words += tokens
                else:
                    exp, tokens = get_exp(j, x_train, y_train, train_masks, i, tokenizer, device, model, explanations)
                    exp = exp.cpu().detach().numpy()
                    # valid = np.where(exp>0.7)[0]
                    valid = np.argsort(x)[::-1][:3]
                    valid = [tokens[v] for v in valid]
                    words += valid
            words = [w for w in words if w not in sw]
            words = [w for w in words if only_letters(w)]
        if args.visualize!=None:
            cx = Counter(words)
            most_occur = cx.most_common(50)
            most_occur = [mo[0] for mo in most_occur]
            write += ','.join(most_occur) + '\n'
        if args.visualize_wordcloud:
            if not os.path.exists(f'{args.graph_save_folder}wordclouds_{args.layer_idx}/'):
                os.makedirs(f'{args.graph_save_folder}wordclouds_{args.layer_idx}/')
            wordcloud = WordCloud(background_color='white').generate(' '.join(words))
            # Display the generated image:
            plt.axis("off")
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.savefig(f'{args.graph_save_folder}wordclouds_{args.layer_idx}/topic_{i}.png')
    if args.visualize!=None:
        #save the results to a file
        text_file = open(save_file, "w")
        n = text_file.write(write)
        text_file.close()
        print('SAVED!')

def read_topics(save_file, topk=10):
    f = open(save_file, 'r')
    lines = f.readlines()
    print(f'{len(lines)} lines in total')
    lines = [lines[i*2+1] for i in range(len(lines)//2)]
    old_lines = [l.split(',') for l in lines]
    lines = []
    for l in old_lines:
        lines.append([w for w in l if only_letters(w)][:topk])
    print(f'{len(lines)} concept keywords extracted')
    return lines