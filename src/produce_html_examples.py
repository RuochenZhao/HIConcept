# import statements
from transformers import BertTokenizer
from BERT_explainability.modules.BERT.ExplanationGenerator import Generator
from BERT_explainability.modules.BERT.BertForSequenceClassification import BertForSequenceClassification
from transformers import DistilBertForSequenceClassification, DistilBertConfig
import visualization
import torch
import torch.nn.functional as F
import numpy as np
from BERT_explainability.modules.layers_ours import *
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import os
from BERT_explainability.modules.layers_ours import NormLayer
import argparse
from text_helper import load_data_text
from pathlib import Path
from visualize import read_topics
from random import shuffle
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from matplotlib import pyplot as plt
import pandas as pd
import copy

def new_model(model, topic_model):  
    model = copy.deepcopy(model)
    # write the topic model to a sequential classifier
    linearlayer1 = Linear(768, args.n_concept, bias = False)
    topic_vector_n = F.normalize(topic_model.topic_vector, dim = 0, p=2).transpose(1,0)
    linearlayer1.weight = nn.Parameter(topic_vector_n)

    thres = topic_model.thres
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
    return model

def no_bpe(tokens, attr_scores, only_delete = False):
    tokens = tokens.copy()
    attr_scores = attr_scores.copy()
    # find the first PAD token
    pad_idx = tokens.index('[PAD]')
    del tokens[pad_idx:]
    del attr_scores[pad_idx:]
    # delete the other tokens
    delete_tokens = ['[UNK]', '[SEP]', '[CLS]']
    for t in delete_tokens:
        while t in tokens:
            idx = tokens.index(t)
            del tokens[idx]
            del attr_scores[idx]
    if only_delete:
        return tokens, attr_scores
    # combine bpe tokens
    while '#' in ''.join(tokens):
        for i, t in enumerate(tokens):
            if '#' in t:
                start_idx = i-1
                tokens = tokens[:start_idx] + [''.join(tokens[i-1:i+1]).replace('#', '')] + tokens[i+1:]
                attr_scores = attr_scores[:start_idx] + [sum(attr_scores[i-1:i+1])] + attr_scores[i+1:]
                break
    return tokens, attr_scores

def reprocess_scores(truth_df):
    all_cc = []
    for cc_scores in truth_df['cc_scores'].values:
        all_cc.append(np.array([float(x) for x in cc_scores.replace('[', '').replace(']', '').split(',')]))
    all_cs = []
    for cs_scores in truth_df['cs_scores'].values:
        all_cs.append([float(x) for x in cs_scores.replace('[', '').replace(']', '').split(',')])
    all_text = []
    for text in truth_df['text'].values:
        all_text.append(text.split(' '))
    truth_df['cc_scores'] = all_cc
    truth_df['cs_scores'] = all_cs
    truth_df['text'] = all_text
    return truth_df


# arguments parser
parser = argparse.ArgumentParser(description="latentRE")
parser.add_argument("--dataset", dest="dataset", type=str,
                    default='news', choices=['news', 'imdb', 'toy'])
parser.add_argument("--dataset_cache_dir", dest="dataset_cache_dir", type=str,
                    default="../../hf_datasets/", help="dataset cache folder")
parser.add_argument("--n_concept", dest="n_concept", type=int,
                    default=10)
parser.add_argument("--n_samples", dest="n_samples", type=int,
                    default=100)
parser.add_argument("--max_features", dest="max_features", type=int,
                    default=100000)
parser.add_argument("--fast", dest="fast", type=int,
                    default=False)
parser.add_argument("--include_keywords", dest="include_keywords", type=bool,
                    default=True)
parser.add_argument("--layer_idx", dest="layer_idx", type=int,
                    default=-1)
parser.add_argument("--seed", dest="seed", type=int,
                    default=11)
parser.add_argument('--wordcloud', action = 'store_true')
parser.add_argument('--txt', action = 'store_true')
parser.add_argument('--reuse', action = 'store_true')
parser.add_argument('--method', type = str, default = None)
# parser.add_argument("--regenerate_topic_words", action = 'store_true')

args = parser.parse_args()

# if args.regenerate_topic_words:
#     if cs_or_cc[i] == 0: #cc
#         data += f'<h3>Other related keywords to the highlighted ones: </h3> \n <body>{topic_keywords_cc[t]}</body>\n'
#     elif cs_or_cc[i] == 1: #cs
#         data += f'<h3>Other related keywords to the highlighted ones: </h3> \n <body>{topic_keywords_cs[t]}</body>\n'
    
args.mask = True #should be default option
# load the bert model of interest
save_dir = f'models/{args.dataset}/bert/imdb_weights'
model = BertForSequenceClassification.from_pretrained(save_dir).to(device)
model.eval() #put in eval mode
# load the topic model of interest
classifier = model.classifier
f_val = torch.from_numpy(np.load(f'models/{args.dataset}/bert/val_embeddings_{args.layer_idx}.npy'))
pred_val = torch.from_numpy(np.load(f'models/{args.dataset}/bert/pred_val.npy'))
if args.fast:
    f_val = f_val[:int(args.fast*0.2)]
    pred_val = pred_val[:int(args.fast*0.2)]
print(os.getcwd())
if args.dataset == 'news':
    graph_save_folder = f'models/{args.dataset}/bert/two_stage/'
    topic_model_cc = torch.load(graph_save_folder + f'two_stage_layer_-1_1.0_1.0_1.0_0.1_0.5.pkl')
    graph_save_folder = f'models/{args.dataset}/bert/conceptshap/'
    topic_model_cs = torch.load(graph_save_folder + f'conceptshap_layer_-1_0_0_1.0_0.1_0.5.pkl')

model_cc = new_model(model, topic_model_cc)
model_cs = new_model(model, topic_model_cs)
# initialize the explanations generator
explanations_cc = Generator(model_cc)
explanations_cs = Generator(model_cs)
if args.dataset == 'imdb':
    classifications = ['positive', 'negative']
else:
    classifications = ['World', 'Sports', 'Business', 'Sci/Tech']

#randomly select some examples 
args.model_name = 'bert'
tokenizer, (x_train, y_train), (x_val, y_val), (train_masks, val_masks) = load_data_text(args)

pred_val = pred_val.argmax(1).numpy()
# false_predictions = range(len(pred_val)) #changed to be everything
false_predictions = [i for i in range(len(pred_val)) if (y_val[i]!=pred_val[i] and y_val[i]==0)]
print('len(false_predictions): ', len(false_predictions))
# raise Exception('end')
print('len(pred_val): ', len(pred_val))
if args.reuse:
    print('REUSING!!')
    # df = pd.read_csv('human_eval_examples/news/txts/df.csv')
    df = pd.read_csv('human_eval_examples/news/txts/df_new.csv')
    print(f'df.head(): {df.head()}')
    # df1 = pd.read_csv('human_eval_examples/news/txts/df_0_50.csv')
    # df2 = pd.read_csv('human_eval_examples/news/txts/df_50_100.csv')
    # df = df1.append(df2)
    samples_to_visualize = list(df[df['ground_truth'] != df['pred']]['sample'].values)
    print('samples_to_visualize: ', samples_to_visualize)
    examples_save_dir = f'human_eval_examples/{args.dataset}/{args.method}/'
    Path(examples_save_dir).mkdir(parents=True, exist_ok=True)
    # first delete everything inside the saved folder
    [f.unlink() for f in Path(examples_save_dir).glob("*") if f.is_file()] 
    if args.method!=None:
        df = reprocess_scores(df)
        # # read in topic keywords
        # save_file = f'models/{args.dataset}/{args.model_name}/conceptshap/topics_conceptshap_{args.layer_idx}.txt'
        # topic_keywords_cs = read_topics(save_file)
        # save_file = f'models/{args.dataset}/{args.model_name}/two_stage/topics_two_stage_{args.layer_idx}.txt'
        # topic_keywords_cc = read_topics(save_file)
        if args.method == 'cc':
            scores = list(df['cc_scores'].values)
            # topic_keywords = topic_keywords_cc
        elif args.method == 'cs':
            scores = list(df['cs_scores'].values)
            # topic_keywords = topic_keywords_cs
        # For a single example
        for i, row in df.iterrows():
            print('i: ', i)
            data = ''
            tokens = row['text']
            expl = np.array(scores[i])
            # normalize scores
            try:
                expl = (expl - expl.min()) / (expl.max() - expl.min() + 1e-8)
            except:
                print(f'expl: {expl}')
                print(f'expl.min(): {expl.min()}')
                print(f'expl.max(): {expl.max()}')
                raise Exception('end')
            vis_data_records = [visualization.VisualizationDataRecord(
                                            expl,
                                            tokens)]
            fig = visualization.visualize_text(vis_data_records, legend = False)
            data += fig.data
            # if args.include_keywords:
            #     data += f'<h3>Other related keywords to the highlighted ones: </h3> \n <body>{topic_keywords[t]}</body>\n'
            with open(f"{examples_save_dir}data_{i}.html", "w") as file:
                file.write(data)
    raise Exception('finished!')
else:
    np.random.seed(args.seed)
    # we should only visualize where there're valid highlights
    samples_to_visualize = np.random.choice(false_predictions, size = args.n_samples, replace = False)
    # samples_to_visualize = [79, 81, 82, 83]
    for (i, s) in enumerate(samples_to_visualize):
        input_ids = torch.from_numpy(x_val[s]).unsqueeze(0).to(device)
        attention_mask = val_masks[s].unsqueeze(0).to(device)
        output = model_cc(input_ids=input_ids, attention_mask=attention_mask)[0].cpu().detach().numpy()
        indices = np.argwhere(output>0)
        t = np.argsort(output)[0][::-1][0]
        expl = explanations_cc.generate_LRP(input_ids=input_ids, attention_mask=attention_mask, start_layer=0, index = t)[0].cpu().detach().numpy()
        print('here')
        if expl.max() - expl.min() < 1e-4:
            # replace it
            print('changing from {}...'.format(samples_to_visualize[i]))
            valid_choices = [sample for sample in false_predictions if sample not in samples_to_visualize]
            samples_to_visualize[i] = np.random.choice(valid_choices)
            print('...to {}'.format(samples_to_visualize[i]))
            # samples_to_visualize.remove(samples_to_visualize[i])
            i = i-1 #check again

if args.txt:
    examples_save_dir = f'human_eval_examples/{args.dataset}/txts/'
    Path(examples_save_dir).mkdir(parents=True, exist_ok=True)
    # df1 = pd.read_csv('human_eval_examples/news/txts/df_0_50.csv')
    # df2 = pd.read_csv('human_eval_examples/news/txts/df_50_100.csv')
    # df = pd.concat([df1,df2])
    # samples_to_visualize = list(df['sample'].values)
    # text = list(df['text'].values)
    # ground_truth = list(df['ground_truth'].values)
    # pred = list(df['pred'].values)
    # dic = {'sample': samples_to_visualize, 'text': text, 'ground_truth': ground_truth, 'pred': pred, 'cc_scores': [], 'cs_scores': []}
    print(f'samples to visualize: {samples_to_visualize}')
    dic = {'sample': samples_to_visualize, 'text': [], 'ground_truth': [], 'pred': [], 'cc_scores': [], 'cs_scores': []}
    # dic = {'sample': samples_to_visualize, 'text': [], 'ground_truth': [], 'pred': [], 'cc_scores': [], 'cs_scores': []}
    for (i, sample) in enumerate(samples_to_visualize):
        input_ids = torch.from_numpy(x_val[sample]).unsqueeze(0).to(device)
        old_tokens = tokenizer.convert_ids_to_tokens(input_ids.flatten())
        # print('input_ids: ', input_ids)
        # print('old_tokens: ', old_tokens)
        dic['ground_truth'].append(classifications[y_val[sample]])
        dic['pred'].append(classifications[pred_val[sample]]) 
        # encode a sentence
        attention_mask = val_masks[sample].unsqueeze(0).to(device)
        # CC SCORE
        output = model_cc(input_ids=input_ids, attention_mask=attention_mask)[0].cpu().detach().numpy()
        output = (output - output.min()) / (output.max() - output.min() + 1e-8)
        # indices = np.argwhere(output>0)
        ts = np.argsort(output)[0][::-1][:3] # get top 3 concepts
        for (i_t, t) in enumerate(ts):
            score = output[0][t]
            expl = explanations_cc.generate_LRP(input_ids=input_ids, attention_mask=attention_mask, start_layer=0, index = t)[0].cpu().detach().numpy()
            expl = (expl - expl.min()) / (expl.max() - expl.min() + 1e-8)
            try:
                tokens, expl = no_bpe(old_tokens, list(expl))
            except:
                print(old_tokens)
                raise Exception('end')
            expl = np.array(expl)
            if i_t ==0:
                cc_scores = (score + 1e-8)*(expl + 1e-8)
            else:
                cc_scores += (score + 1e-8)*(expl + 1e-8)
        dic['cc_scores'].append(','.join(list([str(c) for c in cc_scores])))
        dic['text'].append(' '.join(tokens))

        output = model_cs(input_ids=input_ids, attention_mask=attention_mask)[0].cpu().detach().numpy()
        output = (output - output.min()) / (output.max() - output.min() + 1e-8)
        cs_scores = np.array([0]*len(old_tokens))
        for (i_t, t) in enumerate(ts):
            score = output[0][t]
            expl = explanations_cs.generate_LRP(input_ids=input_ids, attention_mask=attention_mask, start_layer=0, index = t)[0].cpu().detach().numpy()
            expl = (expl - expl.min()) / (expl.max() - expl.min() + 1e-8)
            tokens, expl = no_bpe(old_tokens, list(expl))
            expl = np.array(expl)
            if i_t ==0:
                cs_scores = (score + 1e-8)*(expl + 1e-8)
            else:
                cs_scores += (score + 1e-8)*(expl + 1e-8)
        dic['cs_scores'].append(','.join(list([str(c) for c in cs_scores])))
    df = pd.DataFrame(dic)
    df.to_csv(f'{examples_save_dir}df_new.csv')
    print('SAVED!')
else:
    # read in topic keywords
    save_file = f'models/{args.dataset}/{args.model_name}/conceptshap/topics_conceptshap_{args.layer_idx}.txt'
    topic_keywords_cs = read_topics(save_file)
    save_file = f'models/{args.dataset}/{args.model_name}/two_stage/topics_two_stage_{args.layer_idx}.txt'
    topic_keywords_cc = read_topics(save_file)
    # 100 samples
    # split into two groups
    group1_samples = samples_to_visualize[:int(args.n_samples//2)]
    group2_samples = samples_to_visualize[int(args.n_samples//2):]

    #FIRST SET: cs - group 1; cc - group 2
    write ='Selected samples: \n'
    # save directory
    examples_save_dir = f'human_eval_examples/{args.dataset}/1/'
    Path(examples_save_dir).mkdir(parents=True, exist_ok=True)
    # first delete everything inside the saved folder
    [f.unlink() for f in Path(examples_save_dir).glob("*") if f.is_file()] 
    cs_or_cc = [0]*(args.n_samples//2) + [1]*(args.n_samples//2) #0 is cc
    shuffle(cs_or_cc)
    print('cs_or_cc: ', cs_or_cc)
    cs_order = group1_samples.copy()
    shuffle(cs_order)
    print('cs_order: ', cs_order)
    cc_order = group2_samples.copy()
    shuffle(cc_order)
    print('cc_order: ', cc_order)
    all_samples_to_visualize = []
    cc_i = 0
    cs_i = 0
    for i in cs_or_cc:
        if i == 0: # cc method
            all_samples_to_visualize.append(cc_order[cc_i])
            cc_i += 1
        else: # cs method
            all_samples_to_visualize.append(cs_order[cs_i])
            cs_i += 1
    write += str(all_samples_to_visualize) + '\n'
    print('Samples to visualize: ', all_samples_to_visualize)
    write += 'Ground-Truth labels: \n'+str([y_val[i] for i in all_samples_to_visualize])+'\n'
    write += 'Predicted labels: \n'+ str([pred_val[i] for i in all_samples_to_visualize])+'\n'
    write += 'cs_or_cc: \n' + str(cs_or_cc) + '\n'
    if args.wordcloud:
        if not os.path.exists(f'{examples_save_dir}wordcloud/'):
            os.makedirs(f'{examples_save_dir}wordcloud/')
        with open(f"{examples_save_dir}wordcloud/samples.txt", "w") as file:
            file.write(write)
    else:
        if not os.path.exists(f'{examples_save_dir}htmls/'):
            os.makedirs(f'{examples_save_dir}htmls/')
        with open(f"{examples_save_dir}htmls/samples.txt", "w") as file:
            file.write(write)
    # For a single example
    for i, sample in enumerate(all_samples_to_visualize):
        print('sample: ', sample)
        print('i: ', i)
        text_batch = [tokenizer.decode(x_val[sample])]
        # encode a sentence
        input_ids = torch.from_numpy(x_val[sample]).unsqueeze(0).to(device)
        attention_mask = val_masks[sample].unsqueeze(0).to(device)
        # true_class = classifications[y_val[sample]]
        # get the model classification
        data = ''
        tokens = tokenizer.convert_ids_to_tokens(input_ids.flatten())
        if cs_or_cc[i] == 0: #cc
            output = model_cc(input_ids=input_ids, attention_mask=attention_mask)[0]
            indices = np.argwhere(output.cpu().detach().numpy()>0)
            t = np.argsort(output.cpu().detach().numpy())[0][::-1][0]
            expl = explanations_cc.generate_LRP(input_ids=input_ids, attention_mask=attention_mask, start_layer=0, index = t)[0]
        elif cs_or_cc[i] == 1: # cs
            output = model_cs(input_ids=input_ids, attention_mask=attention_mask)[0]
            indices = np.argwhere(output.cpu().detach().numpy()>0)
            t = np.argsort(output.cpu().detach().numpy())[0][::-1][0]
            expl = explanations_cs.generate_LRP(input_ids=input_ids, attention_mask=attention_mask, start_layer=0, index = t)[0]
        # normalize scores
        expl = (expl - expl.min()) / (expl.max() - expl.min() + 1e-8)
        tokens, expl = no_bpe(tokens, list(expl.cpu().detach().numpy()))
        if args.wordcloud:
            # words = []
            # for (ei, t) in enumerate(tokens):
            #     words += [t]*int(expl[ei]*50)
            # wordcloud = WordCloud(background_color='white').generate(' '.join(words))
            # to avoid nan values
            expl = np.nan_to_num(expl) + 1e-8
            words_dict = dict(zip(tokens, expl))
            wordcloud = WordCloud(background_color='white').generate_from_frequencies(words_dict)
            # Display the generated image:
            plt.axis("off")
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.savefig(f'{examples_save_dir}wordcloud/data_{i}.png')
        else:
            vis_data_records = [visualization.VisualizationDataRecord(
                                            expl,
                                            tokens)]
            fig = visualization.visualize_text(vis_data_records, legend = False)
            data += fig.data
            if args.include_keywords:
                if cs_or_cc[i] == 0: #cc
                    data += f'<h3>Other related keywords to the highlighted ones: </h3> \n <body>{topic_keywords_cc[t]}</body>\n'
                elif cs_or_cc[i] == 1: #cs
                    data += f'<h3>Other related keywords to the highlighted ones: </h3> \n <body>{topic_keywords_cs[t]}</body>\n'
            with open(f"{examples_save_dir}htmls/data_{i}.html", "w") as file:
                file.write(data)

    #SECOND SET: cs - group 2; cc - group 1
    write ='Selected samples: \n'
    # save directory
    examples_save_dir = f'human_eval_examples/{args.dataset}/2/'
    Path(examples_save_dir).mkdir(parents=True, exist_ok=True)
    # first delete everything inside the saved folder
    [f.unlink() for f in Path(examples_save_dir).glob("*") if f.is_file()] 
    cs_or_cc = [0]*(args.n_samples//2) + [1]*(args.n_samples//2)
    shuffle(cs_or_cc)
    # print('cs_or_cc: ', cs_or_cc)
    cs_order = group2_samples.copy()
    shuffle(cs_order)
    # print('cs_order: ', cs_order)
    cc_order = group1_samples.copy()
    shuffle(cc_order)
    # print('cc_order: ', cc_order)
    all_samples_to_visualize = []
    cc_i = 0
    cs_i = 0
    for i in cs_or_cc:
        if i == 0: # cc method
            all_samples_to_visualize.append(cc_order[cc_i])
            cc_i += 1
        else: # cs method
            all_samples_to_visualize.append(cs_order[cs_i])
            cs_i += 1
    write += str(all_samples_to_visualize) + '\n'
    print('Samples to visualize: ', all_samples_to_visualize)
    write += 'Ground-Truth labels: \n'+str([y_val[i] for i in all_samples_to_visualize])+'\n'
    write += 'Predicted labels: \n'+ str([pred_val[i] for i in all_samples_to_visualize])+'\n'
    write += 'cs_or_cc: \n' + str(cs_or_cc) + '\n'
    if args.wordcloud:
        if not os.path.exists(f'{examples_save_dir}wordcloud/'):
            os.makedirs(f'{examples_save_dir}wordcloud/')
        with open(f"{examples_save_dir}wordcloud/samples.txt", "w") as file:
            file.write(write)
    else:
        if not os.path.exists(f'{examples_save_dir}htmls/'):
            os.makedirs(f'{examples_save_dir}htmls/')
        with open(f"{examples_save_dir}htmls/samples.txt", "w") as file:
            file.write(write)
    # For a single example
    for i, sample in enumerate(all_samples_to_visualize):
        # print('sample: ', sample)
        # print('i: ', i)
        text_batch = [tokenizer.decode(x_val[sample])]
        # encode a sentence
        input_ids = torch.from_numpy(x_val[sample]).unsqueeze(0).to(device)
        attention_mask = val_masks[sample].unsqueeze(0).to(device)
        true_class = classifications[y_val[sample]]
        # get the model classification
        data = ''
        tokens = tokenizer.convert_ids_to_tokens(input_ids.flatten())
        if cs_or_cc[i] == 0: #cc
            output = model_cc(input_ids=input_ids, attention_mask=attention_mask)[0]
            indices = np.argwhere(output.cpu().detach().numpy()>0)
            t = np.argsort(output.cpu().detach().numpy())[0][::-1][0]
            expl = explanations_cc.generate_LRP(input_ids=input_ids, attention_mask=attention_mask, start_layer=0, index = t)[0]
        elif cs_or_cc[i] == 1: # cs
            output = model_cs(input_ids=input_ids, attention_mask=attention_mask)[0]
            indices = np.argwhere(output.cpu().detach().numpy()>0)
            t = np.argsort(output.cpu().detach().numpy())[0][::-1][0]
            expl = explanations_cs.generate_LRP(input_ids=input_ids, attention_mask=attention_mask, start_layer=0, index = t)[0]
        # normalize scores
        expl = (expl - expl.min()) / (expl.max() - expl.min() + 1e-8)
        tokens, expl = no_bpe(tokens, list(expl.cpu().detach().numpy()))
        if args.wordcloud:
            # to avoid nan values
            expl = np.nan_to_num(expl) + 1e-8
            words_dict = dict(zip(tokens, expl))
            wordcloud = WordCloud(background_color='white').generate_from_frequencies(words_dict)
            # Display the generated image:
            plt.axis("off")
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.savefig(f'{examples_save_dir}wordcloud/data_{i}.png')
        else:
            vis_data_records = [visualization.VisualizationDataRecord(
                                        expl,
                                        tokens)]
            fig = visualization.visualize_text(vis_data_records, legend = False)
            data += fig.data
            if args.include_keywords:
                if cs_or_cc[i] == 0: #cc
                    data += f'<h3>Other related keywords to the highlighted ones: </h3> \n <body>{topic_keywords_cc[t]}</body>\n'
                elif cs_or_cc[i] == 1: #cs
                    data += f'<h3>Other related keywords to the highlighted ones: </h3> \n <body>{topic_keywords_cs[t]}</body>\n'
            with open(f"{examples_save_dir}data_{i}.html", "w") as file:
                file.write(data)