import os
from torch.utils.data import TensorDataset, DataLoader
from cls_models import *
from utils import *
from tqdm import tqdm, trange
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from concept_models import *
from text_helper import *
from pathlib import Path
from visualize import *
import argparse
import toy_helper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# ARGUMENTS
parser = argparse.ArgumentParser(description="latentRE")
parser.add_argument("--seed", dest="seed", type=int,
                    default=0)
# DATA
parser.add_argument("--fast", dest="fast", type=int,
                    default=False)
parser.add_argument("--max_features", dest="max_features", type=int,
                    default=100000)
parser.add_argument("--short_sentences", dest="short_sentences", type=bool,
                    default=False)
parser.add_argument("--dataset", dest="dataset", type=str,
                    default='news', choices=['news', 'imdb', 'toy', '20news', 'yahoo_answers'])
parser.add_argument("--dataset_cache_dir", dest="dataset_cache_dir", type=str,
                    default="../../hf_datasets/", help="dataset cache folder")
# TOY DATA
parser.add_argument("--cov", dest="cov", type=bool, default=True)
parser.add_argument("--p", dest="p", type=float, default=0.65)
parser.add_argument("--generate_toy_data", action='store_true')
parser.add_argument("--n", dest="n", type=int, default=10000)
parser.add_argument("--toy_data_save_dir", type = str, default = "../DATASETS/toy_data/")
# CLS model
parser.add_argument("--model_name", dest="model_name", type=str,
                    default='bert', choices=['cnn', 'lstm', 'bert', 'distilbert'])
parser.add_argument("--with_pretraining", action='store_true') # whether to use the pre_trained version of BERT
parser.add_argument("--pretrained", action='store_true')
parser.add_argument("--do_inference", action='store_true')
parser.add_argument("--batch_size", dest="batch_size", type=int,
                    default=128)
parser.add_argument("--epochs", dest="epochs", type=int,
                    default=10)
parser.add_argument("--shap_epochs", dest="shap_epochs", type=int,
                    default=-1)
parser.add_argument("--maxlen", dest="maxlen", type=int,
                    default=400)
parser.add_argument("--embedding_dim", dest="embedding_dim", type=int,
                    default=100)
parser.add_argument("--hidden_dim", dest="hidden_dim", type=int,
                    default=500) 
# TOPIC MODEL
parser.add_argument("--freeze", action='store_true')
parser.add_argument("--attack", action='store_true')
parser.add_argument("--train_topic_model", action='store_true')
parser.add_argument("--overall_method", dest="overall_method", type=str,
                    default='two_stage', choices=['conceptnet', 'conceptshap', 'cc', 'two_stage', 'BCVAE', 'pca', 'kmeans'])
parser.add_argument("--masking", dest="masking", type=str,
                    default='max', choices=['max', 'random', 'all', 'mean'])
parser.add_argument("--loss", dest="loss", type=str,
                    default='flip', choices=['flip', 'far'])
parser.add_argument("--random_masking_prob", dest="random_masking_prob", type=float,
                    default=0.2)
parser.add_argument("--n_concept", dest="n_concept", type=int,
                    default=10)
parser.add_argument("--flip_loss_reg", dest="reg_0", type=float,
                    default=0.1)
parser.add_argument("--pred_loss_reg", dest="reg_1", type=float,
                    default=1)
parser.add_argument("--concept_sim_reg", dest="reg_2", type=float,
                    default=0.1)
parser.add_argument("--concept_far_reg", dest="reg_3", type=float,
                    default=0.1)
parser.add_argument("--consistency_reg", dest="reg_4", type=float,
                    default=0)
parser.add_argument("--thres", dest="thres", type=float,
                    default=0.2)
parser.add_argument("--lr", dest="lr", type=float,
                    default=3e-4) # can use 3e-2 instead
parser.add_argument("--layer_idx", dest="layer_idx", type=int,
                    default=-1) # can use 3e-2 instead
parser.add_argument("--extra_layers", dest="extra_layers", type=int,
                    default=0) # can use 3e-2 instead

# POST-HOC ANALYSIS
parser.add_argument("--eval_causal_effect", action='store_true')
parser.add_argument("--causal_effect_filename", dest="causal_effect_filename", type=str,
                    default='causal_effect')
parser.add_argument("--postprocess", action='store_true')
parser.add_argument("--one_correlated_dimension", action='store_true')
parser.add_argument("--early_stopping", action='store_true')
parser.add_argument("--load_cs", action='store_true')
parser.add_argument("--interpret", action='store_true')
parser.add_argument("--PDP", action='store_true')
parser.add_argument("--visualize", type = str, default = None, choices = ['txt', 'most_common'])
parser.add_argument("--visualize_wordcloud", action='store_true')
parser.add_argument("--topk", dest="topk", type=int,
                    default=2000)

args = parser.parse_args()

seed_everything(args)

if args.fast:
    args.n = args.fast
args.n0 = int(args.n * 0.8)
print('args.train_topic_model: ', args.train_topic_model)
if args.shap_epochs == -1:
    args.shap_epochs = args.epochs // 2
# LOAD DATA
if args.dataset == 'toy':
    # if toy dataset, can only use cnn method
    args.model_name = 'cnn'
    path = os.getcwd()
    if args.cov:
        args.dir = f'{toy_data_save_dir}cov/{args.p}'
        args.save_dir = f'../models/toy/cov/{args.p}/'
    else:
        args.dir = f'{toy_data_save_dir}no_cov'
        args.save_dir = '../models/toy/no_cov/'
    if os.path.isdir(args.dir):
        os.chdir(args.dir)
    else:
        os.makedirs(args.dir)
        os.chdir(args.dir)
    if (not (os.path.exists('x_data.npy') and os.path.exists('y_data.npy') and os.path.exists('concept_data.npy'))) or args.generate_toy_data:
        print('Creating data')
        # create dataset
        toy_helper.create_dataset(args.n, args.cov, p = args.p)
    print('Loading data')
    data = toy_helper.load_toy_data(args)
    #change back to original path
    os.chdir(path)
else:
    data = load_data_text(args)
    # LOAD CLS MODEL
    if 'bert' in args.model_name:
        args.save_dir = '../models/' + args.dataset + '/' + args.model_name +'/'
        args.model_save_dir = '../models/' + args.dataset + '/' + args.model_name +'/imdb_weights/'
    else:
        args.save_dir = '../models/' + args.dataset + '/' + args.model_name +'/'
        args.model_save_dir = '../models/' + args.dataset + '/' + args.model_name +'/cls_model.pkl'
# Check if directory exists, if not, create it
if not os.path.isdir(args.save_dir):
    os.makedirs(args.save_dir)
if args.dataset == 'toy':
    model = toy_helper.load_cls_model(args, device, data)
else:
    model = load_cls_model(args, device, data)
if args.short_sentences == True:
    short = 'fragment_'
else:
    short = ''
# print(f'{len(list(model.modules()))} modules')
# print(list(model.modules()))
# raise Exception('end')
# print('para: ', list(model.modules())[-1].weight)
# print('model_dir: ', args.model_save_dir)

args.divide_bert = False
# change according to layer idx
if args.layer_idx == -1:
    classifier = model.classifier
    print('classifier: ', classifier)
elif args.model_name == 'cnn' and args.dataset != 'toy':
    classifier = torch.nn.Sequential(*(list(model.encoder)[2*args.layer_idx+1:]+list(model.classifier)))
    print('classifier: ', classifier)
elif args.model_name == 'bert':
    args.divide_bert = True
    print('DIVIDING BERT')

if 'bert' in args.model_name:
    tokenizer, (x_train, y_train), (x_val, y_val), (train_masks, val_masks) = data
else:
    tokenizer, (x_train, y_train), (x_val, y_val) = data

if not args.divide_bert:
    # DO INFERENCE
    f_train, y_pred_train, f_val, y_pred_val = inference(data, model, classifier, device, args)

    print('y_pred_train.max(1).indices[:10]: ', y_pred_train.max(1).indices[:10])
    pred = y_pred_train.max(1).indices
    print('len(y_pred_train.max(1).indices): ', len(y_pred_train.max(1).indices))
    print('y_train[:10]: ', y_train[:10])
    y = y_train
    print('len(y_train): ', len(y_train))
    acc = (sum([pred[i]==y[i] for i in range(len(pred))]))
    print('acc: ', acc)
    acc = acc / len(y_pred_train)
    print('acc: ', acc)
    # raise Exception('end')

    if args.fast:
        f_train = f_train[:args.fast]
        y_pred_train = y_pred_train[:args.fast]
        f_val = f_val[:int(args.fast*0.2)]
        y_pred_val = y_pred_val[:int(args.fast*0.2)]

print('TESTING {} METHOD NOW'.format(args.overall_method))

args.graph_save_folder = args.save_dir + '{}/'.format(args.overall_method)
Path(args.graph_save_folder).mkdir(parents=True, exist_ok=True)

if args.train_topic_model or args.eval_causal_effect or args.interpret or args.PDP or (args.visualize!=None) or args.visualize_wordcloud:
    print('TRAINING TOPIC MODEL')
    if not args.divide_bert:
        topic_model, topic_vector = load_topic_model(classifier, f_train, y_pred_train, f_val, y_pred_val, model, device, args, toy = (args.dataset=='toy'))
    else:
        topic_model, topic_vector = load_topic_model(None, torch.from_numpy(x_train), torch.from_numpy(y_train), torch.from_numpy(x_val), torch.from_numpy(y_val), model, device, args, toy = False, train_masks = train_masks, val_masks = val_masks)

if args.eval_causal_effect or args.postprocess:
    if args.divide_bert:
        f_val = x_val
    postprocess(topic_model, model, classifier, device, f_val, y_val, topic_vector, args, toy = (args.dataset=='toy'), model_name = args.model_name)

if args.interpret:
    save_file = args.graph_save_folder +'results_' + args.overall_method + f'_{args.layer_idx}.txt'
    concept_analysis(f_train, x_train, topic_vector, save_file, args.model_name)
    print('concept intepretation finished')
if args.PDP:
    topic_classifier = nn.Linear(100, 4, bias = False)
    with torch.no_grad():
        topic_vector_n = F.normalize(topic_model.topic_vector, dim = 0, p=2)
        topic_classifier.weight = nn.Parameter(topic_vector_n.T, requires_grad = True)
    model.classifier = topic_classifier
    # print(model)

    params = list(model.parameters()) # 将参数变换为列表 按照weights bias 排列 池化无参数
    weight_softmax = np.squeeze(params[-1].data.cpu().numpy()) #use the last one because bias=False in our case
    print('weight_softmax.shape: ', weight_softmax.shape) #4, 100

    save_file = args.graph_save_folder +'PDP_' + args.overall_method + '.png'
    # make_PDP(f_train, classifier, topic_vector, save_file, device, overall_method, rge = [-2, 2])
    make_PDP(f_train, classifier, weight_softmax, save_file, device, args.overall_method, args.model_name, rge = [-2, 2])
    print('generated PDP graph')

if (args.visualize!=None) or args.visualize_wordcloud:
    if args.dataset == 'toy':
        toy_helper.visualize_model(x_train, f_train, topic_vector, args.graph_save_folder, args.n_concept, args.overall_method, 
                                        topic_model = topic_model, device = device)
    else:
        if args.visualize=='most_common':
            save_file = args.graph_save_folder +'topics_' + args.overall_method + f'_most_common_{args.layer_idx}.txt'
        else:
            save_file = args.graph_save_folder +'topics_' + args.overall_method + f'_{args.layer_idx}.txt'
        if args.model_name == 'cnn':
            gradcam_analysis(model, tokenizer, topic_model, f_train, x_train, args.maxlen, save_file, args)
        else:
            if args.divide_bert:
                f_train = None
                encoder, classifier = divide_bert_model(model, args)
                bert_topics(encoder, tokenizer, topic_model, f_train, x_train, y_train, train_masks, args, save_file, device, overall_model = model)
            else:
                bert_topics(model, tokenizer, topic_model, f_train, x_train, y_train, train_masks, args, save_file, device)