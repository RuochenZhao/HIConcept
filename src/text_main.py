import os
from helper import *
from utils import *
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from conceptshap import *
from text_helper import *
from pathlib import Path
from visualize import *
import argparse
import toy_helper_v2
import logging
from datasets import load_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)

# ARGUMENTS
parser = argparse.ArgumentParser(description="latentRE")
parser.add_argument("--seed", dest="seed", type=int,
                    default=0)
parser.add_argument("--log_dir", dest="log_dir", type=str,
                        default='../log', help="The path to log dir")
parser.add_argument("--log_name", dest="log_name", type=str,
                    default='dummy', help="The file name of log file")
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
# CLS model
parser.add_argument("--model_name", dest="model_name", type=str,
                    default='bert', choices=['cnn', 'lstm', 'bert', 'distilbert', 'transformer', 't5'])
parser.add_argument("--with_pretraining", action='store_true') # whether to use the pre_trained version of BERT
parser.add_argument("--pretrained", action='store_true')
parser.add_argument("--init_with_pca", action='store_true') 
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
                    default='two_stage', choices=['conceptshap', 'two_stage', 'BCVAE', 'pca', 'kmeans'])
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
parser.add_argument("--ae_loss_reg", dest="ae_loss_reg", type=float,
                    default=1)
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


def set_logger(args):
    global logger
    logger = logging.getLogger('root')
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt = '%m/%d/%Y %H:%M:%S',
        handlers=[
            logging.FileHandler(f"{args.log_dir}/{args.log_name}.log"),
            logging.StreamHandler()
        ]
    )

seed_everything(args)
set_logger(args)
args.logger = logger

if args.fast:
    args.n = args.fast
args.n0 = int(args.n * 0.8)
args.logger.info(f'args.train_topic_model: {args.train_topic_model}')
if args.shap_epochs == -1:
    args.shap_epochs = args.epochs // 2
# LOAD DATA
if args.dataset == 'toy':
    # if toy dataset, can only use cnn method
    args.model_name = 'cnn'
    path = os.getcwd()
    if args.cov:
        args.dir = f'../../DATASETS/toy_data/cov/{args.p}'
        args.save_dir = f'models/toy/cov/{args.p}/'
    else:
        args.dir = '../../DATASETS/toy_data/no_cov'
        args.save_dir = 'models/toy/no_cov/'
    # data = None
    tokenizer = None
    if os.path.isdir(args.dir):
        os.chdir(args.dir)
    else:
        os.makedirs(args.dir)
        os.chdir(args.dir)
    if (not (os.path.exists('x_data.npy') and os.path.exists('y_data.npy') and os.path.exists('concept_data.npy'))) or args.generate_toy_data:
        args.logger.info('Creating data')
        # create dataset
        toy_helper_v2.create_dataset(args.n, args.cov, p = args.p)
    args.logger.info('Loading data')
    data = toy_helper_v2.load_toy_data(args)
    #change back to original path
    os.chdir(path)
else:
    data = load_data_text(args)
    # LOAD CLS MODEL
    if 'bert' in args.model_name:
        args.save_dir = 'models/' + args.dataset + '/' + args.model_name +'/'
        args.model_save_dir = 'models/' + args.dataset + '/' + args.model_name +'/imdb_weights/'
    else:
        args.save_dir = 'models/' + args.dataset + '/' + args.model_name +'/'
        args.model_save_dir = 'models/' + args.dataset + '/' + args.model_name +'/cls_model.pkl'
# Check if directory exists, if not, create it
if not os.path.isdir(args.save_dir):
    os.makedirs(args.save_dir)
if args.dataset == 'toy':
    model = toy_helper_v2.load_cls_model(args, device, data)
    # model = toy_helper_v2.load_cls_model(args, device)
else:
    model = load_cls_model(args, device, data)

args.added = False

if args.short_sentences == True:
    short = 'fragment_'
else:
    short = ''

args.divide_bert = False
# change according to layer idx
if args.layer_idx == -1:
    if args.model_name == 't5':
        classifier = model.lm_head
    else:
        classifier = model.classifier
        args.logger.info(f'classifier: {classifier}')
elif args.model_name == 'cnn' and args.dataset != 'toy':
    classifier = torch.nn.Sequential(*(list(model.encoder)[2*args.layer_idx+1:]+list(model.classifier)))
    args.logger.info(f'classifier: {classifier}')
elif args.model_name == 'bert':
    args.divide_bert = True
    args.logger.info('DIVIDING BERT')
    classifier = None

# if args.dataset != 'toy':
if args.model_name == 't5' and args.dataset == 'news':
    train_dataset, valid_dataset, tokenizer = data
    plain_data = load_dataset('ag_news', cache_dir=args.dataset_cache_dir)
    x_train = plain_data['train']['text']
    y_train = np.array(plain_data['train']['label'])
    x_val = plain_data['test']['text']
    y_val = np.array(plain_data['test']['label'])
elif args.model_name in ['bert', 'distilbert', 't5', 'transformer']:
    tokenizer, (x_train, y_train), (x_val, y_val), (train_masks, val_masks) = data
    if args.fast:
        train_masks = train_masks[:args.fast]
        val_masks = val_masks[:int(args.fast*0.2)]
else:
    tokenizer, (x_train, y_train), (x_val, y_val) = data
if args.fast:
    x_train = x_train[:args.fast]
    y_train = y_train[:args.fast]
    x_val = x_val[:int(args.fast*0.2)]
    y_val = y_val[:int(args.fast*0.2)]

if not args.divide_bert:
    # DO INFERENCE
    f_train, y_pred_train, f_val, y_pred_val = inference(data, model, tokenizer, classifier, device, args)
    if args.fast:
        f_train = f_train[:args.fast]
        f_val = f_val[:int(args.fast*0.2)]
    args.logger.info(f'f_train.shape: {f_train.shape}')
    args.logger.info(f'f_val.shape: {f_val.shape}')
else:
    _, y_pred_train, _, y_pred_val = inference(data, model, tokenizer, None, device, args)
if args.fast:
    y_pred_train = y_pred_train[:args.fast]
    y_pred_val = y_train[:int(args.fast*0.2)]

if args.dataset!='toy' and not args.divide_bert:
    if args.model_name == 't5': #first case: t5, handle differently
        y_pred_val = get_t5_output(y_pred_val, args, logits = False)
        if args.dataset == 'imdb':
            y_val = get_t5_output(y_val, args, logits = False)
        pred = y_pred_val
        args.logger.info(f'y_val3[:10]: {y_val[:10]}')
    elif args.dataset == 'imdb':
        if args.model_name == 'cnn':
            pred = torch.mean(f_val, axis = -1)
            pred = classifier(pred.to(device))
            pred = pred.round().squeeze()
            args.logger.info('HERE')
        elif args.model_name in ['transformer']:
            y_pred_train = y_pred_train.round().squeeze()
            y_pred_val = y_pred_val.round().squeeze()
            pred = y_pred_val
        else: #bert
            y_pred_train = y_pred_train.max(1).indices
            y_pred_val = y_pred_val.max(1).indices
            pred = y_pred_val
    elif args.dataset =='news':
        args.logger.info(f'y_pred_val[:10]: {y_pred_val[:10]}')
        if y_pred_train.dim() ==2:
            y_pred_train = y_pred_train.max(1).indices
        if len(y_pred_val.shape) ==2:
            if type(y_pred_val) is np.ndarray:
                y_pred_val = torch.from_numpy(y_pred_val)
            y_pred_val = y_pred_val.float().max(1).indices
        pred = y_pred_val
    elif args.dataset == 'toy': #rounding: imdb (cnn transformer), toy
        y_pred_train = y_pred_train.round()
        y_pred_val = y_pred_val.round()
        pred = y_pred_val

    args.logger.info(f'y_pred_train.shape: {y_pred_train.shape}')
    args.logger.info(f'y_pred_val.shape: {y_pred_val.shape}')
    args.logger.info(f'y_pred_val: {y_pred_val}')
    args.logger.info(f'y_val: {y_val}')
    acc = (sum([pred[i]==y_val[i] for i in range(len(pred))]))
    args.logger.info(f'validation acc: {acc}')
    acc = acc / len(y_pred_val)
    args.logger.info(f'validation acc: {acc}')
    if not type(y_train) is np.ndarray:
        y_train = y_train.numpy()
    if not type(y_val) is np.ndarray:
        y_val = y_val.numpy()
# raise Exception('end')


args.logger.info('TESTING {} METHOD NOW'.format(args.overall_method))

args.graph_save_folder = args.save_dir + '{}/'.format(args.overall_method)
Path(args.graph_save_folder).mkdir(parents=True, exist_ok=True)

if args.train_topic_model or args.eval_causal_effect or args.interpret or args.PDP or (args.visualize!=None) or args.visualize_wordcloud:
    args.logger.info('TRAINING causal TOPIC MODEL')
    if not args.divide_bert:
        topic_model, topic_vector = load_topic_model(classifier, f_train, y_pred_train, torch.from_numpy(y_train).float(), f_val, y_pred_val, torch.from_numpy(y_val).float(), model, device, args, toy = (args.dataset=='toy'))
    else:
        topic_model, topic_vector = load_topic_model(None, torch.from_numpy(x_train), None, torch.from_numpy(y_train), torch.from_numpy(x_val), None, torch.from_numpy(y_val), model, device, args, toy = False, train_masks = train_masks, val_masks = val_masks)

if not args.added:
    add_activation_hook(model, args.layer_idx, args)
if args.eval_causal_effect or args.postprocess:
    if args.divide_bert:
        f_val = x_val
    if not type(y_pred_val) is np.ndarray:
        y_pred_val = y_pred_val.numpy()
    if not args.divide_bert:
        topic_model = postprocess(topic_model, model, classifier, device, f_val, y_pred_val, topic_vector, args, toy = (args.dataset=='toy'), model_name = args.model_name)
    else:
        if not type(val_masks) is np.ndarray:
            val_masks = val_masks.numpy()
        topic_model = postprocess(topic_model, model, classifier, device, f_val, y_pred_val, topic_vector, args, toy = (args.dataset=='toy'), model_name = args.model_name, val_masks = val_masks)
    
if args.interpret:
    save_file = args.graph_save_folder +'results_' + args.overall_method + f'_{args.layer_idx}.txt'
    concept_analysis(f_train, x_train, topic_vector, save_file, args.model_name)
    args.logger.info('concept intepretation finished')
if args.PDP:
    topic_classifier = nn.Linear(100, 4, bias = False)
    with torch.no_grad():
        topic_vector_n = F.normalize(topic_model.topic_vector, dim = 0, p=2)
        topic_classifier.weight = nn.Parameter(topic_vector_n.T, requires_grad = True)
    model.classifier = topic_classifier

    params = list(model.parameters()) 
    weight_softmax = np.squeeze(params[-1].data.cpu().numpy()) #use the last one because bias=False in our case
    args.logger.info(f'weight_softmax.shape: {weight_softmax.shape}') #4, 100

    save_file = args.graph_save_folder +'PDP_' + args.overall_method + '.png'
    make_PDP(f_train, classifier, weight_softmax, save_file, device, args.overall_method, args.model_name, rge = [-2, 2])
    args.logger.info('generated PDP graph')

if (args.visualize!=None) or args.visualize_wordcloud:
    if args.dataset == 'toy':
        toy_helper_v2.visualize_model(x_train, f_train, topic_vector, args.graph_save_folder, args.n_concept, args.overall_method, 
                                        topic_model = topic_model, device = device)
    else:
        if args.visualize=='most_common':
            save_file = args.graph_save_folder +'topics_' + args.overall_method + f'_most_common_{args.layer_idx}_{args.n_concept}.txt'
        else:
            save_file = args.graph_save_folder +'topics_' + args.overall_method + f'_{args.layer_idx}_{args.n_concept}.txt'
        if args.model_name == 'cnn':
            gradcam_analysis(model, tokenizer, topic_model, f_train, x_train, args.maxlen, save_file, args)
        else:
            if args.divide_bert:
                f_train = None
                encoder, classifier = divide_bert_model(model, args)
                bert_topics(encoder, tokenizer, topic_model, f_train, x_train, y_train, train_masks, args, save_file, device, overall_model = model)
            else:
                bert_topics(model, tokenizer, topic_model, f_train, x_train, y_train, train_masks, args, save_file, device)