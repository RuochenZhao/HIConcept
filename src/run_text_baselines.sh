device="0"
dataset="news" #"imdb", "news", "toy"
model_name='transformer' #'transformer', 'lstm', 'bert', 'distilbert', 'transformer', 't5'
fast='0'
# fast='100'
n_concept='10'

echo "start pca"
CUDA_VISIBLE_DEVICES=$device python text_main.py --dataset=$dataset --pretrained --model_name $model_name\
        --train_topic_model  --epochs 100 --overall_method pca --fast $fast --n_concept $n_concept\
        --eval_causal_effect

echo "start kmeans"
CUDA_VISIBLE_DEVICES=$device python text_main.py --dataset=$dataset --pretrained --model_name $model_name \
        --train_topic_model  --epochs 10 --overall_method kmeans --fast $fast\
        --eval_causal_effect

echo "start BCVAE"
CUDA_VISIBLE_DEVICES=$device python text_main.py --dataset=$dataset --pretrained --model_name $model_name \
        --epochs 100 --overall_method BCVAE  --fast $fast\
        --eval_causal_effect

echo "start conceptshap"
CUDA_VISIBLE_DEVICES=$device python text_main.py --dataset=$dataset --pretrained  --model_name $model_name\
        --epochs 100 --overall_method conceptshap --fast $fast\
        --pred_loss_reg 1 --concept_sim_reg 0.1 --concept_far_reg 0.5\
        --eval_causal_effect

# nohup bash run_text_baselines.sh > ../log/news_trans_baselines_log 2>&1 &