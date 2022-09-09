device="1"
dataset="imdb"

#train cls model, 10 epochs for imdb
CUDA_VISIBLE_DEVICES=$device python main.py --dataset=$dataset --do_inference --epochs 10 --lr 3e-4 --batch_size 2

# nohup bash run_text_exps.sh > ../log/imdb_log 2>&1 &