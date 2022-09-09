device="0"

### if running for the first time, generate data by adding --generate_toy_data

# ### generate data and train cls model
# CUDA_VISIBLE_DEVICES=$device python main.py --generate_toy_data --dataset=toy --p 0.65 --do_inference --epochs 30 --lr 3e-3

### inference only
CUDA_VISIBLE_DEVICES=$device python main.py --dataset=toy --p 0.65 --do_inference --epochs 30 --lr 3e-3 --pretrained

# ### train topic model
# CUDA_VISIBLE_DEVICES=$device python main.py --dataset=toy --p 0.65 --pretrained --epochs 10 --flip_loss_reg 0.5 --train_topic_model --eval_causal_effect --postprocess --one_correlated_dimension

# nohup bash run_toy_exps.sh > ../log/toy_log 2>&1 &