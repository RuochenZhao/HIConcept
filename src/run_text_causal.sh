device="3"
dataset="news"
model_name='bert' #'cnn', 'lstm', 'bert', 'distilbert', 'transformer', 't5'
fast='0'
# fast='100'
ae_loss_reg='1'
pred_loss_reg='1'
flip_loss_reg='1'
concept_sim_reg='0.1'
concept_far_reg='0.5'
random_masking_prob='0.2' #0.2 to be default
batch_size='128'
layer_idx='9'

#train transformer model from scratch
CUDA_VISIBLE_DEVICES=$device python self_train_transformer.py


#finetune t5 model
CUDA_VISIBLE_DEVICES=“0,1,2,3” accelerate launch finetune_t5.py

#  --do_inference --train_topic_model  --freeze
echo "start two_stage model"
CUDA_VISIBLE_DEVICES=$device python text_main.py --dataset=$dataset --pretrained --model_name $model_name --layer_idx $layer_idx\
        --epochs 2 --masking random --batch_size $batch_size\
        --ae_loss_reg $ae_loss_reg --pred_loss_reg $pred_loss_reg --flip_loss_reg $flip_loss_reg\
        --concept_sim_reg $concept_sim_reg --concept_far_reg $concept_far_reg \
        --random_masking_prob $random_masking_prob --fast $fast --freeze\
        --visualize txt --batch_size 8\
        --postprocess --one_correlated_dimension\
        --eval_causal_effect

# ablation studies
# echo "start two_stage model - no ae loss"
# CUDA_VISIBLE_DEVICES=$device python text_main.py --dataset=$dataset --pretrained --model_name $model_name\
#         --train_topic_model --epochs 100 --masking random \
#         --ae_loss_reg 0 --pred_loss_reg $pred_loss_reg --flip_loss_reg $flip_loss_reg\
#         --concept_sim_reg $concept_sim_reg --concept_far_reg $concept_far_reg \
#         --eval_causal_effect --fast $fast\
#         --postprocess --one_correlated_dimension --freeze

# # --train_topic_model 
# echo "start two_stage model - no pred loss"
# CUDA_VISIBLE_DEVICES=$device python text_main.py --dataset=$dataset --pretrained --model_name $model_name\
#         --train_topic_model --epochs 100 --masking random\
#         --ae_loss_reg $ae_loss_reg --pred_loss_reg 0 --flip_loss_reg $flip_loss_reg\
#         --concept_sim_reg $concept_sim_reg --concept_far_reg $concept_far_reg \
#         --eval_causal_effect --fast $fast\
#         --postprocess --one_correlated_dimension --freeze

# echo "start two_stage model - no causal loss"
# CUDA_VISIBLE_DEVICES=$device python text_main.py --dataset=$dataset --pretrained --model_name $model_name\
#         --train_topic_model --epochs 100 --masking random\
#         --ae_loss_reg $ae_loss_reg --pred_loss_reg $pred_loss_reg --flip_loss_reg 0\
#         --concept_sim_reg $concept_sim_reg --concept_far_reg $concept_far_reg \
#         --eval_causal_effect --fast $fast\
#         --postprocess --one_correlated_dimension --freeze

# # --train_topic_model 
# echo "start two_stage model - no regularizer loss"
# CUDA_VISIBLE_DEVICES=$device python text_main.py --dataset=$dataset --pretrained --model_name $model_name\
#         --train_topic_model --epochs 100 --masking random\
#         --ae_loss_reg $ae_loss_reg --pred_loss_reg $pred_loss_reg --flip_loss_reg $flip_loss_reg\
#         --concept_sim_reg 0 --concept_far_reg 0 \
#         --eval_causal_effect --fast $fast\
#         --postprocess --one_correlated_dimension --freeze

# nohup bash run_text_causal.sh > ../log/news_bert_layer3_log 2>&1 &
# nohup bash run_text_causal.sh > ../log/news_bert_causal_ablation_log 2>&1 &
# nohup bash run_text_causal.sh > ../log/news_t5_finetuning_log 2>&1 &
# nohup bash run_text_causal.sh > ../log/news_t5_causal_log 2>&1 &