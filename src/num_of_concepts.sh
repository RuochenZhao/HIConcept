device="3"
dataset="news"
model_name='bert' #'cnn', 'lstm', 'bert', 'distilbert', 'transformer', 't5'
fast='0'
# fast='1000'
# ae_loss_reg='0'
ae_loss_reg='1'
pred_loss_reg='1'
flip_loss_reg='1'
concept_sim_reg='0.1'
concept_far_reg='0.5'
random_masking_prob='0.2' #0.2 to be default
batch_size='128'


n_concept='5'
#  --do_inference --train_topic_model  --freeze
echo "start two_stage model"
CUDA_VISIBLE_DEVICES=$device python text_main.py --dataset=$dataset --pretrained --model_name $model_name\
        --epochs 100 --masking random --batch_size $batch_size\
        --ae_loss_reg $ae_loss_reg --pred_loss_reg $pred_loss_reg --flip_loss_reg $flip_loss_reg\
        --concept_sim_reg $concept_sim_reg --concept_far_reg $concept_far_reg \
        --random_masking_prob $random_masking_prob --fast $fast --freeze\
        --visualize txt --n_concept $n_concept\
        --postprocess --one_correlated_dimension\
        --eval_causal_effect

n_concept='50'
#  --do_inference --train_topic_model  --freeze
echo "start two_stage model"
CUDA_VISIBLE_DEVICES=$device python text_main.py --dataset=$dataset --pretrained --model_name $model_name\
        --epochs 100 --masking random --batch_size $batch_size\
        --ae_loss_reg $ae_loss_reg --pred_loss_reg $pred_loss_reg --flip_loss_reg $flip_loss_reg\
        --concept_sim_reg $concept_sim_reg --concept_far_reg $concept_far_reg \
        --random_masking_prob $random_masking_prob --fast $fast --freeze\
        --visualize txt --n_concept $n_concept\
        --postprocess --one_correlated_dimension\
        --eval_causal_effect

n_concept='100'
#  --do_inference --train_topic_model  --freeze
echo "start two_stage model"
CUDA_VISIBLE_DEVICES=$device python text_main.py --dataset=$dataset --pretrained --model_name $model_name\
        --epochs 100 --masking random --batch_size $batch_size\
        --ae_loss_reg $ae_loss_reg --pred_loss_reg $pred_loss_reg --flip_loss_reg $flip_loss_reg\
        --concept_sim_reg $concept_sim_reg --concept_far_reg $concept_far_reg \
        --random_masking_prob $random_masking_prob --fast $fast --freeze\
        --visualize txt --n_concept $n_concept\
        --postprocess --one_correlated_dimension\
        --eval_causal_effect

# nohup bash num_of_concepts.sh > ../log/news_bert_concepts_ablation_log 2>&1 &