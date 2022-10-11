device="2"
# fast='1000'
fast='0'
n='60000'
# n='1000'
ae_loss_reg='1'
pred_loss_reg='1'
flip_loss_reg='1'
concept_sim_reg='0.1'
concept_far_reg='0.5'
random_masking_prob='0.2' #0.2 to be default
batch_size='128'
p='0.75'

# generate data
echo "generating data"
CUDA_VISIBLE_DEVICES=$device python text_main.py --dataset=toy --n $n --model_name cnn --epochs 30\
        --batch_size 32 --lr 3e-4 --p $p\
        --train_topic_model --epochs 100 --masking random --batch_size $batch_size\
        --ae_loss_reg $ae_loss_reg --pred_loss_reg $pred_loss_reg --flip_loss_reg $flip_loss_reg\
        --concept_sim_reg $concept_sim_reg --concept_far_reg $concept_far_reg \
        --random_masking_prob $random_masking_prob --p $p\
        --eval_causal_effect --fast $fast --freeze\
        --do_inference --generate_toy_data 
        --generate_toy_data 

echo "start two_stage model"
CUDA_VISIBLE_DEVICES=$device python text_main.py --dataset=toy --model_name cnn --pretrained\
        --train_topic_model --epochs 100 --masking random --batch_size $batch_size\
        --ae_loss_reg $ae_loss_reg --pred_loss_reg $pred_loss_reg --flip_loss_reg $flip_loss_reg\
        --concept_sim_reg $concept_sim_reg --concept_far_reg $concept_far_reg \
        --random_masking_prob $random_masking_prob --cov $cov --p $p\
        --eval_causal_effect --fast $fast --freeze\

echo "start conceptshap"
CUDA_VISIBLE_DEVICES=$device python text_main.py --dataset=toy --model_name cnn --pretrained\
        --train_topic_model --epochs 100 --overall_method conceptshap --fast $fast\
        --pred_loss_reg 1 --concept_sim_reg 0.1 --concept_far_reg 0.5 --cov $cov --p $p\
        --eval_causal_effect

# nohup bash run_toy.sh > ../log/toy_0.50_cs_log 2>&1 &
# nohup bash run_toy.sh > ../log/toy_0.65_cs_log 2>&1 &