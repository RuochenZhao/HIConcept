device="2"

# first generate examples
CUDA_VISIBLE_DEVICES=$device python produce_html_examples.py --dataset news --txt

# first generate examples
CUDA_VISIBLE_DEVICES=$device python analyze_csv.py

# generate htmls 
CUDA_VISIBLE_DEVICES=$device python produce_html_examples.py --reuse --method cc

# nohup bash human_eval.sh > ../log/human_eval_log_vis 2>&1 &