# CausalConcept
This is the code repository for paper 'Explaining Language Models with Causal Concepts'

Running instructions:
inside src/, run  `python main.py' with the following arguments:
1. To adjust which dataset: --dataset=imdb / news / toy
    - For toy dataset: --generate_toy_data to re-generate the dataset, --p=0.7 to adjust covariance probability
2. CLS model: --model_name=bert / cnn
    - For toy dataset, only cnn is supported
    - --pretrained if you have already pretrained the CLS model (you could pass this for news dataset as we load the pretrained version from huggingface: https://huggingface.co/fabriceyhc/bert-base-uncased-ag_news)
    - --do_inference if you have not saved inferred results
3. Concept model: --train_topic_model if you have not trained it
    - overall_method=two_stage / conceptshap / BCVAE / kmeans / pca
    - n_concept=10 to adjust number of concepts
    - --flip_loss_reg=0.1 to adjust causal loss coefficient, --concept_sim_reg=0.1 and --concept_far_reg=0.1 to adjust regularizer loss coefficients
    - --lr=3e-4; --epochs=10; --batch_size=128 to adjust training arguments
    - --layer_idx=-1 to adjust which layer to interpret at 
4. post-hoc analysis:
    - postprocess: postprocess to get rid of concepts with causal effect of 0
    - one_correlated_dim: only enforce the causality loss on previous dimensions (excluding last one)
    - visualize=txt: generate the most common words from a topic
    - visualize_wordcloud: generate wordclouds from the topics
    - eval_causal_effect: evaluate causal effects of trained model

For example: To run AG-News with BERT, use our concept model on the last layer, and evaluate its causal effect and visualize, you could run:
python main.py --pretrained --do_inference --train_topic_model --eval_causal_effect --visualize=txt


File structure:
1. src/: source code
    - main.py: main running scripts that loads data, classification model, and topic model with evaluation.
    - concept_models.py: our implementation of concept models
    - cls_models.py: our implementation of CNN-based classification models
    - toy_helper.py: helper functions for toy dataset, such as creating images, datasets, etc.
    - text_helper.py: helper functions for text experiments, such as loading datasets, loading text classification models, etc.
    - visualize.py: functions to visualize topics, including gradcam and BERT visualization
    - bcvae.py & elbo_decomposition.py & lib/: implementation of Beta-TCVAE models from https://github.com/rtqichen/beta-tcvae
    - BERT_explainability/: folder that includes helper functions to achieve BERT transformer visualization, taken from https://github.com/hila-chefer/Transformer-Explainability
2. analysis/: code used to analyze the results
    - analyze_csv.py: script used to analyze human evaluation results, including agreement calculations, etc.
    - wordcloud.ipynb: script used to generate wordcloud images in the appendix.
    - produce_html_examples.py: generates human evaluation examples
3. models/: directory for storing classification models and topic models, after training, cls model will be stored in directory: models/{DATASET}/{CLS_MODEL}/cls_model.pkl, topic model will be stored in directory:models/{DATASET}/{CLS_MODEL}/{TOPIC_MODEL}/topic_model_{TOPIC_MODEL}.pkl. Both will be accompanied by a training graph with loss and accuracy.
4. human_eval_examples/: directory to save generated human evaluation examples.