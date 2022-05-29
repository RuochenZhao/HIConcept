# CausalConcept
This is the code repository for paper 'Explaining Language Models with Causal Concepts'

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