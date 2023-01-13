# HEConcept
This is the code repository for paper 'Explaining Language Models’ Predictions with High-Impact Concepts'

Running instructions:
inside src/, use the following running scripts:
1. For text experiments:
    - baselines: run_text_baselines.sh
    - CausalConcept with ablation studies: run_text_causal.sh
2. For toy experiments: run_toy.sh
3. For human evaluation: human_eval.sh
4. For hyperparameter analysis with number of concepts: num_of_concepts.sh


File structure:
1. src/: source code
    - text_main.py: main running scripts that loads data, classification model, and topic model with evaluation.
    - conceptshp.py: our implementation of concept models, we looked at original conceptSHAP repo: https://github.com/chihkuanyeh/concept_exp; and Berkeley's version: https://github.com/arnav-gudibande/conceptSHAP as references.
    - helper.py: our implementation of self-constructed CNN-based transformer classification models
    - toy_helper.py: helper functions for toy dataset, such as creating images, datasets, etc.
    - text_helper_v2.py: helper functions for text experiments, such as loading datasets, loading text classification models, etc.
    - visualize.py: functions to visualize topics, including gradcam and BERT visualization
    - bcvae.py & elbo_decomposition.py & lib/: implementation of Beta-TCVAE models from https://github.com/rtqichen/beta-tcvae
    - BERT_explainability/ & BERT_rationale_benchmark: folder that includes helper functions to achieve BERT transformer visualization, taken from https://github.com/hila-chefer/Transformer-Explainability
    - code used to analyze the results
        - analyze_csv.py: script used to analyze human evaluation results, including agreement calculations, etc.
        - wordcloud.ipynb: script used to generate wordcloud images in the appendix.
        - produce_html_examples.py: generates human evaluation examples
3. src/models/: directory for storing classification models and topic models, after training, cls model will be stored in directory: models/{DATASET}/{CLS_MODEL}/cls_model.pkl, topic model will be stored in directory:models/{DATASET}/{CLS_MODEL}/{TOPIC_MODEL}/topic_model_{TOPIC_MODEL}.pkl. Both will be accompanied by a training graph with loss and accuracy.
4. src/human_eval_examples/: directory to save generated human evaluation examples.
5. environment.yml: file to create the conda environment. Run ' conda env create -f environment.yml'
5. t5_env.yml: please use this environment to fine-tune t5.
