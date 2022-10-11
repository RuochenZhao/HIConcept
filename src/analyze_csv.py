import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score, precision_recall_fscore_support
import re
from  nltk.metrics import agreement
from nltk.metrics.distance import masi_distance
from nltk.metrics.distance import jaccard_distance

def flat(lst):
    return [item for sublist in lst for item in sublist]

def remove_nan_order(li):
    li = [str(l).replace('.0', '') for l in li]
    li = [value for value in li if value != 'nan']
    li.sort()
    return li

def calculate_agreement_micro(l1, l2):
    total_num = max(len(l1), len(l2))
    match_num = sum([1 if i in l2 else 0 for i in l1])
    return match_num/total_num

def calculate_agreement_macro(answer_df):
    agreements = []
    for index, row in answer_df.iterrows():
        agreements.append(calculate_agreement_micro(row['a_0'], row['a_1']))
    print(f'agreement: {np.mean(agreements)}')

def order_by_intersection(l1, l2, num = 3):
    l1_new = []
    l2_new = []
    for (i, l1_i) in enumerate(l1):
        l2_i = l2[i]
        le = min(len(l1_i), len(l2_i))
        intersection = [ele for ele in l1_i if ele in l2_i]
        intersection.sort()
        l1_new.append(intersection + [ele for ele in l1_i if ele not in intersection] + ['x']*(num-le))
        l2_new.append(intersection + [ele for ele in l2_i if ele not in intersection] + ['x']*(num-le))
    return l1_new, l2_new

def create_annot(an):
    """
    Create frozensets with the unique label
    or with both labels splitting on pipe.
    Unique label has to go in a list so that
    frozenset does not split it into characters.
    """
    if "|" in str(an):
        an = frozenset(an.split("|"))
    else:
        # single label has to go in a list
        # need to cast or not depends on your data
        an = frozenset([str(int(an))])
    return an

def cohen_kappa_agreement(l1, l2):
    l1, l2 = order_by_intersection(l1, l2)
    data = []
    for idx in range(len(l1)):
        data.append(("a1", idx, create_annot('|'.join(l1[idx]))))
        data.append(("a2", idx, create_annot('|'.join(l2[idx]))))
        
    atask = agreement.AnnotationTask(data=data, distance=jaccard_distance)
    print("Pi: {}".format(atask.pi()))
    print("Krippendorff's alpha : {}".format(atask.alpha()))
    print("Cohen's Kappa:", atask.kappa())
    print("Fleiss's Kappa:", atask.multi_kappa())

def get_top_k_words(df, method, k):
    top_k = []
    for (i, row) in df.iterrows():
        scores = row[f'{method}_scores']
        tokens = row['text']
        # make it averaged
        dic = {}
        for (idx, t) in enumerate(tokens):
            if t not in dic.keys():
                dic[t] = [scores[idx]]
            else:
                dic[t].append(scores[idx])
        for key in dic.keys():
            dic[key] = np.mean(dic[key])
        sorted_dict = sorted(dic.items(), key=lambda x: x[1], reverse=True)
        topk = [tok[0] for tok in sorted_dict]
        topk = [tok for tok in topk if re.match('.*[a-zA-Z0-9].*', tok)]
        top_k.append(topk[:k])
    return top_k

def calculate_precision_recall_f1(answer_df, truth_df, k = 6):
    old_answers = answer_df['a_0'] + answer_df['a_1']
    # do a unique superset
    old_answers = [np.unique(answer) for answer in old_answers]
    # do cc first
    topwords_cc = get_top_k_words(truth_df, method = 'cc', k = k)
    answers, topwords_cc = order_by_intersection(old_answers, topwords_cc, num = 6)
    ans = precision_recall_fscore_support(flat(answers), flat(topwords_cc), average='micro')
    print(f'For CC, precision: {ans[0]}, recall: {ans[1]}, f-1: {ans[2]}')
    # do baseline
    topwords_cs = get_top_k_words(truth_df, method = 'cs', k = k)
    answers, topwords_cs = order_by_intersection(old_answers, topwords_cs, num = 6)
    ans = precision_recall_fscore_support(flat(answers), flat(topwords_cs), average='weighted')
    print(ans)
    print(f'For BS, precision: {ans[0]}, recall: {ans[1]}, f-1: {ans[2]}')

def reprocess_scores(truth_df):
    all_cc = []
    for cc_scores in truth_df['cc_scores'].values:
        all_cc.append(np.array([float(x) for x in cc_scores.replace('[', '').replace(']', '').split(',')]))
    all_cs = []
    for cs_scores in truth_df['cs_scores'].values:
        all_cs.append([float(x) for x in cs_scores.replace('[', '').replace(']', '').split(',')])
    all_text = []
    for text in truth_df['text'].values:
        all_text.append(text.split(' '))
    truth_df['cc_scores'] = all_cc
    truth_df['cs_scores'] = all_cs
    truth_df['text'] = all_text
    return truth_df

def process_wrong_scores(arr, len = 512):
    arr1 = arr[len*1:len*2]
    arr2 = arr[len*2:len*3]
    arr3 = arr[len*3:]
    return arr1+arr2+arr3

# SUPERSET ANALYSIS
def superset_analysis(answer_df, truth_df):
    print(f'Number_of_examples: {len(answer_df)}')
    overall_df = truth_df.join(answer_df)
    cc_causal = []
    cs_causal = []
    for index, row in overall_df.iterrows():
        tokens = row['text']
        cc_scores = row['cc_scores']
        cs_scores = row['cs_scores']
        answers = [row[f'a_{n}'] for n in range(num)]
        # do a unique superset
        answers = np.unique(flat(answers))
        for a in answers:
            if (not pd.isna(a)) and (a!='nan') and (a!='x'):
                token_idx = [id for id, t in enumerate(tokens) if a.lower().strip() in t.lower()]
                if len(token_idx)==0:
                    print('a: ', a)
                    print('a!=\'x\': ', a!='x')
                    raise Exception(f'Encountered answer {a} not in list {tokens}!')
                cc_causal.append(np.mean([cc_scores[t] for t in token_idx]))
                cs_causal.append(np.mean([cs_scores[t] for t in token_idx]))
    print('cc_causal average: ', np.mean(cc_causal))
    print('cs_causal average: ', np.mean(cs_causal))

# ANNOTATOR LEVEL ANALYSIS
def annotator_level(answer_df, truth_df, n):
    print(f'Number_of_examples: {len(answer_df)}')
    overall_df = truth_df.join(answer_df)
    cc_causal = []
    cs_causal = []
    for index, row in overall_df.iterrows():
        tokens = row['text']
        cc_scores = row['cc_scores']
        cs_scores = row['cs_scores']
        answers = row[f'a_{n}']
        for a in answers:
            if not pd.isna(a) and a!='nan' and a!='x':
                token_idx = [id for id, t in enumerate(tokens) if a.lower().strip() in t.lower()]
                if len(token_idx)==0:
                    raise Exception(f'Encountered answer {a} not in list {tokens}!')
                cc_causal.append(np.mean([cc_scores[t] for t in token_idx]))
                cs_causal.append(np.mean([cs_scores[t] for t in token_idx]))
    print('cc_causal average: ', np.mean(cc_causal))
    print('bs_causal average: ', np.mean(cs_causal))

answer_df = pd.read_csv('human_eval_examples/answer_all_ordered.csv')
truth_df = pd.read_csv('human_eval_examples/news/txts/df.csv')
truth_df = reprocess_scores(truth_df)
answer_df = answer_df.drop(columns=['Unnamed: 0'])
truth_df = truth_df.drop(columns=['Unnamed: 0'])
num = 2
print(f'{num} responses:')

correct_df = truth_df.loc[truth_df['ground_truth'] == truth_df['pred']]
false_df = truth_df.loc[truth_df['ground_truth'] != truth_df['pred']]

# REPROCESS ANSWER_DF 
answer_df['a_0'] = [a.lower().split(',') for a in list(answer_df['a_0'].values)]
answer_df['a_1'] = [a.lower().split(',') for a in list(answer_df['a_1'].values)]
answer_correct_df = answer_df.loc[truth_df['ground_truth'] == truth_df['pred']]
answer_false_df = answer_df.loc[truth_df['ground_truth'] != truth_df['pred']]


print(f'-----------------SUPERSET ANALYSIS CORRECT------------------')
print('Correct ones:')
superset_analysis(answer_correct_df, correct_df)
calculate_agreement_macro(answer_correct_df)
cohen_kappa_agreement(list(answer_correct_df['a_0'].values), list(answer_correct_df['a_1'].values))
# calculate_precision_recall_f1(answer_correct_df, correct_df)

print('\n')

print('False ones:')
superset_analysis(answer_false_df, false_df)
calculate_agreement_macro(answer_false_df)
cohen_kappa_agreement(list(answer_false_df['a_0'].values), list(answer_false_df['a_1'].values))
# calculate_precision_recall_f1(answer_false_df, false_df)

print('\n')

print('Overall:')
superset_analysis(answer_df, truth_df)
calculate_agreement_macro(answer_df)
cohen_kappa_agreement(list(answer_df['a_0'].values), list(answer_df['a_1'].values))

print(f'-----------------SUPERSET ANALYSIS CORRECT------------------')
print('Overall:')
print(f'-----------------ANNOTATOR 0------------------')
annotator_level(answer_df, truth_df, 0)
print(f'-----------------ANNOTATOR 1------------------')
annotator_level(answer_df, truth_df, 1)