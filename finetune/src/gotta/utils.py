'''
Parts of this code are inspired from below codebases:
https://github.com/huggingface/transformers
https://github.com/Shivanandroy/T5-Finetuning-PyTorch
https://github.com/white127/SQUAD-2.0-bidaf/blob/master/evaluate-v2.0.py
'''


from rich.table import Column, Table
from rich import box
from rich.console import Console

import torch
import json

import collections
import re
import string

def whitespace_tokenize(text):
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens

def postprocess_preds(predictions):
    processed_preds = []
    for p in predictions:
        if 'Answer:' in p:
            processed_preds.append(p.split('Answer:')[1].split('Context')[0].strip())
        else:
            processed_preds.append("")
    return processed_preds

def postprocess_actuals(actuals):
    acts = []
    
    for ac in actuals:
        acts.append([a.split('Answer:')[1].split('Context')[0].strip() for a in ac])

    return acts

# to display dataframe in ASCII format
def display_df(df):
    """display dataframe in ASCII format"""

    console = Console()
    table = Table(
        Column("source_text", justify="center"),
        Column("target_text", justify="center"),
        title="Sample Data",
        pad_edge=False,
        box=box.ASCII,
    )

    for i, row in enumerate(df.values.tolist()):
        table.add_row(row[0], row[1])

    console.print(table)
    
    
def shift_tokens_right(input_ids, pad_token_id):
    """ Shift input ids one token to the right, and wrap the last non pad token (usually <eos>).
      This is taken directly from modeling_bart.py
    """
    prev_output_tokens = input_ids.clone()
    index_of_eos = (input_ids.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
    prev_output_tokens[:, 0] = torch.tensor([pad_token_id] * len(input_ids), device=input_ids.device)#input_ids.gather(1, index_of_eos).squeeze()
    prev_output_tokens[:, 1:] = input_ids[:, :-1]
    return prev_output_tokens


def normalize_answer(s):
  """Lower text and remove punctuation, articles and extra whitespace."""
  def remove_articles(text):
    regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
    return re.sub(regex, ' ', text)
  def white_space_fix(text):
    return ' '.join(text.split())
  def remove_punc(text):
    exclude = set(string.punctuation)
    return ''.join(ch for ch in text if ch not in exclude)
  def lower(text):
    return text.lower()
  return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_tokens(s):
  if not s: return []
  return normalize_answer(s).split()

def compute_exact(a_gold, a_pred):
  return int(normalize_answer(a_gold) == normalize_answer(a_pred))

def compute_f1(a_gold, a_pred):
  gold_toks = get_tokens(a_gold)
  pred_toks = get_tokens(a_pred)
  common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
  num_same = sum(common.values())
  if len(gold_toks) == 0 or len(pred_toks) == 0:
    # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
    return int(gold_toks == pred_toks)
  if num_same == 0:
    return 0
  precision = 1.0 * num_same / len(pred_toks)
  recall = 1.0 * num_same / len(gold_toks)
  f1 = (2 * precision * recall) / (precision + recall)
  return f1

def get_metrics(preds, actual_labels):
    f1_sum = 0.0
    exact_sum = 0.0
    for p, actuals in zip(preds, actual_labels):
        f1_sum += max([compute_f1(a, p) for a in actuals])
        exact_sum += max([compute_exact(a, p) for a in actuals])
    return f1_sum / len(preds), exact_sum / len(preds)


def get_data(input_file, mask_token='<mask>', multi_answer=False):
    lines = open(input_file, "r", encoding='utf-8').read().splitlines()
    srcs = []
    trgs = []
    qa_pairs = []
    for line in lines[1:]:
        paragraph = json.loads(line)
        
        paragraph_text = paragraph["context"]
        qas = paragraph["qas"]
        pt = " ".join(whitespace_tokenize(paragraph_text))

        for qa in qas:
            question_text = qa["question"]

            multi_answers = []
            if multi_answer:
                for answer in qa["answers"]:
                    cleaned_answer_text = " ".join(
                        whitespace_tokenize(answer))
                    multi_answers.append(cleaned_answer_text)
            else:
                answer = qa["answers"][0]
                cleaned_answer_text = " ".join(
                    whitespace_tokenize(answer))

            qt = " ".join(whitespace_tokenize(question_text))
            
            if multi_answer:
                all_answers = []
                for orig_answer_text in multi_answers:
                    at = " ".join(whitespace_tokenize(orig_answer_text))
                    all_answers.append(f"Question: {qt} Answer: {at}")
                prompt = f"Question: {qt} Answer: {mask_token}. Context: {pt}"
                srcs.append(prompt)
                trgs.append(all_answers)
            else:
                answer = qa["answers"][0]
                cleaned_answer_text = " ".join(whitespace_tokenize(answer))
                at = " ".join(whitespace_tokenize(cleaned_answer_text))
                answer = f"Question: {qt} Answer: {at}"
                prompt = f"Question: {qt} Answer: {mask_token}. Context: {pt}"    
                srcs.append(prompt)
                trgs.append(answer)
                
    return srcs, trgs

