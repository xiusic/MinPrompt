#!/usr/bin/env python

'''
Parts of this code are inspired from below codebases:
https://github.com/huggingface/transformers
https://github.com/Shivanandroy/T5-Finetuning-PyTorch
'''

import argparse
import pickle
import time
from transformers import BartTokenizer, BartForConditionalGeneration
import torch
from torch import cuda

# Importing libraries
import numpy as np
import pandas as pd
import torch.nn.functional as F
import random

# rich: for a better display on terminal
from rich.table import Column, Table
from rich import box
from rich.console import Console

import utils
from trainer import run_training_and_eval


if __name__ == "__main__":
    start = time.time()

    # define a rich console logger
    console = Console(record=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/home/QA/FewshotQA_balanced_prompt/FewshotQA/mrqa-few-shot")
    parser.add_argument("--loss_weight", type=float, default=1.0)
    parser.add_argument("--dataset_name", type=str, default="squad", choices=['bioasq' , 'hotpotqa', 'naturalquestions', 'newsqa', 'searchqa', 'squad', 'textbookqa', 'triviaqa'])
    parser.add_argument(
        "--gpu_id",
        default=0,
        type=int
    )

    args = parser.parse_args()

    device = f'cuda:{args.gpu_id}' if cuda.is_available() else 'cpu'

    model_name = "facebook/bart-large"
    model_class = BartForConditionalGeneration

    tokenizer = BartTokenizer.from_pretrained(model_name)
    mask_token = tokenizer.mask_token

    dataset_name = args.dataset_name

    data_seeds = [42, 43, 44, 45, 46] #[42, 43, 44, 45, 46]
    train_sizes = [16, 32, 64, 128] #[16, 32, 64, 128]
    output_suffix = "qa_only"
    import os
    os.makedirs(f"{dataset_name}_{output_suffix}", exist_ok=True)


    model_params = {
        "MODEL": model_name,  # model_type: bart-large etc
        "TRAIN_BATCH_SIZE": 2,  # training batch size
        "VALID_BATCH_SIZE": 32,  # validation batch size
        "TRAIN_EPOCHS": 25,  # number of training epochs
        "VAL_EPOCHS": 1,  # number of validation epochs
        "LEARNING_RATE": 2e-5,  # learning rate
        "SEED": 42,  # set seed for reproducibility,
        "MAX_SOURCE_TEXT_LENGTH": 800
    }


    data_dir = args.data_dir
    all_results = {}
    for train_size in train_sizes:
        seed_f1s = []
        for data_seed in data_seeds:
            console.print(f"Running {train_size} {data_seed}........................")
            output_dir=f"{dataset_name}_seed_results/{train_size}_{data_seed}_{output_suffix}"
            os.makedirs(output_dir, exist_ok=True)

            train_srcs, train_trgs = utils.get_data(f'{data_dir}/{dataset_name}/{dataset_name}-train-seed-{data_seed}-num-examples-{train_size}.jsonl', multi_answer=False)
            _, train_multi_trgs = utils.get_data(f'{data_dir}/{dataset_name}/{dataset_name}-train-seed-{data_seed}-num-examples-{train_size}.jsonl', multi_answer=True)
            f_train_masked_context = f"{data_dir}/{dataset_name}/{dataset_name}-train-num-examples-{train_size}-prompt-masked-answer-context.pkl"
            with open(f_train_masked_context, 'rb') as f:
                train_masked_context = pickle.load(f)
            print(len(train_masked_context))
            # train_samples = list(zip(train_srcs, train_trgs, train_multi_trgs))
#             train_samples = []
            train_samples = list(zip(train_srcs, train_trgs, train_multi_trgs))
#             train_samples += train_masked_context

            #For validation, we always use the seed 42 and size 1024 subset.
            dev_srcs, dev_trgs = utils.get_data(f'{data_dir}/{dataset_name}/{dataset_name}-train-seed-42-num-examples-1024.jsonl', multi_answer=False)
            _, dev_multi_trgs = utils.get_data(f'{data_dir}/{dataset_name}/{dataset_name}-train-seed-42-num-examples-1024.jsonl', multi_answer=True)
            dev_samples = list(zip(dev_srcs, dev_trgs, dev_multi_trgs))
            random.seed(42)
            random.shuffle(dev_samples)
            dev_samples = dev_samples[:len(train_samples)]

            test_srcs, test_trgs = utils.get_data(f'{data_dir}/{dataset_name}/dev.jsonl', multi_answer=False)
            _, test_multi_trgs = utils.get_data(f'{data_dir}/{dataset_name}/dev.jsonl', multi_answer=True)
            test_samples = list(zip(test_srcs, test_trgs, test_multi_trgs)) #[:128]

            src_lens = [len(tokenizer(src)['input_ids']) for (src, trg, _) in dev_samples]
            max_src_len = min(1024, np.max(src_lens))

            trg_lens = [len(tokenizer(trg)['input_ids']) for (src, trg, _) in dev_samples]
            max_trg_len = min(1024, np.max(trg_lens))
            
            console.print(f"#Train:{len(train_samples)}. #Dev: {len(dev_samples)}. #Test: {len(test_samples)}")
            console.print(f"max_src_len:{max_src_len}. max_trg_len: {max_trg_len}.")

            model_params["MAX_TARGET_TEXT_LENGTH"] = max_trg_len

            train_df = pd.DataFrame(train_samples, columns=['source_text', 'target_text', 'multi_target'])
            train_aug_df = pd.DataFrame(train_masked_context, columns=['source_text', 'target_text', 'multi_target'])
            dev_df = pd.DataFrame(dev_samples, columns=['source_text', 'target_text', 'multi_target'])
            test_df = pd.DataFrame(test_samples, columns=['source_text', 'target_text', 'multi_target'])

            test_metrics = run_training_and_eval(
                dataframe=train_df,
                aug_dataframe=train_aug_df,
                val_dataframe=dev_df,
                test_dataframe=test_df,
                source_text="source_text",
                target_text="target_text",
                model_params=model_params,
                output_dir=output_dir,
                model_class=model_class,
                eval_only=False,
                device=device,
                loss_weight=args.loss_weight
            )
            seed_f1s.append(test_metrics[0])
            
            all_results[(train_size, data_seed)] = test_metrics
            
            with open(f"{dataset_name}_{output_suffix}/seed_results.txt", 'a+') as f:
                f.write(f"Size: {train_size}. Seed: {data_seed}. F1: {test_metrics[0]}\n")
        
        all_results[train_size] = (np.mean(seed_f1s), np.std(seed_f1s))
        final_output_dir = f"{dataset_name}_seed_results/{train_size}_{output_suffix}"
        os.makedirs(final_output_dir, exist_ok=True)
        with open(os.path.join(final_output_dir, f"all_results_{args.loss_weight}.p"), 'wb') as results_file:
            pickle.dump(all_results, results_file)
        with open(f"{dataset_name}_{output_suffix}/results.txt", 'a+') as f:
            f.write(f"Size: {train_size}.  Mean: {all_results[train_size][0]}. Std: {all_results[train_size][1]}.\n")

            
    console.log(f"Took {(time.time() - start) / 60} minutes. ")

