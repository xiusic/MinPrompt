'''
Parts of this code are inspired from below codebases:
https://github.com/huggingface/transformers
https://github.com/Shivanandroy/T5-Finetuning-PyTorch
'''

# rich: for a better display on terminal
from rich.table import Column, Table
from rich import box
from rich.console import Console

from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import set_seed

import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

import os
import utils
from text_dataset import TextDataset

import pandas as pd


# define a rich console logger
console = Console(record=True)

MAX_GEN_LEN_TEST = 50


def train(epoch, tokenizer, model, model_aug, device, loader, training_aug_loader, optimizer, training_logger, loss_weight):

    """
    Function to be called for training with the parameters passed from main function

    """

    model.train()
    model_aug.train()
    total_loss = 0.
    dataloader_iterator = iter(training_aug_loader)
    for step, data in enumerate(loader):
        try:
            aug_data = next(dataloader_iterator)
        except StopIteration:
            dataloader_iterator = iter(training_aug_loader)
            aug_data = next(dataloader_iterator)
#     for step, data in enumerate(loader, 0):
        y = data["target_ids"].to(device, dtype=torch.long)
        y[y == tokenizer.pad_token_id] = -100
        ids = data["source_ids"].to(device, dtype=torch.long)
        mask = data["source_mask"].to(device, dtype=torch.long)
        outputs = model(
            input_ids=ids,
            attention_mask=mask,
            labels=y,
        )
        loss = outputs[0]
        
        aug_y = aug_data["target_ids"].to(device, dtype=torch.long)
        aug_y[aug_y == tokenizer.pad_token_id] = -100
        aug_ids = aug_data["source_ids"].to(device, dtype=torch.long)
        aug_mask = aug_data["source_mask"].to(device, dtype=torch.long)
        outputs_aug = model_aug(
            input_ids=aug_ids,
            attention_mask=aug_mask,
            labels=aug_y,
        )
        loss_aug = outputs_aug[0]
        
        loss += loss_weight*loss_aug
      
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    training_logger.add_row(str(epoch), str(step), str(total_loss / (step + 1)))
    console.print(training_logger)


def validate(epoch, tokenizer, model, device, loader, max_gen_len=35):

  """
  Function to evaluate model for predictions

  """
  model.eval()
  predictions = []
  actuals = []
  with torch.no_grad():
      for _, data in enumerate(loader, 0):
          y = data['target_ids'].to(device, dtype = torch.long)
          ids = data['source_ids'].to(device, dtype = torch.long)
          mask = data['source_mask'].to(device, dtype = torch.long)

          generated_ids = model.generate(
              input_ids = ids,
              attention_mask = mask, 
              max_length=max_gen_len, 
              num_beams=1, 
              early_stopping=True,
              #decoder_start_token_id=tokenizer.bos_token_id
              )
          preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
          target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in y]
          if _%100==0:
              console.print(f'Completed {_}')

          predictions.extend(preds)
          actuals.extend(target)
  return predictions, actuals


def run_training_and_eval(
    dataframe, aug_dataframe, val_dataframe, test_dataframe, source_text, target_text, model_params, 
    model_class, device, output_dir="./outputs/", eval_only=False, loss_weight=1.0
):
    set_seed(model_params["SEED"])
    # Set random seeds and deterministic pytorch for reproducibility
    torch.backends.cudnn.deterministic = True

    # logging
    console.log(f"""[Model]: Loading {model_params["MODEL"]}...\n""")

    # tokenzier for encoding the text
    tokenizer = BartTokenizer.from_pretrained(model_params["MODEL"])

    # logging
    console.log(f"[Data]: Reading data...\n")

    # Importing the raw dataset
    dataframe = dataframe[[source_text, target_text]]
    utils.display_df(dataframe.head(2))

    # Creation of Dataset and Dataloader
    # Defining the train size. So 80% of the data will be used for training and the rest for validation.
    #train_size = 0.5
    train_dataset = dataframe
    aug_dataset = aug_dataframe
    val_dataset = val_dataframe
    test_dataset = test_dataframe


    # training logger to log training progress
    training_logger = Table(
        Column("Epoch", justify="center"),
        Column("Steps", justify="center"),
        Column("Loss", justify="center"),
        title="Training Status",
        pad_edge=False,
        box=box.ASCII,
    )

    console.print(f"FULL Dataset: {dataframe.shape}")
    console.print(f"TRAIN Dataset: {train_dataset.shape}")
    console.print(f"TRAIN_AUG Dataset: {aug_dataset.shape}")
    console.print(f"VAL Dataset: {val_dataset.shape}")
    console.print(f"TEST Dataset: {test_dataset.shape}\n")

    # Creating the Training and Validation dataset for further creation of Dataloader
    training_set = TextDataset(
        train_dataset,
        tokenizer,
        model_params["MAX_SOURCE_TEXT_LENGTH"],
        model_params["MAX_TARGET_TEXT_LENGTH"],
        source_text,
        target_text,
    )
    
    training_aug_set = TextDataset(
        aug_dataset,
        tokenizer,
        model_params["MAX_SOURCE_TEXT_LENGTH"],
        model_params["MAX_TARGET_TEXT_LENGTH"],
        source_text,
        target_text,
    )
    
    val_set = TextDataset(
        val_dataset,
        tokenizer,
        model_params["MAX_SOURCE_TEXT_LENGTH"],
        model_params["MAX_TARGET_TEXT_LENGTH"],
        source_text,
        target_text,
    )
    test_set = TextDataset(
        test_dataset,
        tokenizer,
        model_params["MAX_SOURCE_TEXT_LENGTH"],
        model_params["MAX_TARGET_TEXT_LENGTH"],
        source_text,
        target_text,
    )

    # Defining the parameters for creation of dataloaders
    train_params = {
        "batch_size": model_params["TRAIN_BATCH_SIZE"],
        "shuffle": True,
        "num_workers": 0,
    }

    train_aug_params = {
        "batch_size": model_params["TRAIN_BATCH_SIZE"],
        "shuffle": True,
        "num_workers": 0,
    }

    val_params = {
        "batch_size": model_params["VALID_BATCH_SIZE"],
        "shuffle": False,
        "num_workers": 0,
    }

    # Creation of Dataloaders for testing and validation. This will be used down for training and validation stage for the model.
    training_loader = DataLoader(training_set, **train_params)
    training_aug_loader = DataLoader(training_aug_set, **train_aug_params)
    val_loader = DataLoader(val_set, **val_params)
    test_loader = DataLoader(test_set, **val_params)
    
    
    if not eval_only:
        # Defining the model. We are using t5-base model and added a Language model layer on top for generation of Summary.
        # Further this model is sent to device (GPU/TPU) for using the hardware.
        model = model_class.from_pretrained(model_params["MODEL"])
        model = model.to(device)
        
#         model_aug = model_class.from_pretrained(model_params["MODEL"])
#         model_aug = model_aug.to(device)
        
        model_aug = model

        # Defining the optimizer that will be used to tune the weights of the network in the training session.
        optimizer = torch.optim.Adam(
            params=set(list(model.parameters())+list(model_aug.parameters())), lr=model_params["LEARNING_RATE"]
        )
    
        # Training loop
        console.log(f"[Initiating Fine Tuning]...\n")

        best_f1 = -1
        for epoch in range(model_params["TRAIN_EPOCHS"]):
            train(epoch, tokenizer, model, model_aug, device, training_loader, training_aug_loader, optimizer, training_logger, loss_weight)
            predictions, _ = validate(epoch, tokenizer, model, device, val_loader)
            
            processed_preds = utils.postprocess_preds(predictions)
            processed_actuals = utils.postprocess_actuals(val_dataframe['multi_target'])
            cur_metrics = utils.get_metrics(processed_preds, processed_actuals)
            if cur_metrics[0] > best_f1:
                console.log(f"New best: {cur_metrics}")
                console.log(f"[Saving Model]...\n")
                # Saving the model after training
                path = os.path.join(output_dir, "model_files")
                model.save_pretrained(path)
                tokenizer.save_pretrained(path)
                best_f1 = cur_metrics[0]

    model = model_class.from_pretrained(os.path.join(output_dir, "model_files")).to(device)
    # evaluating test dataset
    console.log(f"[Initiating Validation]...\n")
    for epoch in range(model_params["VAL_EPOCHS"]):
        predictions, _ = validate(epoch, tokenizer, model, device, test_loader, max_gen_len=MAX_GEN_LEN_TEST)
        
        generation_data = {"Generated_Text": predictions, "Actual_Text": test_dataframe['multi_target']}
        processed_preds = utils.postprocess_preds(predictions)
        processed_actuals = utils.postprocess_actuals(test_dataframe['multi_target'])
        generation_data["PP_Generated_Text"] = processed_preds
        generation_data["PP_Actual_Text"] = processed_actuals
        generation_df = pd.DataFrame(generation_data)
        generation_df.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)
        test_metrics = utils.get_metrics(processed_preds, processed_actuals)
        console.log(f"Test metrics: {test_metrics}")

    console.save_text(os.path.join(output_dir, "logs.txt"))

    console.log(f"[Validation Completed.]\n")
    console.print(
        f"""[Model] Model saved @ {os.path.join(output_dir, "model_files")}\n"""
    )
    console.print(
        f"""[Validation] Generation on Validation data saved @ {os.path.join(output_dir,'predictions.csv')}\n"""
    )
    console.print(f"""[Logs] Logs saved @ {os.path.join(output_dir,'logs.txt')}\n""")
    
    return test_metrics
