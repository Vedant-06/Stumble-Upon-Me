import config as config
import dataset as dataset
import engine as engine
import torch
import pandas as pd
import torch.nn as nn
import numpy as np

from model import XLMRBase
from sklearn import model_selection
from sklearn import metrics
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
import pickle


def run():

    dfx = pd.read_csv(config.TRAINING_FILE,sep='\t')
    dfx['boilerplate'].replace(to_replace=r'"title":', value="",inplace=True,regex=True)
    dfx['boilerplate'].replace(to_replace=r'"url":',value="",inplace=True,regex=True)

    dfx['boilerplate'].replace(to_replace=r'{|}',value="",inplace=True,regex=True)
    dfx['boilerplate']=dfx['boilerplate'].str.lower()


    df_train, df_valid = model_selection.train_test_split(
        dfx, test_size=0.1, random_state=42, stratify=dfx.label.values
    )
    #Cleaning the test dataframe 
    df_test = pd.read_csv(config.TESTING_FILE,sep='\t',usecols = ['urlid','boilerplate'])
    df_test['boilerplate'].replace(to_replace=r'"title":', value="",inplace=True,regex=True)
    df_test['boilerplate'].replace(to_replace=r'"url":',value="",inplace=True,regex=True)

    df_test['boilerplate'].replace(to_replace=r'{|}',value="",inplace=True,regex=True)
    df_test['boilerplate']=df_test['boilerplate'].str.lower()  
    
    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)

    train_dataset = dataset.XMLRDataset(
        boilerplate=df_train.boilerplate.values, label=df_train.label.values
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.TRAIN_BATCH_SIZE, num_workers=4
    )

    valid_dataset = dataset.XMLRDataset(
        boilerplate=df_valid.boilerplate.values, label=df_valid.label.values
    )

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=config.VALID_BATCH_SIZE, num_workers=1
    )
    
    test_dataset = dataset.XMLRDataset(
        boilerplate=df_test.boilerplate.values
    )
    
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config.TEST_BATCH_SIZE, num_workers=1
    )

    device = torch.device(config.DEVICE)
    model = XLMRBase()
    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.001,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    num_train_steps = int(len(df_train) / config.TRAIN_BATCH_SIZE * config.EPOCHS)
    optimizer = AdamW(optimizer_parameters, lr=3e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )

    best_accuracy = 0
    best_f1_score = 0
    for epoch in range(config.EPOCHS):
        engine.train_fn(train_data_loader, model, optimizer, device, scheduler)
        outputs, targets = engine.eval_fn(valid_data_loader, model, device)
        outputs = np.array(outputs) >= 0.5
        accuracy = metrics.accuracy_score(targets, outputs)
        f1_score = metrics.f1_score(targets,outputs)
        
        print("F1 Score = {}".format(f1_score))
        print("Accuracy Score = {}".format(accuracy))
        
        if accuracy > best_accuracy:
            torch.save(model.state_dict(), config.MODEL_PATH)
            best_accuracy = accuracy

        if f1_score > best_f1_score:
            best_f1_score = f1_score
    
    best_model = torch.load(config.MODEL_PATH)
    outputs, _ = engine.eval_fn(test_data_loader,best_model,device)
    
    df_test['label']= outputs
    df_test.to_csv(config.RESULTS_FILE,columns=['urlid','label'],index=False)
    
    best_model.eval()
    outputs, targets = engine.eval_fn(valid_data_loader,best_model,device)
    print(metrics.classification_report(targets, outputs))
    
if __name__ == "__main__":
    run()
