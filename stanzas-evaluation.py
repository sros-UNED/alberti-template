#!/usr/bin/env python
# coding: utf-8
# conda install pytorch>=1.6 cudatoolkit=10.2 -c pytorch
# wandb login XXX
import json
import logging
import os
import re
import sklearn
import sys
import time
from itertools import product

import numpy as np
import pandas as pd
import wandb
#from IPython import get_ipython
from simpletransformers.classification import ClassificationModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


truthy_values = ("true", "1", "y", "yes")
TAG = os.environ.get("TAG", "bertsification")
# Este es el experimtento y el modelo que vamos usar ?
MODELNAME = os.environ.get("MODELNAME", "bert;bert-base-multilingual-cased")
OVERWRITE = os.environ.get("OVERWRITE", "False").lower() in truthy_values
logging.basicConfig(level=logging.INFO, filename=time.strftime("models/{}-%Y-%m-%dT%H%M%S.log".format(TAG)))
with open('pid', 'w') as pid:
    pid.write(str(os.getpid()))
logging.info("Experiment '{}', (eval_df = {}, pid = {})".format(
    TAG, MODELNAME, str(os.getpid()),
))

# Utils
def clean_text(string):
    output = string.strip()
    # replacements = (("“", '"'), ("”", '"'), ("//", ""), ("«", '"'), ("»",'"'))
    replacements = (
      ("“", ''), ("”", ''), ("//", ""), ("«", ''), ("»",''), (",", ''),
      (";", ''), (".", ''),
    #   ("?", ''), ("¿", ''), ("¡", ''), ("!", ''), ("-", ' '),
    )
    for replacement in replacements:
        output = output.replace(*replacement)
    # Any sequence of two or more spaces should be converted into one space
    output = re.sub(r'(?is)\s+', ' ', output)
    return output.strip()


def clean_labels(label):
    return "unknown" if str(label) == "None" else label

# SE carga en el fichero disco compartido en GCLOUD ***
def prepare_data():
    df = (pd
        .read_csv('/shared/stanzas-evaluation.csv')
        # Srenombrar las columnas
        .rename(columns={"Stanzas_text":"text", "ST_Correct":"stanza"})
        .assign(
            text=lambda x: x["text"].apply(clean_text(x)),
            stanza=lambda x: x["stanza"].apply(clean_text(x)),
        )
    )

    #Codificar las variable del 0 al 46
    label_encoder = LabelEncoder()
    label_encoder.fit(df["stanza"])
    df["labels"] =label_encoder.transform(df["stanzas"])
    train_df, eval_df = train_test_split(
        df, stratify=df["labels"], test_size=0.25, random_state=42
    )
    return train_df, eval_df, label_encoder


def train_model(train_df, num_labels):
    model_type, model_name = MODELNAME.split(";")
    model_output = 'models/{}-{}-{}'.format(TAG, model_type, model_name.replace("/", "-"))
    if OVERWRITE is False and os.path.exists(model_output):
        logging.info("Skipping training of {}".format(model_name))
        sys.exit(0)
    logging.info("Starting training of {}".format(model_name))
    run = wandb.init(project=model_output.split("/")[-1], reinit=True)

    model = ClassificationModel(
        model_type, model_name, num_labels=num_labels, args={
            'output_dir': model_output,
            'best_model_dir': '{}/best'.format(model_output),
            'evaluate_during_training': False,
            'manual_seed': 42,
            # 'learning_rate': 2e-5,  # For BERT, 5e-5, 3e-5, 2e-5
            # For BERT 16, 32. It could be 128, but with gradient_acc_steps set to 2 is equivalent
            'train_batch_size': 8 if "large" in model_name else 32,
            'eval_batch_size': 8 if "large" in model_name else 32,
            # Doubles train_batch_size, but gradients and wrights are calculated once every 2 steps
            'gradient_accumulation_steps': 2 if "large" in model_name else 1,
            'max_seq_length': 64,
            'wandb_project': model_output.split("/")[-1],
            # "adam_epsilon": 3e-5,  # 1e-8
            "silent": False,
            "fp16": False,
            "n_gpu": 1,
    })
    # train the model
    model.train_model(train_df)
    return model, run


def eval_model(model, eval_df, run):
    result, *_ = model.eval_model(eval_df)
    logging.info("Results: {}".format(str(result)))

    eval_df["predicted"], *_ = model.predict(eval_df["text"])

    acc = sum(eval_df.labels == eval_df.predicted) / eval_df.labels.size
    logging.info("Accuracy: {}".format(acc))
    wandb.log({"accuracy_es": acc})

    run.finish()


def main() -> None:
    logging.info("Starting...")
    train_df, eval_df, label_encoder = prepare_data()
    model, run = train_model(train_df, len(label_encoder.classes_))
    eval_model(model, eval_df, run)
    logging.info("Done.")


if __name__ == "__main__":
    main()
