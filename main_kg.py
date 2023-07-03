import os
import json
import time
import os
import random

import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

from transformers import BertTokenizer

import argparse
from model import CrossModel_KG
from model import UGDataset, UGDataModule

def check_statistics(data_rows):
    all_ent = set()
    all_r = set()
    for data in data_rows:
        sh = data['symbol_head']
        st = data['symbol_tail']
        sr = data['symbol_rel']
        all_ent.add(sh)
        all_ent.add(st)
        all_r.add(sr)
    print('Numb. of entity: %s , Numb. of relation: %s' % (len(all_ent), len(all_r)))


def extract_ug_data(data_path: Path):
    data_rows = []
    lst_syms = []
    n_line = 0
    with open(data_path, 'r') as fle:
        for line in fle.readlines()[:]:
            line = line.strip('\n')
            try:
                syms = line.split('\t')
            except ValueError:
                continue
            
            lst_syms.append(syms)

    for syms in lst_syms[:]:
        sym_h, sym_r, sym_t = syms[:3]
        
        data_rows.append({
            'symbol_head': sym_h,
            'symbol_tail': sym_t,
            'symbol_rel': sym_r,
        })

    check_statistics(data_rows)
    return pd.DataFrame(data_rows)
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='parameters for the cross-stitch model')

    parser.add_argument('--surface-cross-stitch-send-layers', type=list, default=[0])
    parser.add_argument('--surface-cross-stitch-receive-layers', type=list, default=[1])
    parser.add_argument('--symbol-cross-stitch-send-layers', type=list, default=[0])
    parser.add_argument('--symbol-cross-stitch-receive-layers', type=list, default=[1])
    parser.add_argument('--surface-loss-weight', type=float, default=0.6)
    parser.add_argument('--symbolic-loss-weight', type=float, default=0.4)
    parser.add_argument('--alignment-loss-weight', type=float, default=0.3)
    parser.add_argument('--stitch-model', type=str, default='none')
    parser.add_argument('--cross-stitch-start-epoch', type=int, default=0)
    parser.add_argument('--freeze-symbol-until-epoch', type=int, default=0)
    parser.add_argument('--freeze-surface-until-epoch', type=int, default=0)
    parser.add_argument('--attn-dim', type=int, default=64)
    parser.add_argument('--alignment-attn-dim', type=int, default=64)
    
    parser.add_argument('--surface_transformer', type=str, default='bert-base-uncased')
    parser.add_argument('--surface-random-init', action='store_true')
    parser.add_argument('--surface-freeze-encoder', action='store_true')

    parser.add_argument('--model', type=str, default='crossmodel')
    parser.add_argument('--lr', type=float, default=2e-5)

    parser.add_argument('--triple-trf-n-hidden', type=int, default=6)
    parser.add_argument('--triple-trf-dropout', type=float, default=0.1)
    parser.add_argument('--triple-trf-n-attn-heads', type=int, default=12)
    parser.add_argument('--head', type=str, default='lmhead')
    parser.add_argument('--pretrained-emb', action='store_true')
    parser.add_argument('--symbolic_transformer', type=str, default='roberta-base')

    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--token_len', type=int, default=10)

    parser.add_argument('--data_addr', type=Path, default='data/kg.txt')
    parser.add_argument('--sym2id_addr', type=Path, default='data/sym2id.json')

    parser.add_argument('--sym_size', type=int, default=10000)
    parser.add_argument('--kg_ckpt', type=str, default='')
    
    args = parser.parse_args()
    
    df = extract_ug_data(args.data_addr)
    with open(args.sym2id_addr) as file_sym2id:
        symbol2id = json.load(file_sym2id)
    symbol2id['[MASK]'] = len(symbol2id)
    args.sym_size = len(symbol2id)
    
    train_df, val_df = train_test_split(df, test_size=0.9)
    train_df = df
    
    tb_logger = pl_loggers.TensorBoardLogger(save_dir="./lightning_logs/logs_kg_%s/" % args.stitch_model)
    
    BATCH_SIZE = args.batch_size
    N_EPOCHS = args.epoch
    TOKEN_LEN = args.token_len

    tx_tokenizer = BertTokenizer.from_pretrained(args.surface_transformer)
    
    data_module = UGDataModule(train_df, val_df, tx_tokenizer, symbol2id, batch_size=BATCH_SIZE, max_token_len=TOKEN_LEN)
    data_module.setup()
    
    model = CrossModel_KG(args) 
    
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints_kg',
        filename='kg_encoder',
        save_top_k=1,
        verbose=True,
        monitor='sym_train_loss',
        mode='min'
    )

    trainer = pl.Trainer(
        checkpoint_callback=checkpoint_callback,
        max_epochs=N_EPOCHS,
        gpus=[args.gpu],
        progress_bar_refresh_rate=30,
        logger=tb_logger
    )

    trainer.fit(model, data_module)


