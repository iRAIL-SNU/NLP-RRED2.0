"""
MedViLL, pre-training model main run.py
"""
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "5,7"


import wandb
import argparse
from datetime import datetime

from dataset_pretrain import VLCXRDataset
from helper import get_transforms
from torch.utils.data import DataLoader

import sys
sys.path.insert(1, '/home/workspace/source/utils')
from utils import *

from train_pretrain import VLCXR_Trainer  # CXR-BERT

from transformers import BertTokenizer, AutoTokenizer
import pandas as pd
import pickle
# from data.extract_medical_vocab import extract_medical_vocab

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler



def train(args):
    wandb.init(config=args, project=args.project_name)

    set_seed(args.seed)

    if args.tokenizer == "microsoft/BiomedVLP-CXR-BERT-specialized": 
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    else:
        tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)
        
    transforms = get_transforms(args)

    print("Load Train dataset", args.train_dataset)
    train_dataset = VLCXRDataset(args.train_dataset, tokenizer, transforms, args)

    print("Load Test dataset", args.test_dataset)
    test_dataset = VLCXRDataset(args.test_dataset, tokenizer, transforms, args) \
        if args.test_dataset is not None else None

    if args.with_cuda and torch.cuda.device_count() > 1:
        dist.init_process_group("nccl")
        args.rank = dist.get_rank()
        torch.cuda.set_device(args.rank)
        args.world_size = dist.get_world_size()

        train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=args.world_size,
        rank=args.rank,
        shuffle=True,
        )   
        test_sampler = DistributedSampler(
        test_dataset,
        num_replicas=args.world_size,
        rank=args.rank,
        shuffle=False,
        )   

        print("Create DataLoader")
        train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_workers, shuffle=False, pin_memory=True)
        test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, sampler=test_sampler, num_workers=args.num_workers, shuffle=False, pin_memory=True) \
            if test_dataset is not None else None
    else:
        print("Create DataLoader")
        train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
        test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False) \
            if test_dataset is not None else None

    print("Creating BERT Trainer")
    trainer = VLCXR_Trainer(args, train_dataloader=train_data_loader, test_dataloader=test_data_loader)

    print("Training Start!")

    for epoch in range(args.epochs):
        trainer.train(epoch)
        trainer.save(epoch, args.output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ### training args ###
    parser.add_argument("--project_name", type=str, default='RRED2.0_pretrain_ex1')

    parser.add_argument("--epochs", type=int, default=100, help='number of epochs')
    parser.add_argument("--batch_size", type=int, default=8, help="number of batch size")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=128)  # loss, optimizer.step() slowly
    parser.add_argument("--warmup", type=float, default=0.1)  # optimizer = BertAdam(warmup=args.warmup)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--dropout_prob", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=123)

    parser.add_argument("--beta1", type=float, default=0.9, help="adams first beta value")
    parser.add_argument("--beta2", type=float, default=0.999, help="adams first beta value")
    parser.add_argument("--eps", type=float, default=1e-6, help="adams epsilon")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="weight_decay of AdamW")  # 0.01 , AdamW
    
    parser.add_argument("--dataset", type=str,
                        default='mimic-cxr', choices=['SNUH', 'mimic-cxr'], 
                        help="dataset for training")
    parser.add_argument("--train_dataset", type=str,
                        default='data/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-metadata_report_AP_ERECT.csv', #MIMIC-AP-ERECT
                        # default='data/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-metadata_report_AP_ERECT_train.csv', #MIMIC-AP-ERECT
                        help="train dataset for training")
    parser.add_argument("--test_dataset", type=str,
                        default='data/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-metadata_report_AP_ERECT.csv', #MIMIC
                        # default='ddata/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-metadata_report_AP_ERECT_test.csv', #MIMIC
                        help='test dataset for evaluating train set')

    output_path = 'pretrain_output/' + str(datetime.now())
    if not os.path.exists(output_path):
        os.mkdir(output_path)
        os.chmod(output_path, 0o777)
    parser.add_argument("--output_path", type=str, default=output_path, help="ex)path/to/save/model")
    parser.add_argument("--log_freq", type=int, default=10, help="printing loss every n inter: setting n")

    parser.add_argument("--with_cuda", type=bool, default=False, help="training with CUDA: True or False")
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=None, help="CUDA device ids")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=32, help="dataloader worker size")

    ## pre_trained_model_path, weight_load
    parser.add_argument("--weight_load", type=bool, default=False, help='pre-trained_model_mid_epoch_load')
    parser.add_argument("--pre_trained_model_path", type=str,
                        default=''
                        )

    ### Image Encoder args ###
    parser.add_argument("--img_size", type=int, default=512, choices=[512])
    parser.add_argument("--patch_size", type=int, default=32, choices=[32])
    parser.add_argument("--img_embed_sz", type=int, default=1024)
    parser.add_argument("--depth_img_enc", type=int, default=12, choices=[12,24,40])
    parser.add_argument("--num_head_img_enc", type=int, default=16, choices=[16,32])
    parser.add_argument("--img_mlp_hidden_sz", type=int, default=2048)
    parser.add_argument("--img_channel", type=int, default=1, choices=[1, 3])


    ### CoCa args ###
    parser.add_argument("--model_dim", type=int, default=512, choices=[512])
    parser.add_argument("--max_seq_len", type=int, default=512, help="total sequence len")
    parser.add_argument("--vocab_size", type=int, default=30522, choices=[30522, 30000, 28996, 250002])  # 28996: clinical bert / 250002: xlm-roberta # / 30522: cxr-bert
    parser.add_argument("--unimodal_depth", type=int, default=6, choices=[6,12,24])
    parser.add_argument("--multimodal_depth", type=int, default=6, choices=[6,12,24])
    parser.add_argument("--num_img_queries", type=int, default=256, choices=[256])
    parser.add_argument("--dim_head", type=int, default=64, choices=[64])
    parser.add_argument("--num_head_dec", type=int, default=8, choices=[8, 12, 24])
    parser.add_argument("--captioin_loss_weight", type=float, default=1.)
    parser.add_argument("--contrastive_loss_weight", type=float, default=1.)

    parser.add_argument("--tokenizer", type=str, default="microsoft/BiomedVLP-CXR-BERT-specialized")
    

    args = parser.parse_args()

    train(args)
