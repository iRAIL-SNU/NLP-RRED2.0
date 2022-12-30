"""
MedViLL, pre-training model main run.py
"""
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


import wandb
import argparse
from datetime import datetime

from dataset_pretrain import VLCXRDataset, VLCXRDataset_biovil
from train_pretrain import VLCXR_Trainer, VLCXR_biovil_Trainer  # CXR-BERT

from health_multimodal.image.data.transforms import create_chest_xray_transform_for_inference

from torch.utils.data import DataLoader

import sys
sys.path.insert(1, 'workspace/source/downstream/utils')
from utils import *

from helpers import get_tokenizer, get_vocab#, get_image_augmentations


import pandas as pd
import pickle
from extract_medical_vocab import extract_medical_vocab

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler



def train(args):
    run = wandb.init(config=args, project=args.project_name)

    set_seed(args.seed)

    tokenizer = get_tokenizer(args)
    vocab = get_vocab(args)
    
    transforms, augmentations_img = None, None
    if args.use_image:
        transforms = create_chest_xray_transform_for_inference(
            resize=args.TRANSFORM_RESIZE,
            center_crop_size=args.TRANSFORM_CENTER_CROP_SIZE
        )
        augmentations_img = get_image_augmentations(args.img_aug) 

    knowledge_vocab = None
    if args.use_MKP:
        with open('data/medical_words/phrase_vocab.pkl', 'rb') as f:
            phrase_vocab = pickle.load(f)
        entity_vocab = extract_medical_vocab('data/medical_words')
        knowledge_vocab = phrase_vocab + entity_vocab

    print("Load Train dataset", args.train_dataset)
    # train_dataset = VLCXRDataset(args.train_dataset, tokenizer, vocab, transforms, args, knowledge_vocab, augmentations_img)
    train_dataset = VLCXRDataset_biovil(args.train_dataset, tokenizer, vocab, transforms, args, knowledge_vocab, augmentations_img)

    print("Load Test dataset", args.test_dataset)
    # test_dataset = VLCXRDataset(args.test_dataset, tokenizer, vocab, transforms, args, knowledge_vocab) \
    test_dataset = VLCXRDataset_biovil(args.test_dataset, tokenizer, vocab, transforms, args, knowledge_vocab) \
        if args.test_dataset is not None else None

    if args.device=='cuda' and torch.cuda.device_count() > 1 and args.use_ddp:
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
    # trainer = VLCXR_Trainer(args, train_dataloader=train_data_loader, test_dataloader=test_data_loader, vocab=vocab, wandb_run=run)
    trainer = VLCXR_biovil_Trainer(args, train_dataloader=train_data_loader, test_dataloader=test_data_loader, vocab=vocab, wandb_run=run)

    print("Training Start!")

    for epoch in range(args.epochs):
        trainer.train(epoch)
        trainer.save(epoch, args.output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ### training args ###
    parser.add_argument("--project_name", type=str, default='CXRLanguageModel_pretrain')
    target_batch_size = 256
    actual_batch_size = 16
    parser.add_argument("--epochs", type=int, default=10, help='number of epochs')
    parser.add_argument("--batch_size", type=int, default=actual_batch_size, help="number of batch size")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=int(target_batch_size/actual_batch_size))  # loss, optimizer.step() slowly

    parser.add_argument("--dropout_prob", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=123)

    parser.add_argument("--beta1", type=float, default=0.9, help="adams first beta value")
    parser.add_argument("--beta2", type=float, default=0.999, help="adams first beta value")
    parser.add_argument("--eps", type=float, default=1e-6, help="adams epsilon")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="weight_decay of AdamW")  # 0.01 , AdamW
    
    parser.add_argument("--dataset", type=str,
                        default='mimic-cxr', choices=['SNUH', 'mimic-cxr'], 
                        help="dataset for training")

    parser.add_argument("--data_path", type=str, default='data/mimic-cxr-jpg/2.0.0/rred',
                        help="dset path for training")
    parser.add_argument("--data_dir_img", type=str, default='data/mimic-cxr-jpg/2.0.0/files',
                        help="dset path for training")

    parser.add_argument("--train_dataset", type=str, default='frontal_train.jsonl', 
                        help="train dataset for training")
    parser.add_argument("--test_dataset", type=str, default='frontal_val.jsonl', 
                        help='test dataset for evaluating train set')

    output_path = 'pretrain_output/' + str(datetime.now())
    if not os.path.exists(output_path):
        os.mkdir(output_path)
        os.chmod(output_path, 0o777)
    parser.add_argument("--output_path", type=str, default=output_path, help="ex)path/to/save/model")
    parser.add_argument("--log_freq", type=int, default=1, help="printing loss every n batch: setting n")

    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--use_ddp", type=str2bool, default=False)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=32, help="dataloader worker size")

    ## pre_trained_model_path, weight_load
    parser.add_argument("--weight_load", type=str2bool, default=False, help='pre-trained_model_mid_epoch_load')
    parser.add_argument("--pre_trained_model_path", type=str,
                        default=None)

    ### Image Encoder args ###
    parser.add_argument("--use_image", type=str2bool, default=False, help='vision-language? or language?')

    parser.add_argument("--TRANSFORM_RESIZE", type=int, default=512)
    parser.add_argument("--TRANSFORM_CENTER_CROP_SIZE", type=int, default=480)
    
    parser.add_argument("--img_aug", type=str, default="all", choices=["affine", "colur", "hflip", "all", "None"])
    parser.add_argument("--txt_aug", type=str, default="sentence_shuffling", choices=["sentence_shuffling","None"])

    parser.add_argument("--img_hidden_sz", type=int, default=768)
    parser.add_argument("--cross_attn_every", type=int, default=3, choices=[1,2,3,4])
    parser.add_argument("--perceiver_depth", type=int, default=1, choices=[1,2,3,4])

    parser.add_argument("--depth_img_enc", type=int, default=12, choices=[12,24,40])
    parser.add_argument("--num_head_img_enc", type=int, default=16, choices=[16,32])
    parser.add_argument("--img_mlp_hidden_sz", type=int, default=2048)
    parser.add_argument("--img_channel", type=int, default=1, choices=[1, 3])



    ### Languagemodel args ###
    parser.add_argument("--use_MKP", type=str2bool, default=True, help='use MKP?')
    parser.add_argument("--use_SET", type=int, default=0, help='use Self-evolution Training after # steps')
    parser.add_argument("--SET_alpha_for_onehot", type=float, default=.9, help='SET_alpha_for_onehot')
    parser.add_argument("--add_impression_rate", type=float, default=.5)

    parser.add_argument("--model", type=str, default="cxr-bert", choices=['mmbt', 'bert', 'clinicalbert', 'roberta', 'gatortron', 'cxr-bert'])
    parser.add_argument("--bert_model", type=str, default="microsoft/BiomedVLP-CXR-BERT-specialized",
                        choices=["bert-base-uncased, emilyalsentzer/Bio_ClinicalBERT", "xlm-roberta-base", '/home/workspace/Multi-modality-Self-supervision/GatorTron', 'microsoft/BiomedVLP-CXR-BERT-specialized'])
    
    ### CoCa args ###
    parser.add_argument("--hidden_sz", type=int, default=768, choices=[768])
    parser.add_argument("--max_seq_len", type=int, default=512, help="total sequence len")

    parser.add_argument("--mlm_loss_weight", type=float, default=1.)
    parser.add_argument("--itm_loss_weight", type=float, default=0.)

    

    args = parser.parse_args()

    train(args)
