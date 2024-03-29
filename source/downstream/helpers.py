import functools
import json
import os
from collections import Counter

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dataset import JsonlDatasetSNUH, JsonlInferDatasetSNUH
from vocab import Vocab

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler

from health_multimodal.image.data.transforms import create_chest_xray_transform_for_inference
from health_multimodal.text.model import CXRBertTokenizer

from torchvision.transforms import Compose, RandomAffine, ColorJitter, RandomHorizontalFlip, RandomResizedCrop
from model import *

def get_model(args):
    if args.multimodal_model_type == "vlbert":
        return VLBertClf(args)
    elif args.multimodal_model_type == "flamingo":
        if args.image_model_type == 'vit':
            # return CXRFlamingo_with_ViT(args)
            return CXRFlamingoForErrorDetection(args)
        elif args.image_model_type == 'resnet':
            return CXRFlamingo(args)
    elif args.multimodal_model_type == "att_pool":
        return VLModelClf(args)
    elif args.multimodal_model_type == "coca":
        return CXRCoCa(args)
    elif args.multimodal_model_type == "xvl":
        return XVL(args)
    else:
        raise(ValueError("Unknown model type: %s" % args.model_type))


def get_transforms(args):
    if args.inference_method == None:
        if args.dataset=='mimic-cxr':
            return transforms.Compose(
                [
                    transforms.Grayscale(num_output_channels=3),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
        else:
            return transforms.Compose(
                [
                    transforms.Grayscale(num_output_channels=3),
                    transforms.RandomCrop([2048,2048], pad_if_needed=True),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]
        )


def get_image_augmentations(aug_method):
    color_jitter = ColorJitter(brightness=.2, contrast=.2, saturation=.2, hue=.2 )
    horizontal_flip = RandomHorizontalFlip(p=0.5)
    random_affine = RandomAffine(degrees=30, translate=(0.1,0.1), scale=(0.8,1.2))
    # random_resized_crop = RandomResizedCrop(size=512, scale=(0.5,1.0))

    augmentations = []
    # if aug_method in ["all","rrc"]:
    #     augmentations.append(random_resized_crop)
    if aug_method in ["all","colur"]:
        augmentations.append(color_jitter)
    if aug_method in ["all","hflip"]:
        augmentations.append(horizontal_flip)
    if aug_method in ["all","affine"]:
        augmentations.append(random_affine)
        
   
    return Compose(augmentations)


def get_labels_and_frequencies(path):
    label_freqs = Counter()
    data_labels = [str(json.loads(line)["label"]) for line in open(path)]
    if type(data_labels) == list:
        for label_row in data_labels:
            if label_row == '':
                label_row = ["'Others'"]
            else:
                label_row = label_row.split(', ')

            label_freqs.update(label_row)
    else:
        pass
    return list(label_freqs.keys()), label_freqs


def get_glove_words(path):
    word_list = []
    for line in open(path):
        w, _ = line.split(" ", 1)
        word_list.append(w)
    return word_list


def get_vocab(args):
    vocab = Vocab()
    if args.model in ["bert", "mmbt", "concatbert"]:
        bert_tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)
        vocab.stoi = bert_tokenizer.vocab
        vocab.itos = bert_tokenizer.ids_to_tokens
        vocab.vocab_sz = len(vocab.itos)
    elif args.model == 'clinicalbert':
        bert_tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)
        vocab.stoi = bert_tokenizer.vocab
        vocab.itos = bert_tokenizer.ids_to_tokens
        vocab.vocab_sz = len(vocab.itos)
    elif args.model == 'gatortron':
        bert_tokenizer = AutoTokenizer.from_pretrained(args.bert_model, do_lower_case=False)
        vocab.stoi = bert_tokenizer.vocab
        vocab.itos = bert_tokenizer.convert_ids_to_tokens
        vocab.vocab_sz = bert_tokenizer.vocab_size
    elif args.model == 'roberta':
        bert_tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
        vocab.itos = bert_tokenizer.convert_ids_to_tokens
        vocab.stoi = bert_tokenizer.get_vocab()  # <unk>, <pad>
        vocab.vocab_sz = len(vocab.stoi)  # 250002
    elif args.model == 'cxr-bert':
        # bert_tokenizer = AutoTokenizer.from_pretrained(args.bert_model, trust_remote_code=True)
        bert_tokenizer = CXRBertTokenizer.from_pretrained(args.bert_model, revision='v1.1')
        vocab.stoi = bert_tokenizer.vocab
        vocab.itos = bert_tokenizer.convert_ids_to_tokens
        vocab.vocab_sz = bert_tokenizer.vocab_size
    elif args.model == 'xvl-bert':
        bert_tokenizer = AutoTokenizer.from_pretrained("Medical_X-VL/my_tokenizer/")
        vocab.stoi = bert_tokenizer.vocab
        vocab.itos = bert_tokenizer.convert_ids_to_tokens
        vocab.vocab_sz = bert_tokenizer.vocab_size
    else:
        word_list = get_glove_words(args.glove_path)
        vocab.add(word_list)

    return vocab


def collate_fn(batch, args):
    ## batch: (sentence_findings, segment_findings, sentence_impression, segment_impression, image_tensor, label)
    def get_text_tensors(batch, n_row):
        """
        :param batch:
        :param n_row:  0:findings, 2:impression
        """
        lens = [len(row[n_row]) for row in batch]  
        bsz, max_seq_len = len(batch), max(lens)
        mask_tensor = torch.zeros(bsz, max_seq_len).long()
        text_tensor = torch.zeros(bsz, max_seq_len).long()
        segment_tensor = torch.zeros(bsz, max_seq_len).long()
        for i, (row, length) in enumerate(zip(batch, lens)):
            tokens, segment = row[n_row:n_row+2]
            text_tensor[i, :length] = tokens
            segment_tensor[i, :length] = segment
            mask_tensor[i, :length] = 1
        
        return text_tensor, segment_tensor, mask_tensor
        
    img_tensor = torch.stack([row[4] for row in batch])
    prev_img_tensor = torch.stack([row[6] for row in batch]) if args.use_prev_img else None

    if args.task_type == "multilabel":
        # Multilabel case
        tgt_tensor = torch.stack([row[5] for row in batch])
    elif args.task_type =='classification' or args.task_type =='binary':
        # Mulitclass case
        tgt_tensor = torch.tensor([row[5] for row in batch]).long()
    else:
        # Single Label case
        tgt_tensor = torch.cat([row[5] for row in batch]).long()

    findings_tensors = get_text_tensors(batch, n_row=0)
    prev_findings_tensors = get_text_tensors(batch, n_row=7) if args.use_prev_txt else None
    impression_tensors = None # get_text_tensors(batch, n_row=2)
    
    

    return findings_tensors, impression_tensors, img_tensor, tgt_tensor, prev_img_tensor, prev_findings_tensors


def get_tokenizer(args):

    if args.model in ["bert", "mmbt", "concatbert"]:
        tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True).tokenize
    elif args.model in ["gatortron"]:
        tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=False).tokenize
    elif args.model in ['clinicalbert', 'roberta']:
        tokenizer = AutoTokenizer.from_pretrained(args.bert_model).tokenize
    elif args.model == 'cxr-bert':
        # tokenizer = AutoTokenizer.from_pretrained(args.bert_model, trust_remote_code=True).tokenize
        tokenizer = CXRBertTokenizer.from_pretrained(args.bert_model, revision='v1.1').tokenize
    elif args.model == 'xvl-bert':
        tokenizer = AutoTokenizer.from_pretrained("Medical_X-VL/my_tokenizer/").tokenize
    else:
        str.split

    return tokenizer


def get_dataset(args):
        
    tokenizer = get_tokenizer(args)
    transforms = create_chest_xray_transform_for_inference(
        resize=args.TRANSFORM_RESIZE,
        center_crop_size=args.TRANSFORM_CENTER_CROP_SIZE
    )
    augmentations_img = None if args.inference else get_image_augmentations(args.img_aug) 

# ###############################TEMP
    args.labels = ["0", "1"]
    args.n_classes = len(args.labels)
#################################

    vocab = get_vocab(args)
    args.vocab = vocab
    args.vocab_sz = vocab.vocab_sz

    if args.inference_method == None:
        train = JsonlDatasetSNUH(
            os.path.join(args.data_path, args.Train_dset0_name),
            tokenizer,
            transforms,
            vocab,
            args,
            os.path.join(args.data_path, args.Train_dset1_name),
            set_type='train',
            augmentations=augmentations_img,
        )

        args.train_data_len = len(train)

        val = JsonlDatasetSNUH(
            os.path.join(args.data_path, args.Valid_dset0_name),
            tokenizer,
            transforms,
            vocab,
            args,
            os.path.join(args.data_path, args.Valid_dset1_name),
            set_type='val',
        )

        args.valid_data_len = len(val)

        return train, val  # , test

    elif args.inference_method == 'single': #for inference

        infer = JsonlInferDatasetSNUH(
            tokenizer,
            transforms,
            vocab,
            args,
            input=args.single_input,
            set_type='test',
        )
        return infer
        
    elif args.inference_method == 'batch': #for inference

        infer = JsonlDatasetSNUH(
            os.path.join(args.data_path, args.Valid_dset0_name),
            tokenizer,
            transforms,
            vocab,
            args,
            os.path.join(args.data_path, args.Valid_dset1_name),
            set_type='test',
        )

        args.valid_data_len = len(infer)

        return infer



def get_data_loaders(args, train=None, val=None):
    collate = functools.partial(collate_fn, args=args)

    if args.inference_method == None:
        if args.use_ddp and torch.cuda.device_count()>1:
            print(f"Lets Use DDP with {torch.cuda.device_count()} GPUs!!")
            dist.init_process_group("nccl")
            args.rank = dist.get_rank()
            torch.cuda.set_device(args.rank)
            args.world_size = dist.get_world_size()

            train_sampler = DistributedSampler(
            train,
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=True,
            )   
            val_sampler = DistributedSampler(
            val,
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=False,
            )   

            train_loader = DataLoader(train, batch_size=args.batch_sz, sampler=train_sampler, num_workers=args.n_workers, shuffle=False, collate_fn=collate, pin_memory=True)
            val_loader = DataLoader(val, batch_size=args.batch_sz, sampler=val_sampler, num_workers=args.n_workers, shuffle=False, collate_fn=collate, pin_memory=True)
        else:
            train_loader = DataLoader(
                train,
                batch_size=args.batch_sz,
                shuffle=True,
                num_workers=args.n_workers,
                collate_fn=collate,
            )
            val_loader = DataLoader(
                val,
                batch_size=args.batch_sz,
                shuffle=False,
                num_workers=args.n_workers,
                collate_fn=collate,
            )

        return train_loader, val_loader  # , test

    elif args.inference_method == 'batch': #for inference
        infer_loader = DataLoader(
            val,
            batch_size=args.batch_sz,
            shuffle=False,
            num_workers=args.n_workers,
            collate_fn=collate,
        )

        return infer_loader


