"""
Construct CXR-BERT or BertForMaskedLM, Training and Saving
"""
import os
import tqdm
import wandb
import datetime
import numpy as np

import torch
import torch.nn as nn

from vit_pytorch import ViT
from vit_pytorch.extractor import Extractor

import sys
sys.path.insert(1, '/home/workspace/source/model/report_coca')
from report_coca import ReportCoCa

# from models.cxrbert_origin import CXRBERT


from transformers.optimization import AdamW
from transformers import AutoModel, BertConfig, AlbertConfig, AutoConfig
from torch.cuda.amp import autocast, GradScaler

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler

class CXRBERT_Trainer():
    def __init__(self, args, train_dataloader, test_dataloader=None):
        self.args = args

        cuda_condition = torch.cuda.is_available() and args.with_cuda

        self.device = torch.device("cuda" if cuda_condition else "cpu")
        # self.device = 'cpu'
        print('Current cuda device ', torch.cuda.current_device())  # check

        if args.weight_load:
            config = AutoConfig.from_pretrained(args.pre_trained_model_path)
            model_state_dict = torch.load(os.path.join(args.pre_trained_model_path, 'pytorch_model.bin'))
            self.model = CXRBERT.from_pretrained(args.pre_trained_model_path, state_dict=model_state_dict,
                                                 config=config, args=args).to(self.device)
            print('training restart with mid epoch')
            print(config)
        else:
            if args.bert_model == "albert-base-v2":
                config = AlbertConfig.from_pretrained(args.bert_model)
            elif args.bert_model in ["emilyalsentzer/Bio_ClinicalBERT", "xlm-roberta-base", "xlm-roberta-large"]:
                config = AutoConfig.from_pretrained(args.bert_model)
            elif args.bert_model == "microsoft/BiomedVLP-CXR-BERT-specialized":
                config = AutoConfig.from_pretrained(args.bert_model, trust_remote_code=True)
            elif args.bert_model == "bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12":
                config = AutoConfig.from_pretrained(args.bert_model)
            elif args.bert_model == "bert-small-scratch":
                config = BertConfig.from_pretrained("google/bert_uncased_L-4_H-512_A-8")
            elif args.bert_model == "bert-base-scratch":
                config = BertConfig.from_pretrained("bert-base-uncased")
            else:
                config = BertConfig.from_pretrained(args.bert_model)  # bert-base, small, tiny
            
            self.model = CXRBERT(config, args).to(self.device)


        wandb.watch(self.model)

        # if args.with_cuda and torch.cuda.device_count() > 1:
        #     print("Using %d GPUS for BERT" % torch.cuda.device_count())
        #     self.model = nn.DataParallel(self.model, device_ids=args.cuda_devices)

        if args.with_cuda and torch.cuda.device_count() > 1:

            print("Using %d GPUS for BERT" % torch.cuda.device_count())
            print("torch.cuda.current_device(): ", torch.cuda.current_device())
            self.model = DistributedDataParallel(self.model, device_ids=[torch.cuda.current_device()], find_unused_parameters=True)


        self.train_data = train_dataloader
        self.test_data = test_dataloader

        self.scaler = GradScaler()
        self.optimizer = AdamW(self.model.parameters(), lr=args.lr)

        self.mlm_criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self.itm_criterion = nn.CrossEntropyLoss()

        self.log_freq = args.log_freq
        self.step_cnt = 0

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self, epoch):

        self.model.train()

        train_losses = []
        train_itm_loss = []
        train_mlm_loss = []

        train_data_iter = tqdm.tqdm(enumerate(self.train_data),
                                    desc=f'EP_:{epoch}',
                                    total=len(self.train_data),
                                    bar_format='{l_bar}{r_bar}')
        total_correct = 0
        total_element = 0
        total_mlm_correct = 0
        total_mlm_element = 0

        total_valid_correct = 0
        total_valid_element = 0
        total_mlm_valid_correct = 0
        total_mlm_valid_element = 0

        for i, data in train_data_iter:
            with autocast():
                cls_tok, input_ids, txt_labels, attn_masks, img, segment, is_aligned, sep_tok, itm_prob = data

                cls_tok = cls_tok.to(self.device)
                input_ids = input_ids.to(self.device)
                txt_labels = txt_labels.to(self.device)
                attn_masks = attn_masks.to(self.device)
                img = img.to(self.device)
                segment = segment.to(self.device)
                is_aligned = is_aligned.to(self.device)
                sep_tok = sep_tok.to(self.device)

                # if self.args.itm_task:
                mlm_output = self.model(cls_tok, input_ids, attn_masks, segment, img, sep_tok)
                # else:
                #     mlm_output = self.model(cls_tok, input_ids, attn_masks, segment, sep_tok)


                if self.args.mlm_task and self.args.itm_task == False:
                    mlm_loss = self.mlm_criterion(mlm_output.transpose(1, 2), txt_labels)
                    loss = mlm_loss
                    # print('only mlm_loss')

                if self.args.itm_task and self.args.mlm_task == False:
                    itm_loss = self.itm_criterion(itm_output, is_aligned)
                    loss = itm_loss
                    print('only itm_loss')

                if self.args.mlm_task and self.args.itm_task:

                    mlm_loss = self.mlm_criterion(mlm_output.transpose(1, 2), txt_labels)
                    train_mlm_loss.append(mlm_loss.item())

                    itm_loss = self.itm_criterion(itm_output, is_aligned)
                    train_itm_loss.append(itm_loss.item())

                    loss = itm_loss + mlm_loss

                if i % 100 == 0:
                    print(f"step {i}, train loss: {loss.item()}")
                train_losses.append(loss.item())    

                loss = loss / self.args.gradient_accumulation_steps
            
            # Accumulates scaled gradients.   
            self.scaler.scale(loss).backward()

            if (i + 1) % self.args.gradient_accumulation_steps == 0:
                # may unscale_ here if desired (e.g., to allow clipping unscaled gradients)
                print('reached to grad acc step')
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            if self.args.itm_task:
                correct = itm_output.argmax(dim=-1).eq(is_aligned).sum().item()
                total_correct += correct
                total_element += is_aligned.nelement()

            if self.args.mlm_task:
                eq = (mlm_output.argmax(dim=-1).eq(txt_labels)).cpu().numpy()
                txt_labels_np = txt_labels.cpu().numpy()
                for bs, label in enumerate(txt_labels_np):
                    index = np.where(label == -100)[0]
                    f_label = np.delete(label, index)
                    f_eq = np.delete(eq[bs], index)
                    total_mlm_correct += f_eq.sum()
                    total_mlm_element += len(f_label)

        print("avg loss per epoch", np.mean(train_losses))
        if self.args.mlm_task and self.args.itm_task:
            print("avg itm acc per epoch", round(total_correct / total_element * 100, 3))
            wandb.log({
                "avg_loss": np.mean(train_losses),
                "avg_mlm_loss": np.mean(train_mlm_loss),
                "avg_itm_loss": np.mean(train_itm_loss),
                "itm_acc": total_correct / total_element * 100,
                "mlm_acc": total_mlm_correct / total_mlm_element * 100
            }, step=epoch)

        if self.args.itm_task and self.args.mlm_task == False:
            print("avg itm acc per epoch", round(total_correct / total_element * 100, 3))

            wandb.log({
                "avg_loss": np.mean(train_losses),
                "itm_epoch_acc": total_correct / total_element * 100
            }, step=epoch)

        if self.args.mlm_task and self.args.itm_task == False:
            wandb.log({
                "avg_loss": np.mean(train_losses),
                "mlm_epoch_acc": total_mlm_correct / total_mlm_element * 100
            }, step=epoch)

        test_data_iter = tqdm.tqdm(enumerate(self.test_data),
                                   desc=f'EP_:{epoch}',
                                   total=len(self.test_data),
                                   bar_format='{l_bar}{r_bar}')
        self.model.eval()
        with torch.no_grad():
            eval_losses = []
            eval_mlm_loss = []
            eval_itm_loss = []
            for i, data in test_data_iter:
                cls_tok, input_ids, txt_labels, attn_masks, img, segment, is_aligned, sep_tok, itm_prob = data
                # cls_tok, input_ids, txt_labels, attn_masks, img, segment, is_aligned, sep_tok = data

                cls_tok = cls_tok.to(self.device)
                input_ids = input_ids.to(self.device)
                txt_labels = txt_labels.to(self.device)
                attn_masks = attn_masks.to(self.device)
                img = img.to(self.device)
                segment = segment.to(self.device)
                is_aligned = is_aligned.to(self.device)
                sep_tok = sep_tok.to(self.device)

                mlm_output = self.model(cls_tok, input_ids, attn_masks, segment, img, sep_tok)

                if self.args.mlm_task and self.args.itm_task == False:
                    valid_mlm_loss = self.mlm_criterion(mlm_output.transpose(1, 2), txt_labels)
                    valid_loss = valid_mlm_loss
                    # print('only valid mlm loss')

                if self.args.itm_task and self.args.mlm_task == False:
                    valid_itm_loss = self.itm_criterion(itm_output, is_aligned)
                    valid_loss = valid_itm_loss
                    print('only valid itm loss')

                if self.args.mlm_task and self.args.itm_task:
                    # TODO: weight each loss, mlm > itm
                    valid_mlm_loss = self.mlm_criterion(mlm_output.transpose(1, 2), txt_labels)
                    valid_itm_loss = self.itm_criterion(itm_output, is_aligned)
                    eval_mlm_loss.append(valid_mlm_loss.item())
                    eval_itm_loss.append(valid_itm_loss.item())

                    valid_loss = valid_itm_loss + valid_mlm_loss

                eval_losses.append(valid_loss.item())

                if self.args.itm_task:
                    valid_correct = itm_output.argmax(dim=-1).eq(is_aligned).sum().item()
                    total_valid_correct += valid_correct
                    total_valid_element += is_aligned.nelement()

                if self.args.mlm_task:
                    eq = (mlm_output.argmax(dim=-1).eq(txt_labels)).cpu().numpy()
                    txt_labels_np = txt_labels.cpu().numpy()
                    for bs, label in enumerate(txt_labels_np):
                        index = np.where(label == -100)[0]
                        f_label = np.delete(label, index)
                        f_eq = np.delete(eq[bs], index)
                        total_mlm_valid_correct += f_eq.sum()
                        total_mlm_valid_element += len(f_label)

            print("avg loss in testset", np.mean(eval_losses))

            if self.args.mlm_task and self.args.itm_task:
                print("avg itm acc in testset", round(total_valid_correct / total_valid_element * 100, 3))
                wandb.log({
                    "eval_avg_loss": np.mean(eval_losses),
                    "eval_mlm_loss": np.mean(eval_mlm_loss),
                    "eval_itm_loss": np.mean(eval_itm_loss),
                    "eval_itm_acc": total_valid_correct / total_valid_element * 100,
                    "eval_mlm_acc": total_mlm_valid_correct / total_mlm_valid_element * 100
                }, step=epoch)

            if self.args.itm_task and self.args.mlm_task == False:
                print("avg itm acc in testset", round(total_valid_correct / total_valid_element * 100, 3))

                wandb.log({
                    "eval_avg_loss": np.mean(eval_losses),
                    "eval_itm_epoch_acc": total_valid_correct / total_valid_element * 100
                }, step=epoch)

            if self.args.mlm_task and self.args.itm_task == False:
                wandb.log({
                    "eval_avg_loss": np.mean(eval_losses),
                    "eval_mlm_epoch_acc": total_mlm_valid_correct / total_mlm_valid_element * 100
                }, step=epoch)

    def save(self, epoch, file_path):
        save_path_per_ep = os.path.join(file_path, str(epoch))
        if not os.path.exists(save_path_per_ep):
            os.mkdir(save_path_per_ep)
            os.chmod(save_path_per_ep, 0o777)

        if torch.cuda.device_count() > 1:
            if self.args.rank == 0:
                self.model.module.save_pretrained(save_path_per_ep)
                print(f'Multi_EP: {epoch} Model saved on {save_path_per_ep}')
                
                os.chmod(save_path_per_ep + '/pytorch_model.bin', 0o777)
        else:
            self.model.save_pretrained(save_path_per_ep)
            print(f'Single_EP: {epoch} Model saved on {save_path_per_ep}')
        
            os.chmod(save_path_per_ep + '/pytorch_model.bin', 0o777)


class VLCXR_Trainer():
    def __init__(self, args, train_dataloader, test_dataloader=None):
        self.args = args

        cuda_condition = torch.cuda.is_available() and args.with_cuda

        self.device = torch.device("cuda" if cuda_condition else "cpu")
        print('Current cuda device ', torch.cuda.current_device())  # check

        if args.weight_load: ## TODO!!!!!!!!!!!!!
            config = AutoConfig.from_pretrained(args.pre_trained_model_path)
            model_state_dict = torch.load(os.path.join(args.pre_trained_model_path, 'pytorch_model.bin'))
            self.model = CXRBERT.from_pretrained(args.pre_trained_model_path, state_dict=model_state_dict,
                                                 config=config, args=args).to(self.device)
            print('training restart with mid epoch')
            print(config)
        else:
            print("Initializing model weights with random values")

            vit = ViT(
                image_size = args.img_size,
                patch_size = args.patch_size,
                num_classes = 2, ## not used
                dim = args.img_embed_sz,
                depth = args.depth_img_enc,
                heads = args.num_head_img_enc,
                mlp_dim = args.img_mlp_hidden_sz
            )

            vit = Extractor(vit, return_embeddings_only = True, detach = False) 

            self.model = ReportCoCa(
                dim = args.model_dim,                        # model dimension
                max_seq_len = args.max_seq_len,                # TODO: max_seq_len should be seperated from dim
                img_encoder = vit,                           # vision transformer - image encoder, returning image embeddings as (batch, seq, dim)
                image_dim = args.img_embed_sz,                # image embedding dimension, if not the same as model dimensions
                num_tokens = args.vocab_size,                  # number of text tokens
                unimodal_depth = args.unimodal_depth,               # depth of the unimodal transformer
                multimodal_depth = args.multimodal_depth,          # depth of the multimodal transformer
                num_img_queries = args.num_img_queries,           # num img_queries
                dim_head = args.dim_head,                         # dimension per attention head
                heads = args.num_head_dec,                        # number of attention heads
                caption_loss_weight = args.captioin_loss_weight,          # weight on the autoregressive caption loss
                contrastive_loss_weight = args.contrastive_loss_weight,  # weight on the contrastive loss between image and text CLS embeddings
            ).to(self.device)

        wandb.watch(self.model)

        if args.with_cuda and torch.cuda.device_count() > 1:
            print("Using %d GPUS for BERT" % torch.cuda.device_count())
            print("torch.cuda.current_device(): ", torch.cuda.current_device())
            self.model = DistributedDataParallel(self.model, device_ids=[torch.cuda.current_device()], find_unused_parameters=True)

        self.train_data = train_dataloader
        self.test_data = test_dataloader

        self.scaler = GradScaler()
        self.optimizer = AdamW(self.model.parameters(), lr=args.lr)

        self.log_freq = args.log_freq
        self.step_cnt = 0

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self, epoch):

        self.model.train()

        train_losses = []
        train_itm_loss = []
        train_mlm_loss = []

        train_data_iter = tqdm.tqdm(enumerate(self.train_data),
                                    desc=f'EP_:{epoch}',
                                    total=len(self.train_data),
                                    bar_format='{l_bar}{r_bar}')
        total_correct = 0
        total_element = 0
        total_mlm_correct = 0
        total_mlm_element = 0

        total_valid_correct = 0
        total_valid_element = 0
        total_mlm_valid_correct = 0
        total_mlm_valid_element = 0

        for i, data in train_data_iter:
            with autocast():
                cls_tok, input_ids, txt_labels, attn_masks, img, segment, is_aligned, sep_tok, itm_prob = data

                cls_tok = cls_tok.to(self.device)
                input_ids = input_ids.to(self.device)
                txt_labels = txt_labels.to(self.device)
                attn_masks = attn_masks.to(self.device)
                img = img.to(self.device)
                segment = segment.to(self.device)
                is_aligned = is_aligned.to(self.device)
                sep_tok = sep_tok.to(self.device)

                # if self.args.itm_task:
                mlm_output = self.model(cls_tok, input_ids, attn_masks, segment, img, sep_tok)
                # else:
                #     mlm_output = self.model(cls_tok, input_ids, attn_masks, segment, sep_tok)


                if self.args.mlm_task and self.args.itm_task == False:
                    mlm_loss = self.mlm_criterion(mlm_output.transpose(1, 2), txt_labels)
                    loss = mlm_loss
                    # print('only mlm_loss')

                if self.args.itm_task and self.args.mlm_task == False:
                    itm_loss = self.itm_criterion(itm_output, is_aligned)
                    loss = itm_loss
                    print('only itm_loss')

                if self.args.mlm_task and self.args.itm_task:

                    mlm_loss = self.mlm_criterion(mlm_output.transpose(1, 2), txt_labels)
                    train_mlm_loss.append(mlm_loss.item())

                    itm_loss = self.itm_criterion(itm_output, is_aligned)
                    train_itm_loss.append(itm_loss.item())

                    loss = itm_loss + mlm_loss

                if i % 100 == 0:
                    print(f"step {i}, train loss: {loss.item()}")
                train_losses.append(loss.item())    

                loss = loss / self.args.gradient_accumulation_steps
            
            # Accumulates scaled gradients.   
            self.scaler.scale(loss).backward()

            if (i + 1) % self.args.gradient_accumulation_steps == 0:
                # may unscale_ here if desired (e.g., to allow clipping unscaled gradients)
                print('reached to grad acc step')
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            if self.args.itm_task:
                correct = itm_output.argmax(dim=-1).eq(is_aligned).sum().item()
                total_correct += correct
                total_element += is_aligned.nelement()

            if self.args.mlm_task:
                eq = (mlm_output.argmax(dim=-1).eq(txt_labels)).cpu().numpy()
                txt_labels_np = txt_labels.cpu().numpy()
                for bs, label in enumerate(txt_labels_np):
                    index = np.where(label == -100)[0]
                    f_label = np.delete(label, index)
                    f_eq = np.delete(eq[bs], index)
                    total_mlm_correct += f_eq.sum()
                    total_mlm_element += len(f_label)

        print("avg loss per epoch", np.mean(train_losses))
        if self.args.mlm_task and self.args.itm_task:
            print("avg itm acc per epoch", round(total_correct / total_element * 100, 3))
            wandb.log({
                "avg_loss": np.mean(train_losses),
                "avg_mlm_loss": np.mean(train_mlm_loss),
                "avg_itm_loss": np.mean(train_itm_loss),
                "itm_acc": total_correct / total_element * 100,
                "mlm_acc": total_mlm_correct / total_mlm_element * 100
            }, step=epoch)

        if self.args.itm_task and self.args.mlm_task == False:
            print("avg itm acc per epoch", round(total_correct / total_element * 100, 3))

            wandb.log({
                "avg_loss": np.mean(train_losses),
                "itm_epoch_acc": total_correct / total_element * 100
            }, step=epoch)

        if self.args.mlm_task and self.args.itm_task == False:
            wandb.log({
                "avg_loss": np.mean(train_losses),
                "mlm_epoch_acc": total_mlm_correct / total_mlm_element * 100
            }, step=epoch)

        test_data_iter = tqdm.tqdm(enumerate(self.test_data),
                                   desc=f'EP_:{epoch}',
                                   total=len(self.test_data),
                                   bar_format='{l_bar}{r_bar}')
        self.model.eval()
        with torch.no_grad():
            eval_losses = []
            eval_mlm_loss = []
            eval_itm_loss = []
            for i, data in test_data_iter:
                cls_tok, input_ids, txt_labels, attn_masks, img, segment, is_aligned, sep_tok, itm_prob = data
                # cls_tok, input_ids, txt_labels, attn_masks, img, segment, is_aligned, sep_tok = data

                cls_tok = cls_tok.to(self.device)
                input_ids = input_ids.to(self.device)
                txt_labels = txt_labels.to(self.device)
                attn_masks = attn_masks.to(self.device)
                img = img.to(self.device)
                segment = segment.to(self.device)
                is_aligned = is_aligned.to(self.device)
                sep_tok = sep_tok.to(self.device)

                mlm_output = self.model(cls_tok, input_ids, attn_masks, segment, img, sep_tok)

                if self.args.mlm_task and self.args.itm_task == False:
                    valid_mlm_loss = self.mlm_criterion(mlm_output.transpose(1, 2), txt_labels)
                    valid_loss = valid_mlm_loss
                    # print('only valid mlm loss')

                if self.args.itm_task and self.args.mlm_task == False:
                    valid_itm_loss = self.itm_criterion(itm_output, is_aligned)
                    valid_loss = valid_itm_loss
                    print('only valid itm loss')

                if self.args.mlm_task and self.args.itm_task:
                    # TODO: weight each loss, mlm > itm
                    valid_mlm_loss = self.mlm_criterion(mlm_output.transpose(1, 2), txt_labels)
                    valid_itm_loss = self.itm_criterion(itm_output, is_aligned)
                    eval_mlm_loss.append(valid_mlm_loss.item())
                    eval_itm_loss.append(valid_itm_loss.item())

                    valid_loss = valid_itm_loss + valid_mlm_loss

                eval_losses.append(valid_loss.item())

                if self.args.itm_task:
                    valid_correct = itm_output.argmax(dim=-1).eq(is_aligned).sum().item()
                    total_valid_correct += valid_correct
                    total_valid_element += is_aligned.nelement()

                if self.args.mlm_task:
                    eq = (mlm_output.argmax(dim=-1).eq(txt_labels)).cpu().numpy()
                    txt_labels_np = txt_labels.cpu().numpy()
                    for bs, label in enumerate(txt_labels_np):
                        index = np.where(label == -100)[0]
                        f_label = np.delete(label, index)
                        f_eq = np.delete(eq[bs], index)
                        total_mlm_valid_correct += f_eq.sum()
                        total_mlm_valid_element += len(f_label)

            print("avg loss in testset", np.mean(eval_losses))

            if self.args.mlm_task and self.args.itm_task:
                print("avg itm acc in testset", round(total_valid_correct / total_valid_element * 100, 3))
                wandb.log({
                    "eval_avg_loss": np.mean(eval_losses),
                    "eval_mlm_loss": np.mean(eval_mlm_loss),
                    "eval_itm_loss": np.mean(eval_itm_loss),
                    "eval_itm_acc": total_valid_correct / total_valid_element * 100,
                    "eval_mlm_acc": total_mlm_valid_correct / total_mlm_valid_element * 100
                }, step=epoch)

            if self.args.itm_task and self.args.mlm_task == False:
                print("avg itm acc in testset", round(total_valid_correct / total_valid_element * 100, 3))

                wandb.log({
                    "eval_avg_loss": np.mean(eval_losses),
                    "eval_itm_epoch_acc": total_valid_correct / total_valid_element * 100
                }, step=epoch)

            if self.args.mlm_task and self.args.itm_task == False:
                wandb.log({
                    "eval_avg_loss": np.mean(eval_losses),
                    "eval_mlm_epoch_acc": total_mlm_valid_correct / total_mlm_valid_element * 100
                }, step=epoch)

    def save(self, epoch, file_path):
        save_path_per_ep = os.path.join(file_path, str(epoch))
        if not os.path.exists(save_path_per_ep):
            os.mkdir(save_path_per_ep)
            os.chmod(save_path_per_ep, 0o777)

        if torch.cuda.device_count() > 1:
            if self.args.rank == 0:
                self.model.module.save_pretrained(save_path_per_ep)
                print(f'Multi_EP: {epoch} Model saved on {save_path_per_ep}')
                
                os.chmod(save_path_per_ep + '/pytorch_model.bin', 0o777)
        else:
            self.model.save_pretrained(save_path_per_ep)
            print(f'Single_EP: {epoch} Model saved on {save_path_per_ep}')
        
            os.chmod(save_path_per_ep + '/pytorch_model.bin', 0o777)
