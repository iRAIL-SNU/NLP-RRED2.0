"""
Construct CXR-BERT or BertForMaskedLM, Training and Saving
"""
import os
import tqdm
import wandb
import numpy as np

import torch
import torch.nn as nn

from helpers import get_language_model
from loss_fn import RadiologySelectionLoss

from torch.optim import AdamW
from transformers import AutoModel, BertConfig, AlbertConfig, AutoConfig
from torch.cuda.amp import autocast, GradScaler

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler
import random


class VLCXR_Trainer():
    def __init__(self, args, train_dataloader, test_dataloader=None, vocab=None, wandb_run=None):
        self.args = args
        self.device = args.device
        self.vocab = vocab

        if args.weight_load: ## TODO!!!!!!!!!!!!!
            config = AutoConfig.from_pretrained(args.pre_trained_model_path)
            model_state_dict = torch.load(os.path.join(args.pre_trained_model_path, 'pytorch_model.bin'))
            self.model = CXRBERT.from_pretrained(args.pre_trained_model_path, state_dict=model_state_dict,
                                                 config=config, args=args).to(self.device)
            print('training restart with mid epoch')
            print(config)
        else:
            print(f"Initializing model weights with {args.model}")

            self.model = get_language_model(args).to(self.device)


        if args.use_ddp:
            print("Using %d GPUS" % torch.cuda.device_count())
            print("torch.cuda.current_device(): ", torch.cuda.current_device())
            self.model = DistributedDataParallel(self.model, device_ids=[torch.cuda.current_device()], find_unused_parameters=True)

        self.train_data = train_dataloader
        self.test_data = test_dataloader

        self.mlm_criterion = nn.CrossEntropyLoss(ignore_index=0) 
        self.itm_criterion = nn.CrossEntropyLoss()
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.softmax = nn.Softmax(dim=-1)

        self.scaler = GradScaler()
        self.optimizer = AdamW(self.model.parameters(), lr=args.lr)

        self.log_freq = args.log_freq
        self.epoch = 0
        self.step_cnt = 0
        
        self.wandb_run = wandb_run
        wandb.watch(self.model, self.mlm_criterion,log="all")
        self.text_table = wandb.Table(columns=["Epoch", "Step", "True", "Mask", "Pred"])

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self, epoch):
        self.epoch = epoch
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
                input_ids, txt_labels, attn_masks, segment, original_ids = data

                input_ids = input_ids.to(self.device)
                original_ids = original_ids.to(self.device) 
                txt_labels = txt_labels.to(self.device)
                attn_masks = attn_masks.to(self.device)
                segment = segment.to(self.device)

                USE_SET = False
                if epoch == 0 and self.step_cnt >= self.args.use_SET:
                    USE_SET = True

                if USE_SET: # Stage 1: self-questioning
                    input_ids, correct_idx, txt_labels = self.self_questioning(input_ids, original_ids, attn_masks,segment)
                    
                mlm_output = self.model(input_ids, attn_masks, segment)['logits']
                
                if USE_SET: # Stage 2: self-evolution training
                    loss = self.self_evolution_training(original_ids, attn_masks,segment, mlm_output, correct_idx, txt_labels)
                else:
                    loss = self.mlm_criterion(mlm_output.transpose(1, 2), txt_labels)

                if i % self.args.log_freq == 0:
                    print(f"step {self.step_cnt}, train loss: {loss.item()}")
                    self.print_examples(input_ids, original_ids, mlm_output, n=5)
                    self.wandb_run.log({"training_samples" : self.text_table})
                    
                    
                train_losses.append(loss.item())    

                loss = loss / self.args.gradient_accumulation_steps
            
            # Accumulates scaled gradients.   
            self.scaler.scale(loss).backward()

            if (i + 1) % self.args.gradient_accumulation_steps == 0:
                # may unscale_ here if desired (e.g., to allow clipping unscaled gradients)
                # print('reached to grad acc step')
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.step_cnt += 1
                

            eq = (mlm_output.argmax(dim=-1).eq(txt_labels)).cpu().numpy()
            txt_labels_np = txt_labels.cpu().numpy()
            for bs, label in enumerate(txt_labels_np):
                index = np.where(label == -100)[0]
                f_label = np.delete(label, index)
                f_eq = np.delete(eq[bs], index)
                total_mlm_correct += f_eq.sum()
                total_mlm_element += len(f_label)

        print("avg loss per epoch", np.mean(train_losses))

        wandb.log({
            "avg_loss": np.mean(train_losses),
            "mlm_epoch_acc": total_mlm_correct / total_mlm_element * 100,
            "epoch": epoch
        }, step=self.step_cnt)
        
        
        test_data_iter = tqdm.tqdm(enumerate(self.test_data),
                                   desc=f'EP_:{epoch}',
                                   total=len(self.test_data),
                                   bar_format='{l_bar}{r_bar}')
        self.model.eval()
        with torch.no_grad():
            eval_losses = []
            eval_mlm_loss = []
            for i, data in test_data_iter:
                input_ids, txt_labels, attn_masks, segment, original_ids = data

                input_ids = input_ids.to(self.device)
                original_ids = original_ids.to(self.device) 
                txt_labels = txt_labels.to(self.device)
                attn_masks = attn_masks.to(self.device)
                segment = segment.to(self.device)

                if USE_SET: # Stage 1: self-questioning
                    input_ids, correct_idx, txt_labels = self.self_questioning(input_ids, original_ids, attn_masks,segment)
                    
                mlm_output = self.model(input_ids, attn_masks, segment)['logits']
                
                if USE_SET: # Stage 2: self-evolution training
                    valid_loss = self.self_evolution_training(original_ids, attn_masks,segment, mlm_output, correct_idx, txt_labels)
                else:
                    valid_loss = self.mlm_criterion(mlm_output.transpose(1, 2), txt_labels)

                eval_losses.append(valid_loss.item())
                
                eq = (mlm_output.argmax(dim=-1).eq(txt_labels)).cpu().numpy()
                txt_labels_np = txt_labels.cpu().numpy()
                for bs, label in enumerate(txt_labels_np):
                    index = np.where(label == -100)[0]
                    f_label = np.delete(label, index)
                    f_eq = np.delete(eq[bs], index)
                    total_mlm_valid_correct += f_eq.sum()
                    total_mlm_valid_element += len(f_label)

            print("avg loss in testset", np.mean(eval_losses))
            self.print_examples(input_ids, original_ids, mlm_output, n=5)
            
            wandb.log({
                "eval_avg_loss": np.mean(eval_losses),
                "eval_mlm_epoch_acc": total_mlm_valid_correct / total_mlm_valid_element * 100,
                "epoch": epoch
        }, step=self.step_cnt)

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

    # def random_word(self, original_ids, correct_idx, masking_prob=0.8):
    def random_word(self, original_ids, input_ids, correct_idx, masking_prob=0.8):
        output_label = []
        tokens = input_ids.clone().detach()
        
        for i, (token, correct) in enumerate(zip(original_ids, correct_idx)):
            prob = random.random()
            if prob < masking_prob and token != 0 and correct==False:
                prob /= masking_prob

                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[i] = self.vocab.stoi["[MASK]"]

                # 10% randomly change token to random token
                elif prob < 0.9:
                    tokens[i] = random.randrange(self.vocab.vocab_sz)

                output_label.append(token)
            else:
                # tokens[i] = token
                output_label.append(-100)  # 0

        if all(o == -100 for o in output_label):  # 0
            # at least one mask
            output_label[0] = original_ids[0]
            tokens[0] = self.vocab.stoi["[MASK]"]

        return tokens, torch.tensor(output_label).to(self.device)

    def self_questioning(self,input_ids, original_ids, attn_masks,segment):
        self.model.eval()
        with torch.no_grad(): 
            # lm_predict = self.model(original_ids, attn_masks, segment)['logits'].argmax(dim=-1) * attn_masks
            lm_predict = self.model(input_ids, attn_masks, segment)['logits'].argmax(dim=-1) * attn_masks
            correct_idx = lm_predict.eq(original_ids)

            # re-Masking the uncorrected tokens
            # new_input_ids_and_new_txt_labels = [self.random_word(x,y) for x,y in zip(original_ids, correct_idx)]
            new_input_ids_and_new_txt_labels = [self.random_word(x,y,z) for x,y,z in zip(original_ids, input_ids, correct_idx)]
            new_input_ids = torch.stack([x[0] for x in new_input_ids_and_new_txt_labels])            
            new_txt_labels = torch.stack([x[1] for x in new_input_ids_and_new_txt_labels])            
            # new_input_ids = (original_ids * correct_idx) + (self.vocab.stoi["[MASK]"] * ~correct_idx)                     
                    
        self.model.train()
        
        return new_input_ids, correct_idx, new_txt_labels

    def self_evolution_training(self, original_ids, attn_masks,segment, mlm_output, correct_idx, txt_labels):
        # with torch.no_grad():
        # lm_original_logits = self.model(original_ids, attn_masks, segment)['logits']
        
        # one_hot_logits = torch.zeros_like(lm_original_logits)
        # for b, batch_ids in enumerate(original_ids):
        #     for j, idx in enumerate(batch_ids[batch_ids!=0]):
        #         one_hot_logits[b][j][idx] = 1
        
        # smoothed_logits = self.args.SET_alpha_for_onehot * one_hot_logits + (1-self.args.SET_alpha_for_onehot) * lm_original_logits
        #### 로스 계산 문제 있음
        # loss = -torch.sum(self.log_softmax(mlm_output[~correct_idx]) * self.softmax(smoothed_logits[~correct_idx])) / mlm_output[~correct_idx].shape[0] 
        # loss = self.mlm_criterion(mlm_output[~correct_idx], smoothed_logits[~correct_idx].argmax(dim=-1))
        loss = self.mlm_criterion(mlm_output.transpose(1,2), original_ids)
        # loss.requires_grad=True
        return loss


    def print_examples(self, input_ids,original_ids, mlm_output, n=3):
        for i in range(n):
            
            true_text = " ".join(self.vocab.itos(original_ids[i][input_ids[i]!=0]))
            mask_text = " ".join(self.vocab.itos(input_ids[i][input_ids[i]!=0]))
            pred_text = " ".join(self.vocab.itos(mlm_output[i].argmax(dim=-1)[input_ids[i]!=0]))
            
            print(f"[[SAMPLE: {i}]]")
            print("True: ", true_text)
            print("--------------------------------------------------")
            print("Mask: ", mask_text)
            print("--------------------------------------------------")
            print("Pred: ", pred_text)
            print("====================================================================================================")
            
            self.text_table.add_data(self.epoch, self.step_cnt, true_text, mask_text, pred_text)
        return 
    

class VLCXR_biovil_Trainer():
    def __init__(self, args, train_dataloader, test_dataloader=None, vocab=None, wandb_run=None):
        self.args = args
        self.device = args.device
        self.vocab = vocab

        if args.weight_load: ## TODO!!!!!!!!!!!!!
            config = AutoConfig.from_pretrained(args.pre_trained_model_path)
            model_state_dict = torch.load(os.path.join(args.pre_trained_model_path, 'pytorch_model.bin'))
            self.model = CXRBERT.from_pretrained(args.pre_trained_model_path, state_dict=model_state_dict,
                                                 config=config, args=args).to(self.device)
            print('training restart with mid epoch')
            print(config)
        else:
            print(f"Initializing model weights with {args.model}")

            self.model = get_language_model(args).to(self.device)


        if args.use_ddp:
            print("Using %d GPUS" % torch.cuda.device_count())
            print("torch.cuda.current_device(): ", torch.cuda.current_device())
            self.model = DistributedDataParallel(self.model, device_ids=[torch.cuda.current_device()], find_unused_parameters=True)

        self.train_data = train_dataloader
        self.test_data = test_dataloader

        self.mlm_criterion = nn.CrossEntropyLoss(ignore_index=0) 
        self.rsm_criterion = RadiologySelectionLoss(t=0.5)
        self.itm_criterion = nn.CrossEntropyLoss()
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.softmax = nn.Softmax(dim=-1)

        self.scaler = GradScaler()
        self.optimizer = AdamW(self.model.parameters(), lr=args.lr)

        self.log_freq = args.log_freq
        self.epoch = 0
        self.step_cnt = 0
        
        self.wandb_run = wandb_run
        wandb.watch(self.model, self.mlm_criterion,log="all")
        self.text_table = wandb.Table(columns=["Epoch", "Step", "True", "Mask", "Pred"])

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self, epoch):
        self.epoch = epoch
        self.model.train()

        train_losses = []
        train_rsm_loss = []
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
                projected_findings, mlm_loss_findings, total_mlm_correct, total_mlm_element, input_ids_f, original_ids_f, mlm_output_f = self.model_forward(data['findings_tensors'], total_mlm_correct, total_mlm_element)
                projected_impression, mlm_loss_impression, total_mlm_correct, total_mlm_element, input_ids_i, original_ids_i, mlm_output_i = self.model_forward(data['impression_tensors'], total_mlm_correct, total_mlm_element)

                mlm_loss = mlm_loss_findings + mlm_loss_impression
                train_mlm_loss.append(mlm_loss.item())    
                
                rsm_loss = self.rsm_criterion(projected_findings, projected_impression, data['chexpert_label']) ## 레이블 확인해서 레이블 같은건 구분할 수 있게 하자? 레이블 같아도 다른정보 있을 수 있으니 no findings만 고려할수있게 하자.
                train_rsm_loss.append(rsm_loss.item())

                total_loss = 0.1 * mlm_loss + rsm_loss
                train_losses.append(total_loss)
                
                # Accumulates scaled gradients.   
                self.scaler.scale(total_loss/self.args.gradient_accumulation_steps).backward()

            if (i + 1) % self.args.gradient_accumulation_steps == 0:              
                # may unscale_ here if desired (e.g., to allow clipping unscaled gradients)
                print('reached to grad acc step')
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                
                if self.step_cnt % self.args.log_freq == 0:
                    print(f"step {self.step_cnt}, train_loss: {total_loss.item()},train MLM loss: {mlm_loss.item()}, train RSM loss: {rsm_loss.item()}")
                    self.print_examples(input_ids_f, original_ids_f, mlm_output_f, n=5)
                    self.print_examples(input_ids_i, original_ids_i, mlm_output_i, n=5)
                    self.wandb_run.log({"training_samples" : self.text_table})
                self.step_cnt += 1

        print("avg loss per epoch", np.mean(train_losses))

        wandb.log({
            "avg_loss": np.mean(train_losses),
            "avg_mlm_loss": np.mean(train_mlm_loss),
            "avg_rsm_loss": np.mean(train_rsm_loss),
            "mlm_epoch_acc": total_mlm_correct / total_mlm_element * 100,
            "epoch": epoch
        }, step=self.step_cnt)
        
        
        test_data_iter = tqdm.tqdm(enumerate(self.test_data),
                                   desc=f'EP_:{epoch}',
                                   total=len(self.test_data),
                                   bar_format='{l_bar}{r_bar}')
        self.model.eval()
        with torch.no_grad():
            eval_losses = []
            eval_mlm_loss = []
            eval_rsm_loss = []
            for i, data in test_data_iter:
                
                projected_findings, mlm_loss_findings, total_mlm_valid_correct, total_mlm_valid_element, input_ids_f, original_ids_f, mlm_output_f = self.model_forward(data['findings_tensors'], total_mlm_valid_correct, total_mlm_valid_element)
                projected_impression, mlm_loss_impression, total_mlm_valid_correct, total_mlm_valid_element, input_ids_i, original_ids_i, mlm_output_i = self.model_forward(data['impression_tensors'], total_mlm_valid_correct, total_mlm_valid_element)

                mlm_loss = mlm_loss_findings + mlm_loss_impression
                eval_mlm_loss.append(mlm_loss.item())    
                
                rsm_loss = self.rsm_criterion(projected_findings, projected_impression)
                eval_rsm_loss.append(rsm_loss.item())

                total_loss = 0.1 * mlm_loss + rsm_loss
                eval_losses.append(total_loss.item())
                
            print("avg loss in testset", np.mean(eval_losses))
            print(f"avg MLM loss: {np.mean(eval_mlm_loss)}, avg RSM loss: {np.mean(eval_rsm_loss)}")
            self.print_examples(input_ids_f, original_ids_f, mlm_output_f, n=5)
            self.print_examples(input_ids_i, original_ids_i, mlm_output_i, n=5)

            
            wandb.log({
                "eval_avg_total_loss": np.mean(eval_losses),
                "eval_avg_mlm_loss": np.mean(eval_mlm_loss),
                "eval_avg_rsm_loss": np.mean(eval_rsm_loss),
                "eval_mlm_epoch_acc": total_mlm_valid_correct / total_mlm_valid_element * 100,
                "epoch": epoch
        }, step=self.step_cnt)

    def model_forward(self, data, total_mlm_correct=None, total_mlm_element=None, return_cls_only=False):
        
        input_ids, txt_labels, attn_masks, segment, original_ids = data

        input_ids = input_ids.to(self.device)
        original_ids = original_ids.to(self.device) 
        txt_labels = txt_labels.to(self.device)
        attn_masks = attn_masks.to(self.device)
        segment = segment.to(self.device)

        USE_SET = False
        if self.step_cnt >= self.args.use_SET:
            USE_SET = True

        if USE_SET: # Stage 1: self-questioning
            input_ids, correct_idx, txt_labels = self.self_questioning(input_ids, original_ids, attn_masks,segment)
            
        model_output = self.model(input_ids, attn_masks, segment, output_cls_projected_embedding=True)
        if return_cls_only:
            return model_output['cls_projected_embedding']
        mlm_output = model_output['logits']
        
        if USE_SET: # Stage 2: self-evolution training
            mlm_loss = self.self_evolution_training(original_ids, attn_masks,segment, mlm_output, correct_idx, txt_labels)
        else:
            mlm_loss = self.mlm_criterion(mlm_output.transpose(1, 2), txt_labels)
        
        eq = (mlm_output.argmax(dim=-1).eq(txt_labels)).cpu().numpy()
        txt_labels_np = txt_labels.cpu().numpy()
        for bs, label in enumerate(txt_labels_np):
            index = np.where(label == -100)[0]
            f_label = np.delete(label, index)
            f_eq = np.delete(eq[bs], index)
            total_mlm_correct += f_eq.sum()
            total_mlm_element += len(f_label)   
        
        return model_output['cls_projected_embedding'], mlm_loss, total_mlm_correct, total_mlm_element, input_ids, original_ids, mlm_output


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

    # def random_word(self, original_ids, correct_idx, masking_prob=0.8):
    def random_word(self, original_ids, input_ids, correct_idx, masking_prob=0.8):
        output_label = []
        tokens = input_ids.clone().detach()
        
        for i, (token, correct) in enumerate(zip(original_ids, correct_idx)):
            prob = random.random()
            if prob < masking_prob and token != 0 and correct==False:
                prob /= masking_prob

                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[i] = self.vocab.stoi["[MASK]"]

                # 10% randomly change token to random token
                elif prob < 0.9:
                    tokens[i] = random.randrange(self.vocab.vocab_sz)

                output_label.append(token)
            else:
                # tokens[i] = token
                output_label.append(-100)  # 0

        if all(o == -100 for o in output_label):  # 0
            # at least one mask
            output_label[0] = original_ids[0]
            tokens[0] = self.vocab.stoi["[MASK]"]

        return tokens, torch.tensor(output_label).to(self.device)

    def self_questioning(self,input_ids, original_ids, attn_masks,segment):
        self.model.eval()
        with torch.no_grad(): 
            # lm_predict = self.model(original_ids, attn_masks, segment)['logits'].argmax(dim=-1) * attn_masks
            lm_predict = self.model(input_ids, attn_masks, segment)['logits'].argmax(dim=-1) * attn_masks
            correct_idx = lm_predict.eq(original_ids)

            # re-Masking the uncorrected tokens
            # new_input_ids_and_new_txt_labels = [self.random_word(x,y) for x,y in zip(original_ids, correct_idx)]
            new_input_ids_and_new_txt_labels = [self.random_word(x,y,z) for x,y,z in zip(original_ids, input_ids, correct_idx)]
            new_input_ids = torch.stack([x[0] for x in new_input_ids_and_new_txt_labels])            
            new_txt_labels = torch.stack([x[1] for x in new_input_ids_and_new_txt_labels])            
            # new_input_ids = (original_ids * correct_idx) + (self.vocab.stoi["[MASK]"] * ~correct_idx)                     
                    
        self.model.train()
        
        return new_input_ids, correct_idx, new_txt_labels

    def self_evolution_training(self, original_ids, attn_masks,segment, mlm_output, correct_idx, txt_labels):
        # with torch.no_grad():
        # lm_original_logits = self.model(original_ids, attn_masks, segment)['logits']
        
        # one_hot_logits = torch.zeros_like(lm_original_logits)
        # for b, batch_ids in enumerate(original_ids):
        #     for j, idx in enumerate(batch_ids[batch_ids!=0]):
        #         one_hot_logits[b][j][idx] = 1
        
        # smoothed_logits = self.args.SET_alpha_for_onehot * one_hot_logits + (1-self.args.SET_alpha_for_onehot) * lm_original_logits
        #### 로스 계산 문제 있음
        # loss = -torch.sum(self.log_softmax(mlm_output[~correct_idx]) * self.softmax(smoothed_logits[~correct_idx])) / mlm_output[~correct_idx].shape[0] 
        # loss = self.mlm_criterion(mlm_output[~correct_idx], smoothed_logits[~correct_idx].argmax(dim=-1))
        return self.mlm_criterion(mlm_output.transpose(1,2), original_ids)

        # loss.requires_grad=True



    def print_examples(self, input_ids,original_ids, mlm_output, n=3):
        if len(input_ids) < n : n = len(input_ids)
        for i in range(n):
            
            true_text = " ".join(self.vocab.itos(original_ids[i][input_ids[i]!=0]))
            mask_text = " ".join(self.vocab.itos(input_ids[i][input_ids[i]!=0]))
            pred_text = " ".join(self.vocab.itos(mlm_output[i].argmax(dim=-1)[input_ids[i]!=0]))
            
            print(f"[[SAMPLE: {i}]]")
            print("True: ", true_text)
            print("--------------------------------------------------")
            print("Mask: ", mask_text)
            print("--------------------------------------------------")
            print("Pred: ", pred_text)
            print("====================================================================================================")
            
            self.text_table.add_data(self.epoch, self.step_cnt, true_text, mask_text, pred_text)
        return 

    def unpack_and_stack(self,stack):                
        new_stack = []
        for i in range(len(stack[0])):
            new_stack.append(torch.cat([f[i] for f in stack]))
        return new_stack