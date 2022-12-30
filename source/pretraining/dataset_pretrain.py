"""
generate dataset
"""
import os
import json
import random
import numpy as np
import pandas as pd
from PIL import Image
from fuzzywuzzy import fuzz

import torch
from torch.utils.data import Dataset
from transformers import BertModel, BertTokenizer, AutoTokenizer
# from transformers.tokenization_albert import AlbertTokenizer
from extract_medical_vocab import where_to_mask
from itertools import chain

import sys
sys.path.insert(1, 'workspace/source/downstream/utils')
from utils import shuffle_sentence

def truncate_txt(txt_tokens, max_seq_len):
    while True:
        if len(txt_tokens) <= max_seq_len:
            break
        else:
            txt_tokens.pop()


class CXRDataset(Dataset):
    def __init__(self, data_path, tokenizer, transforms, args):
        self.args = args
        self.data_dir = os.path.dirname(data_path)
        self.data = [json.loads(l) for l in open(data_path)]

        self.max_seq_len = args.max_seq_len  # 512
        self.max_seq_len -= args.num_image_embeds  # 512 - #img_embeds

        self.seq_len = args.seq_len
        self.transforms = transforms

        self.total_len = self.seq_len + self.args.num_image_embeds + 3
        self._tril_matrix = torch.tril(torch.ones((self.total_len, self.total_len), dtype=torch.long))

        self.tokenizer = tokenizer  # tokenizer = BertTokenizer.from_pretrained('bert-based-uncased').tokenize

        if args.bert_model == "albert-base-v2":
            self.albert_tokenizer = AlbertTokenizer.from_pretrained(args.bert_model)
            self.vocab_stoi = self.albert_tokenizer.get_vocab()  # <unk>, <pad>
            self.vocab_len = len(self.vocab_stoi)  # 30000

        elif args.bert_model == "emilyalsentzer/Bio_ClinicalBERT":
            self.BertTokenizer = AutoTokenizer.from_pretrained(args.bert_model)
            self.vocab_stoi = self.BertTokenizer.vocab
            self.vocab_len = len(self.vocab_stoi)  # 28996

        elif args.bert_model == "bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12":
            self.BertTokenizer = AutoTokenizer.from_pretrained(args.bert_model)
            self.vocab_stoi = self.BertTokenizer.vocab
            self.vocab_len = len(self.vocab_stoi)  # 30522

        elif args.bert_model == "bert-small-scratch":
            self.BertTokenizer = BertTokenizer.from_pretrained("google/bert_uncased_L-4_H-512_A-8")
            self.vocab_stoi = self.BertTokenizer.vocab
            self.vocab_len = len(self.vocab_stoi)  # 30522

        elif args.bert_model == "bert-base-scratch":
            self.BertTokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            self.vocab_stoi = self.BertTokenizer.vocab
            self.vocab_len = len(self.vocab_stoi)  # 30522

        # elif args.bert_model == "load_pretrained_model":
        #     self.BertTokenizer = BertTokenizer.from_pretrained(args.init_model)
        #     self.vocab_stoi = self.BertTokenizer.vocab
        #     self.vocab_len = len(self.vocab_stoi)  # 30522

        else:  # BERT-base, small, tiny
            self.BertTokenizer = BertTokenizer.from_pretrained(args.bert_model)
            self.vocab_stoi = self.BertTokenizer.vocab
            self.vocab_len = len(self.vocab_stoi)  # 30522

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # MLM
        origin_txt, img_path, is_aligned, itm_prob = self.random_pair_sampling(idx)

        if self.args.img_channel == 3:
            image = Image.open(os.path.join(self.data_dir, img_path)).convert("RGB")
        elif self.args.img_channel == 1:
            image = Image.open(os.path.join(self.data_dir, img_path)).convert("RGB")

        image = self.transforms(image)

        tokenized_sentence = self.tokenizer(origin_txt)  # ['i','ate','an','apple'], no special token

        truncate_txt(tokenized_sentence, self.seq_len)

        if self.args.bert_model in ["albert-base-v2", 'xlm-roberta-base', 'xlm-roberta-large']:
            encoded_sentence = [self.vocab_stoi[w] if w in self.vocab_stoi else self.vocab_stoi["<unk>"]
                                for w in tokenized_sentence]
        else:
            encoded_sentence = [self.vocab_stoi[w] if w in self.vocab_stoi else self.vocab_stoi["[UNK]"]
                                for w in tokenized_sentence]  # [178, 8756, 1126, 12075]

        input_ids, txt_labels = self.random_word(encoded_sentence)

        if self.disturbing_mask:
            input_ids = input_ids + [self.vocab_stoi["[SEP]"]]
            txt_labels_t = [-100] + txt_labels + [-100]
            txt_labels_i = [-100] * (self.args.num_image_embeds + 2)
        else:
            input_ids = input_ids + [self.vocab_stoi["[SEP]"]]
            txt_labels_t = txt_labels + [-100]
            txt_labels_i = [-100] * (self.args.num_image_embeds + 2)

        attn_masks_t = [1] * len(input_ids)
        attn_masks_i = [1] * (self.args.num_image_embeds + 2)

        if self.args.bert_model in ["albert-base-v2", 'xlm-roberta-base', 'xlm-roberta-large']:
            padding = [self.vocab_stoi["<pad>"] for _ in range(self.seq_len - len(input_ids) + 1)]  # [SEP]
            label_padding = [-100 for _ in range(self.seq_len - len(input_ids) + 1)]  # [SEP]
        else:
            padding = [self.vocab_stoi["[PAD]"] for _ in range(self.seq_len - len(input_ids) + 1)]  # [SEP]
            label_padding = [-100 for _ in range(self.seq_len - len(input_ids) + 1)]  # [SEP]

        input_ids.extend(padding)
        attn_masks_t.extend(padding)
        txt_labels_t.extend(label_padding)

        txt_labels = txt_labels_i + txt_labels_t
        attn_masks = attn_masks_i + attn_masks_t  # attn_masks [1, 1, 1, 1, 1, 1, 1, 1, 0, 0] -> Img_feat, Token, Pad

        segment = [1 for _ in range(self.seq_len + 1)]  # 2 [SEP]

        cls_tok = [self.vocab_stoi["[CLS]"]]
        cls_tok = torch.tensor(cls_tok)
        input_ids_tensor = torch.tensor(input_ids)
        txt_labels = torch.tensor(txt_labels)
        segment = torch.tensor(segment)
        is_aligned = torch.tensor(is_aligned)

        attn_1d = torch.tensor(attn_masks)

        full_attn = torch.tensor((attn_masks_i + attn_masks_t),
                                 dtype=torch.long).unsqueeze(0).expand(self.total_len, self.total_len).clone()

        extended_attn_masks = torch.zeros(self.total_len, self.total_len, dtype=torch.long)
        second_st, second_end = self.args.num_image_embeds + 2, self.args.num_image_embeds + 2 + len(input_ids)
        extended_attn_masks[:, :self.args.num_image_embeds + 2].fill_(1)
        extended_attn_masks[second_st:second_end, second_st:second_end].copy_(
            self._tril_matrix[:second_end - second_st, :second_end - second_st])
        s2s_attn = extended_attn_masks

        mixed_lst = [full_attn, s2s_attn]

        if self.args.Mixed:
            # print('Mixed attn mask')
            assert (self.args.s2s_prob + self.args.bi_prob) == 1.0
            attn_masks_tensor = random.choices(mixed_lst, weights=[self.args.bi_prob, self.args.s2s_prob])[0]
            # print(f'S2S {self.args.s2s_prob} vs Bi {self.args.bi_prob}')

        elif self.args.BAR_attn:
            # print('BAR_attn attn mask')
            extended_attn_masks[:self.args.num_image_embeds+2, :].fill_(1)
            attn_masks_tensor = extended_attn_masks

        elif self.args.disturbing_mask:
            baseline_attn = torch.zeros(self.total_len, self.total_len, dtype=torch.long)
            baseline_attn[:self.args.num_image_embeds + 2, :self.args.num_image_embeds + 2].fill_(1)
            baseline_attn[self.args.num_image_embeds + 2:, self.args.num_image_embeds + 2:].fill_(1)
            attn_masks_tensor = baseline_attn

        else:
            if self.args.attn_1d:
                # print('1d_bidirectional attn mask')
                attn_masks_tensor = attn_1d  # '1d attention mask'

            else:
                # print('full_bidirecitonal attn mask')
                attn_masks_tensor = full_attn  # 'Full attention mask'

        sep_tok = [self.vocab_stoi["[SEP]"]]
        sep_tok = torch.tensor(sep_tok)

        return cls_tok, input_ids_tensor, txt_labels, attn_masks_tensor, image, segment, is_aligned, sep_tok, itm_prob

    def random_word(self, tokens):
        output_label = []

        for i, token in enumerate(tokens):
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[i] = self.vocab_stoi["[MASK]"]

                # 10% randomly change token to random token
                elif prob < 0.9:
                    tokens[i] = random.randrange(self.vocab_len)

                output_label.append(token)
            else:
                tokens[i] = token
                output_label.append(-100)  # 0

        if all(o == -100 for o in output_label):  # 0
            # at least one mask
            output_label[0] = tokens[0]
            tokens[0] = self.vocab_stoi["[MASK]"]

        return tokens, output_label

    def random_pair_sampling(self, idx):
        _, _, label, txt, img = self.data[idx].keys()  # id, txt, img

        d_label = self.data[idx][label]
        d_txt = self.data[idx][txt]
        d_img = self.data[idx][img]

        itm_prob = random.random()

        if itm_prob > 0.5:
            return d_txt, d_img, 1, itm_prob
        else:
            for itr in range(300):
                random_txt, random_label = self.get_random_line()
                if fuzz.token_sort_ratio(d_label, random_label) != 100:
                    return random_txt, d_img, 0, itm_prob
                    break
                else:
                    pass

    def get_random_line(self):
        rand_num = random.randint(0, len(self.data) - 1)
        txt = self.data[rand_num]['text']
        label = self.data[rand_num]['label']
        return txt, label

class CXRDataset_DABIN(Dataset):
    def __init__(self, data_path, tokenizer, transforms, args, report_img_pair_info, phrase_vocab, entity_vocab):
        self.args = args
        self.data_dir = os.path.dirname(data_path)
        self.data_dir_img = os.path.join(self.data_dir, 'mimic-nlp-jpg')
        self.data = [json.loads(l) for l in open(data_path)]#[:100]
        # print('data is just 100!! '*30)

        self.max_seq_len = args.max_seq_len  # 
        if args.itm_task :
            self.max_seq_len -= args.num_image_embeds  # 512 - #img_embeds

        self.seq_len = args.seq_len -2 #510. 2 for [CLS], [SEP] token
        self.transforms = transforms

        self.total_len = self.seq_len + 2
        if args.itm_task:
            self.total_len += self.args.num_image_embeds + 3
        self._tril_matrix = torch.tril(torch.ones((self.total_len, self.total_len), dtype=torch.long))

        self.tokenizer = tokenizer  # tokenizer = BertTokenizer.from_pretrained('bert-based-uncased').tokenize

        self.r2i = report_img_pair_info
        self.phrase_vocab = phrase_vocab
        self.entity_vocab = entity_vocab

        if args.bert_model == "albert-base-v2":
            self.albert_tokenizer = AlbertTokenizer.from_pretrained(args.bert_model)
            self.vocab_stoi = self.albert_tokenizer.get_vocab()  # <unk>, <pad>
            self.vocab_len = len(self.vocab_stoi)  # 30000

        if args.bert_model in ['xlm-roberta-base', 'xlm-roberta-large']:
            self.roberta_tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
            self.vocab_stoi = self.roberta_tokenizer.get_vocab()  # <unk>, <pad>
            self.vocab_len = len(self.vocab_stoi)  # 250002

        elif args.bert_model == "emilyalsentzer/Bio_ClinicalBERT":
            self.BertTokenizer = AutoTokenizer.from_pretrained(args.bert_model)
            self.vocab_stoi = self.BertTokenizer.vocab
            self.vocab_len = len(self.vocab_stoi)  # 28996

        elif args.bert_model == "bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12":
            self.BertTokenizer = AutoTokenizer.from_pretrained(args.bert_model)
            self.vocab_stoi = self.BertTokenizer.vocab
            self.vocab_len = len(self.vocab_stoi)  # 30522

        elif args.bert_model == "bert-small-scratch":
            self.BertTokenizer = BertTokenizer.from_pretrained("google/bert_uncased_L-4_H-512_A-8")
            self.vocab_stoi = self.BertTokenizer.vocab
            self.vocab_len = len(self.vocab_stoi)  # 30522

        elif args.bert_model == "bert-base-scratch":
            self.BertTokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            self.vocab_stoi = self.BertTokenizer.vocab
            self.vocab_len = len(self.vocab_stoi)  # 30522

        # elif args.bert_model == "load_pretrained_model":
        #     self.BertTokenizer = BertTokenizer.from_pretrained(args.init_model)
        #     self.vocab_stoi = self.BertTokenizer.vocab
        #     self.vocab_len = len(self.vocab_stoi)  # 30522

        else:  # BERT-base, small, tiny
            self.BertTokenizer = BertTokenizer.from_pretrained(args.bert_model)
            self.vocab_stoi = self.BertTokenizer.vocab
            self.vocab_len = len(self.vocab_stoi)  # 30522

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # MLM
        origin_txt, img_path, is_aligned, itm_prob = self.random_pair_sampling(idx)
        # study_id, subject_id, findings, impression, background, label = self.data[idx].keys()  # id, txt, img

        image = torch.tensor([999])
        if self.args.itm_task:
            if img_path is None:
                image = Image.fromarray(128 * np.ones((256, 256, 3), dtype=np.uint8))
            else:
                image = Image.open(
                    os.path.join(self.data_dir_img,img_path)).convert('RGB')

            image = self.transforms(image)

        tokenized_sentence = self.tokenizer(origin_txt)  # ['i','ate','an','apple'], no special token

        truncate_txt(tokenized_sentence, self.seq_len)

        if self.args.bert_model in ["albert-base-v2", 'xlm-roberta-base', 'xlm-roberta-large']:
            encoded_sentence = [self.vocab_stoi[w] if w in self.vocab_stoi else self.vocab_stoi["<unk>"]
                                for w in tokenized_sentence]
        else:
            encoded_sentence = [self.vocab_stoi[w] if w in self.vocab_stoi else self.vocab_stoi["[UNK]"]
                                for w in tokenized_sentence]  # [178, 8756, 1126, 12075]

        # Normal MLM
        # input_ids, txt_labels = self.random_word(encoded_sentence) 

        # Knowledge MLM
        # meaningful_idx = where_to_mask(origin_txt, [], self.tokenizer)
        # meaningful_idx = where_to_mask(origin_txt, self.entity_vocab, self.tokenizer)
        # meaningful_idx = where_to_mask(origin_txt, self.phrase_vocab, self.tokenizer)
        meaningful_idx = where_to_mask(origin_txt, self.entity_vocab + self.phrase_vocab, self.tokenizer)
        input_ids, txt_labels = self.random_knowledge(encoded_sentence, meaningful_idx, num_knowledge=2)

        if self.args.disturbing_mask:
            input_ids = input_ids + [self.vocab_stoi["[SEP]"]]
            txt_labels_t = [-100] + txt_labels + [-100]
            txt_labels_i = [-100] * (self.args.num_image_embeds + 2)
        else:
            if self.args.bert_model in ["albert-base-v2", 'xlm-roberta-base', 'xlm-roberta-large']:
                input_ids = input_ids + [self.vocab_stoi["</s>"]]
            else:
                input_ids = input_ids + [self.vocab_stoi["[SEP]"]]
            txt_labels_t = txt_labels + [-100]
            if self.args.itm_task:
                txt_labels_i = [-100] * (self.args.num_image_embeds + 2) # for [CLS], [SEP]
            else: 
                txt_labels_i = [-100] * (self.args.num_image_embeds + 1) # only for [CLS]

        attn_masks_t = [1] * len(input_ids)

        if self.args.itm_task:
            attn_masks_i = [1] * (self.args.num_image_embeds + 2) # for [CLS], [SEP]
        else:
            attn_masks_i = [1] * (self.args.num_image_embeds + 1) # only for [CLS]

        if self.args.bert_model in ["albert-base-v2", 'xlm-roberta-base', 'xlm-roberta-large']:
            padding = [self.vocab_stoi["<pad>"] for _ in range(self.seq_len - len(input_ids) + 1)]  # [SEP]
            label_padding = [-100 for _ in range(self.seq_len - len(input_ids) + 1)]  # [SEP]
        else:
            padding = [self.vocab_stoi["[PAD]"] for _ in range(self.seq_len - len(input_ids) + 1)]  # [SEP]
            label_padding = [-100 for _ in range(self.seq_len - len(input_ids) + 1)]  # [SEP]

        input_ids.extend(padding)
        attn_masks_t.extend(padding)
        txt_labels_t.extend(label_padding)

        txt_labels = txt_labels_i + txt_labels_t
        attn_masks = attn_masks_i + attn_masks_t  # attn_masks [1, 1, 1, 1, 1, 1, 1, 1, 0, 0] -> Img_feat, Token, Pad

        # sep_indexes = input_ids.index(self.vocab_stoi["[SEP]"])
        # segment = [0 if i <=sep_indexes else 1 for i in range(self.seq_len+1)] # Finding ==0, Impression ==1
        segment = [0 for _ in range(self.seq_len + 1)]  # all 0 mimic-cxr할때는 1이었는데 0이어도 상관없겠지?

        if self.args.bert_model in ["albert-base-v2", 'xlm-roberta-base', 'xlm-roberta-large']:
            cls_tok = [self.vocab_stoi["<s>"]]
        else:
            cls_tok = [self.vocab_stoi["[CLS]"]]
        cls_tok = torch.tensor(cls_tok)
        input_ids_tensor = torch.tensor(input_ids)
        txt_labels = torch.tensor(txt_labels)
        segment = torch.tensor(segment)
        is_aligned = torch.tensor(is_aligned)

        attn_1d = torch.tensor(attn_masks)

        full_attn = torch.tensor((attn_masks_i + attn_masks_t),
                                dtype=torch.long).unsqueeze(0).expand(self.total_len, self.total_len).clone()

        if any((self.args.Mixed, self.args.BAR_attn, self.args.disturbing_mask)):
            extended_attn_masks = torch.zeros(self.total_len, self.total_len, dtype=torch.long)
            second_st, second_end = self.args.num_image_embeds + 2, self.args.num_image_embeds + 2 + len(input_ids)
            extended_attn_masks[:, :self.args.num_image_embeds + 2].fill_(1)
            extended_attn_masks[second_st:second_end, second_st:second_end].copy_(
                self._tril_matrix[:second_end - second_st, :second_end - second_st])
            s2s_attn = extended_attn_masks

            mixed_lst = [full_attn, s2s_attn]

        if self.args.Mixed:
            # print('Mixed attn mask')
            assert (self.args.s2s_prob + self.args.bi_prob) == 1.0
            attn_masks_tensor = random.choices(mixed_lst, weights=[self.args.bi_prob, self.args.s2s_prob])[0]
            # print(f'S2S {self.args.s2s_prob} vs Bi {self.args.bi_prob}')

        elif self.args.BAR_attn:
            # print('BAR_attn attn mask')
            extended_attn_masks[:self.args.num_image_embeds+2, :].fill_(1)
            attn_masks_tensor = extended_attn_masks

        elif self.args.disturbing_mask:
            baseline_attn = torch.zeros(self.total_len, self.total_len, dtype=torch.long)
            baseline_attn[:self.args.num_image_embeds + 2, :self.args.num_image_embeds + 2].fill_(1)
            baseline_attn[self.args.num_image_embeds + 2:, self.args.num_image_embeds + 2:].fill_(1)
            attn_masks_tensor = baseline_attn

        else:
            if self.args.attn_1d:
                # print('1d_bidirectional attn mask')
                attn_masks_tensor = attn_1d  # '1d attention mask'

            else:
                # print('full_bidirecitonal attn mask')
                attn_masks_tensor = full_attn  # 'Full attention mask'
        if self.args.bert_model in ["albert-base-v2", 'xlm-roberta-base', 'xlm-roberta-large']:
            sep_tok = [self.vocab_stoi["</s>"]]
        else:
            sep_tok = [self.vocab_stoi["[SEP]"]]
        sep_tok = torch.tensor(sep_tok)

        return cls_tok, input_ids_tensor, txt_labels, attn_masks_tensor, image, segment, is_aligned, sep_tok, itm_prob

    def random_word(self, tokens):
        output_label = []           

        for i, token in enumerate(tokens):
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[i] = self.vocab_stoi["[MASK]"]

                # 10% randomly change token to random token
                elif prob < 0.9:
                    tokens[i] = random.randrange(self.vocab_len)

                output_label.append(token)
            else:
                tokens[i] = token
                output_label.append(-100)  # 0

        if all(o == -100 for o in output_label):  # 0
            # at least one mask
            output_label[0] = tokens[0]
            tokens[0] = self.vocab_stoi["[MASK]"]

        return tokens, output_label

    def random_knowledge(self, tokens, meaningful_idx=None, num_knowledge=4):
        if self.args.bert_model in ["albert-base-v2", 'xlm-roberta-base', 'xlm-roberta-large']:
            mask_tok = "<mask>"
        else:
            mask_tok = "[MASK]"

        output_label = []
        if meaningful_idx:
            meaningful_idx = np.array(meaningful_idx)
            nonzeros = list(set(meaningful_idx[meaningful_idx!=0]))

        if nonzeros:
            # selected_knowledge_idx = random.choice(nonzeros)
            # selected_token_idx = list(np.where(meaningful_idx==selected_knowledge_idx)[0])
            selected_knowledge_idxs = random.choices(nonzeros, k=num_knowledge)
            selected_token_idxs = [list(np.where(meaningful_idx==idx)[0]) for idx in selected_knowledge_idxs]
            borders = [[span[0]-1, span[-1]+1] for span in selected_token_idxs]
            selected_token_idx = list(chain(*selected_token_idxs))
            border_idx = list(chain(*borders))

            for i, token in enumerate(tokens):
                if i in selected_token_idx: #knowledge MLM
                    tokens[i] = self.vocab_stoi[mask_tok]
                    output_label.append(token)
                elif i in border_idx: # knowledge 주변 단어일 경우 마스킹하지 않도록함
                    tokens[i] = token
                    output_label.append(-100)
                else: # Normal MLM
                    prob = random.random()
                    if prob < 0.15:
                        prob /= 0.15

                        # 80% randomly change token to mask token
                        if prob < 0.8:
                            tokens[i] = self.vocab_stoi[mask_tok]

                        # 10% randomly change token to random token
                        elif prob < 0.9:
                            tokens[i] = random.randrange(self.vocab_len)

                        output_label.append(token)
                    else:
                        tokens[i] = token
                        output_label.append(-100)  # 0
        else: # if there isn't any knowledge word, just do MLM 
            for i, token in enumerate(tokens):
                prob = random.random()
                if prob < 0.15:
                    prob /= 0.15

                    # 80% randomly change token to mask token
                    if prob < 0.8:
                        tokens[i] = self.vocab_stoi[mask_tok]

                    # 10% randomly change token to random token
                    elif prob < 0.9:
                        tokens[i] = random.randrange(self.vocab_len)

                    output_label.append(token)
                else:
                    tokens[i] = token
                    output_label.append(-100)  # 0

        if all(o == -100 for o in output_label):  # 0
            # at least one mask
            output_label[0] = tokens[0]
            tokens[0] = self.vocab_stoi[mask_tok]

        return tokens, output_label

    def random_pair_sampling(self, idx):

        if self.args.dataset == "mimic-cxr":
            study_id, subject_id, findings, impression, background, label = self.data[idx].keys()  # id, txt, img
            origin_txt = self.data[idx][findings] + ' [SEP] ' + self.data[idx][impression]
            d_label = self.data[idx][label] ### Chexpert label
        elif self.args.dataset == "SNUH":
            index, study_id, subject_id, findings, impression, cluster, error_label, label = self.data[idx].keys()  # id, txt, img
            origin_txt = self.data[idx][findings] + ' </s> ' + self.data[idx][impression]
        
        d_txt = origin_txt

        # _, _, label, txt, img = self.data[idx].keys()  # id, txt, img

        if self.args.itm_task:
            ssid = f"{self.data[idx][subject_id]}_{self.data[idx][study_id]}"
            image_file_name = (ssid + '_' + self.r2i[self.r2i.ssid==ssid].frontal.values[0]+'.jpg') if ssid in self.r2i.ssid.values and self.r2i[self.r2i.ssid==ssid].frontal.values[0]!='NAN' else None
            d_img = image_file_name

            itm_prob = random.random()

            if itm_prob > 0.5:
                return d_txt, d_img, 1, itm_prob
            else:
                for itr in range(300):
                    random_txt, random_label = self.get_random_line()
                    if fuzz.token_sort_ratio(d_label, random_label) != 100:
                        return random_txt, d_img, 0, itm_prob
                        break
                    else:
                        pass
        else:
            return d_txt, None, 1, [999]

    def get_random_line(self):
        rand_num = random.randint(0, len(self.data) - 1)
        txt = self.data[rand_num]['findings'] + ' ' + self.data[rand_num]['impression']
        label = self.data[rand_num]['label']
        return txt, label


class VLCXRDataset(Dataset):
    def __init__(self, data_path, tokenizer, vocab, transforms, args, knowledge_vocab, augmentations=None):
        self.args = args
        self.data = [json.loads(l) for l in open(os.path.join(args.data_path,data_path))]#[:100]

        self.max_seq_len = args.max_seq_len  # 512

        self.transforms = transforms
        self.augmentations = augmentations

        self._tril_matrix = torch.tril(torch.ones((self.max_seq_len, self.max_seq_len), dtype=torch.long))

        self.tokenizer = tokenizer
        self.vocab = vocab
        self.knowledge_vocab = knowledge_vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        origin_findings, origin_impression = self.data[idx]['Findings'], self.data[idx]['Impression']
        origin_txt = "[CLS] " + origin_findings
        
        # 일정 확률로 impression을 붙임.
        if random.uniform(0,1) < self.args.add_impression_rate: 
            origin_txt += " [SEP] " + origin_impression
            

        tokenized_sentence = self.tokenizer(origin_txt)  # ['i','ate','an','apple'], no special token
        truncate_txt(tokenized_sentence, self.max_seq_len)

        if self.args.bert_model in ["albert-base-v2", 'xlm-roberta-base', 'xlm-roberta-large']:
            encoded_sentence = [self.vocab.stoi[w] if w in self.vocab.stoi else self.vocab.stoi["<unk>"]
                                for w in tokenized_sentence]
        else:
            encoded_sentence = [self.vocab.stoi[w] if w in self.vocab.stoi else self.vocab.stoi["[UNK]"]
                                for w in tokenized_sentence]  # [178, 8756, 1126, 12075]
        
        original_ids = encoded_sentence.copy()
        
        if self.args.use_MKP:
            meaningful_idx = where_to_mask(origin_txt, self.knowledge_vocab, self.tokenizer)
            input_ids, txt_labels = self.random_knowledge(encoded_sentence, meaningful_idx, num_knowledge=4)
        else:
            input_ids, txt_labels = self.random_word(encoded_sentence)

        attn_masks = [1] * len(input_ids)

        if self.args.bert_model in ["albert-base-v2", 'xlm-roberta-base', 'xlm-roberta-large']:
            padding = [self.vocab.stoi["<pad>"] for _ in range(self.max_seq_len - len(input_ids))]
            label_padding = [-100 for _ in range(self.max_seq_len - len(input_ids))]
        else:
            padding = [self.vocab.stoi["[PAD]"] for _ in range(self.max_seq_len - len(input_ids))]
            label_padding = [-100 for _ in range(self.max_seq_len - len(input_ids))]

        input_ids.extend(padding)
        original_ids.extend(padding)
        attn_masks.extend(padding)
        txt_labels.extend(label_padding)

        segment = [0 for _ in range(self.max_seq_len)]

        input_ids_tensor = torch.tensor(input_ids)
        original_ids_tensor = torch.tensor(original_ids)
        txt_labels = torch.tensor(txt_labels)
        segment = torch.tensor(segment)

        attn_masks = torch.tensor(attn_masks)

        return input_ids_tensor, txt_labels, attn_masks, segment, original_ids_tensor

    def random_pair_sampling(self, idx):
        # suffle image-text pair
        d_label = self.data[idx]['chexpert_label']
        d_findings = self.data[idx]['Findings']
        d_impression = self.data[idx]['Impression']
        d_img = self.data[idx]['dicom_id']

        itm_prob = random.random()

        if itm_prob > 0.5:
            return d_findings, d_impression, d_img, 1, itm_prob
        else:
            for itr in range(300):
                random_findings, random_impression, random_label = self.get_random_line()
                if fuzz.token_sort_ratio(d_label, random_label) != 100:
                    return random_findings, random_impression, d_img, 0, itm_prob
                    break
                else:
                    pass

    def random_knowledge(self, tokens, meaningful_idx=None, num_knowledge=4, masking_prob=0.1):
        if self.args.bert_model in ["albert-base-v2", 'xlm-roberta-base', 'xlm-roberta-large']:
            mask_tok = "<mask>"
        else:
            mask_tok = "[MASK]"

        output_label = []
        if meaningful_idx:
            meaningful_idx = np.array(meaningful_idx)
            nonzeros = list(set(meaningful_idx[meaningful_idx!=0]))

        if nonzeros:
            # selected_knowledge_idx = random.choice(nonzeros)
            # selected_token_idx = list(np.where(meaningful_idx==selected_knowledge_idx)[0])
            selected_knowledge_idxs = random.choices(nonzeros, k=num_knowledge)
            selected_token_idxs = [list(np.where(meaningful_idx==idx)[0]) for idx in selected_knowledge_idxs]
            borders = [[span[0]-1, span[-1]+1] for span in selected_token_idxs]
            selected_token_idx = list(chain(*selected_token_idxs))
            border_idx = list(chain(*borders))

            for i, token in enumerate(tokens):
                if i in selected_token_idx: #knowledge MLM
                    tokens[i] = self.vocab.stoi[mask_tok]
                    output_label.append(token)
                elif i in border_idx: # knowledge 주변 단어일 경우 마스킹하지 않도록함
                    tokens[i] = token
                    output_label.append(-100)
                else: # Normal MLM
                    prob = random.random()
                    if prob < masking_prob:
                        prob /= masking_prob

                        # 80% randomly change token to mask token
                        if prob < 0.8:
                            tokens[i] = self.vocab.stoi[mask_tok]

                        # 10% randomly change token to random token
                        elif prob < 0.9:
                            tokens[i] = random.randrange(self.vocab.vocab_sz)

                        output_label.append(token)
                    else:
                        tokens[i] = token
                        output_label.append(-100)  # 0
        else: # if there isn't any knowledge word, just do MLM 
            for i, token in enumerate(tokens):
                prob = random.random()
                if prob < masking_prob:
                    prob /= masking_prob

                    # 80% randomly change token to mask token
                    if prob < 0.8:
                        tokens[i] = self.vocab.stoi[mask_tok]

                    # 10% randomly change token to random token
                    elif prob < 0.9:
                        tokens[i] = random.randrange(self.vocab.vocab_sz)

                    output_label.append(token)
                else:
                    tokens[i] = token
                    output_label.append(-100)  # 0

        if all(o == -100 for o in output_label):  # 0
            # at least one mask
            output_label[0] = tokens[0]
            tokens[0] = self.vocab.stoi[mask_tok]

        return tokens, output_label

    def random_word(self, tokens):
        output_label = []

        for i, token in enumerate(tokens):
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[i] = self.vocab.stoi["[MASK]"]

                # 10% randomly change token to random token
                elif prob < 0.9:
                    tokens[i] = random.randrange(self.vocab.vocab_sz)

                output_label.append(token)
            else:
                tokens[i] = token
                output_label.append(-100)  # 0

        if all(o == -100 for o in output_label):  # 0
            # at least one mask
            output_label[0] = tokens[0]
            tokens[0] = self.vocab.stoi["[MASK]"]

        return tokens, output_label


    def get_random_line(self):
        rand_num = random.randint(0, len(self.data) - 1)
        findings = self.data[rand_num]['Findings']
        impression = self.data[rand_num]['impression']
        label = self.data[rand_num]['chexpert_label']
        return findings, impression, label
        

class VLCXRDataset_biovil(Dataset):
    def __init__(self, data_path, tokenizer, vocab, transforms, args, knowledge_vocab, augmentations=None):
        self.args = args
        self.data = [json.loads(l) for l in open(os.path.join(args.data_path,data_path))]#[:500] 

        self.max_seq_len = args.max_seq_len  # 512

        self.transforms = transforms
        self.augmentations = augmentations

        self._tril_matrix = torch.tril(torch.ones((self.max_seq_len, self.max_seq_len), dtype=torch.long))

        self.tokenizer = tokenizer
        self.vocab = vocab
        self.knowledge_vocab = knowledge_vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        findings, impression = self.data[idx]['Findings'], self.data[idx]['Impression']
        
        findings_tensors = self.text_to_tensors(findings, num_knowledge=4)
        impression_tensors = self.text_to_tensors(impression, num_knowledge=1)
        
        
        
        return {'findings_tensors': findings_tensors, 
                'impression_tensors':impression_tensors,
                'chexpert_label': self.data[idx]['chexpert_label']}

    def text_to_tensors(self, text, num_knowledge):
        text = "[CLS] " + shuffle_sentence(text)
        tokenized_sentence = self.tokenizer(text)  # ['i','ate','an','apple'], no special token
        truncate_txt(tokenized_sentence, self.max_seq_len)

        if self.args.bert_model in ["albert-base-v2", 'xlm-roberta-base', 'xlm-roberta-large']:
            encoded_sentence = [self.vocab.stoi[w] if w in self.vocab.stoi else self.vocab.stoi["<unk>"]
                                for w in tokenized_sentence]
        else:
            encoded_sentence = [self.vocab.stoi[w] if w in self.vocab.stoi else self.vocab.stoi["[UNK]"]
                                for w in tokenized_sentence]  # [178, 8756, 1126, 12075]
        
        original_ids = encoded_sentence.copy()
        
        if self.args.use_MKP:
            meaningful_idx = where_to_mask(text, self.knowledge_vocab, self.tokenizer)
            input_ids, txt_labels = self.random_knowledge(encoded_sentence, meaningful_idx, num_knowledge=num_knowledge)
        else:
            input_ids, txt_labels = self.random_word(encoded_sentence)

        attn_masks = [1] * len(input_ids)
        
        if self.args.bert_model in ["albert-base-v2", 'xlm-roberta-base', 'xlm-roberta-large']:
            padding = [self.vocab.stoi["<pad>"] for _ in range(self.max_seq_len - len(input_ids))]
            label_padding = [-100 for _ in range(self.max_seq_len - len(input_ids))]
        else:
            padding = [self.vocab.stoi["[PAD]"] for _ in range(self.max_seq_len - len(input_ids))]
            label_padding = [-100 for _ in range(self.max_seq_len - len(input_ids))]
        
        input_ids.extend(padding)
        original_ids.extend(padding)
        attn_masks.extend(padding)
        txt_labels.extend(label_padding)

        segment = [0 for _ in range(self.max_seq_len)]

        input_ids_tensor = torch.tensor(input_ids)
        original_ids_tensor = torch.tensor(original_ids)
        txt_labels = torch.tensor(txt_labels)
        segment = torch.tensor(segment)

        attn_masks = torch.tensor(attn_masks)
        
        return (input_ids_tensor, txt_labels, attn_masks, segment, original_ids_tensor)
            


    def random_knowledge(self, tokens, meaningful_idx=None, num_knowledge=4, masking_prob=0.1):
        if self.args.bert_model in ["albert-base-v2", 'xlm-roberta-base', 'xlm-roberta-large']:
            mask_tok = "<mask>"
        else:
            mask_tok = "[MASK]"

        output_label = []
        if meaningful_idx:
            meaningful_idx = np.array(meaningful_idx)
            nonzeros = list(set(meaningful_idx[meaningful_idx!=0]))

        if nonzeros:
            # selected_knowledge_idx = random.choice(nonzeros)
            # selected_token_idx = list(np.where(meaningful_idx==selected_knowledge_idx)[0])
            selected_knowledge_idxs = random.choices(nonzeros, k=num_knowledge)
            selected_token_idxs = [list(np.where(meaningful_idx==idx)[0]) for idx in selected_knowledge_idxs]
            borders = [[span[0]-1, span[-1]+1] for span in selected_token_idxs]
            selected_token_idx = list(chain(*selected_token_idxs))
            border_idx = list(chain(*borders))

            for i, token in enumerate(tokens):
                if i in selected_token_idx: #knowledge MLM
                    tokens[i] = self.vocab.stoi[mask_tok]
                    output_label.append(token)
                elif i in border_idx: # knowledge 주변 단어일 경우 마스킹하지 않도록함
                    tokens[i] = token
                    output_label.append(-100)
                else: # Normal MLM
                    prob = random.random()
                    if prob < masking_prob:
                        prob /= masking_prob

                        # 80% randomly change token to mask token
                        if prob < 0.8:
                            tokens[i] = self.vocab.stoi[mask_tok]

                        # 10% randomly change token to random token
                        elif prob < 0.9:
                            tokens[i] = random.randrange(self.vocab.vocab_sz)

                        output_label.append(token)
                    else:
                        tokens[i] = token
                        output_label.append(-100)  # 0
        else: # if there isn't any knowledge word, just do MLM 
            for i, token in enumerate(tokens):
                prob = random.random()
                if prob < masking_prob:
                    prob /= masking_prob

                    # 80% randomly change token to mask token
                    if prob < 0.8:
                        tokens[i] = self.vocab.stoi[mask_tok]

                    # 10% randomly change token to random token
                    elif prob < 0.9:
                        tokens[i] = random.randrange(self.vocab.vocab_sz)

                    output_label.append(token)
                else:
                    tokens[i] = token
                    output_label.append(-100)  # 0

        if all(o == -100 for o in output_label):  # 0
            # at least one mask
            output_label[0] = tokens[0]
            tokens[0] = self.vocab.stoi[mask_tok]

        return tokens, output_label

    def random_word(self, tokens):
        output_label = []

        for i, token in enumerate(tokens):
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[i] = self.vocab.stoi["[MASK]"]

                # 10% randomly change token to random token
                elif prob < 0.9:
                    tokens[i] = random.randrange(self.vocab.vocab_sz)

                output_label.append(token)
            else:
                tokens[i] = token
                output_label.append(-100)  # 0

        if all(o == -100 for o in output_label):  # 0
            # at least one mask
            output_label[0] = tokens[0]
            tokens[0] = self.vocab.stoi["[MASK]"]

        return tokens, output_label


    def get_random_line(self):
        rand_num = random.randint(0, len(self.data) - 1)
        findings = self.data[rand_num]['Findings']
        impression = self.data[rand_num]['impression']
        label = self.data[rand_num]['chexpert_label']
        return findings, impression, label
        
        