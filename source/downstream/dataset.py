from genericpath import isfile
import json
import numpy as np
import os
from PIL import Image

import torch
from torch.utils.data import Dataset

from utils.utils import shuffle_sentence, numpy_seed, make_image_path

import random

class JsonlDataset(Dataset):
    def __init__(self, data_path, tokenizer, transforms, vocab, args):
        self.data = [json.loads(l) for l in open(data_path)]
        self.data_dir = os.path.dirname(data_path)
        self.tokenizer = tokenizer
        self.args = args
        self.vocab = vocab
        self.n_classes = len(args.labels)
        self.text_start_token = ["[CLS]"] if args.model != "mmbt" else ["[SEP]"]

        with numpy_seed(0):
            for row in self.data:
                if np.random.random() < args.drop_img_percent:
                    row["img"] = None

        self.max_seq_len = args.max_seq_len
        if args.model == "mmbt":
            self.max_seq_len -= args.num_image_embeds

        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # print('self.max_seq_len:', self.max_seq_len)
        # print('args.num_image_embeds:', self.args.num_image_embeds)
        sentence = (
            self.text_start_token
            + self.tokenizer(self.data[index]["text"])[
                : (self.max_seq_len - 1)
            ] + self.text_start_token
        )
        segment = torch.zeros(len(sentence))
        # print('sentence:', sentence)
        # print('len_seq:', len(sentence))
        # print('**************************************')

        sentence = torch.LongTensor(
            [
                self.vocab.stoi[w] if w in self.vocab.stoi else self.vocab.stoi["[UNK]"]
                for w in sentence
            ]
        )
        if self.args.task_type == "multilabel":
            label = torch.zeros(self.n_classes)
            if self.data[index]["label"] == '':
                self.data[index]["label"] = "'Others'"
            else:
                pass  
            label[
                [self.args.labels.index(tgt) for tgt in self.data[index]["label"].split(', ')]
            ] = 1
        else:
            input("????????? multilabel ?????? ????????? ?????? ????????????~")
            pass

        image = None
        if self.args.model in ["img", "concatbow", "concatbert", "mmbt"]:
            if self.data[index]["img"]:
                image = Image.open(
                    os.path.join(self.data_dir, self.data[index]["img"]))
            else:
                image = Image.fromarray(128 * np.ones((256, 256, 3), dtype=np.uint8))
            image = self.transforms(image)

        if self.args.model == "mmbt":
            # The first SEP is part of Image Token.
            segment = segment[1:]
            sentence = sentence[1:]
            # The first segment (0) is of images.
            segment += 1

        return sentence, segment, image, label


class JsonlDatasetSNUH(Dataset):
    def __init__(self, data_path, tokenizer, transforms, vocab, args, data_path_1=None, set_type='train', augmentations=None):
        self.args = args
        self.set_type = set_type
        self.data_normal = [json.loads(l) for l in open(data_path)]#[:1000]
        # print("????????? 10000?????? ??????!!!!!!!!! "*5)
        if args.make_error:
            print('ADDING ERROR # '*20)
           #### TEMP error file, ???????????? data ????????? ????????? tool ???????????? ?????? ??????????????? ?????????.
            self.data_error = [json.loads(l) for l in open(data_path_1)]#[:1000]
            # print("????????? 10000?????? ??????!!!!!!!!! "*5)

            for idx in range(len(self.data_normal)):
                self.data_normal[idx]['label'] = 0
            for idx in range(len(self.data_error)):
                self.data_error[idx]['label'] = 1

            self.error_sampling = args.error_sampling_train if set_type == 'train' else args.error_sampling_test
                
            if self.error_sampling==0:# or set_type=='test':
                print("Errors are just concatted!")
                self.data = self.data_normal + self.data_error
            elif self.error_sampling!=0 and set_type!='train':
                print("Sampling errors at validation set..")
                self.sample_error()
            else:
                 # self.sample_error() have to be called at each training epochs
                print("# Training errors gonna be sampled WITH DYNAMIC SAMPLING #")
            # print('error is just concatted' * 10)
        else:
            for idx in range(len(self.data_normal)):
                self.data_normal[idx]['label'] = 0
            self.data = self.data_normal
                    
        if args.test_with_bootstrap and set_type=='test':
            self.data = [random.choice(self.data) for i in range(len(self.data))]

        self.data_dir = os.path.dirname(data_path)
        self.tokenizer = tokenizer
        self.args = args
        self.vocab = vocab
        self.n_classes = len(args.labels)

        self.cls_tok = "<s>" if args.model == 'roberta' else "[CLS]"
        self.sep_tok = "</s>" if args.model == 'roberta' else "[SEP]"
        self.unk_tok = "<unk>" if args.model == 'roberta' else "[UNK]"

        self.text_start_token = [self.cls_tok]
        self.image_token = '<image>'
        # self.image_token = ''

        self.max_seq_len = args.max_seq_len


        self.transforms = transforms
        self.augmentations = augmentations

    def __len__(self):
        if self.error_sampling!=0:
            return len(self.data_normal)+int(len(self.data_normal)*self.error_sampling)
        else:
            return len(self.data)


    def __getitem__(self, index):        
        max_seq_len_findings = self.max_seq_len
        max_seq_len_prev_findings = 0

        # sentence shuffling within each section
        if self.set_type=='train' and self.args.txt_aug =='sentence_shuffling':
            self.data[index]["Findings"] = shuffle_sentence(self.data[index]["Findings"])
            # self.data[index]["Impression"] = shuffle_sentence(self.data[index]["Impression"])
            if self.args.use_prev_txt and self.data[index]["prev_Findings"] is not None:
                self.data[index]["prev_Findings"] = shuffle_sentence(self.data[index]["prev_Findings"])
                # self.data[index]["Impression"] = shuffle_sentence(self.data[index]["Impression"])
                
                if len(self.data[index]["Findings"]) + len(self.data[index]["prev_Findings"]) > self.max_seq_len-1:
                    max_seq_len_findings = int(self.max_seq_len * 2/3)
                    max_seq_len_prev_findings = int(self.max_seq_len * 1/3)

            
        sentence_findings = (
            self.text_start_token + self.tokenizer(self.image_token+str(self.data[index]["Findings"]))[: (max_seq_len_findings-1)]
            )
        segment_findings = torch.zeros(len(sentence_findings))
        sentence_findings = torch.LongTensor(
            [
                self.vocab.stoi[w] if w in self.vocab.stoi else self.vocab.stoi[self.unk_tok]
                for w in sentence_findings
            ]
            )
        sentence_prev_findings, segment_prev_findings = None, None
        
        if self.args.use_prev_txt:
            sentence_prev_findings = (
                [self.sep_tok] + self.tokenizer(self.image_token+str(self.data[index]["prev_Findings"]))[: (max_seq_len_prev_findings-1)]
                )
            segment_prev_findings = torch.ones(len(sentence_prev_findings))
            sentence_prev_findings = torch.LongTensor(
                [
                    self.vocab.stoi[w] if w in self.vocab.stoi else self.vocab.stoi[self.unk_tok]
                    for w in sentence_prev_findings
                ]
                )

        # sentence_impression = (
        #     self.text_start_token + self.tokenizer(str(self.data[index]["Impression"]))[: (self.max_seq_len - 1)]
        #     )
        # segment_impression = torch.zeros(len(sentence_impression))
        # sentence_impression = torch.LongTensor(
        #     [
        #         self.vocab.stoi[w] if w in self.vocab.stoi else self.vocab.stoi[self.unk_tok]
        #         for w in sentence_impression
        #     ]
        #     )
        sentence_impression, segment_impression = None, None
        

        if self.args.task_type == "multilabel":
            label = torch.zeros(self.n_classes)
            if self.data[index]["label"] == '':
                self.data[index]["label"] = "'Others'"
            else:
                pass  
            label[
                [self.args.labels.index(tgt) for tgt in str(self.data[index]["label"]).split(', ')]
            ] = 1
        elif self.args.task_type == "classification":
            label = self.args.labels.index(str(self.data[index]['label']))
        elif self.args.task_type == "binary":
            label = self.args.labels.index(str(self.data[index]['label']))
        else:
            print( 'check task_type')

        image_path, prev_image_path = make_image_path(self.data[index], base_dir=self.args.data_dir_img, dataset='mimic-cxr')

        image = self.get_image_with_transform(image_path)
        prev_image = self.get_image_with_transform(prev_image_path) if self.args.use_prev_img else None
        # prev_findings, prev_impression
        
        return sentence_findings, segment_findings, sentence_impression, segment_impression, image, label, prev_image, sentence_prev_findings, segment_prev_findings


    def sample_error(self):
        if self.error_sampling!=0:# and self.set_type!='test':
            print(f'sampling errors for {self.set_type} set!!')
            # sample errors in each epoch
            sampled_error = random.sample(self.data_error, k=int(len(self.data_normal)*self.error_sampling))
            self.data = self.data_normal + sampled_error
        else:
            raise('error occur when sampling errors')

    def get_image_with_transform(self, image_path):
        image = None
        if image_path is not None and os.path.isfile(image_path):
            image = Image.open(image_path)
        else:
            imarray = np.random.rand(2544,3056) * 255
            image = Image.fromarray(imarray.astype('uint8'))
            
        if self.augmentations is not None:
            image = self.augmentations(image)
        image = self.transforms(image)
        
        return image


    def temp_error_sampler(self):
        print("Generating Error...............")
        
        import random
        data_with_error = []
        for idx in range(len(self.data)):
            add_error = bool(random.randint(0,1))
            if add_error:
                data_with_error.append(self.data_error[idx])
                # if random.randint(0,1) == 0: 
                #     data_with_error.append(self.data[idx])
            else: 
                data_with_error.append(self.data[idx])
        
        return data_with_error

    def temp_get_labels_and_frequencies(self):
        from collections import Counter

        label_freqs = Counter()
        if self.args.task_type == "classification":
            data_labels = [str(line["label"]) for line in self.data]
        elif self.args.task_type == "binary":
            data_labels = ["1" if line["label"]!=0 else "0" for line in self.data]
            
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


class JsonlInferDatasetSNUH(Dataset):
    def __init__(self, tokenizer, transforms, vocab, args, report_img_pair_info=None, input=None):
        if args.inference_method == 'single':
            self.data = input
        elif args.inference_method == 'batch':
            self.data = [json.loads(l) for l in open(input)]

        # self.data = self.data[:100]
        ## temp 
        
        self.tokenizer = tokenizer
        self.args = args
        self.vocab = vocab

        self.cls_tok = "<s>" if args.model == 'roberta' else "[CLS]"
        self.sep_tok = "</s>" if args.model == 'roberta' else "[SEP]"
        self.unk_tok = "<unk>" if args.model == 'roberta' else "[UNK]"

        self.text_start_token = [self.cls_tok] if args.model != "mmbt" else [self.sep_tok]

        self.max_seq_len_findings = args.max_seq_len_findings
        self.max_seq_len_impression = args.max_seq_len_impression - 1

        if args.model == "mmbt":
            with numpy_seed(0):
                for row in self.data:
                    if np.random.random() < args.drop_img_percent:
                        row["img"] = None

            self.max_seq_len_findings -= int(args.num_image_embeds/2)
            self.max_seq_len_impression -= int(args.num_image_embeds/2)
            self.r2i = report_img_pair_info 
            self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # print('self.max_seq_len:', self.max_seq_len)
        # print('args.num_image_embeds:', self.args.num_image_embeds)

        sentence_findings = (
            [self.cls_tok]
            + self.tokenizer(self.data[index]["findings"])[
                : (self.max_seq_len_findings-1)
            ] + [self.sep_tok]
        )
        sentence_findings_tokens = sentence_findings[1:]

        segment_findings = torch.zeros(len(sentence_findings))
        # print('sentence:', sentence)
        # print('len_seq:', len(sentence))
        # print('**************************************')

        sentence_findings = torch.LongTensor(
            [
                self.vocab.stoi[w] if w in self.vocab.stoi else self.vocab.stoi[self.unk_tok]
                for w in sentence_findings
            ]
        )

        sentence_impression = (
            self.text_start_token
            + self.tokenizer(self.data[index]["impression"])[
                : (self.max_seq_len_impression - 1)
            ] + [self.sep_tok]
        )

        sentence_impression_tokens = sentence_impression[1:]

        segment_impression = torch.zeros(len(sentence_impression))
        # print('sentence:', sentence)
        # print('len_seq:', len(sentence))
        # print('**************************************')

        sentence_impression = torch.LongTensor(
            [
                self.vocab.stoi[w] if w in self.vocab.stoi else self.vocab.stoi[self.unk_tok]
                for w in sentence_impression
            ]
        )

        if self.data[index]['label'] != 0:
            self.data[index]['label'] = 1
        label = self.args.labels.index(str(self.data[index]['label']))

        image = None
        if self.args.model in ["img", "concatbow", "concatbert", "mmbt"]:
            ########## mimic-cxr-nlp_210904.csv ???????????? ??????

            if self.args.dataset == 'mimic-cxr':
                ssid = f"{self.data[index]['subject_id']}_{self.data[index]['study_id']}"
                image_file_name = (ssid + '_' + self.r2i[self.r2i.ssid==ssid].frontal.values[0]+'.jpg') if ssid in self.r2i.ssid.values and self.r2i[self.r2i.ssid==ssid].frontal.values[0]!='NAN' else None
            elif self.args.dataset == 'indiana':
                uid = self.data[index]['uid']
                image_file_name = (self.r2i[self.r2i.uid==uid].frontal.values[0]+'.jpg') if uid in self.r2i.ssid.values and self.r2i[self.r2i.ssid==ssid].frontal.values[0]!='NAN' else None

            if image_file_name is None:
                image = Image.fromarray(128 * np.ones((256, 256, 3), dtype=np.uint8))
            else:
                image = Image.open(
                    os.path.join(self.data_dir_img,image_file_name))
                
            image = self.transforms(image)

        if self.args.model == "mmbt":
            # The first SEP is part of Image Token.
            segment_findings = segment_findings[1:]
            sentence_findings = sentence_findings[1:]
            # The first segment (0) is of images.
            segment_findings += 1

            # The first SEP is part of findings Token.
            segment_impression = segment_impression[1:]
            sentence_impression = sentence_impression[1:]
            # The first segment (0) is of findings.
            segment_impression += 2

            sentence = torch.cat((sentence_findings, sentence_impression),0)
            segment = torch.cat((segment_findings, segment_impression),0)

        if self.args.model in ["bert", "clinicalbert", "roberta", 'cxr-bert']:
            # The first SEP is part of Image Token.
            segment_findings = segment_findings[0:]
            sentence_findings = sentence_findings[0:]
            # The first segment (0) is of images.
            segment_findings += 0

            # The first SEP is part of findings Token.
            segment_impression = segment_impression[1:-1]
            sentence_impression = sentence_impression[1:-1]
            # The first segment (1) is of findings.

            if self.args.model == "roberta":
                segment_impression += 0
            else:
                segment_impression += 1

            sentence = torch.cat((sentence_findings, sentence_impression),0)
            segment = torch.cat((segment_findings, segment_impression),0)

        # return sentence, segment, image, sentence_findings_tokens, sentence_impression_tokens
        return sentence, segment, image, label



    def temp_error_sampler(self):
        print("Generating Error...............")
        
        import random
        data_with_error = []
        for idx in range(len(self.data)):
            add_error = bool(random.randint(0,1))
            if add_error:
                data_with_error.append(self.data_error[idx])
            else: 
                data_with_error.append(self.data[idx])

        
        return data_with_error

    def temp_get_labels_and_frequencies(self):
        from collections import Counter

        label_freqs = Counter()
        if self.args.task_type == "classification":
            data_labels = [str(line["label"]) for line in self.data]
        elif self.args.task_type == "binary":
            data_labels = ["1" if line["label"]!=0 else "0" for line in self.data]
            
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