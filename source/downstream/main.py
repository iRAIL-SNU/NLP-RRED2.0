import os
from random import choices
import pandas as pd

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import csv
import argparse
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, confusion_matrix
from tqdm import tqdm

import torch.nn as nn
import torch.optim as optim
# from pytorch_pretrained_bert import BertAdam

from helpers import get_data_loaders, get_dataset, get_model
from utils.utils import *

from utils.logger import create_logger
import wandb

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler
from datetime import datetime

# ###
# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# ####
    
def get_args(parser):

    parser.add_argument("--seed", type=int, default=1125)
    parser.add_argument("--batch_sz", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=20)

    parser.add_argument("--model", type=str, default="cxr-bert", choices=['mmbt', 'bert', 'clinicalbert', 'roberta', 'gatortron', 'cxr-bert', 'xvl-bert'])
    parser.add_argument("--task_type", type=str, default="binary", choices=["multilabel", "classification", "binary"])
    parser.add_argument("--n_classes", type=int, default=2)

    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--use_ddp", type=str2bool, default=False)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--n_workers", type=int, default=32)
    parser.add_argument("--patience", type=int, default=10)

    now = datetime.now()
    now = now.strftime('%Y-%m-%d')
    output_path = "training_output/" + str(now)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        os.chmod(output_path, 0o777)

    parser.add_argument("--savedir", type=str, default=output_path)
    # save_name
    parser.add_argument("--save_name", type=str, default='no_name', help='file name to save combination of daset and loaddir name')

    parser.add_argument("--loaddir", type=str, default='NONE')
    # MKP CXR-BERT
    # parser.add_argument("--loaddir", type=str, default="/home/workspace/Multi-modality-Self-supervision/pretrain_output/2022-07-08 14:35:32.870226/49") # knowledg4_pretrained 63epoch

    parser.add_argument("--openi", type=str2bool, default=False)


##########################
##########################
########################## 


    # parser.add_argument("--Train_dset0_name", type=str, default='frontal_train.jsonl',
    parser.add_argument("--Train_dset0_name", type=str, default='frontal_train_w_prev.jsonl',
    # parser.add_argument("--Train_dset0_name", type=str, default='frontal_first_train.jsonl',
                        help="train dset for mimic")
    # parser.add_argument("--Valid_dset0_name", type=str, default='frontal_val.jsonl',
    parser.add_argument("--Valid_dset0_name", type=str, default='frontal_val_w_prev.jsonl',
    # parser.add_argument("--Valid_dset0_name", type=str, default='frontal_first_val.jsonl',
                        help="valid dset for mimic")

    # parser.add_argument("--Train_dset1_name", type=str, default='error_baseline_EasyProblem/frontal_train_error.jsonl',
    # parser.add_argument("--Train_dset1_name", type=str, default='error_baseline_FactualOnly/frontal_train_error.jsonl',
    # parser.add_argument("--Train_dset1_name", type=str, default='error_baseline_FindingsRandomShuffle/frontal_train_error.jsonl',
    # parser.add_argument("--Train_dset1_name", type=str, default='error_baseline_FindingsRandomShuffle_First/frontal_train_error.jsonl',
    # parser.add_argument("--Train_dset1_name", type=str, default='error_baseline_ImpressionRandomShuffle/frontal_train_error.jsonl',
    # parser.add_argument("--Train_dset1_name", type=str, default='error_baseline_PerceptualOnly/frontal_train_error.jsonl',
    # parser.add_argument("--Train_dset1_name", type=str, default='error_baseline_Mixed_FPR/frontal_train_error.jsonl',
    # parser.add_argument("--Train_dset1_name", type=str, default='error_baseline_Mixed_FPI_v0.1/frontal_train_error.jsonl',
    # parser.add_argument("--Train_dset1_name", type=str, default='error_baseline_Mixed_FPI_v0.2/frontal_train_error.jsonl',
    # parser.add_argument("--Train_dset1_name", type=str, default='error_baseline_Mixed_FPI_v0.3/frontal_train_error_v1_to_v10.jsonl',
    # parser.add_argument("--Train_dset1_name", type=str, default='error_baseline_Mixed_FPI_v0.3/frontal_train_error_v1_to_v10_w_prev.jsonl',
    parser.add_argument("--Train_dset1_name", type=str, default='error_baseline_Mixed_FPI_v0.4/reference_dist/frontal_train_error_reference_dist_v1_to_v10_w_prev.jsonl',
    # parser.add_argument("--Train_dset1_name", type=str, default='error_baseline_Mixed_FPI_v0.3/uniform_dist/frontal_train_error_v1_to_v10.jsonl',
                        help="train dset for mimic")
    # parser.add_argument("--Valid_dset1_name", type=str, default='error_baseline_EasyProblem/frontal_val_error.jsonl',
    # parser.add_argument("--Valid_dset1_name", type=str, default='error_baseline_FactualOnly/frontal_val_error.jsonl',
    # parser.add_argument("--Valid_dset1_name", type=str, default='error_baseline_FindingsRandomShuffle/frontal_val_error.jsonl',
    # parser.add_argument("--Valid_dset1_name", type=str, default='error_baseline_FindingsRandomShuffle_First/frontal_val_error.jsonl',
    # parser.add_argument("--Valid_dset1_name", type=str, default='error_baseline_ImpressionRandomShuffle/frontal_val_error.jsonl',
    # parser.add_argument("--Valid_dset1_name", type=str, default='error_baseline_Mixed_FPR/frontal_val_error.jsonl',
    # parser.add_argument("--Valid_dset1_name", type=str, default='error_baseline_Mixed_FPI_v0.1/frontal_val_error.jsonl',
    # parser.add_argument("--Valid_dset1_name", type=str, default='error_baseline_Mixed_FPI_v0.2/frontal_val_error.jsonl',
    # parser.add_argument("--Valid_dset1_name", type=str, default='error_baseline_Mixed_FPI_v0.3/frontal_val_error_v1_to_v10.jsonl',
    # parser.add_argument("--Valid_dset1_name", type=str, default='error_baseline_Mixed_FPI_v0.3/frontal_val_error_v1_to_v10_w_prev.jsonl',
    parser.add_argument("--Valid_dset1_name", type=str, default='error_baseline_Mixed_FPI_v0.4/reference_dist/frontal_val_error_reference_dist_v1_to_v10_w_prev.jsonl',
    # parser.add_argument("--Valid_dset1_name", type=str, default='error_baseline_Mixed_FPI_v0.3/uniform_dist/frontal_val_error_v1_to_v10.jsonl',
                        help="valid dset for mimic")

    parser.add_argument("--dataset", type=str, default='mimic-cxr', choices=['mimic-cxr', 'indiana'],
                        help="mimic-cxr or indiana")
    parser.add_argument("--data_path", type=str, default='data/mimic-cxr-jpg/2.0.0/rred',
                        help="dset path for training")
    parser.add_argument("--data_dir_img", type=str, default='data/mimic-cxr-jpg/2.0.0/files',
                        help="dset path for training")

    parser.add_argument("--test_with_bootstrap", type=str2bool, default=False,
                        help="test with bootstrap")
    parser.add_argument("--make_error", type=str2bool, default=True,
                        help="make error?")
    parser.add_argument("--error_sampling_train", type=float, default=1,
                        help="make error with dinamic sampling?")
    parser.add_argument("--error_sampling_test", type=float, default=1,
                        help="make error with dinamic sampling?")

    parser.add_argument("--error_ids", type=list, default=[],
                        help="error ids")

##########################
##########################
##########################

    parser.add_argument("--JOINT_FEATURE_SIZE", type=int, default=128)
    
    parser.add_argument("--embed_sz", type=int, default=768, choices=[768])
    parser.add_argument("--hidden_sz", type=int, default=768, choices=[768])

    parser.add_argument("--bert_model", type=str, default="microsoft/BiomedVLP-CXR-BERT-specialized",
                        choices=["bert-base-uncased, emilyalsentzer/Bio_ClinicalBERT", "xlm-roberta-base", '/home/workspace/Multi-modality-Self-supervision/GatorTron', 'microsoft/BiomedVLP-CXR-BERT-specialized'])
    parser.add_argument("--init_model", type=str, default="microsoft/BiomedVLP-CXR-BERT-specialized",
                        choices=["bert-base-uncased", "xlm-roberta-base", 'microsoft/BiomedVLP-CXR-BERT-specialized'])

    ### image args ###
    parser.add_argument("--TRANSFORM_RESIZE", type=int, default=240) ## for ViT
    # parser.add_argument("--TRANSFORM_RESIZE", type=int, default=512)
    parser.add_argument("--TRANSFORM_CENTER_CROP_SIZE", type=int, default=224) ## for ViT
    # parser.add_argument("--TRANSFORM_CENTER_CROP_SIZE", type=int, default=480)

    parser.add_argument("--drop_img_percent", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--multimodal_model_type", type=str, default="flamingo", choices=["att_pool", "vlbert", 'flamingo', 'coca'])
    parser.add_argument("--image_model_type", type=str, default="vit", choices=["vit", 'resnet'])
    parser.add_argument("--multimodal_depth", type=int, default=1, choices=[1,2,4,8,12])
    parser.add_argument("--cross_attn_every", type=int, default=1, choices=[1,2,3,4])
    parser.add_argument("--cross_attn_order", type=str, default='single->cross', choices=['cross->single', 'single->cross'])
    parser.add_argument("--perceiver_depth", type=int, default=1, choices=[1,2,3,4,8])
    parser.add_argument("--perceiver_dim_head", type=int, default=64, choices=[64, 128, 256])
    parser.add_argument("--perceiver_num_head", type=int, default=8, choices=[8, 12, 24])
    parser.add_argument("--num_img_token", type=int, default=64, choices=[64, 128, 256])
    parser.add_argument("--max_num_img", type=int, default=2, choices=[2])
    parser.add_argument("--use_prev_img", type=str2bool, default=True)
    parser.add_argument("--use_prev_txt", type=str2bool, default=True)
    parser.add_argument("--img_to_each_perceiver", type=str2bool, default=False)

    parser.add_argument("--img_embed_pool_type", type=str, default="att_txt", choices=["biovil", "att_img", "att_txt"])
    parser.add_argument("--img_aug", type=str, default="all", choices=["affine", "colur", "hflip", "rrc", "all", "None"])
    parser.add_argument("--txt_aug", type=str, default="sentence_shuffling", choices=["sentence_shuffling","None"])
    

    parser.add_argument("--img_hidden_sz", type=int, default=2048)
    parser.add_argument("--include_bn", type=int, default=True)

    parser.add_argument("--freeze_img", type=int, default=0)
    parser.add_argument("--freeze_txt", type=int, default=0)
    parser.add_argument("--freeze_img_all", type=str2bool, default=True)
    parser.add_argument("--freeze_txt_all", type=str2bool, default=True)

    parser.add_argument("--hidden", nargs="*", type=int, default=[])
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_factor", type=float, default=0.5)
    parser.add_argument("--lr_patience", type=int, default=100)

    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--num_image_embeds", type=int, default=0)

    parser.add_argument("--warmup", type=float, default=0.1)
    parser.add_argument("--weight_classes", type=int, default=1)

    parser.add_argument("--inference", type=str2bool, default=False)
    parser.add_argument("--inference_method", type=str, default=None, choices=['batch','single', None])
    parser.add_argument("--test_dset_name", type=str, default='test_error_0509-k1000.jsonl',
                            help="valid dset for SNUH")
     

def get_criterion(args):
    if args.task_type == "multilabel":
        if args.weight_classes:
            freqs = [args.label_freqs[l] for l in args.labels]
            negative = [args.train_data_len - l for l in freqs]
            label_weights = (torch.FloatTensor(freqs) / torch.FloatTensor(negative)) ** -1
            criterion = nn.BCEWithLogitsLoss(pos_weight=label_weights.to(args.device))
        else:
            criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    return criterion


def get_optimizer(model, args):
    if args.model in ["mmbt"]:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
        
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    return optimizer


def get_scheduler(optimizer, args):
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "max", patience=args.lr_patience, verbose=True, factor=args.lr_factor
    )

def model_eval(i_epoch, data, model, args, criterion, store_preds=False):
    with torch.no_grad():
        losses, preds, preds_bool, tgts = [], [], [], []
        outAUROC = []

        for batch in data:
            loss, out, tgt = model_forward(model, args, criterion, batch)
            losses.append(loss.item())

            if args.task_type == "multilabel":
                pred_bool = torch.sigmoid(out).cpu().detach().numpy() > 0.5
                pred = torch.sigmoid(out).cpu().detach().numpy()
                preds_bool.append(pred_bool)
            else:
                pred = torch.nn.functional.softmax(out, dim=1).argmax(dim=1).cpu().detach().numpy()

            preds.append(pred)
            tgt = tgt.cpu().detach().numpy()
            tgts.append(tgt)

    metrics = {"loss": np.mean(losses)}
    classACC = dict()

    if args.task_type == "multilabel":
        tgts = np.vstack(tgts)
        preds = np.vstack(preds)
        preds_bool = np.vstack(preds_bool)

        for i in range(args.n_classes):
            try:
                outAUROC.append(roc_auc_score(tgts[:, i], preds[:, i]))
            except ValueError:
                outAUROC.append(0)
                pass

        for i in range(0, len(outAUROC)):
            assert args.n_classes == len(outAUROC)
            classACC[args.labels[i]] = outAUROC[i]

        metrics["micro_roc_auc"] = roc_auc_score(tgts, preds, average="micro")
        metrics["macro_roc_auc"] = roc_auc_score(tgts, preds, average="macro")
        metrics["macro_f1"] = f1_score(tgts, preds_bool, average="macro")
        metrics["micro_f1"] = f1_score(tgts, preds_bool, average="micro")

    else:
        tgts = [l for sl in tgts for l in sl]
        preds = [l for sl in preds for l in sl]
        metrics["acc"] = accuracy_score(tgts, preds)
        metrics["macro_f1"] = f1_score(tgts, preds, average="macro")
        metrics["micro_f1"] = f1_score(tgts, preds, average="micro")

        matrix = confusion_matrix(tgts, preds)
        print(matrix)
        classACC = matrix.diagonal()/matrix.sum(axis=1)

        print('Accuracy:', metrics["acc"])
    print('Class Accuracy:', classACC)
    print('macro_f1:', metrics["macro_f1"])
    print('micro_f1:', metrics["micro_f1"])
    print('-----------------------------------------------------')

    if store_preds:
        store_preds_to_disk(tgts, preds, args)

    return metrics, classACC, tgts, preds

def freeze_weight(model, args, i_epoch):
    m = model
    if torch.cuda.device_count() > 1 and args.use_ddp==False and args.device != 'cpu':
        m = model.module

    if args.multimodal_model_type == "vlbert":
        m = m.enc

    if i_epoch < args.freeze_img or args.freeze_img_all:
        m.image_model.eval()
        for param in m.image_model.parameters():
            param.requires_grad = False
        print('Freeze image parameters')
    else: 
        m.image_model.train()
        for param in m.image_model.parameters():
            param.requires_grad = True
        print('Unfreeze image parameters')

    if i_epoch < args.freeze_txt or args.freeze_txt_all:
        m.text_model.eval()
        for param in m.text_model.parameters():
            param.requires_grad = False
            
        if args.multimodal_model_type =="coca":
            m.text_embedding.eval()
            for param in m.text_embedding.parameters():
                param.requires_grad = False
        print('Freeze text parameters')

    else: 
        m.text_model.train()
        for param in m.text_model.parameters():
            param.requires_grad = True
            
        if args.multimodal_model_type =="coca":
            m.text_embedding.train()
            for param in m.text_embedding.parameters():
                param.requires_grad = True
        print('Unfreeze text parameters')



def model_forward(model, args, criterion, batch, compute_loss=True):
    findings, impression, img, tgt, prev_img, prev_findings = batch
    device = args.device

    findings = (findings[0].to(device), findings[1].to(device), findings[2].to(device))
    prev_findings = (prev_findings[0].to(device), prev_findings[1].to(device), prev_findings[2].to(device)) if args.use_prev_txt else None
    # impression = (impression[0].to(device), impression[1].to(device), impression[2].to(device))
    impression = None
    
    img = img.to(device)
    prev_img = prev_img.to(device) if args.use_prev_img else None

    out = model((findings, prev_findings), impression, (img, prev_img))

    tgt = tgt.to(device)
    loss = criterion(out, tgt) if compute_loss else None
    return loss, out, tgt

def train(args):
    wandb.init(config=args, project="rred2-biovil", entity="dabinmin", name=args.save_name)
    
    print("Training start!!")
    print(" # PID :", os.getpid())

    set_seed(args.seed)
    args.savedir = os.path.join(args.savedir, args.save_name)
    os.makedirs(args.savedir, exist_ok=True)

    train_dataset, val_dataset = get_dataset(args)
    train_loader, val_loader = get_data_loaders(args, train_dataset, val_dataset)

    model = get_model(args)
    criterion = get_criterion(args)
    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args)

    logger = create_logger("%s/logfile.log" % args.savedir, args)
    torch.save(args, os.path.join(args.savedir, "args.bin"))

    start_epoch, global_step, n_no_improve, best_metric = 0, 0, 0, np.inf

    # print("freeze image?", args.freeze_img_all)
    # print("freeze txt?", args.freeze_txt_all)
    model.to(args.device)
    logger.info("Training..")

    flag_data_parallel = False
    if torch.cuda.device_count() > 1 and args.use_ddp==False and args.device != 'cpu':
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
        flag_data_parallel = True

    if os.path.exists(os.path.join(args.loaddir, "pytorch_model.bin")):
        model.load_state_dict(torch.load(args.loaddir + "/pytorch_model.bin"), strict=True)

        print("This would load the trained model.bin, then fine-tune the model.")
    elif os.path.exists(os.path.join(args.loaddir, "model_best.pt")):
        model.load_state_dict(torch.load(args.loaddir + "/model_best.pt")['state_dict'], strict=True)

        print("This would load the trained model.pt, then fine-tune the model.")
    
    elif os.path.exists(args.loaddir):
        model.load_state_dict(torch.load(args.loaddir)['state_dict'], strict=True)

        print("This would load the trained model.pt, then fine-tune the model.")

    else:
        print("")
        print("")
        print("this option initilize the model with random value. train from scratch.")
        print("Loaded model : ")

    wandb.watch(model,criterion, log="all")

    for i_epoch in range(start_epoch, args.max_epochs):
        if args.error_sampling_train != 0:
            train_dataset.sample_error()
            train_loader, _ = get_data_loaders(args, train_dataset, val_dataset)

        train_losses = []
        print_step = 100*args.gradient_accumulation_steps
        loss_sum = 0

        model.train()
        optimizer.zero_grad()

        if args.multimodal_model_type == "flamingo":
            log_tanh_gating(model,args)
            if not args.freeze_img_all:
                if flag_data_parallel:
                    model.module.unfreeze_image_model()
                else: 
                    model.unfreeze_image_model()
        if args.multimodal_model_type not in ["flamingo"]:
            freeze_weight(model, args, i_epoch) 

        for step, batch in enumerate(tqdm(train_loader, total=len(train_loader))):

            loss, out, target = model_forward(model, args, criterion, batch)
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss_sum += loss.item()
            if step % print_step == 0 and step!=0:
                print(f'avg_train_loss for {print_step} steps: {(loss_sum*args.gradient_accumulation_steps)/print_step}')
                loss_sum = 0
                if args.multimodal_model_type == "flamingo":
                    log_tanh_gating(model, args)

            train_losses.append(loss.item())
            loss.backward()
            global_step += 1
            if global_step % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        model.eval()
        print("epoch ",i_epoch+1," is done. now evaluating..")
        metrics, classACC, tgts, preds = model_eval(i_epoch, val_loader, model, args, criterion)
        logger.info("Train Loss: {:.4f}".format(np.mean(train_losses)))
        log_metrics("Val", metrics,  args, logger)

        tuning_metric = (
            metrics["micro_f1"] if args.task_type == "multilabel" else metrics["loss"]
        )
        scheduler.step(tuning_metric)
        is_improvement = tuning_metric < best_metric 
        if is_improvement:
            best_metric = tuning_metric
            n_no_improve = 0
        else:
            n_no_improve += 1

        if  args.task_type == "multilabel":
            wandb.log({
                "micro_roc_auc": metrics["micro_roc_auc"],
                "macro_roc_auc": metrics["macro_roc_auc"],
                "macro_f1 f1 scroe": metrics["macro_f1"],
                "micro f1 score": metrics["micro_f1"],
                "train loss": np.mean(train_losses),
                "val loss": metrics["loss"],
                "class accuracy 0": classACC[0],  # AUC per class
                "class accuracy 1": classACC[1],  # AUC per class
                "class accuracy 2": classACC[2],  # AUC per class
                "class accuracy 3": classACC[3],  # AUC per class
            })

            csv_save_name = args.save_name
            save_path = args.savedir + '/' + csv_save_name + '.csv'
            f = open(save_path, 'w', encoding='utf-8')
            wr = csv.writer(f)
            key = list(classACC.keys())
            val = list(classACC.values())
            title = ['micro_auc', 'macro_auc', 'micro_f1', 'macro_f1'] + key
            result = [metrics["micro_roc_auc"], metrics["macro_roc_auc"], metrics["micro_f1"], metrics["macro_f1"]] + val

        elif  args.task_type == "classification":
            wandb.log({
                "Macro_f1 f1 scroe": metrics["macro_f1"],
                "Micro f1 score": metrics["micro_f1"],
                "Train loss": np.mean(train_losses),
                "Val loss": metrics["loss"],
                "Accuracy": metrics["acc"],  # AUC per class
                "class accuracy 0": classACC[0],  # AUC per class
                "class accuracy 1": classACC[1],  # AUC per class
                "class accuracy 2": classACC[2],  # AUC per class
                "class accuracy 3": classACC[3],  # AUC per class
            })

            csv_save_name = args.save_name
            save_path = args.savedir + '/' + csv_save_name + '.csv'
            f = open(save_path, 'w', encoding='utf-8')
            wr = csv.writer(f)
            key = ['class_0', 'class_1', 'class_2', 'class_3']
            val = list(classACC)
            title = ['Accuracy', 'micro_f1', 'macro_f1'] + key
            result = [metrics["acc"], metrics["micro_f1"], metrics["macro_f1"]] + val
            
        elif  args.task_type == "binary":
            wandb.log({
                "Macro_f1 f1 scroe": metrics["macro_f1"],
                "Micro f1 score": metrics["micro_f1"],
                "Train loss": np.mean(train_losses),
                "Val loss": metrics["loss"],
                "Accuracy": metrics["acc"],  # AUC per class
            })

            csv_save_name = args.save_name
            save_path = args.savedir + '/' + csv_save_name + '.csv'
            f = open(save_path, 'w', encoding='utf-8')
            wr = csv.writer(f) 
            title = ['Accuracy', 'micro_f1', 'macro_f1']
            result = [metrics["acc"], metrics["micro_f1"], metrics["macro_f1"]]

        wr.writerow(title)
        wr.writerow(result)
        f.close()

        # if torch.cuda.device_count()==1 or args.rank == 0:
        if 1:
            save_checkpoint(
                {
                    "epoch": i_epoch + 1,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "n_no_improve": n_no_improve,
                    "best_metric": best_metric,
                },
                is_improvement,
                args.savedir,
            )

        if n_no_improve >= args.patience:
            logger.info("No improvement. Breaking out of loop.")
            break


def test(args):

    print("Model Test")
    print(" # PID :", os.getpid())
    print('log:', args.Valid_dset_name)
    set_seed(args.seed)
    args.savedir = os.path.join(args.savedir, os.name)
    os.makedirs(args.savedir, exist_ok=True)

    train_loader, val_loader = get_data_loaders(args)
    device = args.device
    model = get_model(args)

    criterion = get_criterion(args, device)

    torch.save(args, os.path.join(args.savedir, "args.bin"))


    if os.path.exists(os.path.join(args.loaddir, "model_best.pt")):
        model.load_state_dict(torch.load(args.loaddir + "/model_best.pt"), strict=False)
        model.expand_token_type_embeddings()

    else:
        print("")
        print("")
        print("this option initilize the model with random value. train from scratch.")
        print("Loaded model : ")

    print("freeze image?", args.freeze_img_all)
    print("freeze txt?", args.freeze_txt_all)
    model.to(device)

    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     model = nn.DataParallel(model)

    load_checkpoint(model, os.path.join(args.loaddir, "model_best.pt"))

    model.eval()
    metrics, classACC, tgts, preds  = model_eval(val_loader, model, args, criterion, device, store_preds=True)

    print('micro_roc_auc:', round(metrics["micro_roc_auc"], 3))
    print('macro_roc_auc:', round(metrics["macro_roc_auc"], 3))
    print('macro_f1 f1 scroe:', round(metrics["macro_f1"], 3))
    print('micro f1 score:', round(metrics["micro_f1"], 3))
    for i in classACC:
        print(i, round(classACC[i], 3))


def cli_main():

    parser = argparse.ArgumentParser(description="Train Models")
    get_args(parser)
    args, remaining_args = parser.parse_known_args()
    assert remaining_args == [], remaining_args

    print('=========INFO==========')
    print('loaddir:', args.loaddir)
    print('openi:', args.openi)
    print('data_path:', args.data_path)
    print('========================')

    train(args)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    cli_main()
