import os
from random import choices
import pandas as pd

from transformers.utils.dummy_tf_objects import WarmUp
from transformers import AutoTokenizer, AutoModel

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
import csv
import argparse
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, confusion_matrix
from tqdm import tqdm
from datetime import datetime

import torch.nn as nn
import torch.optim as optim
# from pytorch_pretrained_bert import BertAdam

from helpers import get_data_loaders
from model import VLModelClf
import sys
sys.path.insert(1, '/home/workspace/source/utils')
from utils import *
from logger import create_logger
import wandb

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler

# ###
# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# ####

def get_args(parser):

    parser.add_argument("--seed", type=int, default=1125)
    parser.add_argument("--batch_sz", type=int, default=32)
    parser.add_argument("--max_epochs", type=int, default=5)

    parser.add_argument("--model", type=str, default="cxr-bert", choices=['mmbt', 'bert', 'clinicalbert', 'roberta', 'gatortron', 'cxr-bert'])
    parser.add_argument("--task_type", type=str, default="binary", choices=["multilabel", "classification", "binary"])

    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--use_ddp", type=bool, default=False)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--n_workers", type=int, default=32)
    parser.add_argument("--patience", type=int, default=100)

    now = datetime.now()
    now = now.strftime('%Y-%m-%d')
    output_path = "workspace/source/downstream/training_output" + str(now)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
        os.chmod(output_path, 0o777)

    parser.add_argument("--savedir", type=str, default=output_path)
    # save_name
    parser.add_argument("--save_name", type=str, default='no_name', help='file name to save combination of daset and loaddir name')

    parser.add_argument("--loaddir", type=str, default='NONE')
    # MKP CXR-BERT
    # parser.add_argument("--loaddir", type=str, default="/home/workspace/Multi-modality-Self-supervision/pretrain_output/2022-07-08 14:35:32.870226/49") # knowledg4_pretrained 63epoch

    parser.add_argument("--openi", type=bool, default=False)


##########################
##########################
########################## 


    parser.add_argument("--Train_dset0_name", type=str, default='frontal_train.jsonl',
                        help="train dset for mimic")
    parser.add_argument("--Valid_dset0_name", type=str, default='frontal_val.jsonl',
                        help="valid dset for mimic")
    parser.add_argument("--Train_dset1_name", type=str, default='error_baseline_FindingsRandomShuffle/frontal_train_error.jsonl',
    # parser.add_argument("--Train_dset1_name", type=str, default='error_baseline_ImpressionRandomShuffle/frontal_train_error.jsonl',
                        help="train dset for mimic")
    parser.add_argument("--Valid_dset1_name", type=str, default='error_baseline_FindingsRandomShuffle/frontal_val_error.jsonl',
    # parser.add_argument("--Valid_dset1_name", type=str, default='error_baseline_ImpressionRandomShuffle/frontal_val_error.jsonl',
                        help="valid dset for mimic")

    parser.add_argument("--dataset", type=str, default='mimic-cxr', choices=['mimic-cxr', 'indiana'],
                        help="mimic-cxr or indiana")
    parser.add_argument("--data_path", type=str, default='data/mimic-cxr-jpg/2.0.0/rred',
                        help="dset path for training")
    parser.add_argument("--data_dir_img", type=str, default='data/mimic-cxr-jpg/2.0.0/files',
                        help="dset path for training")

    parser.add_argument("--test_with_bootstrap", type=bool, default=False,
                        help="test with bootstrap")
    parser.add_argument("--make_error", type=bool, default=True,
                        help="make error?")
    parser.add_argument("--error_sampling", type=int, default=1,
                        help="make error with dinamic sampling?")

    parser.add_argument("--error_ids", type=list, default=[],
                        help="error ids")

##########################
##########################
##########################

    parser.add_argument("--JOINT_FEATURE_SIZE", type=int, default=128)
    
    
    parser.add_argument("--embed_sz", type=int, default=768, choices=[768])
    parser.add_argument("--hidden_sz", type=int, default=768, choices=[768])

    # parser.add_argument("--bert_model", type=str, default="xlm-roberta-base",
    parser.add_argument("--bert_model", type=str, default="microsoft/BiomedVLP-CXR-BERT-specialized",
                        choices=["bert-base-uncased, emilyalsentzer/Bio_ClinicalBERT", "xlm-roberta-base", '/home/workspace/Multi-modality-Self-supervision/GatorTron', 'microsoft/BiomedVLP-CXR-BERT-specialized'])
    # parser.add_argument("--init_model", type=str, default="xlm-roberta-base",
    parser.add_argument("--init_model", type=str, default="microsoft/BiomedVLP-CXR-BERT-specialized",
                        choices=["bert-base-uncased", "xlm-roberta-base", 'microsoft/BiomedVLP-CXR-BERT-specialized'])


    ### image args ###
    parser.add_argument("--TRANSFORM_RESIZE", type=int, default=512)
    parser.add_argument("--TRANSFORM_CENTER_CROP_SIZE", type=int, default=480)

    parser.add_argument("--drop_img_percent", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--freeze_img", type=int, default=0)
    parser.add_argument("--freeze_txt", type=int, default=0)

    parser.add_argument("--freeze_img_all", type=str, default=False)
    parser.add_argument("--freeze_txt_all", type=str, default=False)

    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--hidden", nargs="*", type=int, default=[])

    parser.add_argument("--img_embed_pool_type", type=str, default="max", choices=["max", "avg"])
    parser.add_argument("--img_hidden_sz", type=int, default=2048)
    parser.add_argument("--include_bn", type=int, default=True)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_factor", type=float, default=0.5)
    parser.add_argument("--lr_patience", type=int, default=100)

    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--num_image_embeds", type=int, default=0)

    parser.add_argument("--warmup", type=float, default=0.1)
    parser.add_argument("--weight_classes", type=int, default=1)

    parser.add_argument("--inference", type=bool, default=False)
    parser.add_argument("--inference_method", type=str, default=None, choices=['batch','single'])
    parser.add_argument("--test_dset_name", type=str, default='test_error_0509-k1000.jsonl',
                            help="valid dset for SNUH")

def get_criterion(args, device):
    if args.task_type == "multilabel":
        if args.weight_classes:
            freqs = [args.label_freqs[l] for l in args.labels]
            negative = [args.train_data_len - l for l in freqs]
            label_weights = (torch.FloatTensor(freqs) / torch.FloatTensor(negative)) ** -1
            criterion = nn.BCEWithLogitsLoss(pos_weight=label_weights.to(device))
        else:
            criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    return criterion


def get_optimizer(model, args):
    if args.model in ["mmbt"]:
    #     total_steps = (
    #             args.train_data_len
    #             / args.batch_sz
    #             / args.gradient_accumulation_steps
    #             * args.max_epochs
    #     )
    #     param_optimizer = list(model.named_parameters())
    #     no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    #     optimizer_grouped_parameters = [
    #         {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
    #         {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0, },
    #     ]
    #     optimizer = BertAdam(
    #         optimizer_grouped_parameters,
    #         lr=args.lr,
    #         warmup=args.warmup,
    #         t_total=total_steps,
    #     )
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
        
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    return optimizer


def get_scheduler(optimizer, args):
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "max", patience=args.lr_patience, verbose=True, factor=args.lr_factor
    )

def model_eval(i_epoch, data, model, args, criterion, device, store_preds=False):
    with torch.no_grad():
        losses, preds, preds_bool, tgts = [], [], [], []
        outAUROC = []

        for batch in data:
            loss, out, tgt = model_forward(model, args, criterion, batch, device, i_epoch)
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


def model_forward(model, args, criterion, batch, device, i_epoch):
    findings, impression, img, tgt = batch

    # freeze_img = True if args.freeze_img_all else i_epoch < args.freeze_img
    # print(f'epoch{i_epoch}, freeze_img: {freeze_img}')
    # freeze_txt = True if args.freeze_txt_all else i_epoch < args.freeze_txt
    # print(f'epoch{i_epoch}, freeze_img: {freeze_txt}')


    # printflag_img = True
    # printflag_txt = True

    # if args.num_image_embeds > 0:
        
    #     for param in model.enc.img_encoder.parameters():
    #         param.requires_grad = not freeze_img

    #         if printflag_img: 
    #             print(f"enc.img_encoder freezed?: {freeze_img}, epoch: {i_epoch}")
    #             printflag_img=False
    
    # for param in model.enc.encoder.parameters():
    #     param.requires_grad = not freeze_txt
    #     if printflag_txt: 
    #         print(f"enc.encoder freezed?: {freeze_txt}, epoch: {i_epoch}")
    #         printflag_txt=False

            ####################################### tmp
    # if i_epoch < args.freeze_img:
    #     for param in model.enc.img_encoder.parameters():
    #         param.requires_grad = False
    # else: 
    #     for param in model.enc.img_encoder.parameters():
    #         param.requires_grad = True
    if torch.cuda.device_count() > 1 and args.use_ddp==False and device != torch.device('cpu'):
        if i_epoch < args.freeze_img or args.freeze_img_all:
            for param in model.module.image_model.parameters():
                param.requires_grad = False
        else: 
            for param in model.module.image_model.parameters():
                param.requires_grad = True

        if i_epoch < args.freeze_txt or args.freeze_txt_all:
            for param in model.module.text_model.parameters():
                param.requires_grad = False
        else: 
            for param in model.module.text_model.parameters():
                param.requires_grad = True
        
    else:
        if i_epoch < args.freeze_img or args.freeze_img_all:
            for param in model.image_model.parameters():
                param.requires_grad = False
        else: 
            for param in model.image_model.parameters():
                param.requires_grad = True

        if i_epoch < args.freeze_txt or args.freeze_txt_all:
            for param in model.text_model.parameters():
                param.requires_grad = False
        else: 
            for param in model.text_model.parameters():
                param.requires_grad = True

    findings = (findings[0].to(device), findings[1].to(device), findings[2].to(device))
    impression = (impression[0].to(device), impression[1].to(device), impression[2].to(device))
    img = img.to(device)

    out = model(findings, impression, img)

    tgt = tgt.to(device)
    loss = criterion(out, tgt)
    return loss, out, tgt

def train(args):
    wandb.init(config=args, project="rred2-biovil", entity="dabinmin", name=args.save_name)
    
    print("Training start!!")
    print(" # PID :", os.getpid())

    set_seed(args.seed)
    args.savedir = os.path.join(args.savedir, args.save_name)
    os.makedirs(args.savedir, exist_ok=True)

    train_loader, val_loader = get_data_loaders(args)
    # batch = next(iter(train_loader))

    device = torch.device("cuda" if torch.cuda.is_available() and args.device!='cpu' else "cpu")
    # device = torch.device("cpu")
    args.device = device

    model = VLModelClf(args) 

    criterion = get_criterion(args, device)
    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args)

    logger = create_logger("%s/logfile.log" % args.savedir, args)
    torch.save(args, os.path.join(args.savedir, "args.bin"))

    start_epoch, global_step, n_no_improve, best_metric = 0, 0, 0, np.inf

    if os.path.exists(os.path.join(args.loaddir, "pytorch_model.bin")):
        model.load_state_dict(torch.load(args.loaddir + "/pytorch_model.bin"), strict=False)
        # model.expand_token_type_embeddings()

        print("This would load the trained model.bin, then fine-tune the model.")
    
    elif os.path.exists(args.loaddir):
        model.load_state_dict(torch.load(args.loaddir)['state_dict'], strict=True)
        # model.expand_token_type_embeddings()

        print("This would load the trained model.pt, then fine-tune the model.")

    else:
        print("")
        print("")
        print("this option initilize the model with random value. train from scratch.")
        print("Loaded model : ")



    # print("freeze image?", args.freeze_img_all)
    # print("freeze txt?", args.freeze_txt_all)
    model.to(device)
    logger.info("Training..")

    if torch.cuda.device_count() > 1 and args.use_ddp==False and device != torch.device('cpu'):
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    wandb.watch(model,criterion, log="all")

    for i_epoch in range(start_epoch, args.max_epochs):
        train_losses = []

        model.train()
        optimizer.zero_grad()
        # lr = optimizer.get_lr
        for step, batch in enumerate(tqdm(train_loader, total=len(train_loader))):
            loss, out, target = model_forward(model, args, criterion, batch, device,i_epoch)
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            if step % 10 == 0:
                print(f'train_loss: {loss.item()}')

            train_losses.append(loss.item())
            loss.backward()
            global_step += 1
            if global_step % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        model.eval()
        print("epoch ",i_epoch+1," is done. now evaluating..")
        metrics, classACC, tgts, preds = model_eval(i_epoch, val_loader, model, args, criterion, device)
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
