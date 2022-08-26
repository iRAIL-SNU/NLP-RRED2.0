import os
import pandas as pd

from transformers.utils.dummy_tf_objects import WarmUp

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"

import csv
import argparse
from sklearn.metrics import RocCurveDisplay, accuracy_score, confusion_matrix, plot_roc_curve, precision_recall_curve, PrecisionRecallDisplay, average_precision_score, RocCurveDisplay, auc, roc_curve
from tqdm import tqdm
from datetime import datetime

import torch.nn as nn
import torch.optim as optim

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
from matplotlib import pyplot as plt


def model_eval(args, data):
    device = args.device
    with torch.no_grad():
        preds, tgts, total_outs = [], [], []
        step = 0
        for batch in tqdm(data):
            findings, impression, img, tgt = batch
            step += 1

            # #####################
            # if step > 100:
            #     print('step 100 중지 '*10)
            #     break
            # #####################

            findings = (findings[0].to(device), findings[1].to(device), findings[2].to(device))
            impression = (impression[0].to(device), impression[1].to(device), impression[2].to(device))
            img = img.to(device)

            total_out = torch.zeros([args.batch_sz,2]).to(device)
            single_outs = []

            out = model(findings, impression, img)

            tgt = tgt.to(device)
            tgt = tgt.cpu().detach().numpy()

            pred = torch.nn.functional.softmax(out, dim=1).argmax(dim=1).cpu().detach().numpy()

            preds.append(pred)
            total_outs.append(out)
            tgts.append(tgt)

    return tgts, preds, total_outs



def cal_performance(tgts, total_outs, threshold=0.5, resultdir=None):
    preds = []
    preds_prob = []
    for total_out in total_outs:
        preds.append(torch.nn.functional.softmax(total_out, dim=1)[:,1].cpu().detach().numpy() > threshold)
        preds_prob.append(torch.nn.functional.softmax(total_out, dim=1)[:,1].cpu().detach().numpy())

    metrics = {}

    tgts = [l for sl in tgts for l in sl]
    preds = [l for sl in preds for l in sl]
    preds_prob = [l for sl in preds_prob for l in sl]

    # new_single_preds=[[] for _ in self.model_name_list]
    # for i, p in enumerate(single_preds):
    #     for j, pp in enumerate(p):
    #         new_single_preds[j].append(pp)

    # for i, p in enumerate(new_single_preds):
    #     new_single_preds[i] = [l for sl in p for l in sl]

    metrics["acc"] = accuracy_score(tgts, preds)
    # metrics["single_accs"] = [accuracy_score(tgts, p) for p in new_single_preds]

    matrix = confusion_matrix(tgts, preds)
    print(matrix)
    # classACC = matrix.diagonal()/matrix.sum(axis=1)
    precisions, recalls, thresholds = precision_recall_curve(tgts, preds_prob)
    pr_display = PrecisionRecallDisplay(precisions, recalls).plot()
    pr_display.plot()
    # plt.savefig(os.path.join(resultdir,'PRCurve_mimic_human.png'), dpi=300)

    plt.cla()

    fpr, tpr, thresholds_roc = roc_curve(tgts, preds_prob)
    # roc_display = RocCurveDisplay(fpr, tpr)
    # plt.savefig(os.path.join(resultdir,'ROC_mimic_human.png'), dpi=300)

    import pickle
    # with open('mimic-human_prc_plot.pkl', 'wb') as f:
    #     pickle.dump(pr_display, f)

    metrics["ppv(precision)"] = matrix[1,1]/(matrix[0,1]+matrix[1,1])
    metrics["npv"] = matrix[0,0]/(matrix[0,0]+matrix[1,0])
    metrics["sensitivity(recall)"] = matrix[1,1]/(matrix[1,0]+matrix[1,1])
    metrics["specificity"] = matrix[0,0]/(matrix[0,0]+matrix[0,1])
    metrics["PRAUC"] = average_precision_score(tgts, preds_prob)
    metrics["AUROC"] = auc(fpr, tpr)

    print('Accuracy:', metrics["acc"])
    print(metrics)
    # print('Single Accuracy:', metrics["single_accs"])
    print('-----------------------------------------------------')

    return tgts, preds, metrics, matrix, preds_prob


def get_args(parser):

    parser.add_argument("--seed", type=int, default=1125)
    parser.add_argument("--batch_sz", type=int, default=16)

    parser.add_argument("--model", type=str, default="cxr-bert", choices=['mmbt', 'bert', 'clinicalbert', 'roberta', 'cxr-bert'])
    parser.add_argument("--task_type", type=str, default="binary", choices=["multilabel", "classification", "binary"])

    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--n_workers", type=int, default=16)
    parser.add_argument("--patience", type=int, default=2)

    parser.add_argument("--loaddir", type=str, default=
    # 'workspace/source/downstream/training_output2022-08-22/no_name/model_best.pt'
    # 'workspace/source/downstream/training_output2022-08-23/factualOnly_freeze/model_best.pt'
    # 'workspace/source/downstream/training_output2022-08-25/RandomShuffle_dropna_poolAttTxt_freeze/model_best.pt'
    # 'workspace/source/downstream/training_output2022-08-25/FactualOnly_dropna_poolAttTxt_freeze/model_best.pt'
    'workspace/source/downstream/training_output2022-08-25/RandomShuffle_dropna_poolAttTxt_freeze_augImgTxt/model_best.pt'
    )
    parser.add_argument("--resultdir", type=str, default='workspace/inference_result')


##########################
##########################
########################## 

    parser.add_argument("--Valid_dset0_name", type=str, default='frontal_test.jsonl',
                        help="valid dset for mimic")

    parser.add_argument("--dataset", type=str, default='mimic-cxr', choices=['mimic-cxr', 'indiana'],
                    help="mimic-cxr or indiana")
    parser.add_argument("--openi", type=bool, default=False)
    parser.add_argument("--data_path", type=str, default='data/mimic-cxr-jpg/2.0.0/rred',
                        help="dset path for training")
    parser.add_argument("--data_dir_img", type=str, default='data/mimic-cxr-jpg/2.0.0/files',
                        help="dset path for training")

    parser.add_argument("--test_with_bootstrap", type=bool, default=False,
                        help="test with bootstrap")
    parser.add_argument("--make_error", type=bool, default=False,
                        help="make error?")
    parser.add_argument("--error_sampling", type=int, default=0,
                        help="make error with dinamic sampling?")
    parser.add_argument("--clean_data", type=bool, default=False,
                        help="clean data?")
##########################
##########################
##########################

    parser.add_argument("--JOINT_FEATURE_SIZE", type=int, default=128)

    parser.add_argument("--embed_sz", type=int, default=768, choices=[768])
    parser.add_argument("--hidden_sz", type=int, default=768, choices=[768])

    parser.add_argument("--bert_model", type=str, default="microsoft/BiomedVLP-CXR-BERT-specialized",
                        choices=["bert-base-uncased, emilyalsentzer/Bio_ClinicalBERT", "xlm-roberta-base", "microsoft/BiomedVLP-CXR-BERT-specialized"])
    parser.add_argument("--init_model", type=str, default="microsoft/BiomedVLP-CXR-BERT-specialized",
                        choices=["bert-base-uncased", "xlm-roberta-base", 'microsoft/BiomedVLP-CXR-BERT-specialized'])

    parser.add_argument("--TRANSFORM_RESIZE", type=int, default=512)
    parser.add_argument("--TRANSFORM_CENTER_CROP_SIZE", type=int, default=480)

    parser.add_argument("--multimodal_model_type", type=str, default="att_pool", choices=["att_pool", "transformer"])
    parser.add_argument("--img_embed_pool_type", type=str, default="att_txt", choices=["biovil", "att_img", "att_txt"])

    parser.add_argument("--hidden", nargs="*", type=int, default=[])
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--inference", type=bool, default=True)
    parser.add_argument("--inference_method", type=str, default='batch', choices=['batch','single', None])


if __name__=='__main__':

    parser = argparse.ArgumentParser(description="Evaluating Model")
    get_args(parser)
    args, remaining_args = parser.parse_known_args()

    model_setting_name = args.loaddir.split('/')[-2]

    print("Model Test")
    print(" # PID :", os.getpid())

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # args.device = device
    # device = "cpu" 
    # print('device is cpu'*30)

    args.n_classes = 2
    
    model = VLModelClf(args)
    if torch.cuda.device_count() > 1 and args.device != torch.device('cpu'):
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args.loaddir)['state_dict'], strict=True)
    model.to(args.device)
    model.eval()

    test_iter = 10 if args.test_with_bootstrap else 1
    print(f"Number of testing:{test_iter}")
    metrics_li = []

    for test_i in range(test_iter):
        seed = test_i if args.test_with_bootstrap else args.seed
        set_seed(seed)

        val_loader = get_data_loaders(args)
        tgt, _, total_outs  = model_eval(args, val_loader)
        tgts, preds, metrics, matrix, probs = cal_performance(tgt, total_outs, threshold=0.8, resultdir=args.resultdir)

        metrics_li.append(metrics)

    if args.test_with_bootstrap:
        metrics_avg = metrics.copy()
        metrics_std = metrics.copy()
        metrics_lst = metrics.copy()

        for k in metrics.keys():
            v_li=[]
            for mtr in metrics_li:
                v_li.append(mtr[k])
            metrics_avg[k] = np.mean(v_li)
            metrics_std[k] = np.std(v_li)
            metrics_lst[k] = v_li

        print('##########################')
        print(metrics_avg)
        print(metrics_std)
        print('##########################')
        # pd.DataFrame(metrics_lst).to_csv(f'{n}.csv', header = True, index=False)


    is_correct = np.array(tgts) == np.array(preds)
    wrong_idx = np.where(is_correct==False)[0]

    true1 = np.where(np.array(tgts)==1)[0]
    pred1 = np.where(np.array(preds)==1)[0]
    true0 = np.where(np.array(tgts)==0)[0]
    pred0 = np.where(np.array(preds)==0)[0]


    tp_idx = [p for p in pred1 if p in true1 ]
    fp_idx = [p for p in pred1 if p in true0 ]
    fn_idx = [p for p in pred0 if p in true1 ]


    import json
    data = [json.loads(l) for l in open(os.path.join(args.data_path, args.Valid_dset0_name))]
    data = pd.DataFrame(data)

    fp_dataframe = pd.DataFrame()
    fp_dataframe['idx'] = fp_idx

    fp_dataframe['dicom_id'] = [d for d in data.loc[fp_idx]['dicom_id']]
    fp_dataframe['study_id'] = [d for d in data.loc[fp_idx]['study_id']]
    fp_dataframe['subject_id'] = [d for d in data.loc[fp_idx]['subject_id']]
    fp_dataframe['findings'] = [d for d in data.loc[fp_idx]['Findings']]
    fp_dataframe['impression'] = [d for d in data.loc[fp_idx]['Impression']]
    fp_dataframe['p(error)'] = np.array(probs)[fp_idx]
    # fp_dataframe['background'] = [d for d in data.loc[fp_idx]['background']]
    fp_dataframe=fp_dataframe.sort_values('p(error)', ascending=False)


    import pytz
    now = datetime.now(tz=pytz.timezone('Asia/Tokyo'))
    now = now.strftime("%y%m%d-%H%M%S")
    output_path = os.path.join(args.resultdir, str(now)+'_'+model_setting_name)
    output_img_path = os.path.join(output_path,'images')
    if not os.path.exists(output_path):
        os.mkdir(output_path)
        os.chmod(output_path, 0o777)
        os.mkdir(output_img_path)
        os.chmod(output_img_path, 0o777)
    print('output_path: ', output_path)


    import shutil
    import openpyxl
    from openpyxl.styles import Alignment

    ### Save False positive example file
    filename = 'False_Positive_Examples_rred2.xlsx'

    # fp_dataframe.to_csv('False_Positive_Examples_rred2_0823.csv',header=True, index=False, encoding='utf-8-sig')
    fp_dataframe.to_excel(os.path.join(output_path, filename),header=True, index=False, encoding='utf-8-sig')

    wb = openpyxl.load_workbook(os.path.join(output_path, filename))
    sheet = wb.active

    for i in range(len(fp_dataframe)):
        image_path = make_image_path(fp_dataframe.iloc[i], base_dir=args.data_dir_img, dataset='mimic-cxr')

        shutil.copyfile(image_path, os.path.join(output_img_path, fp_dataframe.iloc[i]['dicom_id']+'.jpg'))
        
        dicom_id = sheet["B"][i+1].value
        print(dicom_id)
        sheet["B"][i+1].value = '=HYPERLINK("{}")'.format(f'images/{fp_dataframe.iloc[i]["dicom_id"]}'+'.jpg')

    sheet.column_dimensions["E"].width = 100
    sheet.column_dimensions["F"].width = 50
    wb.save(os.path.join(output_path, filename))






    data = pd.read_csv('/home/workspace/Multi-modality-Self-supervision/data/mimic-cxr/mimic_test_with_human_error.csv')

    error_types = ['1-A', '1-B', '1-C', '1-D', '2-A', '2-B', '3']
    error_detection_rates = {}
    for type in error_types:
        total = sum(data.iloc[fn_idx]['human_error_type']==type) + sum(data.iloc[tp_idx]['human_error_type']==type)
        rate = sum(data.iloc[tp_idx]['human_error_type']==type) / total

        error_detection_rates[type] = rate
    print(error_detection_rates)
    print('end')