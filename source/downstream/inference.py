import os
import pandas as pd

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"

import argparse
from sklearn.metrics import RocCurveDisplay, accuracy_score, confusion_matrix, plot_roc_curve, precision_recall_curve, PrecisionRecallDisplay, average_precision_score, RocCurveDisplay, auc, roc_curve
from tqdm import tqdm

import torch.nn as nn

from helpers import get_data_loaders, get_model, get_dataset
from utils.utils import *
from utils.logger import create_logger

import torch
from matplotlib import pyplot as plt

import json
import shutil
import openpyxl
import pytz
from datetime import datetime
from main import model_forward


def model_eval(args, data):
    device = args.device
    with torch.no_grad():
        preds, tgts, total_outs = [], [], []
        for batch in tqdm(data):
            _, out, tgt = model_forward(model, args, criterion=None, batch=batch, compute_loss=False)
            
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

    parser.add_argument("--device", type=str, default='cuda', choices=["cuda", 'cpu'])
    parser.add_argument("--n_workers", type=int, default=16)
    parser.add_argument("--patience", type=int, default=2)

    parser.add_argument("--loaddir", type=str, default=
    # 'workspace/source/downstream/training_output2022-08-22/no_name/model_best.pt'
    # 'workspace/source/downstream/training_output2022-08-23/factualOnly_freeze/model_best.pt'
    # 'workspace/source/downstream/training_output2022-08-25/RandomShuffle_dropna_poolAttTxt_freeze/model_best.pt'
    # 'workspace/source/downstream/training_output2022-08-25/FactualOnly_dropna_poolAttTxt_freeze/model_best.pt'
    # 'workspace/source/downstream/training_output2022-08-25/RandomShuffle_dropna_poolAttTxt_freeze_augImgTxt/model_best.pt'
    # 'workspace/source/downstream/training_output2022-08-26/FactualOnly_dropna_VLBERT_augImgTxt/model_best.pt'
    # 'workspace/source/downstream/training_output2022-08-30/FactualOnly_Flamingo_every1/model_best.pt'
    # 'workspace/source/downstream/training_output2022-09-01/FactualOnly_Flamingo_every3_imgtok/model_best.pt'
    # 'workspace/source/downstream/training_output2022-09-01/FactualOnly_Flamingo_every3_withImg/model_best.pt'
    # 'workspace/source/downstream/training_output2022-09-02/PerceptualOnly_Flamingo_every3_withImg/model_best.pt'
    
    # 'workspace/source/downstream/training_output2022-09-06/Mixed_FPR_Flamingo_every3_withImg_again/model_best.pt'
    
    # 'workspace/source/downstream/training_output2022-09-09/Mixed_FPI_v0.1_Flamingo_every3_withImg/model_best.pt'
    # 'workspace/source/downstream/training_output2022-09-14/Mixed_FPI_v0.1_Flamingo_every1_withImg/model_best.pt'
    # 'workspace/source/downstream/training_output2022-09-26/Mixed_FPI_v0.1_Flamingo_every1_withImg_single->cross/model_best.pt'
    # 'workspace/source/downstream/training_output2022-09-15/Mixed_FPI_v0.1_VLBERT_depth1/model_best.pt'
    # 'workspace/source/downstream/training_output2022-09-29/Mixed_FPI_v0.1_CoCa_dpth12_resampler-base_numtok64/model_best.pt'
    # 'workspace/source/downstream/training_output2022-09-30/Mixed_FPI_v0.1_CoCa_dpth12_resampler-large_numtok128/model_best.pt'
    
    # 'workspace/source/downstream/training_output2022-10-10/Mixed_FPI_v0.2_Flamingo_every1_withImg_single->cross/model_best.pt'
    
    # 'workspace/source/downstream/training_output2022-10-27/v0.3_0.08_Flamingo/model_best.pt'
    # 'workspace/source/downstream/training_output2022-10-28/v0.3_1.00_Flamingo/model_best.pt'
    # 'workspace/source/downstream/training_output2022-11-07/v0.3unif_1.0_Flamingo/model_best.pt'
    # 'workspace/source/downstream/training_output2022-11-02/v0.3_10.0_Flamingo/model_best.pt'
    # 'workspace/source/downstream/training_output2022-11-16/v0.3_1.00_flamingo_rrc/model_best.pt'
    # 'workspace/source/downstream/training_output2022-11-16/v0.3_1.00_flamingo_rrc/checkpoint.pt'
    # 'workspace/source/downstream/training_output2022-11-13/v0.3_1.00_prev_flamingo/checkpoint.pt'
    # 'workspace/source/downstream/training_output2022-11-14/v0.3_1.00_prev_flamingo_perceiver4_clstok /checkpoint.pt'
    # 'workspace/source/downstream/training_output2022-11-15/v0.3_1.00_prev_flamingo_perceiver4_dim128_head12 /checkpoint.pt'
    # "workspace/source/downstream/training_output2022-11-17/v0.3_1.00_flamingo_rrc_testsampling1.00/checkpoint.pt"
    # "workspace/source/downstream/training_output2022-11-18/v0.3_1.00_prev_flamingo/model_best.pt"
    "workspace/source/downstream/training_output2022-11-21/v0.3_1.00_prev_flamingo_perceiver_large/model_best.pt"
    
    # 'workspace/source/downstream/training_output2022-11-08/v0.3_1.00_vlbert/model_best.pt'
    # 'workspace/source/downstream/training_output2022-11-08/v0.3_1.00_vlbert/checkpoint.pt'

    # 'workspace/source/downstream/training_output2022-11-09/v0.3_1.00_coca/model_best.pt' #depth 1
    # 'workspace/source/downstream/training_output2022-11-09/v0.3_1.00_coca/checkpoint.pt' #depth 1
    # 'workspace/source/downstream/training_output2022-11-10/v0.3_1.00_coca_depth12/model_best.pt' #depth 12
    # 'workspace/source/downstream/training_output2022-11-10/v0.3_1.00_coca_depth12/checkpoint.pt' #depth 12
    )
    parser.add_argument("--resultdir", type=str, default='inference_result')


##########################
##########################
########################## 

    parser.add_argument("--Valid_dset0_name", type=str, default='frontal_test_w_prev.jsonl',
                        help="valid dset for mimic")
    parser.add_argument("--Valid_dset1_name", type=str, 
        # default='error_baseline_PerceptualOnly/frontal_test_error.jsonl',
        # default='error_baseline_Mixed_FPR/frontal_test_error.jsonl',
        # default='error_baseline_Mixed_FPI_v0.1/frontal_test_error.jsonl',
        # default='error_baseline_Mixed_FPI_v0.2/frontal_test_error.jsonl',
        default='error_baseline_Mixed_FPI_v0.3/frontal_test_error_v1_to_v10_w_prev.jsonl',
        help="valid dset for mimic")

    parser.add_argument("--dataset", type=str, default='mimic-cxr', choices=['mimic-cxr', 'indiana'],
                    help="mimic-cxr or indiana")
    parser.add_argument("--openi", type=str2bool, default=False)
    parser.add_argument("--data_path", type=str, default='data/mimic-cxr-jpg/2.0.0/rred',
                        help="dset path for training")
    parser.add_argument("--data_dir_img", type=str, default='data/mimic-cxr-jpg/2.0.0/files',
                        help="dset path for training")

    parser.add_argument("--test_with_bootstrap", type=str2bool, default=False,
                        help="test with bootstrap")
    parser.add_argument("--make_error", type=str2bool, default=True,
                        help="make error?")
    parser.add_argument("--error_sampling_test", type=int, default=1,
                        help="make error with dinamic sampling?")

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

    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--multimodal_model_type", type=str, default="flamingo", choices=["att_pool", "vlbert", 'flamingo', 'coca'])
    parser.add_argument("--multimodal_depth", type=int, default=12, choices=[1,2,4,8,12])
    parser.add_argument("--cross_attn_every", type=int, default=1, choices=[1,2,3,4])
    parser.add_argument("--cross_attn_order", type=str, default='single->cross', choices=['cross->single', 'single->cross'])
    parser.add_argument("--perceiver_depth", type=int, default=4, choices=[1,2,3,4])
    parser.add_argument("--perceiver_dim_head", type=int, default=128, choices=[64, 128])
    parser.add_argument("--perceiver_num_head", type=int, default=12, choices=[8, 12])
    parser.add_argument("--num_img_token", type=int, default=128, choices=[64, 128])
    parser.add_argument("--max_num_img", type=int, default=2, choices=[2])
    parser.add_argument("--use_prev_img", type=str2bool, default=True)
    parser.add_argument("--use_prev_txt", type=str2bool, default=False)


    parser.add_argument("--img_embed_pool_type", type=str, default="att_txt", choices=["biovil", "att_img", "att_txt"])
    parser.add_argument("--img_hidden_sz", type=int, default=2048)

    parser.add_argument("--hidden", nargs="*", type=int, default=[])
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--inference", type=str2bool, default=True)
    parser.add_argument("--inference_method", type=str, default='batch', choices=['batch','single', None])


if __name__=='__main__':

    parser = argparse.ArgumentParser(description="Evaluating Model")
    get_args(parser)
    args, remaining_args = parser.parse_known_args()

    model_setting_name = args.loaddir.split('/')[-2]

    print("Model Test")
    print(" # PID :", os.getpid())

    args.n_classes = 2
    
    model = get_model(args)
    if torch.cuda.device_count() > 1 and args.device != 'cpu':
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

        val_dataset = get_dataset(args)
        val_loader = get_data_loaders(args, val=val_dataset)
        tgt, _, total_outs  = model_eval(args, val_loader)
        tgts, preds, metrics, matrix, probs = cal_performance(tgt, total_outs, threshold=0.903, resultdir=args.resultdir)

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



    data = [json.loads(l) for l in open(os.path.join(args.data_path, args.Valid_dset0_name))]
    if args.make_error:
        data_error = [json.loads(l) for l in open(os.path.join(args.data_path, args.Valid_dset1_name))]
        data = data+data_error

    data = pd.DataFrame(data)

    def get_result_dataframe(idx):
        dataframe = pd.DataFrame()
        dataframe['idx'] = idx

        dataframe['dicom_id'] = [d for d in data.loc[idx]['dicom_id']]
        dataframe['study_id'] = [d for d in data.loc[idx]['study_id']]
        dataframe['subject_id'] = [d for d in data.loc[idx]['subject_id']]
        dataframe['findings'] = [d for d in data.loc[idx]['Findings']]
        dataframe['impression'] = [d for d in data.loc[idx]['Impression']]
        dataframe['error_type'] = [d for d in data.loc[idx]['error_subtype']]
        dataframe['p(error)'] = np.array(probs)[idx]
        # dataframe['background'] = [d for d in data.loc[idx]['background']]
        dataframe=dataframe.sort_values('p(error)', ascending=False)

        return dataframe

    tp_dataframe = get_result_dataframe(tp_idx)
    fp_dataframe = get_result_dataframe(fp_idx)
    fn_dataframe = get_result_dataframe(fn_idx)

    def get_recall_for_each_type(tp_idx, fn_idx):
        tp_df = get_result_dataframe(tp_idx)
        error_df = get_result_dataframe(tp_idx+fn_idx)
        error_types = list(error_df['error_type'].unique())
        
        total_count = {}
        for error_type in error_types:
            total_count[error_type] = sum(error_df['error_type']==error_type)
            
        total_count = dict(sorted(total_count.items(), key=lambda x: x[1], reverse=True))
        recall = total_count.copy()

        for error_type in error_types:
            recall[error_type] = sum(tp_df['error_type']==error_type) / total_count[error_type]
            
        return recall

    recall_for_each_type = get_recall_for_each_type(tp_idx, fn_idx)
    print(recall_for_each_type)



    def make_infer_output(dataframe, output_path, output_img_path, filename, save_img=True):
        dataframe.to_excel(os.path.join(output_path, filename),header=True, index=False, encoding='utf-8-sig')

        wb = openpyxl.load_workbook(os.path.join(output_path, filename))
        sheet = wb.active

        if save_img:
            for i in tqdm(range(len(dataframe))):
                image_path = make_image_path(dataframe.iloc[i], base_dir=args.data_dir_img, dataset='mimic-cxr')

                shutil.copyfile(image_path, os.path.join(output_img_path, dataframe.iloc[i]['dicom_id']+'.jpg'))
                
                dicom_id = sheet["B"][i+1].value
                sheet["B"][i+1].value = '=HYPERLINK("{}")'.format(f'images/{dataframe.iloc[i]["dicom_id"]}'+'.jpg')

        sheet.column_dimensions["E"].width = 100
        sheet.column_dimensions["F"].width = 50
        wb.save(os.path.join(output_path, filename))


    ### Save
    now = datetime.now(tz=pytz.timezone('Asia/Tokyo'))
    now = now.strftime("%y%m%d-%H%M%S")
    output_path = os.path.join(args.resultdir, str(now)+'_'+model_setting_name)
    output_img_path = os.path.join(output_path,'images')
    if not os.path.exists(output_path):
        os.makedirs(output_img_path)
        os.chmod(output_img_path, 0o777)
    print('output_path: ', output_path)

    make_infer_output(tp_dataframe, output_path, output_img_path, 'True_Positive_Examples_rred2.xlsx', save_img=False)
    make_infer_output(fp_dataframe, output_path, output_img_path, 'False_Positive_Examples_rred2.xlsx', save_img=False)
    make_infer_output(fn_dataframe, output_path, output_img_path, 'False_Negative_Examples_rred2.xlsx', save_img=False)






    # data = pd.read_csv('/home/workspace/Multi-modality-Self-supervision/data/mimic-cxr/mimic_test_with_human_error.csv')

    # error_types = ['1-A', '1-B', '1-C', '1-D', '2-A', '2-B', '3']
    # error_detection_rates = {}
    # for type in error_types:
    #     total = sum(data.iloc[fn_idx]['human_error_type']==type) + sum(data.iloc[tp_idx]['human_error_type']==type)
    #     rate = sum(data.iloc[tp_idx]['human_error_type']==type) / total

    #     error_detection_rates[type] = rate
    # print(error_detection_rates)
    # print('end')