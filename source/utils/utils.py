import os
import random
import shutil
import contextlib
import numpy as np

import torch
from nltk import tokenize
from random import shuffle
import wandb

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_checkpoint(state, is_best, checkpoint_path, filename="checkpoint.pt"):
    filename = os.path.join(checkpoint_path, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(checkpoint_path, "model_best.pt"))

def load_checkpoint(model, path):
    best_checkpoint = torch.load(path)
    model.load_state_dict(best_checkpoint["state_dict"])


def truncate_seq_pair(tokens_a, tokens_b, max_length):
    """
    tokens_a : IMG patches
    tokens_b : TXT tokens
    max_length: bert-base(512)
    using all img patches, only truncate txt tokens if exceed max_length
    """

    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        else:
            tokens_b.pop()

#TODO: def store_preds_to_disk, log_metrics()
def store_preds_to_disk(tgts, preds, args):
    pass

def log_metrics(set_name, metrics, args, logger):
    pass


@contextlib.contextmanager
def numpy_seed(seed, *addl_seeds):
    """Context manager which seeds the NumPy PRNG with the specified seed and
    restores the state afterward"""
    if seed is None:
        yield
        return
    if len(addl_seeds) > 0:
        seed = int(hash((seed, *addl_seeds)) % 1e6)
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)

def make_image_path(data_line, base_dir, dataset='mimic-cxr'):
    if dataset == 'mimic-cxr':
        subject_id = data_line['subject_id']
        study_id = data_line['study_id']
        image_file_name = f"p{str(subject_id)[:2]}/p{subject_id}/s{study_id}/{data_line['dicom_id']}.jpg"
        image_path = os.path.join(base_dir, image_file_name)
    return image_path


def shuffle_sentence(document):
    sentences = tokenize.sent_tokenize(document)
    shuffle(sentences)

    return " ".join(sentences)


def log_tanh_gating(model, args):
    ## 각 gated xattn_layer의 tanh gating 값(att, ffw) wadnb에 로깅
    if torch.cuda.device_count() > 1 and args.device != torch.device('cpu'):
        m = model.module
    else:
        m = model
    
    att_gate_gain = {}
    ffw_gate_gain = {}

    idx_cross_attn_layer = 1 if args.cross_attn_order == 'single->cross' else 0
    for i, layer in enumerate(m.encoder_layers):
        if layer[idx_cross_attn_layer] is not None:
            p = layer[idx_cross_attn_layer].attn_gate[0].detach().cpu().item()
            att_gate_gain[f'Layer {i} Attention tanh gain'] = np.tanh(p)

            p = layer[idx_cross_attn_layer].ff_gate[0].detach().cpu().item()
            ffw_gate_gain[f'Layer {i} Feedforward tanh gain'] = np.tanh(p)
        # if layer[2] is not None:
        #     p = layer[2].attn_gate[0].detach().cpu().item()
        #     att_gate_gain[f'Layer {i} Attention tanh gain'] = np.tanh(p)

        #     p = layer[2].ff_gate[0].detach().cpu().item()
        #     ffw_gate_gain[f'Layer {i} Feedforward tanh gain'] = np.tanh(p)
    # for i in range(len(m.encoder_layers)):
    #     if not (i % args.cross_attn_every):
    #         p = m.encoder_layers[i][0].attn_gate[0].detach().cpu().item()
    #         att_gate_gain[f'Layer {i} Attention tanh gain'] = np.tanh(p)

    #         p = m.encoder_layers[i][0].ff_gate[0].detach().cpu().item()
    #         ffw_gate_gain[f'Layer {i} Feedforward tanh gain'] = np.tanh(p)

    wandb.log(att_gate_gain)
    wandb.log(ffw_gate_gain)



