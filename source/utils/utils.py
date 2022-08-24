import os
import random
import shutil
import contextlib
import numpy as np

import torch

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