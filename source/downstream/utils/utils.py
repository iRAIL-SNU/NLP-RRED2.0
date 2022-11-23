import os
import random
import shutil
import contextlib
import numpy as np

import torch
from nltk import tokenize
from random import shuffle
import wandb

import time
from collections import defaultdict, deque
from datetime import datetime

import torch.distributed as dist
import math
import warnings
import yaml
import cv2


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
        
        if data_line['prev_study_id'] is not None:
            prev_study_id = data_line['prev_study_id']
            prev_image_file_name = f"p{str(subject_id)[:2]}/p{subject_id}/s{prev_study_id}/{data_line['prev_dicom_id']}.jpg"
            prev_image_path = os.path.join(base_dir, prev_image_file_name)
        else:
            prev_image_path = None
            
    return image_path, prev_image_path


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



import numpy as np
import io
import os
import time
from collections import defaultdict, deque
import datetime

import torch
import torch.distributed as dist
import math
import warnings
import yaml
import cv2

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def global_avg(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {:.4f}".format(name, meter.global_avg)
            )
        return self.delimiter.join(loss_str)    
    
    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))
        


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def compute_acc(logits, label, reduction='mean'):
    ret = (torch.argmax(logits, dim=1) == label).float()
    if reduction == 'none':
        return ret.detach()
    elif reduction == 'mean':
        return ret.mean().item()

def compute_n_params(model, return_str=True):
    tot = 0
    for p in model.parameters():
        w = 1
        for x in p.shape:
            w *= x
        tot += w
    if return_str:
        if tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}K'.format(tot / 1e3)
    else:
        return tot

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def load_yaml(file_name):
    with open(file_name, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.SafeLoader)
    return config


def border_pad(image, cfg):
    h, w, c = image.shape

    if cfg.border_pad == 'zero':
        image = np.pad(image, ((0, cfg.long_side - h),
                               (0, cfg.long_side - w), (0, 0)),
                       mode='constant',
                       constant_values=0.0)
    elif cfg.border_pad == 'pixel_mean':
        image = np.pad(image, ((0, cfg.long_side - h),
                               (0, cfg.long_side - w), (0, 0)),
                       mode='constant',
                       constant_values=cfg.pixel_mean)
    else:
        image = np.pad(image, ((0, cfg.long_side - h),
                               (0, cfg.long_side - w), (0, 0)),
                       mode=cfg.border_pad)

    return image


def fix_ratio(image, cfg):
    h, w, c = image.shape

    if h >= w:
        ratio = h * 1.0 / w
        h_ = cfg.long_side
        w_ = round(h_ / ratio)
    else:
        ratio = w * 1.0 / h
        w_ = cfg.long_side
        h_ = round(w_ / ratio)

    image = cv2.resize(image, dsize=(w_, h_), interpolation=cv2.INTER_LINEAR)
    image = border_pad(image, cfg)

    return image


def gen_sev_arr_from_map(sev_maps, m, device):
    # Load mask
    masks = m.squeeze(1).cpu().detach().numpy()
    sev_maps = sev_maps.squeeze(1)

    sev_arrs_max = []
    for i in range(masks.shape[0]):
        m = masks[i]
        both, left, right = only_lung(m)

        both = torch.from_numpy(both).to(device)
        left = torch.from_numpy(left).to(device)
        right = torch.from_numpy(right).to(device)

        # Calculate the line
        if len(both.nonzero(as_tuple=True)[0]) == 0:
            both_max = 0
            both_min = 0
        else:
            both_max = both.nonzero(as_tuple=True)[0][-1]
            both_min = both.nonzero(as_tuple=True)[0][0]

        both_3_12 = int(3 * (both_max - both_min) // 12)
        both_4_12 = int(4 * (both_max - both_min) // 12)
        both_5_12 = int(5 * (both_max - both_min) // 12)

        both_7_12 = int(7 * (both_max - both_min) // 12)
        both_8_12 = int(8 * (both_max - both_min) // 12)
        both_9_12 = int(9 * (both_max - both_min) // 12)

        # Load Brixia map
        sm = sev_maps[i]
        left_bm = left * sm
        right_bm = right * sm

        a_max, b_max, c_max, d_max, e_max, f_max = torch.zeros(1, requires_grad=True).to(device), \
                                                   torch.zeros(1, requires_grad=True).to(device), \
                                                   torch.zeros(1, requires_grad=True).to(device), \
                                                   torch.zeros(1, requires_grad=True).to(device), \
                                                   torch.zeros(1, requires_grad=True).to(device), \
                                                   torch.zeros(1, requires_grad=True).to(device)

        if len(left.nonzero(as_tuple=True)[0]) > 0:
            left_max = left.nonzero(as_tuple=True)[0][-1]
            left_min = left.nonzero(as_tuple=True)[0][0]

            try:
                a_max = left_bm[left_min:both_5_12].max()
            except:
                pass
            try:
                b_max = left_bm[both_5_12:both_8_12].max()
            except:
                pass
            try:
                c_max = left_bm[both_8_12:left_max].max()
            except:
                pass

        if len(right.nonzero(as_tuple=True)[0]) > 0:
            right_max = right.nonzero(as_tuple=True)[0][-1]
            right_min = right.nonzero(as_tuple=True)[0][0]

            try:
                d_max = right_bm[right_min:both_5_12].max()
            except:
                pass
            try:
                e_max = right_bm[both_5_12:both_8_12].max()
            except:
                pass
            try:
                f_max = right_bm[both_8_12:right_max].max()
            except:
                pass


        sev_arr_max = torch.stack([torch.cat([a_max.view(-1), d_max.view(-1)]),
                                   torch.cat([b_max.view(-1), e_max.view(-1)]),
                                   torch.cat([c_max.view(-1), f_max.view(-1)])])
        sev_arrs_max.append(sev_arr_max)

    sev_arrs_max = torch.stack(sev_arrs_max)

    return sev_arrs_max


def only_lung(mask, blur=False):
    if blur:
        m = cv2.blur(mask.copy(), ksize=(200, 200))
    else:
        m = mask.copy()
    m = (m*255).astype(np.uint8)
    ret, markers = cv2.connectedComponents(m)

    npoints = []
    for i in range(ret):
        npoints.append(len(markers[markers == i]))

    sorted_labels = np.argsort(npoints)

    half1 = np.zeros_like(m)
    half2 = np.zeros_like(m)
    if len(sorted_labels) >= 3:
        half1[markers == sorted_labels[-2]] = 1
        half2[markers == sorted_labels[-3]] = 1

        flag1 = half1.nonzero()[1].max()
        flag2 = half2.nonzero()[1].max()

        if flag1 > flag2:
            left = half2  # right lung in reality
            right = half1  # left lung in reality
        else:
            left = half1
            right = half2

    elif len(sorted_labels) == 2:
        try:
            half1[markers == sorted_labels[-2]] = 1
            flag_max = half1.nonzero()[1].max()
            flag_min = half1.nonzero()[1].min()
            if half1[:, flag_min].argmax() > half1[:, flag_max].argmax():
                left = half1
                right = half2
            else:
                left = half2
                right = half1

        except:
            left = half1
            right = half2
    else:
        left = half1
        right = half2

    both = left + right

    return both, left, right


def transform(image, cfg):
    assert image.ndim == 2, "image must be gray image"
    if cfg.use_equalizeHist:
        image = cv2.equalizeHist(image)

    if cfg.gaussian_blur > 0:
        image = cv2.GaussianBlur(
            image,
            (cfg.gaussian_blur, cfg.gaussian_blur), 0)

    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    image = fix_ratio(image, cfg)
    # augmentation for train or co_train

    # normalization
    image = image.astype(np.float32) - cfg.pixel_mean
    # vgg and resnet do not use pixel_std, densenet and inception use.
    if cfg.pixel_std:
        image /= cfg.pixel_std
    # normal image tensor :  H x W x C
    # torch image tensor :   C X H X W
    image = image.transpose((2, 0, 1))

    return image


def post_process(answer):
    # [Prepocess] Unify expression with same semantic meaning
    if answer == 'pa':
        answer = 'posterior anterior'
    elif answer == 'x ray':
        answer = 'x-ray'
    elif answer == 'xray':
        answer = 'x-ray'
    elif answer == 'xr':
        answer = 'x-ray'
    elif answer == 'plain film x ray':
        answer = 'x-ray'
    elif answer == 'plain film':
        answer = 'x-ray'
    elif answer == 't2 mri':
        answer = 't2 weighted'
    elif answer == 't2 weighted mri':
        answer = 't2 weighted'
    elif answer == 'mr flair':
        answer = 'flair'
    elif answer == 'ct with contrast':
        answer = 'ct'
    elif answer == 'axial plane':
        answer = 'axial'
    return answer