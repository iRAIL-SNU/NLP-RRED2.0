from health_multimodal.text.model import CXRBertTokenizer, CXRBertModel
from transformers import AutoTokenizer
from torchvision.transforms import Compose, RandomAffine, ColorJitter, RandomHorizontalFlip, RandomResizedCrop

import sys
sys.path.append('workspace/source/downstream')
from vocab import Vocab

def get_language_model(args):
    if args.model == "cxr-bert":
        return CXRBertModel.from_pretrained('microsoft/BiomedVLP-CXR-BERT-specialized', revision="v1.1")

    else:
        raise(ValueError("Unknown model type: %s" % args.model_type))
    
    
def get_tokenizer(args):
    if args.model == 'cxr-bert':
        # tokenizer = AutoTokenizer.from_pretrained(args.bert_model, trust_remote_code=True).tokenize
        tokenizer = CXRBertTokenizer.from_pretrained(args.bert_model, revision='v1.1').tokenize
    elif args.model == 'xvl-bert':
        tokenizer = AutoTokenizer.from_pretrained("Medical_X-VL/my_tokenizer/").tokenize
    else:
        str.split

    return tokenizer



def get_vocab(args):
    vocab = Vocab()
    if args.model == 'cxr-bert':
        # bert_tokenizer = AutoTokenizer.from_pretrained(args.bert_model, trust_remote_code=True)
        bert_tokenizer = CXRBertTokenizer.from_pretrained(args.bert_model, revision='v1.1')
        vocab.stoi = bert_tokenizer.vocab
        vocab.itos = bert_tokenizer.convert_ids_to_tokens
        vocab.vocab_sz = bert_tokenizer.vocab_size
    elif args.model == 'xvl-bert':
        bert_tokenizer = AutoTokenizer.from_pretrained("Medical_X-VL/my_tokenizer/")
        vocab.stoi = bert_tokenizer.vocab
        vocab.itos = bert_tokenizer.convert_ids_to_tokens
        vocab.vocab_sz = bert_tokenizer.vocab_size
    else:
        assert('correct model needed')

    return vocab

def get_image_augmentations(aug_method):
    color_jitter = ColorJitter(brightness=.2, contrast=.2, saturation=.2, hue=.2 )
    horizontal_flip = RandomHorizontalFlip(p=0.5)
    random_affine = RandomAffine(degrees=30, translate=(0.1,0.1), scale=(0.8,1.2))
    # random_resized_crop = RandomResizedCrop(size=512, scale=(0.5,1.0))

    augmentations = []
    # if aug_method in ["all","rrc"]:
    #     augmentations.append(random_resized_crop)
    if aug_method in ["all","colur"]:
        augmentations.append(color_jitter)
    if aug_method in ["all","hflip"]:
        augmentations.append(horizontal_flip)
    if aug_method in ["all","affine"]:
        augmentations.append(random_affine)
        
   
    return Compose(augmentations)