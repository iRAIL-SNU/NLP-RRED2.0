from health_multimodal.text.model import CXRBertTokenizer, CXRBertModel
from transformers import AutoTokenizer

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
