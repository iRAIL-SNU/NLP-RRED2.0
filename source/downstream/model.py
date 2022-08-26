import torch
import torch.nn as nn
import torch.nn.functional as F


from health_multimodal.text.model import CXRBertModel
from health_multimodal.image.model import get_biovil_resnet

import sys
sys.path.insert(1, '/home/workspace/source/model/report_coca')
from report_coca import CrossAttention, LayerNorm, Residual, ParallelTransformerBlock
from einops import rearrange, repeat

from transformers import BertConfig, BertModel

MODEL_TYPE = "resnet50"


class VLModelClf(nn.Module):
    def __init__(self, args):
        super(VLModelClf, self).__init__()
        
        self.args = args
        
        self.text_model = CXRBertModel.from_pretrained(args.bert_model, revision="v1.1")
        
        self.image_model = get_biovil_resnet()

        ######### image pooler 
        if self.args.img_embed_pool_type=='biovil':
            1
        else:
            # use attention pooling
            self.dim = 768
            self.img_attn_pool = CrossAttention(dim=self.dim, context_dim=2048, dim_head=64, heads=8, norm_context=True)
            self.img_attn_pool_norm = LayerNorm(self.dim)
    
            self.attn_last_pool = nn.Linear(self.dim, args.JOINT_FEATURE_SIZE)
            self.txt_pool_layer = nn.Linear(self.dim, args.JOINT_FEATURE_SIZE)

            if self.args.img_embed_pool_type=='att_img':
                # use random learnable query vector  
                self.img_queries = nn.Parameter(torch.randn(1,self.dim))

        self.clf = nn.Linear(args.JOINT_FEATURE_SIZE*2, args.n_classes)

    def forward(self, findings, impression, image):
        image_embed = self.image_model(image)
        txt, _, mask = findings

        if self.args.img_embed_pool_type=='biovil':
            image_embed = image_embed.projected_global_embedding
            findings_embed = self.text_model.get_projected_text_embeddings(txt, mask)
        else:
            findings_embed = self.text_model(txt,mask).last_hidden_state[:,0,:]
            image_embed = image_embed.patch_embedding
            image_embed = image_embed.resize(image_embed.shape[0],15*15,2048)

            if self.args.img_embed_pool_type=='att_txt':
                # use [cls] feature for query
                queries = findings_embed.reshape(findings_embed.shape[0], 1, -1)

            elif self.args.img_embed_pool_type=='att_img':
                # use random learnable query vector
                queries = repeat(self.img_queries, 'n d -> b n d', b=image_embed.shape[0])

            pooled_img = self.img_attn_pool(queries, image_embed)
            pooled_img = self.img_attn_pool_norm(pooled_img)[:,0]

            image_embed = self.attn_last_pool(pooled_img)
            findings_embed = self.txt_pool_layer(findings_embed)

        return self.clf(torch.cat((image_embed, findings_embed), 1))

        # txt, _, mask = impression
        # impression_embed = self.text_model.get_projected_text_embeddings(txt, mask)
        # return self.clf(torch.cat((image_embed, impression_embed), 1))


class ImageBertEmbeddings(nn.Module):
    def __init__(self, args, embeddings):
        super(ImageBertEmbeddings, self).__init__()
        self.args = args

        # use attention pooling for Multimodal transformer
        self.dim = 768
        self.img_attn_pool = CrossAttention(dim=self.dim, context_dim=2048, dim_head=64, heads=8, norm_context=True)
        self.img_attn_pool_norm = LayerNorm(self.dim)
        self.img_queries = nn.Parameter(torch.randn(128,self.dim))

        self.token_type_embeddings = embeddings.token_type_embeddings
        self.word_embeddings = embeddings.word_embeddings
        self.LayerNorm = embeddings.LayerNorm
        self.dropout = nn.Dropout(p=args.dropout)

    def forward(self, input_imgs, token_type_ids):
        bsz = input_imgs.size(0)
        seq_length = 128 + 1  # +1 for SEP Token

        sep_id = torch.LongTensor([self.args.vocab.stoi["[SEP]"]]).to(self.args.device)
        sep_id = sep_id.unsqueeze(0).expand(bsz, 1)
        sep_token_embeds = self.word_embeddings(sep_id)

        queries = repeat(self.img_queries, 'n d -> b n d', b=input_imgs.shape[0])
        imgs_embeddings = self.img_attn_pool(queries, input_imgs)
        imgs_embeddings = self.img_attn_pool_norm(imgs_embeddings)

        token_embeddings = torch.cat(
            [sep_token_embeds, imgs_embeddings], dim=1)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = token_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class VLBertEncoder(nn.Module):
    def __init__(self, args):
        super(VLBertEncoder, self).__init__()
        self.args = args

        config = BertConfig(num_hidden_layers=self.args.multimodal_depth)
        vlbert = BertModel(config)
        
        self.txt_embeddings = vlbert.embeddings
        self.dropout = nn.Dropout(p=args.dropout)
        self.text_model = CXRBertModel.from_pretrained(args.bert_model, revision="v1.1")

        self.img_embeddings = ImageBertEmbeddings(args, self.txt_embeddings)
        self.image_model = get_biovil_resnet()
        
        self.vl_encoder = vlbert.encoder
        self.vl_pooler = vlbert.pooler

    def forward(self, findings, impression, image):
        txt, _, mask = findings

        bsz = txt.size(0)
        # attention_mask = torch.cat(
        #     [
        #         torch.ones(bsz, self.args.num_image_embeds + 2).long().cuda(),
        #         attention_mask,
        #     ],
        #     dim=1)
        # extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # try:
        #     extended_attention_mask = extended_attention_mask.to(
        #         dtype=next(self.parameters()).dtype)  # fp16 compatibility
        # except StopIteration:
        #     extended_attention_mask = extended_attention_mask.to(dtype=torch.float16)

        # extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        img_tok = (
            torch.LongTensor(txt.size(0), 128 + 1)
            .fill_(0)
            .to(self.args.device))
        txt_tok = (
            torch.LongTensor(txt.size(0), txt.size(1))
            .fill_(1)
            .to(self.args.device))

        img = self.image_model(image).patch_embedding  
        img = img.resize(img.shape[0],15*15,2048)# img: Bx15x15x2048

        img_embed_out = self.img_embeddings(img, img_tok) # img_embed_out: Bx128x768
       
        findings_embed = self.text_model(txt,mask).last_hidden_state

        findings_embed = findings_embed + self.txt_embeddings.token_type_embeddings(txt_tok)
        findings_embed = self.txt_embeddings.LayerNorm(findings_embed)
        findings_embed = self.dropout(findings_embed)

        encoder_input = torch.cat([findings_embed, img_embed_out], 1)  # Bx(TEXT+IMG)xHID
        encoded_layers = self.vl_encoder(encoder_input)#, extended_attention_mask)
        return self.vl_pooler(encoded_layers[-1])


class VLBertClf(nn.Module):
    def __init__(self, args):
        super(VLBertClf, self).__init__()
        self.args = args
        self.enc = VLBertEncoder(args)
        self.clf = nn.Linear(args.hidden_sz, args.n_classes)

    def forward(self, findings, impression, image):
        x = self.enc(findings, impression, image)
        return self.clf(x)

