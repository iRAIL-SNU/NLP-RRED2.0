import torch
import torch.nn as nn
import torch.nn.functional as F

from health_multimodal.text.model import CXRBertModel
from health_multimodal.image.model import get_biovil_resnet

import sys
sys.path.insert(1, '/home/workspace/source/model/report_coca')
from report_coca import CrossAttention, LayerNorm
from einops import rearrange, repeat

MODEL_TYPE = "resnet50"


class VLModelClf(nn.Module):
    def __init__(self, args):
        super(VLModelClf, self).__init__()
        
        self.args = args
        
        self.text_model = CXRBertModel.from_pretrained(args.bert_model, revision="v1.1")
        
        self.image_model = get_biovil_resnet()

        ######### pooler 
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
        #########        

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
