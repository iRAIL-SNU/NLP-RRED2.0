import torch
import torch.nn as nn

from health_multimodal.text.model import CXRBertModel
from health_multimodal.image.model import get_biovil_resnet

from report_coca.report_coca import CrossAttention, LayerNorm
from flamingo_pytorch import PerceiverResampler, GatedCrossAttentionBlock
from flamingo_pytorch.flamingo_palm import *

# sys.path.insert(1, '/home/workspace/Medical_X-VL')
# from models.ibot_vit import VisionTransformer, interpolate_pos_embed, vit_base, vit_small
# from models.xbert import BertConfig, BertModel
# from models.model_retrieval import XVLModel
from ibot_vit import vit_small, interpolate_pos_embed


from einops import rearrange, repeat

import transformers

import xbert

MODEL_TYPE = "resnet50"

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


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
            self.img_attn_pool = CrossAttention(dim=self.dim, context_dim=args.img_hidden_sz, dim_head=64, heads=8, norm_context=True)
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
            image_embed = image_embed.resize(image_embed.shape[0],15*15,self.args.img_hidden_sz)

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
        self.img_attn_pool = CrossAttention(dim=self.dim, context_dim=args.img_hidden_sz, dim_head=64, heads=8, norm_context=True)
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

        config = transformers.BertConfig(num_hidden_layers=self.args.multimodal_depth)
        vlbert = transformers.BertModel(config)
        
        self.txt_embeddings = vlbert.embeddings
        self.dropout = nn.Dropout(p=args.dropout)
        self.text_model = CXRBertModel.from_pretrained(args.bert_model, revision="v1.1")

        self.img_embeddings = ImageBertEmbeddings(args, self.txt_embeddings)
        self.image_model = get_biovil_resnet()
        
        self.vl_encoder = vlbert.encoder
        self.vl_pooler = vlbert.pooler

    def forward(self, findings, impression, image):
        txt, _, mask = findings

        img_tok = (
            torch.LongTensor(txt.size(0), 128 + 1)
            .fill_(0)
            .to(self.args.device))
        txt_tok = (
            torch.LongTensor(txt.size(0), txt.size(1))
            .fill_(1)
            .to(self.args.device))

        img = self.image_model(image).patch_embedding  
        img = img.resize(img.shape[0],15*15,self.args.img_hidden_sz)# img: Bx15x15x2048

        img_embed_out = self.img_embeddings(img, img_tok) # img_embed_out: Bx(128+1)x768
       
        findings_embed = self.text_model(txt,mask).last_hidden_state
        ## TODO: findings_embed length가 너무 길어서 len(encoder_input)>512인 경우 에러 발생할수도. truncate해야할듯

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


class CXRFlamingo(nn.Module):
    def __init__(self, args):
        super(CXRFlamingo,self).__init__()
        self.args = args

        language_model = CXRBertModel.from_pretrained(args.bert_model, revision="v1.1")
        self.image_model = get_biovil_resnet()
                
        if self.args.inference or self.args.freeze_txt_all:
            freeze_model_and_make_eval_(language_model)
            print("Language model is freezed")
        if self.args.inference or self.args.freeze_img_all:
            freeze_model_and_make_eval_(self.image_model)
            print("Image model is freezed")

        self.dim = args.hidden_sz # 768
        self.img_attn_pool = CrossAttention(dim=self.dim, context_dim=args.img_hidden_sz, dim_head=64, heads=8, norm_context=True)
        self.img_attn_pool_norm = LayerNorm(self.dim)
        self.img_queries = nn.Parameter(torch.randn(15*15,self.dim))

        self.perceiver_resampler = PerceiverResampler(
            dim=self.dim,
            depth=args.perceiver_depth,
            dim_head=args.perceiver_dim_head,
            heads=args.perceiver_num_head,
            num_latents=args.num_img_token,       # the number of latents to shrink your media sequence to, perceiver style
            num_media_embeds = args.max_num_img, ## max number of images per example
        )

        self.img_attn_pool_last = CrossAttention(dim=self.dim, context_dim=self.dim, dim_head=64, heads=8, norm_context=True)
        self.img_attn_pool_norm_last = LayerNorm(self.dim)

        self.text_embdding = language_model.bert.embeddings
        lm_layers = language_model.bert.encoder.layer
        
        self.encoder_layers = nn.ModuleList([])

        if self.args.cross_attn_order == 'single->cross':
            for i in range(len(lm_layers)):
                self.encoder_layers.append(nn.ModuleList([
                    lm_layers[i],
                    GatedCrossAttentionBlock(dim=self.dim, dim_head=64, heads=12, only_attend_immediate_media=True) if not (i % args.cross_attn_every) else None
                ])
            )
        elif self.args.cross_attn_order == 'cross->single':
            for i in range(len(lm_layers)):
                self.encoder_layers.append(nn.ModuleList([
                    GatedCrossAttentionBlock(dim=self.dim, dim_head=64, heads=12, only_attend_immediate_media=True) if not (i % args.cross_attn_every) else None,
                    lm_layers[i]
                ])
            )

        self.to_logits = nn.Sequential(
            # LayerNorm(self.dim),
            # nn.Linear(self.dim, args.n_classes, bias=False)
            LayerNorm(self.dim*2),
            nn.Linear(self.dim*2, args.n_classes, bias=False)
        )

    def forward(self, findingss, impression, images):
        findings, prev_findings = findingss

        txt, segment, attention_mask = findings
        findings_embed = self.text_embdding(txt, token_type_ids=segment)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        if self.args.use_prev_txt:
            prev_txt, prev_segment, prev_attention_mask = prev_findings
            prev_findings_embed = self.text_embdding(prev_txt, token_type_ids=prev_segment)

            prev_extended_attention_mask = prev_attention_mask.unsqueeze(1).unsqueeze(2)
            prev_extended_attention_mask = (1.0 - prev_extended_attention_mask) * -10000.0

            findings_embed = torch.cat((findings_embed, prev_findings_embed),dim = 1)
            extended_attention_mask = torch.cat((extended_attention_mask, prev_extended_attention_mask), dim=3)
            segment = torch.cat((segment, prev_segment), dim = 1)
        # impression_embed = self.text_embdding(txt)

        media_locations = torch.zeros_like(segment).bool().to(self.args.device)
        for i, media_location in enumerate(media_locations):
            media_location[0] = True
            media_location[segment[i].argmax()] = True

        image, prev_image = images
        img = self.image_model(image).patch_embedding  # Bxn_channelxWxH --> Bx2048x15x15
        img = img.resize(img.shape[0],img.shape[2]*img.shape[3],self.args.img_hidden_sz)# Bx2048x15x15 --> Bx15x15x2048

        queries = repeat(self.img_queries, 'n d -> b n d', b=img.shape[0])
        img = self.img_attn_pool(queries, img)
        img = self.img_attn_pool_norm(img) # Bx225x768
        
        #use prev report
        if self.args.use_prev_img:
            prev_img = self.image_model(prev_image).patch_embedding  # Bxn_channelxWxH --> Bx2048x15x15
            prev_img = prev_img.resize(prev_img.shape[0],prev_img.shape[2]*prev_img.shape[3],self.args.img_hidden_sz)# Bx2048x15x15 --> Bx15x15x2048

            prev_queries = repeat(self.img_queries, 'n d -> b n d', b=prev_img.shape[0])
            prev_img = self.img_attn_pool(prev_queries, prev_img)
            prev_img = self.img_attn_pool_norm(prev_img) # Bx225x768

            img = rearrange(img, 'b n d -> b 1 n d')
            prev_img = rearrange(prev_img, 'b n d -> b 1 n d')
                    
            if self.args.img_to_each_perceiver:
                prev_img= self.perceiver_resampler(prev_img) 
            else:
                img = torch.cat((img,prev_img), dim=1)
                    
        img = self.perceiver_resampler(img) 
        if self.args.img_to_each_perceiver:
            img = torch.cat((img,prev_img), dim=1)

        if self.args.cross_attn_order == 'cross->single':
            for xattn_layer, lm_layer in self.encoder_layers:
                if exists(xattn_layer) and exists(img):
                    findings_embed = xattn_layer(findings_embed, img, media_locations=media_locations)
                findings_embed = lm_layer(findings_embed, extended_attention_mask)[0]
        elif self.args.cross_attn_order == 'single->cross':
            for lm_layer, xattn_layer in self.encoder_layers:
                findings_embed = lm_layer(findings_embed, extended_attention_mask)[0]
                if exists(xattn_layer) and exists(img):
                    findings_embed = xattn_layer(findings_embed, img, media_locations=media_locations)

        img = self.img_attn_pool_last(findings_embed[:,0,:].unsqueeze(1), img.reshape(img.shape[0],-1,img.shape[-1]))
        img = self.img_attn_pool_norm_last(img)

        return self.to_logits(torch.cat((img.squeeze(1), findings_embed[:,2,:]), 1)) #### 이거 findings_embed[:,0,:] 으로 해보자
        # return self.to_logits(torch.cat((img.squeeze(1), findings_embed[:,0,:]), 1)) 

    def unfreeze_image_model(self):
        self.image_model.train()
        unfreeze_all_layers_(self.image_model)
        print("Image model is unfreezed")
            
            
class CXRFlamingo_with_ViT(nn.Module):
    def __init__(self, args):
        super(CXRFlamingo_with_ViT,self).__init__()
        self.args = args

        get_text_encoder = True if self.args.model == "xvl-bert" else False
        self.image_model, language_model = get_XVL(get_text_encoder=get_text_encoder)
        
        if self.args.model =="cxr-bert":
           language_model = CXRBertModel.from_pretrained(args.bert_model, revision="v1.1")
                
        if self.args.inference or self.args.freeze_txt_all:
            freeze_model_and_make_eval_(language_model)
            print("Language model is freezed")
        if self.args.inference or self.args.freeze_img_all:
            freeze_model_and_make_eval_(self.image_model)
            print("Image model is freezed")

        self.dim = 384 if self.args.model == 'xvl-bert' else 768
        self.perceiver_resampler = PerceiverResampler(
            dim=384, ##384 for vit
            depth=args.perceiver_depth,
            dim_head=args.perceiver_dim_head,
            heads=args.perceiver_num_head,
            num_latents=args.num_img_token *2,       # the number of latents to shrink your media sequence to, perceiver style
            num_media_embeds = args.max_num_img, ## max number of images per example
        )

        self.img_attn_pool_last = CrossAttention(dim=self.dim, context_dim=self.dim, dim_head=64, heads=8, norm_context=True)
        self.img_attn_pool_norm_last = LayerNorm(self.dim)

        if self.args.model == "xvl-bert":
            self.text_embdding = language_model.embeddings
            lm_layers = language_model.encoder.layer
        else:
            self.text_embdding = language_model.bert.embeddings
            lm_layers = language_model.bert.encoder.layer
        
        self.encoder_layers = nn.ModuleList([])

        if self.args.cross_attn_order == 'single->cross':
            for i in range(len(lm_layers)):
                self.encoder_layers.append(nn.ModuleList([
                    lm_layers[i],
                    GatedCrossAttentionBlock(dim=self.dim, dim_head=64, heads=12, only_attend_immediate_media=False) if not (i % args.cross_attn_every) else None
                ])
            )
        elif self.args.cross_attn_order == 'cross->single':
            for i in range(len(lm_layers)):
                self.encoder_layers.append(nn.ModuleList([
                    GatedCrossAttentionBlock(dim=self.dim, dim_head=64, heads=12, only_attend_immediate_media=False) if not (i % args.cross_attn_every) else None,
                    lm_layers[i]
                ])
            )

        self.to_logits = nn.Sequential(
            LayerNorm(self.dim*2),
            nn.Linear(self.dim*2, args.n_classes, bias=False)
        )

    def forward(self, findingss, impression, images):
        findings, prev_findings = findingss

        txt, segment, attention_mask = findings
        findings_embed = self.text_embdding(txt, token_type_ids=segment)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        if self.args.use_prev_txt:
            prev_txt, prev_segment, prev_attention_mask = prev_findings
            prev_findings_embed = self.text_embdding(prev_txt, token_type_ids=prev_segment)

            prev_extended_attention_mask = prev_attention_mask.unsqueeze(1).unsqueeze(2)
            prev_extended_attention_mask = (1.0 - prev_extended_attention_mask) * -10000.0

            findings_embed = torch.cat((findings_embed, prev_findings_embed),dim = 1)
            extended_attention_mask = torch.cat((extended_attention_mask, prev_extended_attention_mask), dim=3)
            segment = torch.cat((segment, prev_segment), dim = 1)
        # impression_embed = self.text_embdding(txt)

        media_locations = torch.zeros_like(segment).bool().to(self.args.device)
        for i, media_location in enumerate(media_locations):
            media_location[0] = True
            media_location[segment[i].argmax()] = True

        image, prev_image = images
        img = self.image_model(image)  # Bxn_channelxWxH --> Bx2048x15x15
        
        #use prev image
        if self.args.use_prev_img:
            prev_img = self.image_model(prev_image)  # Bxn_channelxWxH --> Bx2048x15x15

            img = rearrange(img, 'b n d -> b 1 n d')
            prev_img = rearrange(prev_img, 'b n d -> b 1 n d')
                    
            if self.args.img_to_each_perceiver:
                prev_img= self.perceiver_resampler(prev_img) 
            else:
                img = torch.cat((img,prev_img), dim=1)
                    
        img = self.perceiver_resampler(img) 
        if self.args.img_to_each_perceiver:
            img = torch.cat((img,prev_img), dim=1)

        img = torch.reshape(img, (img.shape[0], img.shape[1], -1, self.dim))

        if self.args.cross_attn_order == 'cross->single':
            for xattn_layer, lm_layer in self.encoder_layers:
                if exists(xattn_layer) and exists(img):
                    findings_embed = xattn_layer(findings_embed, img, media_locations=media_locations)
                findings_embed = lm_layer(findings_embed, extended_attention_mask)[0]
        elif self.args.cross_attn_order == 'single->cross':
            for lm_layer, xattn_layer in self.encoder_layers:
                findings_embed = lm_layer(findings_embed, extended_attention_mask)[0]
                if exists(xattn_layer) and exists(img):
                    findings_embed = xattn_layer(findings_embed, img, media_locations=media_locations)

        img = self.img_attn_pool_last(findings_embed[:,0,:].unsqueeze(1), img.reshape(img.shape[0],-1,img.shape[-1]))
        # img = self.img_attn_pool_last(findings_embed[:,0,:].unsqueeze(1), img[:,0]) # Last image at Last
        img = self.img_attn_pool_norm_last(img)

        return self.to_logits(torch.cat((img.squeeze(1), findings_embed[:,2,:]), 1)) #### 이거 findings_embed[:,0,:] 으로 해보자
        # return self.to_logits(torch.cat((img.squeeze(1), findings_embed[:,0,:]), 1)) 

    def unfreeze_image_model(self):
        self.image_model.train()
        unfreeze_all_layers_(self.image_model)
        print("Image model is unfreezed")
            
            
            
class CXRFlamingoForErrorDetection(nn.Module):
    def __init__(self,args):
        super(CXRFlamingoForErrorDetection,self).__init__()
        self.args = args
        self.encoder = CXRFlamingoEncoder(args)

        self.to_logits = nn.Sequential(
            LayerNorm(self.dim*2),
            nn.Linear(self.dim*2, args.n_classes, bias=False)
        )

    def forward(self, findings, impression, image):
        img, findings_embed, impression_embed = self.encoder(findings, impression, image)
        return self.to_logits(torch.cat((img.squeeze(1), findings_embed[:,2,:]), 1))


class CXRFlamingoForPreTraining(nn.Module):
    def __init__(self,args):
        super(CXRFlamingoForPreTraining,self).__init__()
        self.args = args
        self.encoder = CXRFlamingoEncoder(args)
        self.mlm = CXRFlamingoMLMHead(args, self.encoder.text_embdding.word_embeddings.weight)
        self.itm = nn.Sequential(LayerNorm(args.hidden_sz*2), 
                                nn.Linear(args.hidden_sz*2, 2))


    def forward(self, findings, impression, image):
        img, findings_embed, _ = self.encoder(findings, impression, image)

        prediction_scores_masked, _ = self.mlm(findings_embed[-1])
        predict_itm = self.itm(img, findings_embed)
        return prediction_scores_masked, predict_itm


class CXRFlamingoPredictionHeadTransform(nn.Module):
    def __init__(self, args):
        super(CXRFlamingoPredictionHeadTransform, self).__init__()
        self.transform_act_fn = gelu
        hid_size = args.hidden_sz
        self.dense = nn.Linear(args.hidden_sz, hid_size)
        self.LayerNorm = nn.LayerNorm(hid_size, eps=1e-5)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class CXRFlamingoLMPredictionHead(nn.Module):
    def __init__(self, args, bert_model_embedding_weights):
        super(CXRFlamingoLMPredictionHead, self).__init__()
        self.transform = CXRFlamingoPredictionHeadTransform(args)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(bert_model_embedding_weights.size(1),
                                 bert_model_embedding_weights.size(0),
                                 bias=False)
        self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(
            bert_model_embedding_weights.size(0)))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states

class CXRFlamingoMLMHead(nn.Module):
    def __init__(self, args, bert_model_embedding_weights):
        super(CXRFlamingoMLMHead, self).__init__()
        self.predictions = CXRFlamingoLMPredictionHead(
            args, bert_model_embedding_weights)
    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = None
        return prediction_scores, seq_relationship_score



class CXRFlamingoEncoder(nn.Module):
    def __init__(self, args):
        super(CXRFlamingoEncoder,self).__init__()
        self.args = args

        language_model = CXRBertModel.from_pretrained(args.bert_model, revision="v1.1")
        self.image_model = get_biovil_resnet()
        freeze_model_and_make_eval_(language_model)
        freeze_model_and_make_eval_(self.image_model)

        self.dim = args.hidden_sz # 768
        self.img_attn_pool = CrossAttention(dim=self.dim, context_dim=args.img_hidden_sz, dim_head=64, heads=8, norm_context=True)
        self.img_attn_pool_norm = LayerNorm(self.dim)
        self.img_queries = nn.Parameter(torch.randn(15*15,self.dim))

        self.img_attn_pool_last = CrossAttention(dim=self.dim, context_dim=self.dim, dim_head=64, heads=8, norm_context=True)
        self.img_attn_pool_norm_last = LayerNorm(self.dim)

        self.perceiver_resampler = PerceiverResampler(
            dim=self.dim,
            depth=args.perceiver_depth,
            dim_head=64,
            heads=8,
            num_latents=64,       # the number of latents to shrink your media sequence to, perceiver style
            num_media_embeds = 2, ## max number of images per example
        )

        self.text_embdding = language_model.bert.embeddings
        lm_layers = language_model.bert.encoder.layer
        
        self.encoder_layers = nn.ModuleList([])
        for i in range(len(lm_layers)):
            self.encoder_layers.append(nn.ModuleList([
                GatedCrossAttentionBlock(dim=self.dim, dim_head=64, heads=12, only_attend_immediate_media=False) if not (i % args.cross_attn_every) else None,
                lm_layers[i]
            ])
        )


    def forward(self, findings, impression, image):
        txt, _, attention_mask = findings

        img = self.image_model(image).patch_embedding  
        img = img.resize(img.shape[0],15*15,self.args.img_hidden_sz)# img: Bx15x15x2048

        queries = repeat(self.img_queries, 'n d -> b n d', b=img.shape[0])
        img = self.img_attn_pool(queries, img)
        img = self.img_attn_pool_norm(img)

        img = self.perceiver_resampler(img)

        findings_embed = self.text_embdding(txt)
        impression_embed = None

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        for xattn_layer_front, lm_layer in self.encoder_layers:
            if exists(xattn_layer_front) and exists(img):
                findings_embed = xattn_layer_front(findings_embed, img)

            findings_embed = lm_layer(findings_embed, extended_attention_mask)[0]

        img = self.img_attn_pool_last(findings_embed[:,0,:].unsqueeze(1), img.squeeze(1))
        img = self.img_attn_pool_norm_last(img)

        return img, findings_embed, impression_embed


class CXRCoCa(nn.Module):
    def __init__(self, args):
        super(CXRCoCa,self).__init__()
        self.args = args
        self.pad_id = 0
        self.caption_loss_weight = 1.
        self.contrastive_loss_weight = 1.

        language_model = CXRBertModel.from_pretrained(args.bert_model, revision="v1.1")
        self.image_model = get_biovil_resnet()

        self.dim = args.hidden_sz # 768
        
        self.img_attn_pool = CrossAttention(dim=self.dim, context_dim=args.img_hidden_sz, dim_head=64, heads=8, norm_context=True)
        self.img_attn_pool_norm = LayerNorm(self.dim)
        self.text_cls_norm = LayerNorm(self.dim)
        self.img_queries = nn.Parameter(torch.randn(15*15,self.dim))

        self.perceiver_resampler = PerceiverResampler(
            dim=self.dim,
            depth=args.perceiver_depth,
            dim_head=args.perceiver_dim_head,
            heads=args.perceiver_num_head,
            num_latents=args.num_img_token+1,       # the number of latents to shrink your media sequence to, perceiver style 
                                                    # +1 for contrastive loss
            num_media_embeds = args.max_num_img, ## max number of images per example
        )

        self.text_embedding = language_model.bert.embeddings
        self.text_model = language_model.bert.encoder
        # freeze_model_and_make_eval_(self.text_embedding)
        # freeze_model_and_make_eval_(self.text_model)
        # freeze_model_and_make_eval_(self.image_model)
        
        # contrastive learning temperature
        self.temperature = nn.Parameter(torch.Tensor([1.]))

        # multimodal layers
        self.multimodal_layers = nn.ModuleList([])
        for ind in range(self.args.multimodal_depth):
            self.multimodal_layers.append(nn.ModuleList([
                Residual(ParallelTransformerBlock(dim=self.dim, dim_head=64, heads=8, ff_mult=4)),
                XResidual(CrossAttention(dim=self.dim, dim_head=64, heads=8, parallel_ff=True, ff_mult=4)) ## TODO: Apply GatedCrossAttentionBlock 
            ]))

        self.to_logits = nn.Sequential(
            LayerNorm(self.dim),
            nn.Linear(self.dim, args.n_classes, bias=False)
        )

    def embed_text(self, text, attention_mask):
        findings_embed = self.text_embedding(text)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        encoded_text = self.text_model(findings_embed, extended_attention_mask).last_hidden_state

        return self.text_cls_norm(encoded_text[:,0]), encoded_text

    def embed_image(self, image):

        img = self.image_model(image).patch_embedding  
        img = img.resize(img.shape[0],15*15,self.args.img_hidden_sz)# img: Bx15x15x2048

        queries = repeat(self.img_queries, 'n d -> b n d', b=img.shape[0])
        img = self.img_attn_pool(queries, img)
        img = self.img_attn_pool_norm(img)

        img = self.perceiver_resampler(img)
        img = img.squeeze(1)

        ## TODO: img_embeds normalization 필요한가?
        return img[:,-1], img[:,:-1] ##퍼시벌리셈플러 64+1개 만들어서 1개는 image_embeds로, 64개는 image_tokens로 리턴

    def forward(self, findings, impression, image):
        return_loss=False,
        return_embeddings=False

        txt, _, attention_mask = findings

        text_embeds, text_tokens = self.embed_text(txt, attention_mask)
        # impression_embed = self.text_embdding(txt)

        image_embeds, image_tokens = self.embed_image(image)

        # return embeddings if that is what the researcher wants
        if return_embeddings:
            return text_embeds, image_embeds

        for attn_ff, cross_attn in self.multimodal_layers:
            text_tokens = attn_ff(text_tokens)
            text_tokens = cross_attn(text_tokens, image_tokens)

        logits = self.to_logits(text_tokens[:,0])

        return logits
        # if not return_loss:
        #     return logits

        #### Pretrain 용 Loss 계산
        # # shorthand

        # ce = F.cross_entropy

        # # calculate caption loss (cross entropy loss)

        # logits = rearrange(logits, 'b n c -> b c n')
        # caption_loss = ce(logits, labels, ignore_index=self.pad_id)
        # caption_loss = caption_loss * self.caption_loss_weight

        # # calculate contrastive loss

        # sim = einsum('i d, j d -> i j', text_embeds, image_embeds)
        # sim = sim * self.temperature.exp()
        # contrastive_labels = torch.arange(batch, device=device)

        # contrastive_loss = (ce(sim, contrastive_labels) + ce(sim.t(), contrastive_labels)) * 0.5
        # contrastive_loss = contrastive_loss * self.contrastive_loss_weight

        # return caption_loss + contrastive_loss


# class XVL(nn.Module):
#     def __init__(self, args):
#         super(XVL,self).__init__()
#         import argparse
#         import ruamel.yaml as yaml
#         from transformers import AutoTokenizer
        
#         parser = argparse.ArgumentParser()
#         parser.add_argument('--config', default='./configs/Detection.yaml')
#         parser.add_argument('--output_dir', default='output/Detection')
#         parser.add_argument('--checkpoint', default='')
#         parser.add_argument('--text_encoder', default='bert-base-uncased')
#         parser.add_argument('--evaluate', action='store_true')
#         parser.add_argument('--device', default='cuda')
#         parser.add_argument('--seed', default=42, type=int)
#         parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
#         parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
#         parser.add_argument('--distributed', default=True, type=bool)
#         args = parser.parse_args()

#         config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
        
#         tokenizer = AutoTokenizer.from_pretrained("./my_tokenizer/")
#         xvlmodel = XVLModel(config, tokenizer)
    
#         print('done')

def get_XVL(get_text_encoder=False):
        visual_encoder = vit_small(
            img_size=(224, 224),
            patch_size=16,
            drop_path_rate=0.1,
            return_all_tokens=True,
            masked_im_modeling=False,
        )
        #visual_encoder = ibot_utils.MultiCropWrapper(visual_encoder, None)
        state_dict = torch.load('workspace/VLP_chest.pth')['model']
        pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.backbone.pos_embed'], visual_encoder)
        state_dict['visual_encoder.backbone.pos_embed'] = pos_embed_reshaped
        
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        for key in list(state_dict.keys()):
            if 'visual_encoder' in key:
                encoder_key = key.replace('visual_encoder.', '')
                state_dict[encoder_key] = state_dict[key]
                del state_dict[key]
            else:
                if not get_text_encoder:
                    del state_dict[key] 

        visual_encoder.load_state_dict(state_dict, strict=False)
        
        text_encoder=None
        if get_text_encoder:
            
            config = {
                "attention_probs_dropout_prob": 0.1,
                "hidden_act": "gelu",
                "hidden_dropout_prob": 0.1,
                "hidden_size": 384,
                "initializer_range": 0.02,
                "intermediate_size": 3072,
                "layer_norm_eps": 1e-12,
                "max_position_embeddings": 512,
                "model_type": "bert",
                "num_attention_heads": 12,
                "num_hidden_layers": 12,
                "pad_token_id": 0,
                "type_vocab_size": 2,
                "vocab_size": 30522,
                "fusion_layer": 12,
                "encoder_width": 384
            }
            
            bert_config = xbert.BertConfig.from_dict(config)
            text_encoder = xbert.BertModel(config=bert_config, add_pooling_layer=False)
            
            for key in list(state_dict.keys()):
                if 'text_encoder.bert.' in key:
                    encoder_key = key.replace('text_encoder.bert.', '')
                    state_dict[encoder_key] = state_dict[key]
                    del state_dict[key]
            
            text_encoder.load_state_dict(state_dict, strict=False)
        
        return visual_encoder, text_encoder