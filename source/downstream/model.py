import torch
import torch.nn as nn
import torch.nn.functional as F

from health_multimodal.text.model import CXRBertModel
from health_multimodal.image.model import get_biovil_resnet

MODEL_TYPE = "resnet50"


class VLModelClf(nn.Module):
    def __init__(self, args):
        super(VLModelClf, self).__init__()
        
        self.args = args
        
        self.text_model = CXRBertModel.from_pretrained(args.bert_model, revision="v1.1")
        
        self.image_model = get_biovil_resnet()
        
        # resnet_checkpoint_path = _download_biovil_image_model_weights()# if pretrained else None
        # self.image_model = ImageModel(
        #     img_model_type=MODEL_TYPE,
        #     joint_feature_size=args.JOINT_FEATURE_SIZE,
        #     pretrained_model_path=resnet_checkpoint_path,
        # )

        self.clf = nn.Linear(args.JOINT_FEATURE_SIZE*2, args.n_classes)

    def forward(self, findings, impression, image):
        image_embed = self.image_model(image)
        image_embed = image_embed.projected_global_embedding

        txt, _, mask = findings
        findings_embed = self.text_model.get_projected_text_embeddings(txt, mask)
        return self.clf(torch.cat((image_embed, findings_embed), 1))

        # txt, _, mask = impression
        # impression_embed = self.text_model.get_projected_text_embeddings(txt, mask)
        # return self.clf(torch.cat((image_embed, impression_embed), 1))
