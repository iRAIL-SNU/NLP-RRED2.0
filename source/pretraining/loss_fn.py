import torch
import torch.nn as nn


class RadiologySelectionLoss(nn.Module):
    def __init__(self, t=.5):
        super(RadiologySelectionLoss, self).__init__()
        self.t = t
        self.f2i = nn.LogSoftmax(dim=1)
        self.i2f = nn.LogSoftmax(dim=0)       

    def forward(self, findings_embeddings, impression_embeddings, chexpert_label): 
        n = findings_embeddings.shape[0]
        
        # if chexpert_label is same, do not consider as negative sample
        label_equal_matrix = []
        for i in range(n):
            label_equal_matrix.append([chexpert_label[i]==chexpert_label[j] if i!=j else False for j in range(n)])
        
        dot_mat = torch.matmul(findings_embeddings,impression_embeddings.transpose(1,0))/self.t
        
        dot_mat = dot_mat * ~torch.tensor(label_equal_matrix).to('cuda') 
        
        loss = -torch.trace(self.f2i(dot_mat) + self.i2f(dot_mat)) / n
        
        return loss   