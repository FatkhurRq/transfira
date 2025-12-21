import torch.nn as nn
from typing import Any, Dict


class RecognizabilityPredictionNetwork(nn.Module):
    """
    TransFIRA Recognizability Prediction Network.

    Predicts CCS (Class Center Similarity) and CCAS (Class Center Angular Separation)
    for face image recognizability assessment as described in the TransFIRA paper.
    """

    def __init__(
        self,
        gpu_id: int,
        backbone,
        outdim,
        kwargs: Dict[str, Any] = {},
    ):
        super(RecognizabilityPredictionNetwork, self).__init__()

        self.backbone = backbone
        self.backbone.train()
        for param in self.backbone.parameters():
            param.requires_grad = True

        # CCS prediction head
        self.ccs_layer = nn.Linear(outdim, 1, bias=True)
        nn.init.xavier_uniform_(self.ccs_layer.weight)
        nn.init.constant_(self.ccs_layer.bias, 0.0)
        self.ccs_layer.weight.requires_grad_(True)
        self.ccs_layer.bias.requires_grad_(True)

        # CCAS prediction head
        self.ccas_layer = nn.Linear(outdim, 1, bias=True)
        nn.init.xavier_uniform_(self.ccas_layer.weight)
        nn.init.constant_(self.ccas_layer.bias, 0.0)
        self.ccas_layer.weight.requires_grad_(True)
        self.ccas_layer.bias.requires_grad_(True)

        self.backbone.to(gpu_id)
        self.ccs_layer.to(gpu_id)
        self.ccas_layer.to(gpu_id)
        self.to(gpu_id)

    def forward(self, img):
        feature = self.backbone(img)
        ccs = self.ccs_layer(feature)
        ccas = self.ccas_layer(feature)
        return ccs, ccas
