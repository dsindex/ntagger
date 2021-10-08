# ---------------------------------------------------------------------------------------------------------- #
# source from https://github.com/dlmacedo/entropic-out-of-distribution-detection/blob/master/losses/isomax.py
# ---------------------------------------------------------------------------------------------------------- #

import torch
import torch.nn as nn
import torch.nn.functional as F

class IsoMaxLoss(nn.Module):
    """Replaces the nn.CrossEntropyLoss()"""
    def __init__(self, model_classifier):
        super(IsoMaxLoss, self).__init__()
        self.model_classifier = model_classifier
        self.entropic_scale = 10.0

    def forward(self, logits, targets, debug=False):
        """Probabilities and logarithms are calculate separately and sequentially!!!"""
        """Therefore, nn.CrossEntropyLoss() must not be used to calculate the loss!!!"""
        probabilities_for_training = nn.Softmax(dim=1)(self.entropic_scale * logits[:len(targets)])
        probabilities_at_targets = probabilities_for_training[range(logits.size(0)), targets]
        loss = -torch.log(probabilities_at_targets).mean()
        if not debug:
            return loss
        else:
            targets_one_hot = torch.eye(self.model_classifier.prototypes.size(0))[targets].long().cuda()
            intra_inter_logits = torch.where(targets_one_hot != 0, -logits[:len(targets)], torch.Tensor([float('Inf')]).cuda())
            inter_intra_logits = torch.where(targets_one_hot != 0, torch.Tensor([float('Inf')]).cuda(), -logits[:len(targets)])
            intra_logits = intra_inter_logits[intra_inter_logits != float('Inf')]
            inter_logits = inter_intra_logits[inter_intra_logits != float('Inf')]
            distance_scale = 1
            return loss, distance_scale, intra_logits, inter_logits

