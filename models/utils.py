import torch
import torch.nn.functional as F
import ipdb 


def match_aware_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2, match_scores=None):
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    
    ipdb.set_trace()
    
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes