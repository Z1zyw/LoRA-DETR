# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou, box_iou


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self,
                 cost_class: float = 1,
                 cost_bbox: float = 1,
                 cost_giou: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    def forward(self, outputs, targets, return_iou=False):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        with torch.no_grad():
            bs, num_queries = outputs["pred_logits"].shape[:2]

            # We flatten to compute the cost matrices in a batch
            out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()
            out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

            # Also concat the target labels and boxes
            tgt_ids = torch.cat([v["labels"] for v in targets])
            tgt_bbox = torch.cat([v["boxes"] for v in targets])

            # Compute the classification cost. # Focal loss
            alpha = 0.25
            gamma = 2.0
            neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

            # Compute the L1 cost between boxes
            cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

            # Compute the giou cost betwen boxes
            if return_iou:
                cost_giou, iou = generalized_box_iou(box_cxcywh_to_xyxy(out_bbox),
                                             box_cxcywh_to_xyxy(tgt_bbox), return_iou=True)
                cost_giou = -cost_giou
            else:                
                cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox),
                                             box_cxcywh_to_xyxy(tgt_bbox))

            # Final cost matrix
            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
            C = C.view(bs, num_queries, -1).cpu()

            sizes = [len(v["boxes"]) for v in targets]
            indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]

            if return_iou:
                iou = iou.view(bs, num_queries, -1)
                iou = [iou_[i].T for i, iou_ in enumerate(iou.split(sizes, -1))]
                
                cls_score = out_prob[:, tgt_ids]
                cls_score = cls_score.view(bs, num_queries, -1)
                cls_score = [cls_[i].T for i, cls_ in enumerate(cls_score.split(sizes, -1))]
                
                return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices], iou, cls_score, C.split(sizes, -1)
            
            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


class CollaborativeHungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self,
        cost_class: float = 1,
        cost_bbox: float = 1,
        cost_giou: float = 1,
        col_alpha: float = 0.5
    ):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.col_alpha = col_alpha
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    def update_cost_class(self, cost_class):
        """ Update the cost class """
        self.cost_class = cost_class
    
    def forward(self, outputs, targets, cost_col=None, return_iou=False, use_quality_score=False):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        with torch.no_grad():
            bs, num_queries = outputs["pred_logits"].shape[:2]

            # We flatten to compute the cost matrices in a batch
            out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()
            out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

            # Also concat the target labels and boxes
            tgt_ids = torch.cat([v["labels"] for v in targets])
            tgt_bbox = torch.cat([v["boxes"] for v in targets])

            # Compute the giou cost betwen boxes
            if return_iou:
                cost_giou, iou = generalized_box_iou(box_cxcywh_to_xyxy(out_bbox),
                                             box_cxcywh_to_xyxy(tgt_bbox), return_iou=True)
                cost_giou = -cost_giou
            else:                
                cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox),
                                             box_cxcywh_to_xyxy(tgt_bbox))
            
            if not use_quality_score:
                # Compute the classification cost. # Focal loss
                alpha = 0.25
                gamma = 2.0
                neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
                pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
                cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]
            else:
                # Use quality score as the cost class
                assert return_iou
                # v1：match_gamma = 0.5
                # v2 添加： match_gamma = 1  cost_class *= 0.5
                base_scale = 0.25
                focal_gamma = 1.5 # for Negative
                match_gamma = 1.0  # iou ** match_gamma
                neg_cost_class = (out_prob ** focal_gamma) * (-(1 - out_prob + 1e-8).log())
                out_prob_tgt = out_prob[:, tgt_ids]
                iou_tgt = iou ** match_gamma
                pos_cost_class = - iou_tgt * (out_prob_tgt + 1e-8).log() - (1-iou_tgt) * (1 - out_prob_tgt + 1e-8).log()
                cost_class = pos_cost_class * base_scale - neg_cost_class[:, tgt_ids]
                cost_class = cost_class * 0.5 # 由于 原始 cost_class 是 2.0， 而 loss 的权重是 1.0
                

            # Compute the L1 cost between boxes
            cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)


            # Final cost matrix
            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
            C = C.view(bs, num_queries, -1).cpu()
            
            C_output = C
            if cost_col is not None:
                C = self.col_alpha * C + (1 - self.col_alpha) * cost_col.view(bs, num_queries, -1).cpu()

            sizes = [len(v["boxes"]) for v in targets]
            indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]

            if return_iou:
                iou = iou.view(bs, num_queries, -1)
                iou = [iou_[i].T for i, iou_ in enumerate(iou.split(sizes, -1))]
                
                cls_score = out_prob[:, tgt_ids]
                cls_score = cls_score.view(bs, num_queries, -1)
                cls_score = [cls_[i].T for i, cls_ in enumerate(cls_score.split(sizes, -1))]
                
                return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices], iou, cls_score, C.split(sizes, -1), C_output
            
            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

def build_matcher(args):
    return HungarianMatcher(cost_class=args.set_cost_class,
                            cost_bbox=args.set_cost_bbox,
                            cost_giou=args.set_cost_giou)
    
def build_matcher_collaborative(args): 
    # import ipdb; ipdb.set_trace()
    return CollaborativeHungarianMatcher(cost_class=args.set_cost_class,
                                          cost_bbox=args.set_cost_bbox,
                                          cost_giou=args.set_cost_giou,
                                          col_alpha=1.0)

class SampleHungarianMatcher(nn.Module):
    def __init__(self,
                 coef_box: float = 0.7,
                 coef_cls: float = 0.3):
        super().__init__()
        self.coef_box = coef_box
        self.coef_cls = coef_cls
    
    
    def forward(self, outputs, targets, return_cost_matrix=True):
        with torch.no_grad():
            bs, num_queries = outputs["pred_logits"].shape[:2]
            indices = []
            iou_matrices = []
            cls_matrices = []
            cost_matrices = [] 

            for b in range(bs):
                pred_boxes = outputs["pred_boxes"][b]
                pred_logits = outputs["pred_logits"][b]
                
                gt_boxes = targets[b]["boxes"]
                gt_labels = targets[b]["labels"]
                
                # Compute the cost
                out_prob = pred_logits.sigmoid()
                out_bbox = pred_boxes
                
                cost_box = box_iou(
                    box_cxcywh_to_xyxy(out_bbox),
                    box_cxcywh_to_xyxy(gt_boxes))[0] # iou
                
                iou_matrices.append(cost_box.T)
                
                cost_cls = out_prob[:, gt_labels]
                cls_matrices.append(cost_cls.T)
                
                cost = self.coef_box * cost_box + self.coef_cls * cost_cls
                
                cost = cost.view(num_queries, -1)

                # find the best matching by Hungarian algorithm
                index = linear_sum_assignment(cost.cpu(), maximize=True)
                indices.append(index)
                cost_matrices.append(cost.T)
                
            indices = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
            
            if return_cost_matrix:
                return indices, iou_matrices, cls_matrices, cost_matrices
                
            return indices