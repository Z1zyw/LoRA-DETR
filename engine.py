# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable

import torch
import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator
from datasets.data_prefetcher import data_prefetcher

# USING_CONFLICT_FREE = True
# from conflictfree.grad_operator import ConFIG_update
# from conflictfree.utils import get_gradient_vector,apply_gradient_vector

RECORD_GRAD_NORM = True
# GRAD_LIST = [
#     'backbone.0',
#     'encoder',
#     'decoder',
#     'lora',
# ]
NAMES = ['module.transformer.level_embed', 'module.transformer.encoder.layers.0.self_attn.sampling_offsets.weight', 'module.transformer.encoder.layers.0.self_attn.sampling_offsets.bias', 'module.transformer.encoder.layers.0.self_attn.attention_weights.weight', 'module.transformer.encoder.layers.0.self_attn.attention_weights.bias', 'module.transformer.encoder.layers.0.self_attn.value_proj.weight', 'module.transformer.encoder.layers.0.self_attn.value_proj.bias', 'module.transformer.encoder.layers.0.self_attn.output_proj.weight', 'module.transformer.encoder.layers.0.self_attn.output_proj.bias', 'module.transformer.encoder.layers.0.norm1.weight', 'module.transformer.encoder.layers.0.norm1.bias', 'module.transformer.encoder.layers.0.linear1.weight', 'module.transformer.encoder.layers.0.linear1.bias', 'module.transformer.encoder.layers.0.linear2.weight', 'module.transformer.encoder.layers.0.linear2.bias', 'module.transformer.encoder.layers.0.norm2.weight', 'module.transformer.encoder.layers.0.norm2.bias', 'module.transformer.encoder.layers.1.self_attn.sampling_offsets.weight', 'module.transformer.encoder.layers.1.self_attn.sampling_offsets.bias', 'module.transformer.encoder.layers.1.self_attn.attention_weights.weight', 'module.transformer.encoder.layers.1.self_attn.attention_weights.bias', 'module.transformer.encoder.layers.1.self_attn.value_proj.weight', 'module.transformer.encoder.layers.1.self_attn.value_proj.bias', 'module.transformer.encoder.layers.1.self_attn.output_proj.weight', 'module.transformer.encoder.layers.1.self_attn.output_proj.bias', 'module.transformer.encoder.layers.1.norm1.weight', 'module.transformer.encoder.layers.1.norm1.bias', 'module.transformer.encoder.layers.1.linear1.weight', 'module.transformer.encoder.layers.1.linear1.bias', 'module.transformer.encoder.layers.1.linear2.weight', 'module.transformer.encoder.layers.1.linear2.bias', 'module.transformer.encoder.layers.1.norm2.weight', 'module.transformer.encoder.layers.1.norm2.bias', 'module.transformer.encoder.layers.2.self_attn.sampling_offsets.weight', 'module.transformer.encoder.layers.2.self_attn.sampling_offsets.bias', 'module.transformer.encoder.layers.2.self_attn.attention_weights.weight', 'module.transformer.encoder.layers.2.self_attn.attention_weights.bias', 'module.transformer.encoder.layers.2.self_attn.value_proj.weight', 'module.transformer.encoder.layers.2.self_attn.value_proj.bias', 'module.transformer.encoder.layers.2.self_attn.output_proj.weight', 'module.transformer.encoder.layers.2.self_attn.output_proj.bias', 'module.transformer.encoder.layers.2.norm1.weight', 'module.transformer.encoder.layers.2.norm1.bias', 'module.transformer.encoder.layers.2.linear1.weight', 'module.transformer.encoder.layers.2.linear1.bias', 'module.transformer.encoder.layers.2.linear2.weight', 'module.transformer.encoder.layers.2.linear2.bias', 'module.transformer.encoder.layers.2.norm2.weight', 'module.transformer.encoder.layers.2.norm2.bias', 'module.transformer.encoder.layers.3.self_attn.sampling_offsets.weight', 'module.transformer.encoder.layers.3.self_attn.sampling_offsets.bias', 'module.transformer.encoder.layers.3.self_attn.attention_weights.weight', 'module.transformer.encoder.layers.3.self_attn.attention_weights.bias', 'module.transformer.encoder.layers.3.self_attn.value_proj.weight', 'module.transformer.encoder.layers.3.self_attn.value_proj.bias', 'module.transformer.encoder.layers.3.self_attn.output_proj.weight', 'module.transformer.encoder.layers.3.self_attn.output_proj.bias', 'module.transformer.encoder.layers.3.norm1.weight', 'module.transformer.encoder.layers.3.norm1.bias', 'module.transformer.encoder.layers.3.linear1.weight', 'module.transformer.encoder.layers.3.linear1.bias', 'module.transformer.encoder.layers.3.linear2.weight', 'module.transformer.encoder.layers.3.linear2.bias', 'module.transformer.encoder.layers.3.norm2.weight', 'module.transformer.encoder.layers.3.norm2.bias', 'module.transformer.encoder.layers.4.self_attn.sampling_offsets.weight', 'module.transformer.encoder.layers.4.self_attn.sampling_offsets.bias', 'module.transformer.encoder.layers.4.self_attn.attention_weights.weight', 'module.transformer.encoder.layers.4.self_attn.attention_weights.bias', 'module.transformer.encoder.layers.4.self_attn.value_proj.weight', 'module.transformer.encoder.layers.4.self_attn.value_proj.bias', 'module.transformer.encoder.layers.4.self_attn.output_proj.weight', 'module.transformer.encoder.layers.4.self_attn.output_proj.bias', 'module.transformer.encoder.layers.4.norm1.weight', 'module.transformer.encoder.layers.4.norm1.bias', 'module.transformer.encoder.layers.4.linear1.weight', 'module.transformer.encoder.layers.4.linear1.bias', 'module.transformer.encoder.layers.4.linear2.weight', 'module.transformer.encoder.layers.4.linear2.bias', 'module.transformer.encoder.layers.4.norm2.weight', 'module.transformer.encoder.layers.4.norm2.bias', 'module.transformer.encoder.layers.5.self_attn.sampling_offsets.weight', 'module.transformer.encoder.layers.5.self_attn.sampling_offsets.bias', 'module.transformer.encoder.layers.5.self_attn.attention_weights.weight', 'module.transformer.encoder.layers.5.self_attn.attention_weights.bias', 'module.transformer.encoder.layers.5.self_attn.value_proj.weight', 'module.transformer.encoder.layers.5.self_attn.value_proj.bias', 'module.transformer.encoder.layers.5.self_attn.output_proj.weight', 'module.transformer.encoder.layers.5.self_attn.output_proj.bias', 'module.transformer.encoder.layers.5.norm1.weight', 'module.transformer.encoder.layers.5.norm1.bias', 'module.transformer.encoder.layers.5.linear1.weight', 'module.transformer.encoder.layers.5.linear1.bias', 'module.transformer.encoder.layers.5.linear2.weight', 'module.transformer.encoder.layers.5.linear2.bias', 'module.transformer.encoder.layers.5.norm2.weight', 'module.transformer.encoder.layers.5.norm2.bias', 'module.transformer.decoder.layers.0.cross_attn.sampling_offsets.weight', 'module.transformer.decoder.layers.0.cross_attn.sampling_offsets.bias', 'module.transformer.decoder.layers.0.cross_attn.attention_weights.weight', 'module.transformer.decoder.layers.0.cross_attn.attention_weights.bias', 'module.transformer.decoder.layers.0.cross_attn.value_proj.weight', 'module.transformer.decoder.layers.0.cross_attn.value_proj.bias', 'module.transformer.decoder.layers.0.cross_attn.output_proj.weight', 'module.transformer.decoder.layers.0.cross_attn.output_proj.bias', 'module.transformer.decoder.layers.0.norm1.weight', 'module.transformer.decoder.layers.0.norm1.bias', 'module.transformer.decoder.layers.0.self_attn.in_proj_weight', 'module.transformer.decoder.layers.0.self_attn.in_proj_bias', 'module.transformer.decoder.layers.0.self_attn.out_proj.weight', 'module.transformer.decoder.layers.0.self_attn.out_proj.bias', 'module.transformer.decoder.layers.0.norm2.weight', 'module.transformer.decoder.layers.0.norm2.bias', 'module.transformer.decoder.layers.0.linear1.weight', 'module.transformer.decoder.layers.0.linear1.bias', 'module.transformer.decoder.layers.0.linear2.weight', 'module.transformer.decoder.layers.0.linear2.bias', 'module.transformer.decoder.layers.0.norm3.weight', 'module.transformer.decoder.layers.0.norm3.bias', 'module.transformer.decoder.layers.0.linear1_lora.A', 'module.transformer.decoder.layers.0.linear1_lora.B', 'module.transformer.decoder.layers.0.linear2_lora.A', 'module.transformer.decoder.layers.0.linear2_lora.B', 'module.transformer.decoder.layers.1.cross_attn.sampling_offsets.weight', 'module.transformer.decoder.layers.1.cross_attn.sampling_offsets.bias', 'module.transformer.decoder.layers.1.cross_attn.attention_weights.weight', 'module.transformer.decoder.layers.1.cross_attn.attention_weights.bias', 'module.transformer.decoder.layers.1.cross_attn.value_proj.weight', 'module.transformer.decoder.layers.1.cross_attn.value_proj.bias', 'module.transformer.decoder.layers.1.cross_attn.output_proj.weight', 'module.transformer.decoder.layers.1.cross_attn.output_proj.bias', 'module.transformer.decoder.layers.1.norm1.weight', 'module.transformer.decoder.layers.1.norm1.bias', 'module.transformer.decoder.layers.1.self_attn.in_proj_weight', 'module.transformer.decoder.layers.1.self_attn.in_proj_bias', 'module.transformer.decoder.layers.1.self_attn.out_proj.weight', 'module.transformer.decoder.layers.1.self_attn.out_proj.bias', 'module.transformer.decoder.layers.1.norm2.weight', 'module.transformer.decoder.layers.1.norm2.bias', 'module.transformer.decoder.layers.1.linear1.weight', 'module.transformer.decoder.layers.1.linear1.bias', 'module.transformer.decoder.layers.1.linear2.weight', 'module.transformer.decoder.layers.1.linear2.bias', 'module.transformer.decoder.layers.1.norm3.weight', 'module.transformer.decoder.layers.1.norm3.bias', 'module.transformer.decoder.layers.1.linear1_lora.A', 'module.transformer.decoder.layers.1.linear1_lora.B', 'module.transformer.decoder.layers.1.linear2_lora.A', 'module.transformer.decoder.layers.1.linear2_lora.B', 'module.transformer.decoder.layers.2.cross_attn.sampling_offsets.weight', 'module.transformer.decoder.layers.2.cross_attn.sampling_offsets.bias', 'module.transformer.decoder.layers.2.cross_attn.attention_weights.weight', 'module.transformer.decoder.layers.2.cross_attn.attention_weights.bias', 'module.transformer.decoder.layers.2.cross_attn.value_proj.weight', 'module.transformer.decoder.layers.2.cross_attn.value_proj.bias', 'module.transformer.decoder.layers.2.cross_attn.output_proj.weight', 'module.transformer.decoder.layers.2.cross_attn.output_proj.bias', 'module.transformer.decoder.layers.2.norm1.weight', 'module.transformer.decoder.layers.2.norm1.bias', 'module.transformer.decoder.layers.2.self_attn.in_proj_weight', 'module.transformer.decoder.layers.2.self_attn.in_proj_bias', 'module.transformer.decoder.layers.2.self_attn.out_proj.weight', 'module.transformer.decoder.layers.2.self_attn.out_proj.bias', 'module.transformer.decoder.layers.2.norm2.weight', 'module.transformer.decoder.layers.2.norm2.bias', 'module.transformer.decoder.layers.2.linear1.weight', 'module.transformer.decoder.layers.2.linear1.bias', 'module.transformer.decoder.layers.2.linear2.weight', 'module.transformer.decoder.layers.2.linear2.bias', 'module.transformer.decoder.layers.2.norm3.weight', 'module.transformer.decoder.layers.2.norm3.bias', 'module.transformer.decoder.layers.2.linear1_lora.A', 'module.transformer.decoder.layers.2.linear1_lora.B', 'module.transformer.decoder.layers.2.linear2_lora.A', 'module.transformer.decoder.layers.2.linear2_lora.B', 'module.transformer.decoder.layers.3.cross_attn.sampling_offsets.weight', 'module.transformer.decoder.layers.3.cross_attn.sampling_offsets.bias', 'module.transformer.decoder.layers.3.cross_attn.attention_weights.weight', 'module.transformer.decoder.layers.3.cross_attn.attention_weights.bias', 'module.transformer.decoder.layers.3.cross_attn.value_proj.weight', 'module.transformer.decoder.layers.3.cross_attn.value_proj.bias', 'module.transformer.decoder.layers.3.cross_attn.output_proj.weight', 'module.transformer.decoder.layers.3.cross_attn.output_proj.bias', 'module.transformer.decoder.layers.3.norm1.weight', 'module.transformer.decoder.layers.3.norm1.bias', 'module.transformer.decoder.layers.3.self_attn.in_proj_weight', 'module.transformer.decoder.layers.3.self_attn.in_proj_bias', 'module.transformer.decoder.layers.3.self_attn.out_proj.weight', 'module.transformer.decoder.layers.3.self_attn.out_proj.bias', 'module.transformer.decoder.layers.3.norm2.weight', 'module.transformer.decoder.layers.3.norm2.bias', 'module.transformer.decoder.layers.3.linear1.weight', 'module.transformer.decoder.layers.3.linear1.bias', 'module.transformer.decoder.layers.3.linear2.weight', 'module.transformer.decoder.layers.3.linear2.bias', 'module.transformer.decoder.layers.3.norm3.weight', 'module.transformer.decoder.layers.3.norm3.bias', 'module.transformer.decoder.layers.3.linear1_lora.A', 'module.transformer.decoder.layers.3.linear1_lora.B', 'module.transformer.decoder.layers.3.linear2_lora.A', 'module.transformer.decoder.layers.3.linear2_lora.B', 'module.transformer.decoder.layers.4.cross_attn.sampling_offsets.weight', 'module.transformer.decoder.layers.4.cross_attn.sampling_offsets.bias', 'module.transformer.decoder.layers.4.cross_attn.attention_weights.weight', 'module.transformer.decoder.layers.4.cross_attn.attention_weights.bias', 'module.transformer.decoder.layers.4.cross_attn.value_proj.weight', 'module.transformer.decoder.layers.4.cross_attn.value_proj.bias', 'module.transformer.decoder.layers.4.cross_attn.output_proj.weight', 'module.transformer.decoder.layers.4.cross_attn.output_proj.bias', 'module.transformer.decoder.layers.4.norm1.weight', 'module.transformer.decoder.layers.4.norm1.bias', 'module.transformer.decoder.layers.4.self_attn.in_proj_weight', 'module.transformer.decoder.layers.4.self_attn.in_proj_bias', 'module.transformer.decoder.layers.4.self_attn.out_proj.weight', 'module.transformer.decoder.layers.4.self_attn.out_proj.bias', 'module.transformer.decoder.layers.4.norm2.weight', 'module.transformer.decoder.layers.4.norm2.bias', 'module.transformer.decoder.layers.4.linear1.weight', 'module.transformer.decoder.layers.4.linear1.bias', 'module.transformer.decoder.layers.4.linear2.weight', 'module.transformer.decoder.layers.4.linear2.bias', 'module.transformer.decoder.layers.4.norm3.weight', 'module.transformer.decoder.layers.4.norm3.bias', 'module.transformer.decoder.layers.4.linear1_lora.A', 'module.transformer.decoder.layers.4.linear1_lora.B', 'module.transformer.decoder.layers.4.linear2_lora.A', 'module.transformer.decoder.layers.4.linear2_lora.B', 'module.transformer.decoder.layers.5.cross_attn.sampling_offsets.weight', 'module.transformer.decoder.layers.5.cross_attn.sampling_offsets.bias', 'module.transformer.decoder.layers.5.cross_attn.attention_weights.weight', 'module.transformer.decoder.layers.5.cross_attn.attention_weights.bias', 'module.transformer.decoder.layers.5.cross_attn.value_proj.weight', 'module.transformer.decoder.layers.5.cross_attn.value_proj.bias', 'module.transformer.decoder.layers.5.cross_attn.output_proj.weight', 'module.transformer.decoder.layers.5.cross_attn.output_proj.bias', 'module.transformer.decoder.layers.5.norm1.weight', 'module.transformer.decoder.layers.5.norm1.bias', 'module.transformer.decoder.layers.5.self_attn.in_proj_weight', 'module.transformer.decoder.layers.5.self_attn.in_proj_bias', 'module.transformer.decoder.layers.5.self_attn.out_proj.weight', 'module.transformer.decoder.layers.5.self_attn.out_proj.bias', 'module.transformer.decoder.layers.5.norm2.weight', 'module.transformer.decoder.layers.5.norm2.bias', 'module.transformer.decoder.layers.5.linear1.weight', 'module.transformer.decoder.layers.5.linear1.bias', 'module.transformer.decoder.layers.5.linear2.weight', 'module.transformer.decoder.layers.5.linear2.bias', 'module.transformer.decoder.layers.5.norm3.weight', 'module.transformer.decoder.layers.5.norm3.bias', 'module.transformer.decoder.layers.5.linear1_lora.A', 'module.transformer.decoder.layers.5.linear1_lora.B', 'module.transformer.decoder.layers.5.linear2_lora.A', 'module.transformer.decoder.layers.5.linear2_lora.B', 'module.transformer.decoder.bbox_embed.0.layers.0.weight', 'module.transformer.decoder.bbox_embed.0.layers.0.bias', 'module.transformer.decoder.bbox_embed.0.layers.1.weight', 'module.transformer.decoder.bbox_embed.0.layers.1.bias', 'module.transformer.decoder.bbox_embed.0.layers.2.weight', 'module.transformer.decoder.bbox_embed.0.layers.2.bias', 'module.transformer.decoder.bbox_embed.1.layers.0.weight', 'module.transformer.decoder.bbox_embed.1.layers.0.bias', 'module.transformer.decoder.bbox_embed.1.layers.1.weight', 'module.transformer.decoder.bbox_embed.1.layers.1.bias', 'module.transformer.decoder.bbox_embed.1.layers.2.weight', 'module.transformer.decoder.bbox_embed.1.layers.2.bias', 'module.transformer.decoder.bbox_embed.2.layers.0.weight', 'module.transformer.decoder.bbox_embed.2.layers.0.bias', 'module.transformer.decoder.bbox_embed.2.layers.1.weight', 'module.transformer.decoder.bbox_embed.2.layers.1.bias', 'module.transformer.decoder.bbox_embed.2.layers.2.weight', 'module.transformer.decoder.bbox_embed.2.layers.2.bias', 'module.transformer.decoder.bbox_embed.3.layers.0.weight', 'module.transformer.decoder.bbox_embed.3.layers.0.bias', 'module.transformer.decoder.bbox_embed.3.layers.1.weight', 'module.transformer.decoder.bbox_embed.3.layers.1.bias', 'module.transformer.decoder.bbox_embed.3.layers.2.weight', 'module.transformer.decoder.bbox_embed.3.layers.2.bias', 'module.transformer.decoder.bbox_embed.4.layers.0.weight', 'module.transformer.decoder.bbox_embed.4.layers.0.bias', 'module.transformer.decoder.bbox_embed.4.layers.1.weight', 'module.transformer.decoder.bbox_embed.4.layers.1.bias', 'module.transformer.decoder.bbox_embed.4.layers.2.weight', 'module.transformer.decoder.bbox_embed.4.layers.2.bias', 'module.transformer.decoder.bbox_embed.5.layers.0.weight', 'module.transformer.decoder.bbox_embed.5.layers.0.bias', 'module.transformer.decoder.bbox_embed.5.layers.1.weight', 'module.transformer.decoder.bbox_embed.5.layers.1.bias', 'module.transformer.decoder.bbox_embed.5.layers.2.weight', 'module.transformer.decoder.bbox_embed.5.layers.2.bias', 'module.transformer.decoder.bbox_embed.6.layers.0.weight', 'module.transformer.decoder.bbox_embed.6.layers.0.bias', 'module.transformer.decoder.bbox_embed.6.layers.1.weight', 'module.transformer.decoder.bbox_embed.6.layers.1.bias', 'module.transformer.decoder.bbox_embed.6.layers.2.weight', 'module.transformer.decoder.bbox_embed.6.layers.2.bias', 'module.transformer.decoder.class_embed.0.weight', 'module.transformer.decoder.class_embed.0.bias', 'module.transformer.decoder.class_embed.1.weight', 'module.transformer.decoder.class_embed.1.bias', 'module.transformer.decoder.class_embed.2.weight', 'module.transformer.decoder.class_embed.2.bias', 'module.transformer.decoder.class_embed.3.weight', 'module.transformer.decoder.class_embed.3.bias', 'module.transformer.decoder.class_embed.4.weight', 'module.transformer.decoder.class_embed.4.bias', 'module.transformer.decoder.class_embed.5.weight', 'module.transformer.decoder.class_embed.5.bias', 'module.transformer.decoder.class_embed.6.weight', 'module.transformer.decoder.class_embed.6.bias', 'module.transformer.enc_output.weight', 'module.transformer.enc_output.bias', 'module.transformer.enc_output_norm.weight', 'module.transformer.enc_output_norm.bias', 'module.transformer.pos_trans.weight', 'module.transformer.pos_trans.bias', 'module.transformer.pos_trans_norm.weight', 'module.transformer.pos_trans_norm.bias', 'module.input_proj.0.0.weight', 'module.input_proj.0.0.bias', 'module.input_proj.0.1.weight', 'module.input_proj.0.1.bias', 'module.input_proj.1.0.weight', 'module.input_proj.1.0.bias', 'module.input_proj.1.1.weight', 'module.input_proj.1.1.bias', 'module.input_proj.2.0.weight', 'module.input_proj.2.0.bias', 'module.input_proj.2.1.weight', 'module.input_proj.2.1.bias', 'module.input_proj.3.0.weight', 'module.input_proj.3.0.bias', 'module.input_proj.3.1.weight', 'module.input_proj.3.1.bias']
# def get_params_from_name(model, name):
#     params = []
#     import ipdb; ipdb.set_trace()
#     for n, p in model.named_parameters():
#         if name in n:
#             params.append(p)
#     return params
 
# def get_grad_norm_from_name(model, name):
#     params = get_params_from_name(model, name)
#     grad_norm = utils.get_total_grad_norm(params, norm_type=2)
#     return grad_norm

def get_all_params_grad_norm_dict(model, norm_type=2):
    grad_norm_dict = {}
    for n, p in model.named_parameters():
        if p.grad is not None:
            grad_norm_dict[n] = torch.norm(p.grad.detach(), p=norm_type)

    # sum_grad = 0
    # for k, v in grad_norm_dict.items():
    #     sum_grad += v ** 2
    # sum_grad = sum_grad ** 0.5
    
    return grad_norm_dict

def train_one_epoch_test_time(model: torch.nn.Module, criterion: torch.nn.Module,   
                                          data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):        
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    
    header = 'Epoch: [{}]'.format(epoch)
    
    total_test_iter = 500
    print_freq = 50
    
    prefetcher = data_prefetcher(data_loader, device, prefetch=True)
    samples, targets = prefetcher.next()
    
    print('Start test time: Total', total_test_iter, ' iterations')
    import time
    time_dict = {
        'forward': 0,
        'criterion': 0, #
        'backward': 0,
        'update': 0,
        'data_prefetch': 0,
        'log': 0,
    }
    cnt = 0
    
    for _ in metric_logger.log_every(range(len(data_loader)), print_freq, header):
        now = time.time()
        outputs = model(samples)
        
        time_dict['forward'] += time.time() - now
        now = time.time()
        
        loss_dict = criterion(outputs, targets, model)
        
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                        for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)
        
        time_dict['criterion'] += time.time() - now
        now = time.time()
        
        optimizer.zero_grad()
        losses.backward()
        
        time_dict['backward'] += time.time() - now
        now = time.time()
        
        
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), norm_type=2, max_norm=max_norm)
            
        optimizer.step()
        time_dict['update'] += time.time() - now
        
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)
        time_dict['log'] += time.time() - now
        now = time.time()
        
        
        samples, targets = prefetcher.next()
        time_dict['data_prefetch'] += time.time() - now
            
        cnt += 1
        if cnt > total_test_iter:
            break   
    time_dict['total'] = sum(time_dict.values())
    print(time_dict)
    return time_dict
        


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    # for name in NAMES:
    #     metric_logger.add_meter(f'grad_norm_{name}', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 50

    prefetcher = data_prefetcher(data_loader, device, prefetch=True)
    samples, targets = prefetcher.next()

    # for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
    for _ in metric_logger.log_every(range(len(data_loader)), print_freq, header):
        outputs = model(samples)
        loss_dict = criterion(outputs, targets) 
        weight_dict = criterion.weight_dict

        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)    

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        # gard_dict = get_all_params_grad_norm_dict(model)
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), norm_type=2, max_norm=max_norm)
     
        # for name in GRAD_LIST:
        #     grad_norm = get_grad_norm_from_name(model, name)
        #     metric_logger.update(**{f'grad_norm_{name}': grad_norm})
        
        optimizer.step()


    

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)
        # for k, v in gard_dict.items():
        #     if 'backbone' in k:
        #         continue
        #     metric_logger.update(**{f'grad_norm_{k}': v})

        samples, targets = prefetcher.next()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples,targets=targets)
        loss_dict = criterion(outputs, targets, model)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]
    return stats, coco_evaluator
