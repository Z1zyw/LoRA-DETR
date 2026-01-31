# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

from .deformable_detr import build
from .lora_multi_deformable_detr_w_mfl_2 import build as build_multi_lora_w_mfl_2
from .naive_multi_branch_deformable_detr import build as build_multi_naive

def build_model(args):
    return build(args)

