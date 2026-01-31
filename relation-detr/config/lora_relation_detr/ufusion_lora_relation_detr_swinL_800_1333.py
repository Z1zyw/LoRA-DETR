from torch import nn

from models.backbones.swin import SwinTransformerBackbone
from models.bricks.position_encoding import PositionEmbeddingSine
from models.bricks.post_process import PostProcess
from models.bricks.relation_transformer import (
    UFusionRelationTransformerWithLoRA,
    UFusionRelationTransformerDecoderWithLoRA,
    UFusionRelationTransformerEncoder,
    RelationTransformerDecoderLayerWithLoRA,
    RelationTransformerEncoderLayer,
)
from models.bricks.set_criterion import HybridSetCriterion, HybirdSetCriterionWithO2M
from models.detectors.relation_detr import RelationDETRwithLoRA
from models.matcher.hungarian_matcher import HungarianMatcher
from models.matcher.o2m_matcher import Stage2Assigner as One2ManyMatcher
from models.necks.channel_mapper import ChannelMapper

# mostly changed parameters
embed_dim = 256
num_classes = 91
num_queries = 900
hybrid_num_proposals = 1500
hybrid_assign = 6
num_feature_levels = 4
transformer_enc_layers = 6
transformer_dec_layers = 6
num_heads = 8
dim_feedforward = 2048

# lora parameters
rank_lora = 32
num_lora = 3

# instantiate model components
position_embedding = PositionEmbeddingSine(
    embed_dim // 2, temperature=10000, normalize=True, offset=-0.5
)

backbone = SwinTransformerBackbone(arch="swin_l", return_indices=(1, 2, 3), freeze_indices=(0,))

neck = ChannelMapper(
    in_channels=backbone.num_channels,
    out_channels=embed_dim,
    num_outs=num_feature_levels,
)

transformer = UFusionRelationTransformerWithLoRA(
    encoder=UFusionRelationTransformerEncoder(
        encoder_layer=RelationTransformerEncoderLayer(
            embed_dim=embed_dim,
            n_heads=num_heads,
            dropout=0.0,
            activation=nn.ReLU(inplace=True),
            n_levels=num_feature_levels,
            n_points=4,
            d_ffn=dim_feedforward,
        ),
        num_layers=transformer_enc_layers,
    ),
    decoder=UFusionRelationTransformerDecoderWithLoRA(
        decoder_layer=RelationTransformerDecoderLayerWithLoRA(
            embed_dim=embed_dim,
            n_heads=num_heads,
            dropout=0.0,
            activation=nn.ReLU(inplace=True),
            n_levels=num_feature_levels,
            n_points=4,
            d_ffn=dim_feedforward,
            num_lora=num_lora,
            rank_lora=rank_lora,
        ),
        num_layers=transformer_dec_layers,
        num_classes=num_classes,
    ),
    num_classes=num_classes,
    num_feature_levels=num_feature_levels,
    two_stage_num_proposals=num_queries,
    hybrid_num_proposals=hybrid_num_proposals,
)

matcher = HungarianMatcher(
    cost_class=2, cost_bbox=5, cost_giou=2, focal_alpha=0.25, focal_gamma=2.0
)

if num_lora == 3:
    matcher_o2m = [
        One2ManyMatcher(threshold=0.4, k=2),
        One2ManyMatcher(threshold=0.4, k=4),
        One2ManyMatcher(threshold=0.4, k=6),
    ] 
elif num_lora == 2:
    matcher_o2m = [
        One2ManyMatcher(threshold=0.4, k=3),
        One2ManyMatcher(threshold=0.4, k=6),
    ]
elif num_lora == 1:
    matcher_o2m = [One2ManyMatcher(threshold=0.4, k=6)]
else:
    raise NotImplementedError

assert len(matcher_o2m) == num_lora

# follow ms-detr setting
matcher_enc = HungarianMatcher(
    cost_class=0, cost_bbox=5, cost_giou=2, focal_alpha=0.25, focal_gamma=2.0
)

# construct weight_dict for loss
# weight_dict = {"loss_class": 1, "loss_bbox": 5, "loss_giou": 2}
# weight_dict.update({"loss_class_dn": 1, "loss_bbox_dn": 5, "loss_giou_dn": 2})
# aux_weight_dict = {}
# for i in range(transformer.decoder.num_layers - 1):
#     aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
# weight_dict.update(aux_weight_dict)
# weight_dict.update({"loss_class_enc": 1, "loss_bbox_enc": 5, "loss_giou_enc": 2})
# weight_dict.update({k + "_hybrid": v for k, v in weight_dict.items()})
# weight_dict.update({k + "_lora": v for k, v in weight_dict.items()})

weight_dict = {"loss_class": 0.5, "loss_bbox": 5, "loss_giou": 2} # follow ms-detr setting, one-to-one use smaller weight
weight_dict.update({"loss_class_dn": 1, "loss_bbox_dn": 5, "loss_giou_dn": 2})
aux_weight_dict = {}
for i in range(transformer.decoder.num_layers - 1):
    aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
weight_dict.update(aux_weight_dict)
weight_dict.update({"loss_class_enc": 1, "loss_bbox_enc": 5, "loss_giou_enc": 2})

tmp_weight_dict = {"loss_class": 1, "loss_bbox": 5, "loss_giou": 2}
aux_tmp_weight_dict = {}
for i in range(transformer.decoder.num_layers - 1):
    aux_tmp_weight_dict.update({k + f"_{i}": v for k, v in tmp_weight_dict.items()})
tmp_weight_dict.update(aux_tmp_weight_dict)
tmp_weight_dict.update({"loss_class_enc": 1, "loss_bbox_enc": 5, "loss_giou_enc": 2})

weight_dict.update({k + "_hybrid": v for k, v in tmp_weight_dict.items()})
for lora_id in range(num_lora):
    # weight_dict.update({k + "_lora": v / 2. / num_lora for k, v in tmp_weight_dict.items()})
    weight_dict.update({k + f"_lora_{lora_id}": v / 2. / num_lora for k, v in tmp_weight_dict.items()})

criterion = HybirdSetCriterionWithO2M(
    num_classes=num_classes, matcher=matcher, matcher_o2m=matcher_o2m, matcher_enc=matcher_enc,
    weight_dict=weight_dict, alpha=0.25, gamma=2.0,
    use_indices_merge=False
)
postprocessor = PostProcess(select_box_nums_for_evaluation=300)

# combine above components to instantiate the model
model = RelationDETRwithLoRA(
    backbone=backbone,
    neck=neck,
    position_embedding=position_embedding,
    transformer=transformer,
    criterion=criterion,
    postprocessor=postprocessor,
    num_classes=num_classes,
    num_queries=num_queries,
    hybrid_assign=hybrid_assign,
    denoising_nums=100,
    min_size=800,
    max_size=1333,
)
