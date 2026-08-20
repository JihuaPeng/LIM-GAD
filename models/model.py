from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BitsAndBytesConfig, CLIPVisionModel

from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_PATCH_TOKEN)

from .llava.model.language_model.llava_llama import (LlavaLlamaForCausalLM,
                                                     LlavaLlamaModel)


from roi_align.roi_align import RoIAlign

from .backbone import build_backbone
from .group_transformer import build_group_transformer
from .feed_forward import MLP



from models.group_matcher import build_group_matcher
from models.criterion import SetCriterion
from util import box_ops
from util.misc import (accuracy, get_world_size, is_dist_avail_and_initialized)

from .cafe_models import build_model




def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    scale=1000,  # 100000.0,
    eps=1e-6,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1, 2)
    targets = targets.flatten(1, 2)
    numerator = 2 * (inputs / scale * targets).sum(-1)
    denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
    loss = 1 - (numerator + eps) / (denominator + eps)
    loss = loss.sum() / (num_masks + 1e-8)
    return loss


def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = loss.flatten(1, 2).mean(1).sum() / (num_masks + 1e-8)
    return loss


class LisaMetaModel:
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(LisaMetaModel, self).__init__(config)

        self.config = config
        self.config.out_dim = kwargs["out_dim"]
        self.config.num_class = kwargs["num_class"]
        self.config.num_boxes = kwargs["num_boxes"]
        self.config.num_frame = kwargs["num_frame"]
        self.config.hidden_dim = kwargs["hidden_dim"]
        self.config.crop_size = kwargs["crop_size"]
        self.config.drop_rate = kwargs["drop_rate"]
        self.config.num_group_tokens = kwargs["num_group_tokens"]
        self.config.num_act_tokens = kwargs["num_act_tokens"]

        self.config.distance_threshold = kwargs["distance_threshold"]
        self.config.position_embedding = kwargs["position_embedding"]
        self.config.frozen_batch_norm = kwargs["frozen_batch_norm"]
        self.config.backbone = kwargs["backbone"]
        self.config.dilation = kwargs["dilation"]
        self.config.gar_nheads = kwargs["gar_nheads"]
        self.config.gar_ffn_dim = kwargs["gar_ffn_dim"]
        self.config.gar_enc_layers = kwargs["gar_enc_layers"]
        self.config.ce_loss_coef = kwargs["loss_ce"]
        self.config.group_ce_loss_coef = kwargs["loss_group_ce"]
        self.config.group_code_loss_coef = kwargs["loss_group_code"]
        self.config.consistency_loss_coef = kwargs["loss_consistency"]
        self.config.eos_coef = kwargs["eos_coef"]
        self.config.group_eos_coef = kwargs["group_eos_coef"]
        self.config.temperature = 0.2
        self.config.llava_loss_weight = kwargs["llava_loss_weight"]
        self.config.set_cost_group_class = kwargs["set_cost_group_class"]
        self.config.set_cost_membership = kwargs["set_cost_membership"]

        self.initialize_lisa_modules(self.config)

    def initialize_lisa_modules(self, config):

        #CAFE
        self.cafe_model, self.criterion = build_model(config)
        for param in self.cafe_model.parameters():
            param.requires_grad = True
        for param in self.criterion.parameters():
            param.requires_grad = False


        self.llava_loss_weight = config.llava_loss_weight


        # Projection layer
        in_dim = config.hidden_size
        out_dim = config.out_dim
        text_fc = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0),
        ]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])
        self.text_hidden_fcs.train()
        for param in self.text_hidden_fcs.parameters():
            param.requires_grad = True

        

class LisaModel(LisaMetaModel, LlavaLlamaModel):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(LisaModel, self).__init__(config, **kwargs)

        self.config.use_cache = False
        self.config.vision_tower = self.config.mm_vision_tower
        self.config.mm_vision_select_feature = "patch"
        self.config.image_aspect_ratio = "square"
        self.config.image_grid_pinpoints = None
        self.config.tune_mm_mlp_adapter = False
        self.config.freeze_mm_mlp_adapter = True
        self.config.pretrain_mm_mlp_adapter = None
        self.config.mm_use_im_patch_token = False


class LISAForCausalLM(LlavaLlamaForCausalLM):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        if not hasattr(config, "train_mask_decoder"):
            config.mm_use_im_start_end = kwargs.pop("use_mm_start_end", True)
            config.mm_vision_tower = kwargs.get(
                "vision_tower", "openai/clip-vit-large-patch14"
            )
            self.ce_loss_weight = kwargs.pop("ce_loss_weight", None)
            self.dice_loss_weight = kwargs.pop("dice_loss_weight", None)
            self.bce_loss_weight = kwargs.pop("bce_loss_weight", None)
        else:
            config.mm_vision_tower = config.vision_tower
            
        self.group_token_idx = kwargs.pop("group_token_idx")

        self.act_token_idx = kwargs.pop("act_token_idx")

        super().__init__(config)

        self.model = LisaModel(config, **kwargs)

        self.post_init()


    def calculate_pairwise_distnace(self, boxes):
        bs = boxes.shape[0]

        rx = boxes.pow(2).sum(dim=2).reshape((bs, -1, 1))
        ry = boxes.pow(2).sum(dim=2).reshape((bs, -1, 1))

        dist = rx - 2.0 * boxes.matmul(boxes.transpose(1, 2)) + ry.transpose(1, 2)

        return torch.sqrt(dist)
    

    def forward(self, **kwargs):
        return self.model_forward(**kwargs)


    def model_forward(
        self,
        images: torch.FloatTensor,
        images_clip: torch.FloatTensor,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        attention_masks: torch.LongTensor,
        offset: torch.LongTensor,
        boxes,
        dummy_mask,
        targets,
        inference: bool = False,
        **kwargs,
    ):

        # get group token id
        group_token_mask = torch.isin(input_ids[:, 1:], torch.tensor(self.group_token_idx, device=input_ids.device))
        group_token_mask = torch.cat(
            [
                group_token_mask,
                torch.zeros((group_token_mask.shape[0], 1)).bool().cuda(),
            ], 
            dim=1,
        )
         # hack for IMAGE_TOKEN_INDEX (we suppose that there is only one image, and it is in the front)
        group_token_mask = torch.cat(
            [torch.zeros((group_token_mask.shape[0], 255)).bool().cuda(), group_token_mask],
            dim=1,
        )

        # get actor token id
        act_token_mask = torch.isin(input_ids[:, 1:], torch.tensor(self.act_token_idx, device=input_ids.device))
        act_token_mask = torch.cat(
            [
                act_token_mask,
                torch.zeros((act_token_mask.shape[0], 1)).bool().to(input_ids.device),
            ], 
            dim=1,
        )
        act_token_mask = torch.cat(
            [torch.zeros((act_token_mask.shape[0], 255)).bool().to(input_ids.device), act_token_mask], 
            dim=1,
        )


        # extract frame features by vision backbone
        bs, t, _, h, w = images.shape    
        n = boxes.shape[2]

        boxes = torch.reshape(boxes, (-1, 4))                                       
        boxes_flat = boxes.clone().detach()
        boxes_idx = [i * torch.ones(n, dtype=torch.int) for i in range(bs * t)]
        boxes_idx = torch.stack(boxes_idx).to(device=boxes.device)
        boxes_idx_flat = torch.reshape(boxes_idx, (bs * t * n, ))  

        features, pos = self.model.cafe_model.backbone(images)    
        _, c, oh, ow = features.shape                 

        src = self.model.cafe_model.input_proj(features)
        src = torch.reshape(src, (bs, t, -1, oh, ow))                     

        # calculate distance & distance mask
        boxes_center = boxes.clone().detach()
        boxes_center = torch.reshape(boxes_center[:, :2], (-1, n, 2))
        boxes_distance = self.calculate_pairwise_distnace(boxes_center)

        distance_mask = (boxes_distance > self.model.cafe_model.distance_threshold)

        # ignore dummy boxes (padded boxes to match the number of actors)
        dummy_mask = dummy_mask.unsqueeze(1).repeat(1, t, 1).reshape(-1, n)
        actor_dummy_mask = (~dummy_mask.unsqueeze(2)).float() @ (~dummy_mask.unsqueeze(1)).float()
        dummy_diag = (dummy_mask.unsqueeze(2).float() @ dummy_mask.unsqueeze(1).float()).nonzero(as_tuple=True)
        actor_mask = ~(actor_dummy_mask.bool())
        actor_mask[dummy_diag] = False
        actor_mask = distance_mask + actor_mask
        group_dummy_mask = dummy_mask

        boxes_flat[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) * ow
        boxes_flat[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * oh
        boxes_flat[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2) * ow
        boxes_flat[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2) * oh

        boxes_flat.requires_grad = False
        boxes_idx_flat.requires_grad = False
        
        features = features.float()

        # extract actor features
        actor_features = self.model.cafe_model.roi_align(features, boxes_flat, boxes_idx_flat)

        actor_features = actor_features.half()

        actor_features = torch.reshape(actor_features, (bs * t * n, -1))
        actor_features = self.model.cafe_model.fc_emb(actor_features)
        actor_features = F.relu(actor_features)
        actor_features = self.model.cafe_model.drop_emb(actor_features)
        actor_features = actor_features.reshape(bs, t, n, self.model.cafe_model.hidden_dim)

        boxes = boxes.half()

        # add positional information to box features
        box_pos_emb = self.model.cafe_model.box_pos_emb(boxes)
        box_pos_emb = torch.reshape(box_pos_emb, (bs, t, n, -1))                        
        actor_features = actor_features + box_pos_emb


        if inference:
            n_batch = 1
            length = input_ids.shape[0]
            images_clip_extend = images_clip.expand(length, -1, -1, -1).contiguous()

            output_hidden_states = []
            for i in range(n_batch):
                start_i, end_i = i * length, min((i + 1) * length, input_ids.shape[0])

                output_i = super().forward(
                    images=images_clip_extend[: end_i - start_i],
                    attention_mask=attention_masks[start_i:end_i],
                    input_ids=input_ids[start_i:end_i],
                    output_hidden_states=True,
                )
                output_hidden_states.append(output_i.hidden_states)
                torch.cuda.empty_cache()
            
            output_hidden_states_list = []
            output_hidden_states_level = torch.cat(output_hidden_states, dim=0)
            output_hidden_states_list.append(output_hidden_states_level)
            output_hidden_states = output_hidden_states_list
            output = None
            
        else:
            images_clip_list = []
            for i in range(len(offset) - 1):
                start_i, end_i = offset[i], offset[i + 1]
                images_clip_i = (
                    images_clip[i]
                    .unsqueeze(0)
                    .expand(end_i - start_i, -1, -1, -1)
                    .contiguous()
                )
                images_clip_list.append(images_clip_i)
            images_clip = torch.cat(images_clip_list, dim=0)     

            #llava output
            output = super().forward(
                images=images_clip,               
                attention_mask=attention_masks,   
                input_ids=input_ids,              
                labels=labels,                   
                output_hidden_states=True,
            )
            output_hidden_states = output.hidden_states



        hidden_states = []

        hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states[-1]))

        last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)     


        # group llava embeddings
        pred_embeddings_group = last_hidden_state[group_token_mask]
        group_token_counts = group_token_mask.int().sum(-1)  
        group_token_offset = group_token_counts.cumsum(-1)
        group_token_offset = torch.cat(
            [torch.zeros(1).long().cuda(), group_token_offset], dim=0
        )
        group_token_offset = group_token_offset[offset]

        pred_embeddings_group_ = []
        for i in range(len(group_token_offset) - 1):
            start_i, end_i = group_token_offset[i], group_token_offset[i + 1]
            pred_embeddings_group_.append(pred_embeddings_group[start_i:end_i])
        pred_embeddings_group = torch.stack(pred_embeddings_group_, dim=0)            # llava output [batch, frame*group_token, dim]
        
        pred_embeddings_group_new = pred_embeddings_group.view(bs, t, self.model.cafe_model.num_group_tokens, -1)
        pred_embeddings_group_new = pred_embeddings_group_new.permute(2, 0, 1, 3)
        pred_embeddings_group_new = pred_embeddings_group_new.reshape(self.model.cafe_model.num_group_tokens, -1, self.model.cafe_model.hidden_dim)


        # [ACT] llava embeddings
        pred_embeddings = last_hidden_state[act_token_mask]
        act_token_counts = act_token_mask.int().sum(-1)
        act_token_offset = act_token_counts.cumsum(-1)
        act_token_offset = torch.cat(
            [torch.zeros(1).long().cuda(), act_token_offset], dim=0)
        act_token_offset = act_token_offset[offset]

        pred_embeddings_act_ = []
        for i in range(len(act_token_offset) - 1):
            start_i, end_i = act_token_offset[i], act_token_offset[i + 1]
            pred_embeddings_act_.append(pred_embeddings_act[start_i:end_i])
        pred_embeddings_act = torch.stack(pred_embeddings_act_, dim=0)
        
        pred_embeddings_act = pred_embeddings_act.reshape(-1, self.model.cafe_model.hidden_dim)
        pred_embeddings_act_new = pred_embeddings_act.unsqueeze(0)


        # group transformer
        hs, actor_att, feature_att = self.model.cafe_model.group_transformer(pred_embeddings_group_new, pred_embeddings_act_new, src, actor_mask, group_dummy_mask,
                                                            self.model.cafe_model.group_query_emb.weight, pos, actor_features)
        # [1, bs * t, n + k, f'], [1, bs * t, k, n], [1, bs * t, n + k, oh x ow]   M: # group tokens, K: # boxes

        actor_hs = hs[0, :, :n]
        group_hs = hs[0, :, n:]

        actor_hs = actor_hs.reshape(bs, t, n, -1)
        actor_hs = actor_features + actor_hs

        # normalize
        inst_repr = F.normalize(actor_hs.reshape(bs, t, n, -1).mean(dim=1), p=2, dim=2)
        group_repr = F.normalize(group_hs.reshape(bs, t, self.model.cafe_model.num_group_tokens, -1).mean(dim=1), p=2, dim=2)


        # prediction heads
        outputs_class = self.model.cafe_model.class_emb(actor_hs)
        outputs_group_class = self.model.cafe_model.group_emb(group_hs)


        outputs_actor_emb = self.model.cafe_model.actor_match_emb(inst_repr)
        outputs_group_emb = self.model.cafe_model.group_match_emb(group_repr)

        membership = torch.bmm(outputs_group_emb, outputs_actor_emb.transpose(1, 2))
        membership = F.softmax(membership, dim=1)


        # [ACT] embedding activity heads
        act_class = self.model.cafe_model.act_emb_class(pred_embeddings_act_new)
        act_class = act_class.squeeze(0)


        out = {
            "pred_actions": outputs_class.reshape(bs, t, self.model.cafe_model.num_boxes, self.model.cafe_model.num_class + 1).mean(dim=1),
            "pred_activities": outputs_group_class.reshape(bs, t, self.model.cafe_model.num_group_tokens, self.model.cafe_model.num_class + 1).mean(dim=1),
            "membership": membership.reshape(bs, self.model.cafe_model.num_group_tokens, self.model.cafe_model.num_boxes),
            "actor_embeddings": F.normalize(actor_hs.reshape(bs, t, n, -1).mean(dim=1), p=2, dim=2),
            "pred_act_class": act_class.reshape(bs, t, self.model.cafe_model.num_class + 1).mean(dim=1),
        }

        return out