import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

import os
import math
import sys
import copy
import time
import random
import numpy as np
import argparse

import deepspeed


from models_lisa.build_criterion_actloss import build_model
from util.utils import *
import util.misc as utils
import util.logger as loggers
from dataloader.dataloader_videolisa_12group_boxes_prompt import read_dataset
import evaluation.cafe_eval as evaluation

import transformers

from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         AverageMeter, ProgressMeter, Summary, dict_to_cuda, move_to_device,
                         intersectionAndUnionGPU)


from models_lisa.llava.constants import (DEFAULT_IMAGE_TOKEN, IGNORE_INDEX,
                                   IMAGE_TOKEN_INDEX)

from models_lisa.llava import conversation_vl as conversation_lib
from models_lisa.llava.mm_utils import tokenizer_image_token
from models_lisa.LISA_cafe_actors_12group_qwen2_llava_actor_mid_layer3_actloss import LISAForCausalLM

from peft import LoraConfig, get_peft_model


# from utils.dataset_cafe import HybridDataset, ValDataset, collate_fn

from functools import partial


import pdb

parser = argparse.ArgumentParser(description='Group Activity Detection train code', add_help=False)

# Dataset specification
parser.add_argument('--dataset', default='cafe', type=str, help='dataset name')
parser.add_argument('--val_mode', action='store_true')
parser.add_argument('--split', default='place', type=str, help='dataset split. place or view')
parser.add_argument('--data_path', default='../Dataset/', type=str, help='data path')
parser.add_argument('--image_width', default=1280, type=int, help='Image width to resize')
parser.add_argument('--image_height', default=720, type=int, help='Image height to resize')
parser.add_argument('--random_sampling', action='store_true', help='random sampling strategy')
parser.add_argument('--num_frame', default=5, type=int, help='number of frames for each clip')
parser.add_argument('--num_class', default=6, type=int, help='number of activity classes')

# Backbone parameters
parser.add_argument('--backbone', default='resnet18', type=str, help='feature extraction backbone')
parser.add_argument('--dilation', action='store_true', help='use dilation or not')
parser.add_argument('--frozen_batch_norm', action='store_true', help='use frozen batch normalization')
parser.add_argument('--hidden_dim', default=256, type=int, help='transformer channel dimension')

# RoI Align parameters
parser.add_argument('--num_boxes', default=14, type=int, help='maximum number of actors')
parser.add_argument('--crop_size', default=5, type=int, help='roi align crop size')

# Group Transformer
parser.add_argument('--gar_nheads', default=4, type=int, help='number of heads')
parser.add_argument('--gar_enc_layers', default=6, type=int, help='number of group transformer layers')
parser.add_argument('--gar_ffn_dim', default=512, type=int, help='feed forward network dimension')
parser.add_argument('--position_embedding', default='sine', type=str, help='various position encoding')
parser.add_argument('--num_group_tokens', default=12, type=int, help='number of group tokens')
parser.add_argument('--aux_loss', action='store_true')
parser.add_argument('--group_threshold', default=0.5, type=float, help='post processing threshold')
parser.add_argument('--distance_threshold', default=0.2, type=float, help='distance mask threshold')

# Loss option
parser.add_argument('--temperature', default=0.2, type=float, help='consistency loss temperature')

# Loss coefficients (Individual)
parser.add_argument('--ce_loss_coef', default=1, type=float)
parser.add_argument('--eos_coef', default=1, type=float,
                    help="Relative classification weight of the no-object class")

# Loss coefficients (Group)
parser.add_argument('--group_eos_coef', default=1, type=float)
parser.add_argument('--group_ce_loss_coef', default=1, type=float)
parser.add_argument('--group_code_loss_coef', default=5, type=float)
parser.add_argument('--consistency_loss_coef', default=2, type=float)

# Matcher (Group)
parser.add_argument('--set_cost_group_class', default=1, type=float,
                    help="Class coefficient in the matching cost")
parser.add_argument('--set_cost_membership', default=1, type=float,
                    help="Membership coefficient in the matching cost")

# Training parameters
parser.add_argument('--random_seed', default=42, type=int, help='random seed for reproduction')
parser.add_argument('--epochs', default=30, type=int, help='Max epochs')
parser.add_argument('--test_freq', default=1, type=int, help='print frequency')
parser.add_argument('--batch', default=4, type=int, help='Batch size')
parser.add_argument('--test_batch', default=4, type=int, help='Test batch size')
parser.add_argument('--lr', default=1e-5, type=float, help='Initial learning rate')
parser.add_argument('--max_lr', default=1e-4, type=float, help='Max learning rate')
parser.add_argument('--lr_step', default=4, type=int, help='step size for learning rate scheduler')
parser.add_argument('--lr_step_down', default=25, type=int, help='step down size (cyclic) for learning rate scheduler')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay')
parser.add_argument('--drop_rate', default=0.1, type=float, help='Dropout rate')
parser.add_argument('--gradient_clipping', action='store_true', help='use gradient clipping')
parser.add_argument('--max_norm', default=1.0, type=float, help='gradient clipping max norm')

# GPU
parser.add_argument('--device', default="0", type=str, help='GPU device')
parser.add_argument('--distributed', action='store_true')

# Load model
parser.add_argument('--load_model', action='store_true', help='load model')
parser.add_argument('--model_path', default="", type=str, help='pretrained model path')

# Visualization
parser.add_argument('--result_path', default="./outputs_llava3B_actloss/")

# Evaluation
parser.add_argument('--groundtruth', default='./evaluation/gt_tracks.txt', type=argparse.FileType("r"))
parser.add_argument('--labelmap', default='./label_map/group_action_list.pbtxt', type=argparse.FileType("r"))
parser.add_argument('--giou_thresh', default=1.0, type=float)
parser.add_argument('--eval_type', default="gt_base", type=str, help='gt_based or detection_based')


# Llava
parser.add_argument('--llava_gar_enc_layers', default=1, type=int, help='number of group transformer layers')

parser.add_argument(
        "--version", default="MBZUAI/LLaVA-Phi-3-mini-4k-instruct"
    )
parser.add_argument("--model_max_length", default=2048, type=int)
parser.add_argument("--use_mm_start_end", action="store_true", default=True)
parser.add_argument(
        "--conv_type",
        default="phi3_instruct",
        type=str,
        # choices=["llava_v1", "llava_llama_2"],
    )
parser.add_argument("--local_rank", default=0, type=int, help="node rank")
parser.add_argument("--num_workers", default=8, type=int, help="worker numbers")

parser.add_argument(
        "--precision",
        default="fp16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
parser.add_argument("--out_dim", default=256, type=int)
parser.add_argument("--ce_loss_weight", default=1.0, type=float)
parser.add_argument("--dice_loss_weight", default=0.5, type=float)
parser.add_argument("--bce_loss_weight", default=2.0, type=float)
parser.add_argument("--eval_only", action="store_true", default=False)

# lora
parser.add_argument("--lora_target_modules", default="q_proj,v_proj", type=str)
parser.add_argument("--lora_r", default=8, type=int)
parser.add_argument("--lora_alpha", default=16, type=int)
parser.add_argument("--lora_dropout", default=0.05, type=float)
parser.add_argument(
        "--grad_accumulation_steps",
        default=1,
        type=int,
    )
parser.add_argument("--beta1", default=0.9, type=float)
parser.add_argument("--beta2", default=0.999, type=float)
parser.add_argument("--steps_per_epoch", default=500, type=int)
parser.add_argument("--auto_resume", action="store_true", default=True)
parser.add_argument("--resume", default="", type=str)
parser.add_argument("--exp_name", default="lisa", type=str)
parser.add_argument("--log_base_dir", default="./runs", type=str)

parser.add_argument("--llava_loss_weight", default=1.0, type=float)

parser.add_argument('--num_act_tokens', default=1, type=int, help='number of activity tokens')


# llava text [ACT] loss
parser.add_argument('--text_act_loss_coef', default=1, type=float)



args = parser.parse_args()
path = None

SEQS_CAFE = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]

# SEQS_CAFE = [1, 3, 4]

ACTIVITIES = ['Queueing', 'Ordering', 'Drinking', 'Working', 'Fighting', 'Selfie', 'Individual', 'No']


import tokenizers
from packaging import version
IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')


import pdb


def collate_fn(
    batch, tokenizer, conv_type="llava_v1", use_mm_start_end=True, local_rank=-1
):
    image_path_list = []
    images_list = []
    images_clip_list = []
    conversation_list = []
    masks_list = []
    label_list = []
    resize_list = []
    questions_list = []
    sampled_classes_list = []
    offset_list = [0]
    cnt = 0
    inferences = []
    targets_list = []
    infos_list = []

    for (
        images,
        images_clip,
        targets,
        infos,
        conversations,
        questions,
        inference,
    ) in batch:
        images_list.append(images)
        images_clip_list.append(images_clip)
        conversation_list.extend(conversations)
        questions_list.append(questions)
        cnt += len(conversations)
        offset_list.append(cnt)
        targets_list.append(targets)
        infos_list.append(infos)
        inferences.append(inference)


    if use_mm_start_end:
        # replace <image> token
        for i in range(len(conversation_list)):
            replace_token = DEFAULT_IMAGE_TOKEN
            replace_token = (
                DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            )
            conversation_list[i] = conversation_list[i].replace(
                DEFAULT_IMAGE_TOKEN, replace_token
            )

    input_ids = [
        tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
        for prompt in conversation_list
    ]
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    attention_masks = input_ids.ne(tokenizer.pad_token_id)

    targets_id = input_ids.clone()

    conv = conversation_lib.default_conversation.copy()

    # if conv_type == "phi3_instruct":
    #     sep = conv.sep + conv.roles[1] + ": "
    # else:
    #     sep = "[/INST] "
    sep = conv.sep + conv.roles[1]

    for conversation, target in zip(conversation_list, targets_id):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])]  # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx + 2]))  # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX

        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)

            assert len(parts) == 2, (len(parts), rou)
            parts[0] += sep

            if DEFAULT_IMAGE_TOKEN in conversation:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            # ------------- adapted from LLaVA Phi-3 ------------------
            if i == 0:
                round_len += 1
                instruction_len += 1
            else:
                round_len -= 2
                instruction_len -= 2

            if i != 0 and getattr(tokenizer, 'legacy', False) and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len += 1
                instruction_len += 1
            # ------------- end line ------------------

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            assert cur_len == total_len, (cur_len, total_len)

    if inferences[0] == False:
        truncate_len = tokenizer.model_max_length - 255
        if input_ids.shape[1] > truncate_len:
            input_ids = input_ids[:, :truncate_len]
            targets_id = targets_id[:, :truncate_len]
            attention_masks = attention_masks[:, :truncate_len]

    llava_input = {
        "images": torch.stack(images_list, dim=0),
        "images_clip": torch.stack(images_clip_list, dim=0),
        "input_ids": input_ids,
        "labels": targets_id,
        "attention_masks": attention_masks,
        "offset": torch.LongTensor(offset_list),
        "questions_list": questions_list,
        "conversation_list": conversation_list,
        }

    return llava_input, targets_list, infos_list




def main():
    global args, path

    torch.cuda.empty_cache()
    torch.backends.cuda.max_split_size_mb = 32

    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # if args.local_rank == 0:
    time_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    exp_name = '%s_GAD_%s' % (args.dataset, time_str)
    save_path = './result/%s' % exp_name

    # args.log_dir = os.path.join(args.log_base_dir, args.exp_name)


    # set random seed
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

    base_token = "[GROUP"

    custom_tokens = []
    for i in range(args.num_group_tokens):
        custom_tokens.append(f"{base_token}{i}]")


    # Create model
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=None,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    num_added_tokens = tokenizer.add_tokens(custom_tokens)
    args.group_token_idx = {token: tokenizer(token, add_special_tokens=False).input_ids[0]
    for token in custom_tokens}
    
    args.group_token_idx = list(args.group_token_idx.values())


    num_added_tokens2 = tokenizer.add_tokens("[ACTORS]")
    args.actor_token_idx = tokenizer("[ACTORS]", add_special_tokens=False).input_ids[0]


    best_group_map_05 = 0.0
    # best_outlier_miou = 0.0


    if args.use_mm_start_end:
        tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
        )


    model_args = {
        ## lisa
        "out_dim": args.out_dim,
        "ce_loss_weight": args.ce_loss_weight,
        "dice_loss_weight": args.dice_loss_weight,
        "bce_loss_weight": args.bce_loss_weight,
        "group_token_idx": args.group_token_idx,
        "actor_token_idx": args.actor_token_idx,
        "use_mm_start_end": args.use_mm_start_end,
        "llava_loss_weight": args.llava_loss_weight,
        ## cafe
        "num_class": args.num_class,
        "num_frame": args.num_frame,
        "num_boxes": args.num_boxes,
        "hidden_dim": args.hidden_dim,
        "crop_size": args.crop_size,
        "drop_rate": args.drop_rate,
        "num_group_tokens": args.num_group_tokens,
        "num_act_tokens": args.num_act_tokens,
        "distance_threshold": args.distance_threshold,
        "position_embedding": args.position_embedding,
        "frozen_batch_norm": args.frozen_batch_norm,
        "backbone": args.backbone,
        "dilation": args.dilation,
        "gar_nheads": args.gar_nheads,
        "gar_ffn_dim": args.gar_ffn_dim,
        "gar_enc_layers": args.gar_enc_layers,
        "llava_gar_enc_layers":args.llava_gar_enc_layers,
        # Set loss coefficients
        "loss_ce": args.ce_loss_coef,
        "loss_group_ce": args.group_ce_loss_coef,
        "loss_group_code": args.group_code_loss_coef,
        "loss_consistency": args.consistency_loss_coef,
        # Group matching
        "set_cost_group_class": args.set_cost_group_class,
        "set_cost_membership": args.set_cost_membership,
        # Loss functions
        "eos_coef": args.eos_coef,
        "group_eos_coef": args.group_eos_coef,
        "temperature": args.temperature,
    }


    torch_dtype = torch.float16
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half
    elif args.precision == "fp32":
        torch_dtype = torch.float


    model = LISAForCausalLM.from_pretrained(
        args.version, torch_dtype=torch_dtype, low_cpu_mem_usage=True, **model_args
    )

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()


    # debug
    # pdb.set_trace()

    # print(f"Local rank: {args.local_rank}, Total GPUs: {torch.cuda.device_count()}")


    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype, device=args.local_rank)
    if not args.eval_only:
        model.get_model().initialize_lisa_modules(model.get_model().config)

    for p in vision_tower.parameters():
        p.requires_grad = False
    for p in model.get_model().mm_projector.parameters():
        p.requires_grad = False


    train_set, test_set = read_dataset(args)


    conversation_lib.default_conversation = conversation_lib.conv_templates[
        args.conv_type
    ]



    # lora finetune
    lora_r = args.lora_r
    if lora_r > 0:

        def find_linear_layers(model, lora_target_modules):
            cls = torch.nn.Linear
            lora_module_names = set()
            for name, module in model.named_modules():
                if (
                    isinstance(module, cls)
                    and all(
                        [
                            x not in name
                            for x in [
                                "visual_model",
                                "vision_tower",
                                "mm_projector",
                                "text_hidden_fcs",
                                "cafe_model",
                                "criterion",
                            ]
                        ]
                    )
                    and any([x in name for x in lora_target_modules])
                ):
                    lora_module_names.add(name)
            return sorted(list(lora_module_names))

        lora_alpha = args.lora_alpha
        lora_dropout = args.lora_dropout
        lora_target_modules = find_linear_layers(
            model, args.lora_target_modules.split(",")
        )
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()



    model.resize_token_embeddings(len(tokenizer))

    # make text_hidden_fcs, mask_decoder, lm_head, embed_tokens trainable
    for n, p in model.named_parameters():
        if any(
            [
                x in n
                for x in ["lm_head", "embed_tokens", "text_hidden_fcs", "cafe_model", "group_query_emb"]
            ]
        ):
            print("n: ", n, "p.shape: ", p.shape)
            p.requires_grad = True


    model_cafe, criterion = build_model(args)
    # model_cafe = torch.nn.DataParallel(model_cafe).cuda()
    
    

    args.steps_per_epoch = len(train_set) // (args.batch * args.grad_accumulation_steps)
    args.warmup_epochs = 2


    def calculate_cyclic_lr_per_epoch(step, lr, max_lr, steps_per_epoch, warmup_epochs, total_epochs, mode):
    
        current_epoch = step // steps_per_epoch  
        total_cycles = total_epochs              

        cycle_progress = current_epoch % total_cycles
        if cycle_progress < warmup_epochs:
            current_lr = lr + (max_lr - lr) * (cycle_progress / warmup_epochs)
        else:
            current_lr = max_lr - (max_lr - lr) * ((cycle_progress - warmup_epochs) / (total_cycles - warmup_epochs))
    
        if mode == 'triangular2':
            cycle_num = current_epoch // total_cycles
            current_lr = current_lr / (2 ** cycle_num)
    
        return current_lr


    ds_config = {
        "train_micro_batch_size_per_gpu": args.batch,
        "gradient_accumulation_steps": args.grad_accumulation_steps,
        "fp16": {
        "enabled": args.precision == "fp16",
        # "initial_scale_power": 10,
        "min_loss_scale": 1e-6,
        "loss_scale": 0,
        "hysteresis": 2, 
        "loss_scale_window": 1000,
        # "verbose": True,
        },
        "gradient_clipping": 1.0,
        "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
        "device": "none",
        },
        # "offload_param": {
        # "device": "cpu",  
        # "pin_memory": True
        # },
        "contiguous_gradients": True,
        "overlap_comm": True,
        "reduce_bucket_size": 2e8,  
        "allgather_bucket_size": 2e8
        },
        # "activation_checkpointing": {
        # "partition_activations": True,
        # "contiguous_memory_optimization": True
        # },
        # "aio": {
        # "block_size": 1e6,  
        # "queue_depth": 8,
        # "thread_count": 4,
        # "single_submit": False
        # }
    }


    # get the number of model parameters
    parameters = 'Number of full model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()]))
    print_log(save_path, '--------------------Number of parameters--------------------')
    print_log(save_path, parameters)

    # define loss function and optimizer
    optimizer = torch.optim.Adam(model.parameters(), args.lr, betas=(0.9, 0.999), eps=1e-8,
                                 weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, args.lr, args.max_lr, step_size_up=args.lr_step,
                                                  step_size_down=args.lr_step_down, mode='triangular2',
                                                  cycle_momentum=False)


    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        model_parameters=model.parameters(),
        config=ds_config,
    )


    world_size = torch.cuda.device_count()
    args.distributed = world_size > 1

    # for variable length input
    if args.distributed:
        sampler_train = data.DistributedSampler(train_set, shuffle=True)
        sampler_test = data.DistributedSampler(test_set, shuffle=False)
    else:
        sampler_train = data.RandomSampler(train_set)
        sampler_test = data.RandomSampler(test_set)

    batch_sampler_train = data.BatchSampler(sampler_train, args.batch, drop_last=True)

    train_loader = data.DataLoader(train_set, batch_sampler=batch_sampler_train,
                                   collate_fn=partial(collate_fn, 
                                  tokenizer=tokenizer, 
                                  conv_type=args.conv_type, 
                                  use_mm_start_end=args.use_mm_start_end, 
                                  local_rank=args.local_rank,
        ), num_workers=args.num_workers, pin_memory=True)
    test_loader = data.DataLoader(test_set, args.test_batch, sampler=sampler_test, drop_last=False,
                                  collate_fn=partial(collate_fn, 
                                  tokenizer=tokenizer, 
                                  conv_type=args.conv_type, 
                                  use_mm_start_end=args.use_mm_start_end, 
                                  local_rank=args.local_rank,
        ), num_workers=args.num_workers, pin_memory=True)


    if args.load_model:
        checkpoint = torch.load(args.model_path)
        model.load_state_dict(checkpoint['state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
    else:
        start_epoch = 1

    path = args.result_path + exp_name
    if not os.path.exists(path):
        os.makedirs(path)
    # if args.local_rank == 0:
    #     os.makedirs(path)

    metrics = evaluation.GAD_Evaluation(args)

    # training phase
    for epoch in range(start_epoch, args.epochs + 1):
        if args.local_rank == 0:
            print_log(save_path, '----- %s at epoch #%d' % ("Train", epoch))
        
        # with torch.cuda.amp.autocast():
        train_log = train(train_loader, model_engine, criterion, optimizer, epoch)
    
        if args.local_rank == 0:
            print_log(save_path, 'Loss: %.4f' % (train_log['loss']))
            print_log(save_path, 'Group class error: %.2f' % (train_log['group_class_error']))
            # print('Current learning rate is %f' % scheduler.get_last_lr()[0])
        # scheduler.step()
            print('Current learning rate is %f' % model_engine.get_lr()[0])
        
        scheduler.step()


        if epoch % args.test_freq == 0:
            if args.local_rank == 0:
                print_log(save_path, '----- %s at epoch #%d' % ("Test", epoch))
            
            test_log, result = validate(test_loader, model_engine, criterion, metrics, epoch)

            if args.local_rank == 0:
                print_log(save_path, 'Loss: %.4f' % (test_log['loss']))
                print_log(save_path, 'Group class error: %.2f' % (test_log['group_class_error']))
                print_log(save_path, "group mAP at 1.0: %.2f" % result['group_mAP_1.0'])
                print_log(save_path, "group mAP at 0.5: %.2f" % result['group_mAP_0.5'])
                print_log(save_path, "outlier mIoU: %.2f" % result['outlier_mIoU'])

            # state = {
            #     'epoch': epoch,
            #     'state_dict': model.state_dict(),
            #     'optimizer': optimizer.state_dict(),
            #     'scheduler': scheduler.state_dict(),
            # }
            current_group_map_05 = result['group_mAP_0.5']
            # current_outlier_iou = result['outlier_mIoU']

            if current_group_map_05 >= best_group_map_05:
                best_group_map_05 = current_group_map_05
                # best_outlier_miou = current_outlier_iou

                result_path = save_path + '/epoch%d.pth' % epoch
                torch.distributed.barrier()
                # torch.save(state, result_path)
                model_engine.save_checkpoint(result_path, tag=f'epoch{epoch}')


def train(train_loader, model, criterion, optimizer, epoch):
    model.train()
    criterion.train()

    # logger
    metric_logger = loggers.MetricLogger(mode="train", delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    space_fmt = str(len(str(args.epochs)))
    header = 'Epoch [{start_epoch: >{fill}}/{end_epoch}]'.format(start_epoch=epoch, end_epoch=args.epochs,
                                                                 fill=space_fmt)

    print_freq = len(train_loader)

    device = next(model.parameters()).device

    criterion = criterion.to(device)

    accumulation_steps = args.grad_accumulation_steps
    accumulation_counter = 0

    weight_dict = {}
    weight_dict['loss_ce'] = args.ce_loss_coef
    weight_dict['loss_group_ce'] = args.group_ce_loss_coef
    weight_dict['loss_group_code'] = args.group_code_loss_coef
    weight_dict['loss_consistency'] = args.consistency_loss_coef

    for i, (llava_input, targets, infos) in enumerate(metric_logger.log_every(train_loader, print_freq, header)):
        
       
        llava_input = move_to_device(llava_input, device)

        if args.precision == "fp16":
            llava_input["images"] = llava_input["images"].half()              # [16, 5, 3, 720, 1280]
            llava_input["images_clip"] = llava_input["images_clip"].half()    # [16, 5, 3, 224, 224]
        elif args.precision == "bf16":
            llava_input["images"] = llava_input["images"].bfloat16()
            llava_input["images_clip"] = llava_input["images_clip"].bfloat16()
        else:
            llava_input["images"] = llava_input["images"].float()
            llava_input["images_clip"] = llava_input["images_clip"].float()

        # images = images.cuda()  # [B, T, 3, H, W]
        
        # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        targets = move_to_device(targets, device)

        boxes = torch.stack([t['boxes'] for t in targets])
        dummy_mask = torch.stack([t['actions'] == args.num_class + 1 for t in targets]).squeeze()


        num_batch = llava_input["images"].shape[0]
        num_frame = llava_input["images"].shape[1]

        bs, t, _, hc, wc = llava_input["images_clip"].shape

        input_dict = {
        "images": llava_input["images"],
        "images_clip": llava_input["images_clip"].reshape(bs * t, 3, hc, wc),
        "input_ids": llava_input["input_ids"],
        "labels": llava_input["labels"],
        "attention_masks": llava_input["attention_masks"],
        "offset": llava_input["offset"],
        "boxes": boxes,
        "dummy_mask": dummy_mask,
        "targets": targets,
        "inference": False,
        }

        outputs = model(**input_dict)

        # outputs = model(input_dict["images"], input_dict["boxes"], input_dict["dummy_mask"])


        loss_dict = criterion(outputs, targets, log=False)

        weight_dict = criterion.weight_dict


        loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # loss = loss_dict['loss_backward']


        # deepspeed 
        model.backward(loss)
        # accumulation_counter += 1

        if args.gradient_clipping:
            nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)

        model.step()
        model.zero_grad()  

        # if model.is_gradient_accumulation_boundary():
            # reduce losses over all GPUs for logging purposes

        # del loss_dict['loss_backward']
        loss_dict_reduced = utils.reduce_dict(loss_dict)
            # loss_dict_reduced = model.reduce_tensor(loss_dict)

        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()

        # debug
        # pdb.set_trace()
        
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # compute gradient and do SGD step
        # optimizer.zero_grad()
        # loss.backward()
        # if args.gradient_clipping:
        #     nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        # optimizer.step()
            # if args.local_rank == 0:
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(group_class_error=loss_dict_reduced['group_class_error'])
        # metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(lr=model.get_lr()[0])

    metric_logger.synchronize_between_processes()
    # if args.local_rank == 0:
    print("Averaged stats:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validate(test_loader, model, criterion, metrics, epoch):
    model.eval()
    criterion.eval()

    # debug
    # pdb.set_trace()

    metric_logger = loggers.MetricLogger(mode="test", delimiter="  ")
    metric_logger.add_meter('group_class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Evaluation Inference: '

    print_freq = len(test_loader)
    name_to_vid = {name: i + 1 for i, name in enumerate(SEQS_CAFE)}
    file_path = path + '/pred_group_epoch_%d.txt' % epoch


    device = next(model.parameters()).device

    criterion = criterion.to(device)

    weight_dict = {}
    weight_dict['loss_ce'] = args.ce_loss_coef
    weight_dict['loss_group_ce'] = args.group_ce_loss_coef
    weight_dict['loss_group_code'] = args.group_code_loss_coef
    weight_dict['loss_consistency'] = args.consistency_loss_coef

    for i, (llava_input, targets, infos) in enumerate(metric_logger.log_every(test_loader, print_freq, header)):
        # images = images.cuda()  # [B, T, 3, H, W]
        
        torch.cuda.empty_cache()

        llava_input = move_to_device(llava_input, device)

        if args.precision == "fp16":
            llava_input["images"] = llava_input["images"].half()              # [16, 5, 3, 720, 1280]
            llava_input["images_clip"] = llava_input["images_clip"].half()    # [16, 5, 3, 224, 224]
        elif args.precision == "bf16":
            llava_input["images"] = llava_input["images"].bfloat16()
            llava_input["images_clip"] = llava_input["images_clip"].bfloat16()
        else:
            llava_input["images"] = llava_input["images"].float()
            llava_input["images_clip"] = llava_input["images_clip"].float()


        # targets = [{k: v.cuda() for k, v in t.items()} for t in targets]
        device = next(model.parameters()).device
        # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        targets = move_to_device(targets, device)


        boxes = torch.stack([t['boxes'] for t in targets])
        dummy_mask = torch.stack([t['actions'] == args.num_class + 1 for t in targets]).squeeze()


        bs, t, _, hc, wc = llava_input["images_clip"].shape

        input_dict = {
        "images": llava_input["images"],
        "images_clip": llava_input["images_clip"].reshape(bs * t, 3, hc, wc),
        "input_ids": llava_input["input_ids"],
        "labels": llava_input["labels"],
        "attention_masks": llava_input["attention_masks"],
        "offset": llava_input["offset"],
        "boxes": boxes,
        "dummy_mask": dummy_mask,
        "targets": targets,
        "inference": True,
        }


        # compute output
        outputs = model(**input_dict)

        # outputs = model(input_dict["images"], input_dict["boxes"], input_dict["dummy_mask"])


        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict


        # outputs = loss_dict["outputs"]
        # del loss_dict['outputs']
        

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)

        metric_logger.update(group_class_error=loss_dict_reduced['group_class_error'])
        
        if args.local_rank == 0:
            make_txt(boxes, infos, outputs, name_to_vid, file_path)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    if args.local_rank == 0:
        print("Averaged stats:", metric_logger)

    # if args.local_rank == 0:
    detections = open(file_path, "r")
    result = metrics.evaluate(detections)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, result


def make_txt(boxes, infos, outputs, name_to_vid, file_path):
    for b in range(boxes.shape[0]):
        for t in range(boxes.shape[1]):
            image_w, image_h = args.image_width, args.image_height

            pred_group_actions = outputs['pred_activities'][b]
            pred_group_actions = F.softmax(pred_group_actions, dim=1)
            members = outputs['membership'][b]

            pred_membership = torch.argmax(members.transpose(0, 1), dim=1).detach().cpu()
            keep_membership = members.transpose(0, 1).max(-1).values > args.group_threshold
            pred_group_action = torch.argmax(pred_group_actions, dim=1).detach().cpu()

            for box_idx in range(boxes.shape[2]):
                x, y, w, h = boxes[b][t][box_idx]
                x1, y1, x2, y2 = (x - w / 2) * image_w, (y - h / 2) * image_h, (x + w / 2) * image_w, (
                            y + h / 2) * image_h

                pred_group_id = pred_membership[box_idx]
                pred_group_action_idx = pred_group_action[pred_group_id]
                pred_group_action_prob = pred_group_actions[pred_group_id][pred_group_action_idx]

                if not (x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0):
                    if pred_group_action_idx != (pred_group_actions.shape[-1] - 1):
                        if bool(keep_membership[box_idx]) is False:
                            pred_group_id = -1
                            pred_group_action_idx = args.num_class

                    pred_list = [name_to_vid[infos[b]['vid']], infos[b]['sid'], infos[b]['fid'][t],
                                 int(x1), int(y1), int(x2), int(y2),
                                 int(pred_group_id), int(pred_group_action_idx) + 1,
                                 float(pred_group_action_prob)]
                    str_to_be_added = [str(k) for k in pred_list]
                    str_to_be_added = (" ".join(str_to_be_added))

                    f = open(file_path, "a+")
                    f.write(str_to_be_added + "\r\n")
                    f.close()


if __name__ == '__main__':
    main()
