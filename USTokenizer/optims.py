# This script is from https://github.com/salesforce/LAVIS/blob/main/lavis/common/optims.py

import math
import logging

import torch


class LinearWarmupStepLRScheduler:
    def __init__(
        self,
        optimizer,
        max_epoch,
        min_lr,
        init_lr,
        decay_rate=1,
        warmup_start_lr=-1,
        warmup_steps=0,
        **kwargs
    ):
        self.optimizer = optimizer

        self.max_epoch = max_epoch
        self.min_lr = min_lr

        self.decay_rate = decay_rate

        self.init_lr = init_lr
        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr if warmup_start_lr >= 0 else init_lr

    def step(self, cur_epoch, cur_step):
        if cur_epoch == 0:
            warmup_lr_schedule(
                step=cur_step,
                optimizer=self.optimizer,
                max_step=self.warmup_steps,
                init_lr=self.warmup_start_lr,
                max_lr=self.init_lr,
            )
        else:
            step_lr_schedule(
                epoch=cur_epoch,
                optimizer=self.optimizer,
                init_lr=self.init_lr,
                min_lr=self.min_lr,
                decay_rate=self.decay_rate,
            )


class LinearWarmupCosineLRScheduler:
    def __init__(
        self,
        optimizer,
        max_epoch,
        iters_per_epoch,
        min_lr,
        init_lr,
        warmup_steps=0,
        warmup_start_lr=-1,
        **kwargs
    ):
        self.optimizer = optimizer

        self.max_epoch = max_epoch
        self.iters_per_epoch = iters_per_epoch
        self.min_lr = min_lr

        self.init_lr = init_lr
        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr if warmup_start_lr >= 0 else init_lr

    def step(self, cur_epoch, cur_step):
        total_cur_step = cur_epoch * self.iters_per_epoch + cur_step
        if total_cur_step < self.warmup_steps:
            warmup_lr_schedule(
                step=cur_step,
                optimizer=self.optimizer,
                max_step=self.warmup_steps,
                init_lr=self.warmup_start_lr,
                max_lr=self.init_lr,
            )
        else:
            cosine_lr_schedule(
                epoch=total_cur_step,
                optimizer=self.optimizer,
                max_epoch=self.max_epoch * self.iters_per_epoch,
                init_lr=self.init_lr,
                min_lr=self.min_lr,
            )


def cosine_lr_schedule(optimizer, epoch, max_epoch, init_lr, min_lr):
    """Decay the learning rate"""
    for param_group in optimizer.param_groups:
        # 从参数组中读取 init_lr 和 min_lr（若未定义则使用默认值）
        if "warmup_start_lr" in param_group:
            init_lr = param_group["init_lr"]
            min_lr = param_group["min_lr"]
        lr = (init_lr - min_lr) * 0.5 * (
            1.0 + math.cos(math.pi * epoch / max_epoch)
        ) + min_lr
        param_group["lr"] = lr


def warmup_lr_schedule(optimizer, step, max_step, init_lr, max_lr):
    """Warmup the learning rate"""
    for param_group in optimizer.param_groups:
        if "warmup_start_lr" in param_group:
            init_lr = param_group["warmup_start_lr"]
            max_lr = param_group["init_lr"]
        lr = min(max_lr, init_lr + (max_lr - init_lr) * step / max(max_step, 1))
        param_group["lr"] = lr


def step_lr_schedule(optimizer, epoch, init_lr, min_lr, decay_rate):
    """Decay the learning rate"""
    lr = max(min_lr, init_lr * (decay_rate**epoch))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def get_optimizer(model, config):
    num_parameters = 0
    p_wd, p_non_wd = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue  # frozen weights
        print(n)
        if p.ndim < 2 or "bias" in n or "ln" in n or "bn" in n:
            p_non_wd.append(p)
        else:
            p_wd.append(p)
        num_parameters += p.data.nelement()
    logging.info("number of trainable parameters: %d" % num_parameters)
    optim_params = [
        {
            "params": p_wd,
            "weight_decay": float(config.weight_decay),
        },
        {"params": p_non_wd, "weight_decay": 0},
    ]
    beta2 = config.get("beta2", 0.999)
    optimizer = torch.optim.AdamW(
        optim_params,
        lr=float(config.init_lr),
        weight_decay=float(config.weight_decay),
        betas=(0.9, beta2),
    )

    return optimizer

def get_optimizer_low_lr_whisper(model, config):
    num_parameters = 0
    # 判断模型是否为DDP包装（参数名是否包含"module."前缀）
    has_module = any(name.startswith("module.") for name, _ in model.named_parameters())
    prefix = "module." if has_module else ""
    speech_encoder_prefix = f"{prefix}speech_encoder."

    p_wd_non_speech = []
    p_non_wd_non_speech = []
    p_wd_speech = []
    p_non_wd_speech = []

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue  
        num_parameters += p.data.nelement()
        
        # 判断参数是否属于 speech_encoder
        is_speech = n.startswith(speech_encoder_prefix)
        
        if p.ndim < 2 or "bias" in n or "ln" in n or "bn" in n:
            if is_speech:
                p_non_wd_speech.append(p)
            else:
                p_non_wd_non_speech.append(p)
        else:
            if is_speech:
                p_wd_speech.append(p)
            else:
                p_wd_non_speech.append(p)

    logging.info(f"number of trainable parameters: {num_parameters}")

    optim_params = [
        { 
            "params": p_wd_non_speech,
            "weight_decay": float(config.weight_decay),
            # "init_lr": float(config.init_lr),   
            # "min_lr": float(config.min_lr),    
        },
        {
            "params": p_non_wd_non_speech,
            "weight_decay": 0,
            # "init_lr": float(config.init_lr),   
            # "min_lr": float(config.min_lr),   
        },
        { 
            "params": p_wd_speech,
            "weight_decay": float(config.weight_decay),
            "init_lr": float(config.init_lr) * 0.1,  
            "min_lr": float(config.min_lr) * 0.1,  
            "warmup_start_lr": float(config.warmup_start_lr) * 0.1, 
        },
        { 
            "params": p_non_wd_speech,
            "weight_decay": 0,
            "init_lr": float(config.init_lr) * 0.1,
            "min_lr": float(config.min_lr) * 0.1,
            "warmup_start_lr": float(config.warmup_start_lr) * 0.1,
        },
    ]

    beta2 = config.get("beta2", 0.999)
    optimizer = torch.optim.AdamW(
        optim_params,
        lr=float(config.init_lr),  
        betas=(0.9, beta2),
        weight_decay=float(config.weight_decay),
    )

    return optimizer