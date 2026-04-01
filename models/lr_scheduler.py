import torch.optim.lr_scheduler as lr_scheduler
from typing import Literal, Optional
from prodigyopt import Prodigy
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import LambdaLR
import math

def create_scheduler(optimizer, args, total_steps: Optional[int]):
    """
    根据 args.scheduler 创建学习率调度器

    Args:
        optimizer: 优化器
        args: 包含 scheduler 相关配置的参数对象
        total_steps: 总训练步数 (len(train_loader)*num_epochs)
    """
    scheduler_type = args.scheduler.lower()

    if scheduler_type == 'cosine_warmup':
        return create_warmup_cosine_scheduler(
            optimizer,
            warmup_steps=args.warmup_steps,
            total_steps=total_steps,
            warmup_start_lr_ratio=args.warmup_ratio,
            min_lr_ratio=args.min_lr_ratio,
        )
    elif scheduler_type == 'constant_warmup':
        return create_warmup_constant_scheduler(
            optimizer,
            warmup_steps=args.warmup_steps,
            warmup_start_lr_ratio=args.warmup_ratio
        )
    elif scheduler_type == 'constant':
        return create_constant_scheduler(optimizer)
    else:
        raise NotImplementedError(f"Unrecognized scheduler: {args.scheduler}")


def create_warmup_cosine_scheduler(optimizer,
                                   warmup_steps: int,
                                   total_steps: int,
                                   warmup_start_lr_ratio: float=0.01,
                                   min_lr_ratio: float=0.01):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return warmup_start_lr_ratio + (current_step / warmup_steps) * (1 - warmup_start_lr_ratio)
        else:
            progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
            return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


def create_constant_scheduler(optimizer):
    return LambdaLR(optimizer, lambda current_step: 1.0)


def create_warmup_constant_scheduler(optimizer,
                                     warmup_steps: int,
                                     warmup_start_lr_ratio: float=0.01):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return warmup_start_lr_ratio + (current_step / warmup_steps) * (1 - warmup_start_lr_ratio)
        else:
            return 1.0
    return LambdaLR(optimizer, lr_lambda)

def get_lr_scheduler(optimizer,
        scheduler_type:Literal['constant','step','mstep','exponential','cosine','cosine-warmup','poly'],
        **argv):
    if scheduler_type == 'constant':
        return lr_scheduler.ConstantLR(optimizer, **argv)
    elif scheduler_type == 'step':
        return lr_scheduler.StepLR(optimizer, **argv)
    elif scheduler_type == 'mstep':
        return lr_scheduler.MultiStepLR(optimizer, **argv)
    elif scheduler_type == 'exponential':
        return lr_scheduler.ExponentialLR(optimizer, **argv)
    elif scheduler_type == 'cosine':
        return lr_scheduler.CosineAnnealingLR(optimizer, **argv)
    elif scheduler_type == 'cosine-warmup':
        return create_warmup_cosine_scheduler(optimizer, **argv)
    elif scheduler_type == 'poly':
        return lr_scheduler.PolynomialLR(optimizer, **argv)
    raise NotImplementedError("Unrecognized scheduler type:{}".format(scheduler_type))

def get_optimizer(parameters,
        optimizer_type:Literal['adamw','adam','prodigy'],
        **argv):
    if optimizer_type == 'adamw':
        return AdamW(parameters, **argv)
    elif optimizer_type == 'adam':
        return Adam(parameters, **argv)
    elif optimizer_type == 'prodigy':
        return Prodigy(parameters, **argv)
    else:
        raise NotImplementedError("Unrecognized Optimizer type:{}".format(optimizer_type))