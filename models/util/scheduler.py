import torch
import math

class WarmupCosineRestarts(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self, optimizer, T_0, iters_per_epoch, T_mult=1, eta_min=0, warmup_ratio=0.1, warmup_lr_init=1e-7, last_epoch=-1
    ):
        # Similar to torch.optim.lr_scheduler.OneCycleLR()
        # But allow multiple cycles and a warmup
        self.T_0 = T_0 * iters_per_epoch
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.warmup_iters = int(T_0 * warmup_ratio * iters_per_epoch)
        self.warmup_lr_init = warmup_lr_init
        super(WarmupCosineRestarts, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.T_mult == 1:
            i_restart = self.last_epoch // self.T_0
            T_cur = self.last_epoch - i_restart * self.T_0
        else:
            n = int(math.log((self.last_epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
            T_cur = self.last_epoch - self.T_0 * (self.T_mult**n - 1) // (self.T_mult - 1)

        if T_cur < self.warmup_iters:
            warmup_ratio = T_cur / self.warmup_iters
            return [self.warmup_lr_init + (base_lr - self.warmup_lr_init) * warmup_ratio for base_lr in self.base_lrs]
        else:
            T_cur_adjusted = T_cur - self.warmup_iters
            T_i = self.T_0 - self.warmup_iters
            return [
                self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * T_cur_adjusted / T_i)) / 2
                for base_lr in self.base_lrs
            ]
