import math

def cosine_annealing(step, cfg):    
    if step < cfg.warmup_steps:
        return (step+1)/cfg.warmup_steps
    current_lr = cfg.min_lr + (0.5 * (cfg.max_lr - cfg.min_lr) * (1 + math.cos(math.pi * (step/cfg.max_steps))))
    return current_lr/cfg.max_lr   