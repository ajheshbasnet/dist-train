from configs import configs

def cosine_annealing(step):    
    if step < warmup_steps:
        return step/warmup_steps
    current_lr = min_lr + (0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * (step/max_steps))))
    return current_lr/configs.lr   