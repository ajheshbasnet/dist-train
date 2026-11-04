from configs import configs

def cosine_annealing(step):    
    if step < configs.warmup_steps:
        return step/configs.warmup_steps
    current_lr = configs.min_lr + (0.5 * (configs.max_lr - configs.min_lr) * (1 + math.cos(math.pi * (step/configs.max_steps))))
    return current_lr/configs.lr   