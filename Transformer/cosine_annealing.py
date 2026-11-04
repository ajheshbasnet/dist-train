from configs import configs

def cosine_annealing(step):    
    if step < self.warmup_steps:
        return step/self.warmup_steps
    current_lr = self.min_lr + (0.5 * (self.max_lr - self.min_lr) * (1 + math.cos(math.pi * (step/self.max_steps))))
    return current_lr/configs.lr   