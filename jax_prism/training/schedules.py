import optax

def warmup_cosine_schedule(
    peak_lr: float,
    total_steps: int,
    warmup_steps: int,
    min_lr: float = 1e-6
) -> optax.Schedule:
    """Create a warmup → cosine decay learning rate schedule.

    Args:
        peak_lr: Maximum learning rate after warmup
        total_steps: Total number of training steps
        warmup_steps: Number of linear warmup steps
        min_lr (optional): Minimum learning rate at end of decay.

    Returns:
        An optax Schedule (callable: step → lr).
    """
    
    # 1. Create linear schedule from min_lr to peak_lr
    # over warmup_steps
    
    schedules = [
        optax.linear_schedule(
            init_value=min_lr, 
            end_value=peak_lr,
            transition_steps=warmup_steps
        ),
        optax.cosine_decay_schedule(
            init_value=peak_lr,
            decay_steps=total_steps - warmup_steps,
            alpha=min_lr / peak_lr
        )
    ]
    
    return optax.join_schedules(
        schedules, 
        boundaries=[warmup_steps]
    )

def warmup_plateau_cosine_schedule(
    peak_lr: float,
    total_steps: int,
    warmup_steps: int,
    plateau_steps: int,
    min_lr: float = 1e-6
) -> optax.Schedule:
    """Create a warmup → cosine decay learning rate schedule.

    Args:
        peak_lr: Maximum learning rate after warmup
        total_steps: Total number of training steps
        warmup_steps: Number of linear warmup steps
        plateau_steps: Number of training steps at peak_lr
        min_lr (optional): Minimum learning rate at end of decay.

    Returns:
        An optax Schedule (callable: step → lr).
    """
    
    # 1. Create linear schedule from min_lr to peak_lr
    # over warmup_steps
    
    schedules = [
        optax.linear_schedule(
            init_value=min_lr, 
            end_value=peak_lr,
            transition_steps=warmup_steps
        ),
        optax.constant_schedule(peak_lr),
        optax.cosine_decay_schedule(
            init_value=peak_lr,
            decay_steps=total_steps - warmup_steps - plateau_steps,
            alpha=min_lr / peak_lr
        )
    ]
    
    return optax.join_schedules(
        schedules, 
        boundaries=[warmup_steps, warmup_steps + plateau_steps]
    )