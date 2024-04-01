from torch.optim.lr_scheduler import LambdaLR, _LRScheduler
from torch.optim.optimizer import Optimizer


class InverseSquareRootLRScheduler(LambdaLR):

    def __init__(self, optimizer, warmup_step):
        r"""
        """

        def lr_lambda(step):
            return 1 / (max(step, warmup_step)**0.5)

        super().__init__(optimizer, lr_lambda, -1)


class DecayLRScheduler(LambdaLR):

    def __init__(self, optimizer, rate):
        r"""
        """

        def lr_lambda(step):
            return rate**step

        super().__init__(optimizer, lr_lambda, -1)


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    r"""
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


class LRScheduler(_LRScheduler):

    def __init__(self, optimizer, last_epoch=-1):
        r"""
        Check if using mixed precision training
        """
        self.mixed_training = False
        base_optimizer = optimizer

        # Check that optimizer param is valid
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(type(optimizer).__name__))

        super(LRScheduler, self).__init__(base_optimizer, last_epoch)

    def step(self, epoch=None):
        r"""
        Set the current training step
        ('epoch' is used to be consistent with _LRScheduler)
        """

        if self.mixed_training:
            # The assumption is that the step will be constant
            state_dict = self.optimizer.state[self.optimizer.param_groups[0]['params'][0]]
            if 'step' in state_dict:
                self.last_epoch = state_dict['step'] + 1
            else:
                self.last_epoch = 1
        else:
            self.last_epoch = epoch if epoch is not None else self.last_epoch + 1

        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


class StaticScheduler(LRScheduler):
    r"""
    Applies a warm up period to the learning rate.
    """

    def __init__(self, optimizer, warmup, total_steps, last_epoch=-1):
        super(StaticScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr for base_lr in self.base_lrs]


class LinearWarmUpScheduler(LRScheduler):
    r"""
    Applies a warm up period to the learning rate.
    """

    def __init__(self, optimizer, warmup, total_steps, last_epoch=-1):
        self.warmup = warmup
        self.total_steps = total_steps
        super(LinearWarmUpScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        progress = self.last_epoch / self.total_steps
        if progress < self.warmup:
            return [base_lr * progress / self.warmup for base_lr in self.base_lrs]
        else:
            return [base_lr * max((progress - 1.0) / (self.warmup - 1.0), 0.) for base_lr in self.base_lrs]


class LinearWarmupPolyDecayScheduler(LRScheduler):
    r"""
    Applies a warm up period to the learning rate.
    """

    def __init__(self,
                 optimizer,
                 start_warmup_steps,
                 warmup_steps,
                 total_steps,
                 end_learning_rate=0.0,
                 degree=1.0,
                 last_epoch=-1):
        self.num_warmup_updates = warmup_steps
        self.start_warmup_steps = start_warmup_steps
        self.total_steps = total_steps
        self.end_learning_rate = end_learning_rate
        self.degree = degree
        super(LinearWarmupPolyDecayScheduler, self).__init__(optimizer, last_epoch)

    def step(self, **kwargs):
        param_group = self.optimizer.param_groups[0]
        if 'step' in param_group:
            self.last_epoch = param_group['step'] + 1
        else:
            self.last_epoch = 1

        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

    def get_lr(self):
        mod_step = self.last_epoch - self.start_warmup_steps
        if mod_step < self.num_warmup_updates:
            progress = mod_step / self.num_warmup_updates
            return [(base_lr * progress) for base_lr in self.base_lrs]
        else:
            progress = min(self.last_epoch / self.total_steps, 1.0)
            return [(base_lr - self.end_learning_rate) * (1 - progress)**self.degree + self.end_learning_rate
                    for base_lr in self.base_lrs]
