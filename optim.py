from transformers.optimization import AdamW
import torch


def is_vision(args, param: str):
    if hasattr(args, 'vision_lr') and ('vision_encoder' in param):
        return True
    else:
        return False


def is_text(args, param: str):
    if hasattr(args, 'text_lr') and ('text_encoder' in param):
        return True
    else:
        return False


def is_cross(args, param: str):
    if hasattr(args, 'cross_lr') and ('cross_encoder' in param):  # only supports XVLMPlusBase
        return True
    else:
        return False


def create_optimizer(args, model):
    lr = args.lr
    wd = args.weight_decay
    lr_mult = getattr(args, 'lr_mult', 1)
    print(f"### lr: {args.lr},  lr_mult: {lr_mult}")

    optimizer_grouped_parameters = [
        {"params": [], "weight_decay": wd, "lr": lr},
        {"params": [], "weight_decay": 0.0, "lr": lr},
        {"params": [], "weight_decay": wd, "lr": lr * lr_mult},
        {"params": [], "weight_decay": 0.0, "lr": lr * lr_mult}
    ]

    if hasattr(args, 'vision_lr'):
        # 4 & 5
        print("### vision_lr: ", args.vision_lr)
        optimizer_grouped_parameters.append({"params": [], "weight_decay": wd, "lr": args.vision_lr})
        optimizer_grouped_parameters.append({"params": [], "weight_decay": 0.0, "lr": args.vision_lr})

        # 6 & 7
        assert hasattr(args, 'text_lr')
        print("### text_lr: ", args.text_lr)
        optimizer_grouped_parameters.append({"params": [], "weight_decay": wd, "lr": args.text_lr})
        optimizer_grouped_parameters.append({"params": [], "weight_decay": 0.0, "lr": args.text_lr})

        # 8 & 9
        if not hasattr(args, 'cross_lr'):
            args.cross_lr = args.text_lr

        print("### cross_lr: ", args.cross_lr, flush=True)
        optimizer_grouped_parameters.append({"params": [], "weight_decay": wd, "lr": args.cross_lr})
        optimizer_grouped_parameters.append({"params": [], "weight_decay": 0.0, "lr": args.cross_lr})

    no_decay = {"bias",
        "LayerNorm.bias",
        "LayerNorm.weight",
        "norm.bias",
        "norm.weight",
        "norm1.bias",
        "norm1.weight",
        "norm2.bias",
        "norm2.weight"}

    if hasattr(model, 'init_params'):
        large_lr = model.init_params
        print("### model has 'init_params', ", len(large_lr))
    else:
        large_lr = {}

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue  # frozen weights

        if any(nd in n for nd in no_decay):
            if is_vision(args, n):
                optimizer_grouped_parameters[5]['params'].append(p)
            elif is_text(args, n):
                optimizer_grouped_parameters[7]['params'].append(p)
            elif is_cross(args, n):
                optimizer_grouped_parameters[9]['params'].append(p)
            elif n in large_lr:
                optimizer_grouped_parameters[3]['params'].append(p)
            else:
                optimizer_grouped_parameters[1]['params'].append(p)
        else:  # decay
            if is_vision(args, n):
                optimizer_grouped_parameters[4]['params'].append(p)
            elif is_text(args, n):
                optimizer_grouped_parameters[6]['params'].append(p)
            elif is_cross(args, n):
                optimizer_grouped_parameters[8]['params'].append(p)
            elif n in large_lr:
                optimizer_grouped_parameters[2]['params'].append(p)
            else:
                optimizer_grouped_parameters[0]['params'].append(p)

    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=1e-8, betas=(0.9, 0.98))

    return optimizer


class LARS(torch.optim.Optimizer):
    """
    LARS optimizer, no rate scaling or weight decay for parameters <= 1D.
    """
    def __init__(self, params, lr=0, weight_decay=0, momentum=0.9, trust_coefficient=0.001):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, trust_coefficient=trust_coefficient)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if p.ndim > 1: # if not normalization gamma/beta or bias
                    dp = dp.add(p, alpha=g['weight_decay'])
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                    (g['trust_coefficient'] * param_norm / update_norm), one),
                                    one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)
                p.add_(mu, alpha=-g['lr'])
