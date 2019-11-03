
# sourced from:
# https://github.com/oguiza/DataAugmentation/blob/master/ImageDataAugmentation.py

from fastai.torch_core import *
from fastai.callback import *
from fastai.callbacks.mixup import MixUpLoss
from fastai.basic_train import Learner, LearnerCallback

class CutMixCallback(LearnerCallback):
    "Callback that creates the cutmixed input and target."
    def __init__(self, learn:Learner, α:float=1., stack_y:bool=True, true_λ:bool=True):
        super().__init__(learn)
        self.α,self.stack_y,self.true_λ = α,stack_y,true_λ

    def on_train_begin(self, **kwargs):
        if self.stack_y: self.learn.loss_func = MixUpLoss(self.learn.loss_func)

    def on_batch_begin(self, last_input, last_target, train, **kwargs):
        "Applies cutmix to `last_input` and `last_target` if `train`."
        if not train: return
        λ = np.random.beta(self.α, self.α)
        λ = max(λ, 1- λ)
        shuffle = torch.randperm(last_target.size(0)).to(last_input.device)
        x1, y1 = last_input[shuffle], last_target[shuffle]
        #Get new input
        last_input_size = last_input.shape
        bbx1, bby1, bbx2, bby2 = rand_bbox(last_input.size(), λ)
        new_input = last_input.clone()
        new_input[:, ..., bby1:bby2, bbx1:bbx2] = last_input[shuffle, ..., bby1:bby2, bbx1:bbx2]
        λ = last_input.new([λ])
        if self.true_λ:
            λ = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (last_input_size[-1] * last_input_size[-2]))
            λ = last_input.new([λ])
        if self.stack_y:
            new_target = torch.cat([last_target.unsqueeze(1).float(), y1.unsqueeze(1).float(),
                                    λ.repeat(last_input_size[0]).unsqueeze(1).float()], 1)
        else:
            if len(last_target.shape) == 2:
                λ = λ.unsqueeze(1).float()
            new_target = last_target.float() * λ + y1.float() * (1-λ)
        return {'last_input': new_input, 'last_target': new_target}

    def on_train_end(self, **kwargs):
        if self.stack_y: self.learn.loss_func = self.learn.loss_func.get_old()


def rand_bbox(last_input_size, λ):
    '''lambd is always between .5 and 1'''

    W = last_input_size[-1]
    H = last_input_size[-2]
    cut_rat = np.sqrt(1. - λ) # 0. - .707
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmix(learn:Learner, α:float=1., stack_x:bool=False, stack_y:bool=True, true_λ:bool=True) -> Learner:
    "Add mixup https://arxiv.org/pdf/1905.04899.pdf to `learn`."
    learn.callback_fns.append(partial(CutMixCallback, α=α, stack_y=stack_y, true_λ=true_λ))
    return learn

Learner.cutmix = cutmix


from fastai.torch_core import *
from fastai.callback import *
from fastai.basic_train import Learner, LearnerCallback

class RicapLoss(nn.Module):
    "Adapt the loss function `crit` to go with ricap data augmentations."

    def __init__(self, crit, reduction='mean'):
        super().__init__()
        if hasattr(crit, 'reduction'):
            self.crit = crit
            self.old_red = crit.reduction
            setattr(self.crit, 'reduction', 'none')
        else:
            self.crit = partial(crit, reduction='none')
            self.old_crit = crit
        self.reduction = reduction

    def forward(self, output, target):
        if target.ndim == 2:
            c_ = target[:, 1:5]
            W_ = target[:, 5:]
            loss = [W_[:, k] * self.crit(output, c_[:, k].long()) for k in range(4)]
            d = torch.mean(torch.stack(loss))
        else: d = self.crit(output, target)
        if self.reduction == 'mean': return d.mean()
        elif self.reduction == 'sum': return d.sum()
        return d

    def get_old(self):
        if hasattr(self, 'old_crit'): return self.old_crit
        elif hasattr(self, 'old_red'):
            setattr(self.crit, 'reduction', self.old_red)
            return self.crit

class RicapCallback(LearnerCallback):
    "Callback that creates the ricap input and target."
    def __init__(self, learn:Learner, β:float=.3, stack_y:bool=True):
        super().__init__(learn)
        self.β,self.stack_y = β,stack_y

    def on_train_begin(self, **kwargs):
        if self.stack_y: self.learn.loss_func = RicapLoss(self.learn.loss_func)

    def on_batch_begin(self, last_input, last_target, train, **kwargs):
        "Applies ricap to `last_input` and `last_target` if `train`."
        if not train: return
        I_x, I_y = last_input.size()[2:]
        w = int(np.round(I_x * np.random.beta(self.β, self.β)))
        h = int(np.round(I_y * np.random.beta(self.β, self.β)))
        w_ = [w, I_x - w, w, I_x - w]
        h_ = [h, h, I_y - h, I_y - h]
        cropped_images = {}
        bs = last_input.size(0)
        c_ = torch.zeros((bs, 4)).float().to(last_input.device)
        W_ = torch.zeros(4).float().to(last_input.device)
        for k in range(4):
            idx = torch.randperm(bs)
            x_k = np.random.randint(0, I_x - w_[k] + 1)
            y_k = np.random.randint(0, I_y - h_[k] + 1)
            cropped_images[k] = last_input[idx][:, :, x_k:x_k + w_[k], y_k:y_k + h_[k]]
            c_[:, k] = last_target[idx].float()
            W_[k] = w_[k] * h_[k] / (I_x * I_y)
        patched_images = torch.cat(
            (torch.cat((cropped_images[0], cropped_images[1]), 2),
             torch.cat((cropped_images[2], cropped_images[3]), 2)), 3)  #.cuda()
        if self.stack_y:
                new_target = torch.cat((last_target[:,None].float(), c_,
                                        W_[None].repeat(last_target.size(0), 1)), dim=1)
        else:
            new_target = c_ * W_
        return {'last_input': patched_images, 'last_target': new_target}

    def on_train_end(self, **kwargs):
        if self.stack_y: self.learn.loss_func = self.learn.loss_func.get_old()


def ricap(learn:Learner, stack_y:bool=True) -> Learner:
    "Add ricap https://arxiv.org/pdf/1811.09030.pdf to `learn`."
    learn.callback_fns.append(partial(RicapCallback, stack_y=stack_y))
    return learn

Learner.ricap = ricap