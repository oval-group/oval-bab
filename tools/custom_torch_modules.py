from torch import nn
import torch
import numpy as np
import copy


class Flatten(nn.Module):
    def forward(self, x):
        self.in_shape = (-1, *x.shape[1:])
        return torch.flatten(x, start_dim=1)

    def inverse(self, inp):
        # Manually provide inverse operator.
        return inp.reshape(self.in_shape)


class View(nn.Module):
    '''
    This is necessary in order to reshape "flat activations" such as used by
    nn.Linear with those that comes from MaxPooling
    '''
    def __init__(self, out_shape):
        super(View, self).__init__()
        self.out_shape = out_shape

    def forward(self, inp):
        # We make the assumption that all the elements in the tuple have
        # the same batchsize and need to be brought to the same size

        # We assume that the first dimension is the batch size
        batch_size = inp.size(0)
        out_size = (batch_size, ) + self.out_shape
        out = inp.view(out_size)
        return out


class Transpose(nn.Module):
    def __init__(self, perm_shape):
        super(Transpose, self).__init__()
        self.perm_shape = perm_shape

    def forward(self, inp):
        out = inp.permute(self.perm_shape)
        return out

    def inverse(self, inp):
        # Manually provide inverse operator.
        inverse_shape = [0] * len(self.perm_shape)
        for idx, cs in enumerate(self.perm_shape):
            inverse_shape[cs] = idx
        out = inp.permute(tuple(inverse_shape))
        return out


class Reshape(nn.Module):
    '''
    This is necessary in order to reshape "flat activations" such as used by
    nn.Linear with those that comes from MaxPooling
    '''
    def __init__(self, out_shape):
        super(Reshape, self).__init__()
        self.out_shape = out_shape

    def forward(self, inp):
        # Handle failing reshapes because of mismatched batch sizes at construction.
        if np.prod(inp.shape) != np.prod(self.out_shape):
            if inp.shape[0] != self.out_shape[0]:
                self.out_shape = (-1, *self.out_shape[1:])
        self.in_shape = (-1, *inp.shape[1:])
        out = inp.reshape(self.out_shape)
        return out

    def inverse(self, inp):
        # Manually provide inverse operator.
        return inp.reshape(self.in_shape)


class Add(nn.Module):
    def __init__(self, const):
        super(Add, self).__init__()
        self.const = const

    def forward(self, inp):
        return inp + self.const

    def inverse(self, inp):
        # Manually provide inverse operator.
        return inp - self.const

    def cuda(self, device=None):
        super().cuda()
        self.const = self.const.cuda(device)
        return self

    def cpu(self):
        super().cpu()
        self.const = self.const.cpu()
        return self


class Mul(nn.Module):
    def __init__(self, const):
        super(Mul, self).__init__()
        self.const = const

    def forward(self, inp):
        return inp * self.const

    def inverse(self, inp):
        # Manually provide inverse operator.
        return inp / self.const

    def cuda(self, device=None):
        super().cuda()
        self.const = self.const.cuda(device)
        return self

    def cpu(self):
        super().cpu()
        self.const = self.const.cpu()
        return self


def unify_math_transforms(m_transforms, additive, multiplicative):
    for op in m_transforms:
        if isinstance(op, Add):
            # ax + b ->  ax + b + c = a x + (b + c)
            additive += op.const
        elif isinstance(op, Mul):
            # ax + b -> c( ax + b) = c a x + bc
            multiplicative *= op.const
            additive *= op.const
        else:
            raise ValueError('expected Add or Mul within unify_math_transforms.')
    return additive, multiplicative


def build_unified_math_transforms(layers, to_skip_count):
    operators = []
    for count, clayer in enumerate(layers):
        if isinstance(clayer, nn.ReLU) or isinstance(clayer, (nn.Linear, nn.Conv2d)):
            break
        if isinstance(clayer, math_transforms):
            operators.append(clayer)
        else:
            raise ValueError(f'unexpected operator {clayer} while processing add/mul')
        to_skip_count += 1
    if len(operators) > 0:
        additive, multiplicative = unify_math_transforms(operators, 0., 1.)
        return (additive, multiplicative), to_skip_count
    else:
        return None, to_skip_count


def parse_post_linear_math_transform(layer, additive, multiplicative):
    assert isinstance(layer, (nn.Linear, nn.Conv2d))
    with torch.no_grad():
        layer = copy.deepcopy(layer)
        multiplicative = multiplicative.squeeze()
        additive = additive.squeeze()
        layer.weight *= multiplicative.view(multiplicative.shape + (1,)*(layer.weight.dim() - multiplicative.dim()))
        layer.bias.data = layer.bias.data * multiplicative + additive
    return layer


supported_transforms = (Transpose, Reshape, Flatten, Add, Mul)
shape_transforms = (Transpose, Reshape, Flatten)
math_transforms = (Add, Mul)
