import math
import numbers
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

if torch.__version__ >= '1.4.0':
    kwargs = {'align_corners': False}
else:
    kwargs = {}


def get_bounds(mean, std):
    '''
    get bound of dataset
    :param mean: list, float
    :param std: list, float
    :return: list, [lower_bound, upper_bound]
    '''
    bounds = [-1, 1]
    if type(mean) == type(()):
        c = len(mean)
        _min = (np.zeros([c]) - np.array(mean)) / np.array([std])
        _max = (np.ones([c]) - np.array(mean)) / np.array([std])
        bounds = [np.min(_min).item(), np.max(_max).item()]
    elif type(mean) == float:
        bounds = [(0.0 - mean) / std, (1.0 - mean) / std]
    return bounds


def rgb2hsv(rgb):
    """Convert a 4-d RGB tensor to the HSV counterpart.
    Here, we compute hue using atan2() based on the definition in [1],
    instead of using the common lookup table approach as in [2, 3].
    Those values agree when the angle is a multiple of 30°,
    otherwise they may differ at most ~1.2°.

    References
    [1] https://en.wikipedia.org/wiki/Hue
    [2] https://www.rapidtables.com/convert/color/rgb-to-hsv.html
    [3] https://github.com/scikit-image/scikit-image/blob/master/skimage/color/colorconv.py#L212
    """
    r, g, b = rgb[:, 0, :, :], rgb[:, 1, :, :], rgb[:, 2, :, :]
    Cmax = rgb.max(1)[0]
    Cmin = rgb.min(1)[0]
    delta = Cmax - Cmin
    hue = torch.atan2(math.sqrt(3) * (g - b), 2 * r - g - b)
    hue = (hue % (2 * math.pi)) / (2 * math.pi)
    saturate = delta / Cmax
    value = Cmax
    hsv = torch.stack([hue, saturate, value], dim=1)
    hsv[~torch.isfinite(hsv)] = 0.
    return hsv


def hsv2rgb(hsv):
    """Convert a 4-d HSV tensor to the RGB counterpart.
    >>> %timeit hsv2rgb(hsv)
    2.37 ms ± 13.4 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
    >>> %timeit rgb2hsv_fast(rgb)
    298 µs ± 542 ns per loop (mean ± std. dev. of 7 runs, 1000 loops each)
    >>> torch.allclose(hsv2rgb(hsv), hsv2rgb_fast(hsv), atol=1e-6)
    True

    References
    [1] https://en.wikipedia.org/wiki/HSL_and_HSV#HSV_to_RGB_alternative
    """
    h, s, v = hsv[:, [0]], hsv[:, [1]], hsv[:, [2]]
    c = v * s

    n = hsv.new_tensor([5, 3, 1]).view(3, 1, 1)
    k = (n + h * 6) % 6
    t = torch.min(k, 4 - k)
    t = torch.clamp(t, 0, 1)

    return v - c * t


class RandomResizedCropLayer(nn.Module):
    def __init__(self, size=None, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.)):
        '''
            Inception Crop
            size (tuple): size of fowarding image (C, W, H)
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped
        '''
        super(RandomResizedCropLayer, self).__init__()

        _eye = torch.eye(2, 3)
        self.size = size
        self.register_buffer('_eye', _eye)
        self.scale = scale
        self.ratio = ratio

    def forward(self, inputs, whbias=None):
        _device = inputs.device
        N = inputs.size(0)
        _theta = self._eye.repeat(N, 1, 1)

        if whbias is None:
            whbias = self._sample_latent(inputs)

        _theta[:, 0, 0] = whbias[:, 0]
        _theta[:, 1, 1] = whbias[:, 1]
        _theta[:, 0, 2] = whbias[:, 2]
        _theta[:, 1, 2] = whbias[:, 3]

        grid = F.affine_grid(_theta, inputs.size(), **kwargs).to(_device)
        output = F.grid_sample(inputs, grid, padding_mode='reflection', **kwargs)

        if self.size is not None:
            output = F.adaptive_avg_pool2d(output, self.size)

        return output

    def _clamp(self, whbias):
        w = whbias[:, 0]
        h = whbias[:, 1]
        w_bias = whbias[:, 2]
        h_bias = whbias[:, 3]
        # Clamp with scale
        w = torch.clamp(w, *self.scale)
        h = torch.clamp(h, *self.scale)
        # Clamp with ratio
        w = self.ratio[0] * h + torch.relu(w - self.ratio[0] * h)
        w = self.ratio[1] * h - torch.relu(self.ratio[1] * h - w)
        # Clamp with bias range: w_bias \in (w - 1, 1 - w), h_bias \in (h - 1, 1 - h)
        w_bias = w - 1 + torch.relu(w_bias - w + 1)
        w_bias = 1 - w - torch.relu(1 - w - w_bias)
        h_bias = h - 1 + torch.relu(h_bias - h + 1)
        h_bias = 1 - h - torch.relu(1 - h - h_bias)
        whbias = torch.stack([w, h, w_bias, h_bias], dim=0).t()
        return whbias

    def _sample_latent(self, inputs):
        _device = inputs.device
        N, _, width, height = inputs.shape
        # N * 10 trial
        area = width * height
        target_area = np.random.uniform(*self.scale, N * 10) * area
        log_ratio = (math.log(self.ratio[0]), math.log(self.ratio[1]))
        aspect_ratio = np.exp(np.random.uniform(*log_ratio, N * 10))
        # If doesn't satisfy ratio condition, then do central crop
        w = np.round(np.sqrt(target_area * aspect_ratio))
        h = np.round(np.sqrt(target_area / aspect_ratio))
        cond = (0 < w) * (w <= width) * (0 < h) * (h <= height)
        w = w[cond]
        h = h[cond]
        cond_len = w.shape[0]
        if cond_len >= N:
            w = w[:N]
            h = h[:N]
        else:
            w = np.concatenate([w, np.ones(N - cond_len) * width])
            h = np.concatenate([h, np.ones(N - cond_len) * height])
        w_bias = np.random.randint(w - width, width - w + 1) / width
        h_bias = np.random.randint(h - height, height - h + 1) / height
        w = w / width
        h = h / height
        whbias = np.column_stack([w, h, w_bias, h_bias])
        whbias = torch.tensor(whbias, device=_device)
        return whbias

class HorizontalFlipRandomCrop(nn.Module):
    def __init__(self, max_range):
        super(HorizontalFlipRandomCrop, self).__init__()
        self.max_range = max_range
        _eye = torch.eye(2, 3)
        self.register_buffer('_eye', _eye)

    def forward(self, input, sign=None, bias=None, rotation=None):
        _device = input.device
        N = input.size(0)
        _theta = self._eye.repeat(N, 1, 1)
        if sign is None:
            sign = torch.bernoulli(torch.ones(N, device=_device) * 0.5) * 2 - 1
        if bias is None:
            bias = torch.empty((N, 2), device=_device).uniform_(-self.max_range, self.max_range)
        _theta[:, 0, 0] = sign
        _theta[:, :, 2] = bias
        if rotation is not None:
            _theta[:, 0:2, 0:2] = rotation
        grid = F.affine_grid(_theta, input.size(), **kwargs).to(_device)
        output = F.grid_sample(input, grid, padding_mode='reflection', **kwargs)
        return output

    def _sample_latent(self, N, device=None):
        sign = torch.bernoulli(torch.ones(N, device=device) * 0.5) * 2 - 1
        bias = torch.empty((N, 2), device=device).uniform_(-self.max_range, self.max_range)
        return sign, bias


class Rotation(nn.Module):
    def __init__(self, max_range = 4):
        super(Rotation, self).__init__()
        self.max_range = max_range
        self.prob = 0.5

    def forward(self, input, aug_index=None):
        _device = input.device
        _, _, H, W = input.size()
        if aug_index is None:
            aug_index = np.random.randint(4)
            output = torch.rot90(input, aug_index, (2, 3))
            _prob = input.new_full((input.size(0),), self.prob)
            _mask = torch.bernoulli(_prob).view(-1, 1, 1, 1)
            output = _mask * input + (1-_mask) * output
        else:
            aug_index = aug_index % self.max_range
            output = torch.rot90(input, aug_index, (2, 3))
        return output


class CutPerm(nn.Module):
    def __init__(self, max_range = 4):
        super(CutPerm, self).__init__()
        self.max_range = max_range
        self.prob = 0.5

    def forward(self, input, aug_index=None):
        _device = input.device
        _, _, H, W = input.size()
        if aug_index is None:
            aug_index = np.random.randint(4)
            output = self._cutperm(input, aug_index)
            _prob = input.new_full((input.size(0),), self.prob)
            _mask = torch.bernoulli(_prob).view(-1, 1, 1, 1)
            output = _mask * input + (1 - _mask) * output
        else:
            aug_index = aug_index % self.max_range
            output = self._cutperm(input, aug_index)
        return output

    def _cutperm(self, inputs, aug_index):
        _, _, H, W = inputs.size()
        h_mid = int(H / 2)
        w_mid = int(W / 2)
        jigsaw_h = aug_index // 2
        jigsaw_v = aug_index % 2
        if jigsaw_h == 1:
            inputs = torch.cat((inputs[:, :, h_mid:, :], inputs[:, :, 0:h_mid, :]), dim=2)
        if jigsaw_v == 1:
            inputs = torch.cat((inputs[:, :, :, w_mid:], inputs[:, :, :, 0:w_mid]), dim=3)
        return inputs


class HorizontalFlipLayer(nn.Module):
    def __init__(self):
        """
        img_size : (int, int, int)
            Height and width must be powers of 2.  E.g. (32, 32, 1) or
            (64, 128, 3). Last number indicates number of channels, e.g. 1 for
            grayscale or 3 for RGB
        """
        super(HorizontalFlipLayer, self).__init__()
        _eye = torch.eye(2, 3)
        self.register_buffer('_eye', _eye)

    def forward(self, inputs):
        _device = inputs.device
        N = inputs.size(0)
        _theta = self._eye.repeat(N, 1, 1)
        r_sign = torch.bernoulli(torch.ones(N, device=_device) * 0.5) * 2 - 1
        _theta[:, 0, 0] = r_sign
        grid = F.affine_grid(_theta, inputs.size(), **kwargs).to(_device)
        inputs = F.grid_sample(inputs, grid, padding_mode='reflection', **kwargs)
        return inputs


class RandomColorGrayLayer(nn.Module):
    def __init__(self, p):
        super(RandomColorGrayLayer, self).__init__()
        self.prob = p
        _weight = torch.tensor([[0.299, 0.587, 0.114]])
        self.register_buffer('_weight', _weight.view(1, 3, 1, 1))

    def forward(self, inputs, aug_index=None):
        if aug_index == 0:
            return inputs
        l = F.conv2d(inputs, self._weight)
        gray = torch.cat([l, l, l], dim=1)
        if aug_index is None:
            _prob = inputs.new_full((inputs.size(0),), self.prob)
            _mask = torch.bernoulli(_prob).view(-1, 1, 1, 1)
            gray = inputs * (1 - _mask) + gray * _mask
        return gray


class ColorJitterLayer(nn.Module):
    def __init__(self, p, brightness, contrast, saturation, hue):
        super(ColorJitterLayer, self).__init__()
        self.prob = p
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)

    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))
        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    def adjust_contrast(self, x):
        if self.contrast:
            factor = x.new_empty(x.size(0), 1, 1, 1).uniform_(*self.contrast)
            means = torch.mean(x, dim=[2, 3], keepdim=True)
            x = (x - means) * factor + means
        return torch.clamp(x, 0, 1)

    def adjust_hsv(self, x):
        f_h = x.new_zeros(x.size(0), 1, 1)
        f_s = x.new_ones(x.size(0), 1, 1)
        f_v = x.new_ones(x.size(0), 1, 1)

        if self.hue:
            f_h.uniform_(*self.hue)
        if self.saturation:
            f_s = f_s.uniform_(*self.saturation)
        if self.brightness:
            f_v = f_v.uniform_(*self.brightness)

        return RandomHSVFunction.apply(x, f_h, f_s, f_v)

    def transform(self, inputs):
        # Shuffle transform
        if np.random.rand() > 0.5:
            transforms = [self.adjust_contrast, self.adjust_hsv]
        else:
            transforms = [self.adjust_hsv, self.adjust_contrast]

        for t in transforms:
            inputs = t(inputs)

        return inputs

    def forward(self, inputs):
        _prob = inputs.new_full((inputs.size(0),), self.prob)
        _mask = torch.bernoulli(_prob).view(-1, 1, 1, 1)
        return inputs * (1 - _mask) + self.transform(inputs) * _mask


class RandomHSVFunction(Function):
    @staticmethod
    def forward(ctx, x, f_h, f_s, f_v):
        # ctx is a context object that can be used to stash information
        # for backward computation
        x = rgb2hsv(x)
        h = x[:, 0, :, :]
        h += (f_h * 255. / 360.)
        h = (h % 1)
        x[:, 0, :, :] = h
        x[:, 1, :, :] = x[:, 1, :, :] * f_s
        x[:, 2, :, :] = x[:, 2, :, :] * f_v
        x = torch.clamp(x, 0, 1)
        x = hsv2rgb(x)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        # We return as many input gradients as there were arguments.
        # Gradients of non-Tensor arguments to forward must be None.
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.clone()
        return grad_input, None, None, None


class NormalizeLayer(nn.Module):
    """
    In order to certify radii in original coordinates rather than standardized coordinates, we
    add the Gaussian noise _before_ standardizing, which is why we have standardization be the first
    layer of the classifier rather than as a part of preprocessing as is typical.
    """
    def __init__(self):
        super(NormalizeLayer, self).__init__()

    def forward(self, inputs):
        return (inputs - 0.5) / 0.5



class Adv(nn.Module):
    def __init__(self, model, bounds):
        super(Adv, self).__init__()
        self.model = model
        self.bounds = bounds
        self.device = next(model.parameters()).device

    def pgd(self, images, labels, eps=8./255, alpha=10./255., steps=0, random_start=True):
        steps = random.randint(30, 60) if steps == 0 else steps
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        loss = nn.CrossEntropyLoss()
        adv_images = images.clone().detach()
        if random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-eps, eps)
            adv_images = torch.clamp(adv_images, min=self.bounds[0], max=self.bounds[1]).detach()
        for i in range(steps):
            adv_images.requires_grad = True
            outputs = self.model(adv_images)
            cost = -1. * loss(outputs, labels)
            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]
            adv_images = adv_images.detach() - alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-eps, max=eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()
        return adv_images

    def fgsm(self, images, labels, eps=8.0/255.):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        loss = nn.CrossEntropyLoss()
        images.requires_grad = True
        outputs = self.model(images)
        cost = -1. * loss(outputs, labels)
        grad = torch.autograd.grad(cost, images,
                                   retain_graph=False, create_graph=False)[0]
        adv_images = images - eps * grad.sign()
        adv_images = torch.clamp(adv_images, min=self.bounds[0], max=self.bounds[1]).detach()
        return adv_images

    def bim(self, images, labels, eps=8.0/255, alpha=10.0/255, steps=0):
        steps = random.randint(5, 10) if steps == 0 else steps
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        loss = torch.nn.CrossEntropyLoss()
        ori_images = images.clone().detach()

        for i in range(steps):
            images.requires_grad = True
            outputs = self.model(images)
            cost = (-1.0) * loss(outputs, labels)

            grad = torch.autograd.grad(cost, images,
                                       retain_graph=False,
                                       create_graph=False)[0]
            adv_images = images - alpha*grad.sign()
            # a = max(ori_images-eps, 0)
            a = torch.clamp(ori_images - eps, min=self.bounds[0])
            # b = max(adv_images, a) = max(adv_images, ori_images-eps, 0)
            b = (adv_images >= a).float()*adv_images \
                + (adv_images < a).float()*a
            # c = min(ori_images+eps, b) = min(ori_images+eps, max(adv_images, ori_images-eps, 0))
            c = (b > ori_images+eps).float()*(ori_images+eps) \
                + (b <= ori_images + eps).float()*b
            # images = max(1, c) = min(1, ori_images+eps, max(adv_images, ori_images-eps, 0))
            images = torch.clamp(c, max=self.bounds[1]).detach()
        return images

    def cw(self, images, labels, c=1e-4, kappa=0, lr=0.01, steps=0):
        steps = random.randint(200, 800) if steps == 0 else steps
        self.c = c
        self.kappa = kappa
        self.steps = steps
        self.lr = lr

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        # w = torch.zeros_like(images).detach() # Requires 2x times
        w = self.inverse_tanh_space(images).detach()
        w.requires_grad = True

        best_adv_images = images.clone().detach()
        best_L2 = 1e10*torch.ones((len(images))).to(self.device)
        prev_cost = 1e10
        dim = len(images.shape)

        MSELoss = torch.nn.MSELoss(reduction='none')
        Flatten = torch.nn.Flatten()
        optimizer = torch.optim.Adam([w], lr=lr)

        for step in range(steps):
            # Get Adversarial Images
            adv_images = self.tanh_space(w)

            current_L2 = MSELoss(Flatten(adv_images),
                                 Flatten(images)).sum(dim=1)
            L2_loss = current_L2.sum()
            outputs = self.model(adv_images)
            f_loss = self.f(outputs, labels).sum()
            cost = L2_loss + c*f_loss
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            # Update Adversarial Images
            _, pre = torch.max(outputs.detach(), 1)
            correct = (pre == labels).float()

            mask = (1-correct)*(best_L2 > current_L2.detach())
            best_L2 = mask*current_L2.detach() + (1-mask)*best_L2

            mask = mask.view([-1]+[1]*(dim-1))
            best_adv_images = mask*adv_images.detach() + (1-mask)*best_adv_images

            # Early Stop when loss does not converge.
            if step % (steps//10) == 0:
                if cost.item() > prev_cost:
                    return best_adv_images
                prev_cost = cost.item()
        return best_adv_images

    def tanh_space(self, x):
        return 1/2*(torch.tanh(x) + 1)

    def inverse_tanh_space(self, x):
        # torch.atanh is only for torch >= 1.7.0
        return self.atanh(x*2-1)

    def atanh(self, x):
        return 0.5*torch.log((1+x)/(1-x))

    # f-function in the paper
    def f(self, outputs, labels):
        one_hot_labels = torch.eye(len(outputs[0]))[labels].to(self.device)
        i, _ = torch.max((1-one_hot_labels)*outputs, dim=1)
        j = torch.masked_select(outputs, one_hot_labels.bool())
        return torch.clamp(-1.0*(i-j), min=-self.kappa)

    def forward(self, images, labels, aug_index=None):
        self.device = images.device
        if aug_index == 0:
            return self.pgd(images, labels)
        elif aug_index == 1:
            return self.fgsm(images, labels)
        elif aug_index == 2:
            return self.bim(images, labels)
        else:
            return images






















