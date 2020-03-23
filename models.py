# Copyright 2020 Lorna Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import fuse_conv_and_bn
from utils import model_info
from utils import parse_model_config

ONNX_EXPORT = False


def create_modules(module_defs, img_size):
    # Constructs module list of layer blocks from module configuration in module_defs

    hyperparams = module_defs.pop(0)
    output_filters = [int(hyperparams["channels"])]
    module_list = nn.ModuleList()
    routs = []  # list of layers which rout to deeper layers
    yolo_index = -1

    for i, module in enumerate(module_defs):
        modules = nn.Sequential()

        if module["type"] == "convolutional":
            bn = module["batch_normalize"]
            filters = module["filters"]
            size = module["size"]
            stride = module["stride"] if "stride" in module else (
                module["stride_y"], module["stride_x"])
            modules.add_module("Conv2d",
                               nn.Conv2d(in_channels=output_filters[-1],
                                         out_channels=filters,
                                         kernel_size=size,
                                         stride=stride,
                                         padding=(size - 1) // 2 if module["pad"]
                                         else 0,
                                         groups=module[
                                             "groups"] if "groups" in module else 1,
                                         bias=not bn))
            if bn:
                modules.add_module("BatchNorm2d",
                                   nn.BatchNorm2d(filters, momentum=0.003,
                                                  eps=1E-4))
            else:
                routs.append(i)  # detection output (goes into yolo layer)

            if module["activation"] == "leaky":
                modules.add_module("activation",
                                   nn.LeakyReLU(0.1, inplace=True))
            elif module["activation"] == "swish":
                modules.add_module("activation", Swish())
            elif module["activation"] == "mish":
                modules.add_module("activation", Mish())

        elif module["type"] == "maxpool":
            size = module["size"]
            stride = module["stride"]
            maxpool = nn.MaxPool2d(kernel_size=size, stride=stride,
                                   padding=(size - 1) // 2)
            if size == 2 and stride == 1:  # yolov3-tiny
                modules.add_module("ZeroPad2d", nn.ZeroPad2d((0, 1, 0, 1)))
                modules.add_module("MaxPool2d", maxpool)
            else:
                modules = maxpool

        elif module["type"] == "upsample":
            if ONNX_EXPORT:  # explicitly state size, avoid scale_factor
                g = (yolo_index + 1) * 2 / 32  # gain
                # image_size = (320, 192)
                modules = nn.Upsample(size=tuple(int(x * g) for x in img_size))
            else:
                modules = nn.Upsample(scale_factor=module["stride"])

        # nn.Sequential() placeholder for "route" layer
        elif module["type"] == "route":
            layers = module['layers']
            filters = sum([output_filters[l + 1 if l > 0 else l] for l in layers])
            routs.extend([i + l if l < 0 else l for l in layers])

        # nn.Sequential() placeholder for "shortcut" layer
        elif module["type"] == "shortcut":
            layers = module['from']
            filters = output_filters[-1]
            routs.extend([i + l if l < 0 else l for l in layers])
            modules = WeightFeatureFusion(layers=layers,
                                          weight='weights_type' in module)

        elif module["type"] == "reorg3d":  # yolov3-spp-pan-scale
            pass

        elif module["type"] == "yolo":
            yolo_index += 1
            l = module["from"] if "from" in module else []
            modules = YOLOLayer(anchors=module["anchors"][module["mask"]],
                                # anchor list
                                nc=module["classes"],  # number of classes
                                img_size=img_size,  # (416, 416)
                                yolo_index=yolo_index,  # 0, 1, 2...
                                layers=l)  # output layers

            # Initialize preceding Conv2d() bias (https://arxiv.org/pdf/1708.02002.pdf section 3.3)
            try:
                bo = -4.5  #  obj bias
                # cls bias: class probability is sigmoid(p) = 1/nc
                bc = math.log(1 / (modules.nc - 0.99))

                j = l[yolo_index] if "from" in module else -1
                bias_ = module_list[j][0].bias  # shape(255,)
                # shape(3,85)
                bias = bias_[:modules.no * modules.na].view(modules.na, -1)
                bias[:, 4] += bo - bias[:, 4].mean()  # obj
                # cls, view with utils.print_model_biases(model)
                bias[:, 5:] += bc - bias[:, 5:].mean()
                module_list[j][0].bias = torch.nn.Parameter(
                    bias_, requires_grad=bias_.requires_grad)
            except:
                print("WARNING: smart bias initialization failure.")

        else:
            print(f"Warning: Unrecognized Layer Type: {module['type']}")

        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)

    routs_binary = [False] * (i + 1)
    for i in routs:
        routs_binary[i] = True
    return module_list, routs_binary


class WeightFeatureFusion(nn.Module):
    """Weighted sum of 2 or more layers https://arxiv.org/abs/1911.09070"""
    def __init__(self, layers, weight=False):
        super(WeightFeatureFusion, self).__init__()
        self.layers = layers  # layer indices
        self.weight = weight  # apply weights boolean
        self.n = len(layers) + 1  # number of layers
        if weight:
            # layer weights
            self.w = torch.nn.Parameter(torch.zeros(self.n), requires_grad=True)

    def forward(self, x, outputs):
        # Weights
        if self.weight:
            w = torch.sigmoid(self.w) * (2 / self.n)  # sigmoid weights (0-1)
            x = x * w[0]

        # Fusion
        nc = x.shape[1]  # input channels
        for i in range(self.n - 1):
            # feature to add
            a = outputs[self.layers[i]] * w[i + 1] if self.weight else outputs[
                self.layers[i]]
            ac = a.shape[1]  # feature channels
            dc = nc - ac  # delta channels

            # Adjust channels
            if dc > 0:  # slice input
                # or a = nn.ZeroPad2d((0, 0, 0, 0, 0, dc))(a); x = x + a
                x[:, :ac] = x[:, :ac] + a
            elif dc < 0:  # slice feature
                x = x + a[:, :nc]
            else:  # same shape
                x = x + a
        return x


class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        ctx.save_for_backward(i)
        return i * torch.sigmoid(i)

    @staticmethod
    def backward(ctx, grad_output):
        sigmoid_i = torch.sigmoid(ctx.saved_variables[0])
        return grad_output * (
                sigmoid_i * (1 + ctx.saved_variables[0] * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


class Swish(nn.Module):
    def forward(self, x):
        return x.mul_(torch.sigmoid(x))


# Source https://github.com/digantamisra98/Mish/blob/master/Mish/Torch/mish.py
class Mish(nn.Module):
    """
    Applies the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    Examples:
        >>> m = Mish()
        >>> inputs = torch.randn(2)
        >>> output = m(inputs)
    """

    def __init__(self):
        """
        Init method.
        """
        super().__init__()

    def forward(self, inputs):
        """
        Forward pass of the function.
        """
        return inputs * torch.tanh(F.softplus(inputs))


class YOLOLayer(nn.Module):
    def __init__(self, anchors, nc, img_size, yolo_index, layers):
        super(YOLOLayer, self).__init__()
        self.anchors = torch.Tensor(anchors)
        self.index = yolo_index  # index of this layer in layers
        self.layers = layers  # model output layer indices
        self.nl = len(layers)  # number of output layers (3)
        self.na = len(anchors)  # number of anchors (3)
        self.nc = nc  # number of classes (80)
        self.no = nc + 5  # number of outputs (85)
        self.nx = 0  # initialize number of x gridpoints
        self.ny = 0  # initialize number of y gridpoints

        if ONNX_EXPORT:
            stride = [32, 16, 8][yolo_index]  # stride of this layer
            nx = img_size[1] // stride  # number x grid points
            ny = img_size[0] // stride  # number y grid points
            create_grids(self, img_size, (nx, ny))

    def forward(self, p, img_size, out):
        ASFF = False  # https://arxiv.org/abs/1911.09516
        if ASFF:
            i, n = self.index, self.nl  # index in layers, number of layers
            p = out[self.layers[i]]
            bs, _, ny, nx = p.shape  # bs, 255, 13, 13
            if (self.nx, self.ny) != (nx, ny):
                create_grids(self, img_size, (nx, ny), p.device, p.dtype)

            # outputs and weights
            # w = F.softmax(p[:, -n:], 1)  # normalized weights
            w = torch.sigmoid(p[:, -n:]) * (2 / n)  # sigmoid weights (faster)
            # w = w / w.sum(1).unsqueeze(1)  # normalize across layer dimension

            # weighted ASFF sum
            p = out[self.layers[i]][:, :-n] * w[:, i:i + 1]
            for j in range(n):
                if j != i:
                    p += w[:, j:j + 1] * F.interpolate(
                        input=out[self.layers[j]][:, :-n],
                        size=[ny, nx],
                        mode="bilinear",
                        align_corners=False)

        elif ONNX_EXPORT:
            bs = 1  # batch size
        else:
            bs, _, ny, nx = p.shape  # bs, 255, 13, 13
            if (self.nx, self.ny) != (nx, ny):
                create_grids(self, img_size, (nx, ny), p.device, p.dtype)

        # p.view(bs, 255, 13, 13) -- > (bs, 3, 13, 13, 85)  # (bs, anchors, grid, grid, classes + xywh)
        p = p.view(bs, self.na, self.no, self.ny, self.nx).permute(
            0, 1, 3, 4, 2).contiguous()  # prediction

        if self.training:
            return p

        elif ONNX_EXPORT:
            # Avoid broadcasting for ANE operations
            m = self.na * self.nx * self.ny
            ng = 1 / self.ng.repeat((m, 1))
            grid_xy = self.grid_xy.repeat((1, self.na, 1, 1, 1)).view(m, 2)
            anchor_wh = self.anchor_wh.repeat(
                (1, 1, self.nx, self.ny, 1)).view(m, 2) * ng

            p = p.view(m, self.no)
            xy = torch.sigmoid(p[:, 0:2]) + grid_xy  # x, y
            wh = torch.exp(p[:, 2:4]) * anchor_wh  # width, height
            p_cls = torch.sigmoid(p[:, 4:5]) if self.nc == 1 else torch.sigmoid(
                p[:, 5:self.no]) * torch.sigmoid(p[:, 4:5])  # conf
            return p_cls, xy * ng, wh

        else:  # inference
            io = p.clone()  # inference output
            io[..., :2] = torch.sigmoid(io[..., :2]) + self.grid_xy  # xy
            # wh yolo method
            io[..., 2:4] = torch.exp(io[..., 2:4]) * self.anchor_wh
            io[..., :4] *= self.stride
            torch.sigmoid_(io[..., 4:])
            # view [1, 3, 13, 13, 85] as [1, 507, 85]
            return io.view(bs, -1, self.no), p


class Darknet(nn.Module):
    # YOLOv3 object detection model

    def __init__(self, cfg, image_size=(416, 416)):
        super(Darknet, self).__init__()

        self.module_defs = parse_model_config(cfg)
        self.module_list, self.routs = create_modules(self.module_defs,
                                                      image_size)
        self.yolo_layers = get_yolo_layers(self)

        # Darknet Header https://github.com/AlexeyAB/darknet/issues/2914#issuecomment-496675346
        self.version = np.array([0, 2, 5],
                                dtype=np.int32)  # (int32) version info: major, minor, revision
        self.seen = np.array([0],
                             dtype=np.int64)  # (int64) number of images seen during training
        self.info()  # print model description

    def forward(self, x, verbose=False):
        img_size = x.shape[-2:]
        yolo_out, out = [], []
        if verbose:
            str = ""
            print("0", x.shape)

        for i, (mdef, module) in enumerate(
                zip(self.module_defs, self.module_list)):
            mtype = mdef["type"]
            if mtype in ["convolutional", "upsample", "maxpool"]:
                x = module(x)
            elif mtype == "shortcut":  # sum
                if verbose:
                    l = [i - 1] + module.layers  # layers
                    s = [list(x.shape)] + [list(out[i].shape) for i in
                                           module.layers]  # shapes
                    str = " >> " + " + ".join(
                        ["layer %g %s" % x for x in zip(l, s)])
                x = module(x, out)  # weightedFeatureFusion()
            elif mtype == "route":  # concat
                layers = mdef["layers"]
                if verbose:
                    l = [i - 1] + layers  # layers
                    s = [list(x.shape)] + [list(out[i].shape) for i in
                                           layers]  # shapes
                    str = " >> " + " + ".join(
                        ["layer %g %s" % x for x in zip(l, s)])
                if len(layers) == 1:
                    x = out[layers[0]]
                else:
                    try:
                        x = torch.cat([out[i] for i in layers], 1)
                    except:  # apply stride 2 for darknet reorg layer
                        out[layers[1]] = F.interpolate(
                            out[layers[1]], scale_factor=[0.5, 0.5])
                        x = torch.cat([out[i] for i in layers], 1)
                    # print(""), [print(out[i].shape) for i in layers], print(x.shape)
            elif mtype == "yolo":
                yolo_out.append(module(x, img_size, out))
            out.append(x if self.routs[i] else [])
            if verbose:
                print(f"%g/%g %s -" % (i, len(self.module_list), mtype),
                      list(x.shape), str)
                str = ""

        if self.training:  # train
            return yolo_out
        elif ONNX_EXPORT:  # export
            x = [torch.cat(x, 0) for x in zip(*yolo_out)]
            return x[0], torch.cat(x[1:3], 1)  # scores, boxes: 3780x80, 3780x4
        else:  # test
            io, p = zip(*yolo_out)  # inference output, training output
            return torch.cat(io, 1), p

    def fuse(self):
        # Fuse Conv2d + BatchNorm2d layers throughout model
        print("Fusing Conv2d() and BatchNorm2d() layers...")
        fused_list = nn.ModuleList()
        for a in list(self.children())[0]:
            if isinstance(a, nn.Sequential):
                for i, b in enumerate(a):
                    if isinstance(b, nn.modules.batchnorm.BatchNorm2d):
                        # fuse this bn layer with the previous conv2d layer
                        conv = a[i - 1]
                        fused = fuse_conv_and_bn(conv, b)
                        a = nn.Sequential(fused, *list(a.children())[i + 1:])
                        break
            fused_list.append(a)
        self.module_list = fused_list
        self.info()  # yolov3-spp reduced from 225 to 152 layers

    def info(self, verbose=False):
        model_info(self, verbose)


def get_yolo_layers(model):
    return [i for i, x in enumerate(model.module_defs) if
            x["type"] == "yolo"]  # [82, 94, 106] for yolov3


def create_grids(self, image_size=416, ng=(13, 13), device="cpu",
                 type=torch.float32):
    nx, ny = ng  # x and y grid size
    self.img_size = max(image_size)
    self.stride = self.img_size / max(ng)

    # build xy offsets
    yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
    self.grid_xy = torch.stack((xv, yv), 2).to(device).type(type).view(
        (1, 1, ny, nx, 2))

    # build wh gains
    self.anchor_vec = self.anchors.to(device) / self.stride
    self.anchor_wh = self.anchor_vec.view(1, self.na, 1, 1, 2).type(type)
    self.ng = torch.Tensor(ng).to(device)
    self.nx = nx
    self.ny = ny


def load_darknet_weights(self, weights, cutoff=-1):
    # Parses and loads the weights stored in "weights"

    # Establish cutoffs (load layers between 0 and cutoff. if cutoff = -1 all are loaded)
    file = Path(weights).name
    if file == "darknet53.conv.74":
        cutoff = 75
    elif file == "yolov3-tiny.conv.15":
        cutoff = 15

    # Read weights file
    with open(weights, "rb") as f:
        # Read Header https://github.com/AlexeyAB/darknet/issues/2914#issuecomment-496675346
        self.version = np.fromfile(f, dtype=np.int32,
                                   count=3)  # (int32) version info: major, minor, revision
        self.seen = np.fromfile(f, dtype=np.int64,
                                count=1)  # (int64) number of images seen during training

        weights = np.fromfile(f, dtype=np.float32)  # the rest are weights

    ptr = 0
    for i, (mdef, module) in enumerate(
            zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
        if mdef["type"] == "convolutional":
            conv = module[0]
            if mdef["batch_normalize"]:
                # Load BN bias, weights, running mean and running variance
                bn = module[1]
                nb = bn.bias.numel()  # number of biases
                # Bias
                bn.bias.data.copy_(
                    torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.bias))
                ptr += nb
                # Weight
                bn.weight.data.copy_(
                    torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.weight))
                ptr += nb
                # Running Mean
                bn.running_mean.data.copy_(
                    torch.from_numpy(weights[ptr:ptr + nb]).view_as(
                        bn.running_mean))
                ptr += nb
                # Running Var
                bn.running_var.data.copy_(
                    torch.from_numpy(weights[ptr:ptr + nb]).view_as(
                        bn.running_var))
                ptr += nb
            else:
                # Load conv. bias
                nb = conv.bias.numel()
                conv_b = torch.from_numpy(weights[ptr:ptr + nb]).view_as(
                    conv.bias)
                conv.bias.data.copy_(conv_b)
                ptr += nb
            # Load conv. weights
            nw = conv.weight.numel()  # number of weights
            conv.weight.data.copy_(
                torch.from_numpy(weights[ptr:ptr + nw]).view_as(conv.weight))
            ptr += nw


def save_weights(self, path="model.weights", cutoff=-1):
    # Converts a PyTorch model to Darket format (*.pt to *.weights)
    # Note: Does not work if model.fuse() is applied
    with open(path, "wb") as f:
        # Write Header https://github.com/AlexeyAB/darknet/issues/2914#issuecomment-496675346
        self.version.tofile(f)  # (int32) version info: major, minor, revision
        self.seen.tofile(f)  # (int64) number of images seen during training

        # Iterate through layers
        for i, (mdef, module) in enumerate(
                zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
            if mdef["type"] == "convolutional":
                conv_layer = module[0]
                # If batch norm, load bn first
                if mdef["batch_normalize"]:
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(f)
                    bn_layer.weight.data.cpu().numpy().tofile(f)
                    bn_layer.running_mean.data.cpu().numpy().tofile(f)
                    bn_layer.running_var.data.cpu().numpy().tofile(f)
                # Load conv bias
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(f)
                # Load conv weights
                conv_layer.weight.data.cpu().numpy().tofile(f)


def convert(cfg="cfg/yolov3-spp.cfg", weights="weights/yolov3-spp.weights"):
    # Converts between PyTorch and Darknet format per extension (i.e. *.weights convert to *.pt and vice versa)
    # from models import *; convert("cfg/yolov3-spp.cfg", "weights/yolov3-spp.weights")

    # Initialize model
    model = Darknet(cfg)

    # Load weights and save
    if weights.endswith(".pth"):
        model.load_state_dict(torch.load(weights, map_location="cpu")["model"])
        save_weights(model, path="converted.weights", cutoff=-1)
        print(f"Success: converted `{weights}` to `converted.weights`")

    elif weights.endswith(".weights"):
        load_darknet_weights(model, weights)

        state = {"epoch": -1,
                 "best_fitness": None,
                 "training_results": None,
                 "model": model.state_dict(),
                 "optimizer": None}

        torch.save(state, "converted.pth")
        print(f"Success: converted `{weights}` to `converted.pth`")

    else:
        print("Error: extension not supported.")
