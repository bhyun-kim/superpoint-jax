import numpy as np
import torch

import jax.numpy as jnp
from flax import nnx

from superpoint_jax.model import VGGBlockNNX, VGGBlockTorch


def set_conv_params(nnx_conv: nnx.Conv, torch_conv: torch.nn.Conv2d):
    """Copy PyTorch Conv2d weights to Flax NNX Conv.

    This function copies the weights and bias (if present) from a PyTorch Conv2d
    layer to a Flax NNX Conv layer. The PyTorch weight shape is (out_ch, in_ch, kH, kW)
    and is transposed to (kH, kW, in_ch, out_ch) for Flax.

    Args:
        nnx_conv (nnx.Conv): Flax convolution layer.
        torch_conv (torch.nn.Conv2d): PyTorch convolution layer.
    """
    w_torch = torch_conv.weight.data.cpu().numpy()
    w_flax = np.transpose(w_torch, (2, 3, 1, 0))
    nnx_conv.kernel = nnx.Param(jnp.array(w_flax))
    if torch_conv.bias is not None:
        b_torch = torch_conv.bias.data.cpu().numpy()
        nnx_conv.bias = nnx.Param(jnp.array(b_torch))


def set_bn_params(nnx_bn: nnx.BatchNorm, torch_bn: torch.nn.BatchNorm2d):
    """Copy PyTorch BatchNorm2d parameters to Flax NNX BatchNorm.

    This function copies the weight, bias, running mean, and running variance
    from a PyTorch BatchNorm2d layer to a Flax NNX BatchNorm layer.

    Args:
        nnx_bn (nnx.BatchNorm): Flax batch normalization layer.
        torch_bn (torch.nn.BatchNorm2d): PyTorch batch normalization layer.
    """
    scale = torch_bn.weight.data.cpu().numpy()
    bias = torch_bn.bias.data.cpu().numpy()
    mean = torch_bn.running_mean.data.cpu().numpy()
    var = torch_bn.running_var.data.cpu().numpy()
    
    nnx_bn.scale = nnx.Param(jnp.array(scale))
    nnx_bn.bias = nnx.Param(jnp.array(bias))
    nnx_bn.mean = nnx.BatchStat(jnp.array(mean))
    nnx_bn.var = nnx.BatchStat(jnp.array(var))


def get_vgg_block_flax(nnx_vgg):
    """Return the convolution and batch normalization layers from a Flax NNX VGGBlock.

    Args:
        nnx_vgg: A Flax NNX VGGBlock.

    Returns:
        tuple: (nnx.Conv, nnx.BatchNorm)
    """
    return nnx_vgg.conv, nnx_vgg.bn


def get_vgg_block_torch(pytorch_vgg: VGGBlockTorch):
    """Return the convolution and batch normalization layers from a PyTorch VGGBlock.

    Args:
        pytorch_vgg (VGGBlockTorch): A PyTorch VGGBlock.

    Returns:
        tuple: (conv_layer, bn_layer)
    """
    conv = pytorch_vgg.conv
    bn = pytorch_vgg.bn
    return conv, bn


def convert_superpoint_weights(pytorch_model, flax_model):
    """Convert and copy weights from a PyTorch SuperPoint model to a Flax SuperPoint model.

    Both models must be constructed with the same architecture.

    Args:
        pytorch_model: The PyTorch SuperPoint model.
        flax_model: The Flax SuperPoint model.

    Returns:
        The Flax SuperPoint model with weights copied from the PyTorch model.
    """
    backbone_blocks_torch = []
    for i, stage_seq in enumerate(pytorch_model.backbone):
        stage_blocks = []
        for layer in stage_seq:
            if isinstance(layer, VGGBlockTorch):
                stage_blocks.append(layer)
        backbone_blocks_torch.append(stage_blocks)

    backbone_blocks_flax = []
    for i, stage_seq in enumerate(flax_model.backbone.layers):
        stage_blocks = []
        for submod in stage_seq.layers:
            if isinstance(submod, nnx.Sequential):
                for subsubmod in submod.layers:
                    if isinstance(subsubmod, VGGBlockNNX):
                        stage_blocks.append(subsubmod)
        backbone_blocks_flax.append(stage_blocks)

    det_blocks_torch = []
    for layer in pytorch_model.detector:
        if isinstance(layer, VGGBlockTorch):
            det_blocks_torch.append(layer)

    det_blocks_flax = []
    for submod in flax_model.detector.layers:
        if isinstance(submod, VGGBlockNNX):
            det_blocks_flax.append(submod)

    desc_blocks_torch = []
    for layer in pytorch_model.descriptor:
        if isinstance(layer, VGGBlockTorch):
            desc_blocks_torch.append(layer)

    desc_blocks_flax = []
    for submod in flax_model.descriptor.layers:
        if isinstance(submod, VGGBlockNNX):
            desc_blocks_flax.append(submod)

    def load_vgg_block(pytorch_block, flax_block):
        conv_torch, bn_torch = get_vgg_block_torch(pytorch_block)
        conv_flax, bn_flax = get_vgg_block_flax(flax_block)
        set_conv_params(conv_flax, conv_torch)
        set_bn_params(bn_flax, bn_torch)

    for stage_pt, stage_fx in zip(backbone_blocks_torch, backbone_blocks_flax):
        for block_pt, block_fx in zip(stage_pt, stage_fx):
            load_vgg_block(block_pt, block_fx)

    for pt_block, fx_block in zip(det_blocks_torch, det_blocks_flax):
        load_vgg_block(pt_block, fx_block)

    for pt_block, fx_block in zip(desc_blocks_torch, desc_blocks_flax):
        load_vgg_block(pt_block, fx_block)

    return flax_model
