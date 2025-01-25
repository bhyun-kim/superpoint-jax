import numpy as np
import torch

import jax
import jax.numpy as jnp
from flax import nnx

from superpoint_jax.model import VGGBlockNNX, VGGBlockTorch


def set_conv_params(nnx_conv: nnx.Conv, torch_conv: torch.nn.Conv2d):
    """
    Copy PyTorch Conv2d weights (out_ch, in_ch, kH, kW) to Flax NNX Conv (kH, kW, in_ch, out_ch).
    Also copies the bias if present.
    """
    w_torch = torch_conv.weight.data.cpu().numpy()  # (out_ch, in_ch, kH, kW)
    # Transpose => (kH, kW, in_ch, out_ch)
    w_flax = np.transpose(w_torch, (2, 3, 1, 0))
    nnx_conv.kernel = nnx.Param(jnp.array(w_flax))

    if torch_conv.bias is not None:
        b_torch = torch_conv.bias.data.cpu().numpy()  # (out_ch,)
        nnx_conv.bias = nnx.Param(jnp.array(b_torch))


def set_bn_params(nnx_bn: nnx.BatchNorm, torch_bn: torch.nn.BatchNorm2d):
    """
    Copy PyTorch BatchNorm2d parameters to Flax NNX BatchNorm:
      - weight => scale
      - bias => bias
      - running_mean => mean
      - running_var => var
    """
    scale = torch_bn.weight.data.cpu().numpy()       # (num_features,)
    bias = torch_bn.bias.data.cpu().numpy()         # (num_features,)
    mean = torch_bn.running_mean.data.cpu().numpy() # (num_features,)
    var  = torch_bn.running_var.data.cpu().numpy()  # (num_features,)

    # print('setting values')

    nnx_bn.scale = nnx.Param(jnp.array(scale))
    nnx_bn.bias =  nnx.Param(jnp.array(bias))
    nnx_bn.mean =  nnx.BatchStat(jnp.array(mean))
    nnx_bn.var =   nnx.BatchStat(jnp.array(var))


def get_vgg_block_flax(nnx_vgg):
    """
    Returns (nnx.Conv, nnx.BatchNorm) from a single Flax NNX VGGBlock.
    """
    return nnx_vgg.conv, nnx_vgg.bn

def get_vgg_block_torch(pytorch_vgg: VGGBlockTorch):
    """
    Returns (conv_layer, bn_layer) from a single PyTorch VGGBlock.
    'activation' is omitted since it has no trainable params.
    """
    conv = pytorch_vgg.conv
    bn = pytorch_vgg.bn
    return conv, bn


def convert_superpoint_weights(pytorch_model, flax_model):
    """
    Copies all PyTorch parameters from 'pytorch_model' (SuperPoint)
    into 'flax_model' (SuperPointNNX) in-place.

    NOTE: Both models must already be constructed with the same architecture
    (same # of channels, same # of blocks, etc.).
    """

    # 1) GATHER BACKBONE SUBMODULES
    # PyTorch: model.backbone is a nn.Sequential of "stages".
    # Each stage => nn.Sequential([VGGBlock, VGGBlock, optional MaxPool])
    # We want just the 2 VGGBlocks from each stage.
    # Let's collect them in a list like: backbone_blocks_torch[i][j],
    # where i is stage index, j is block index in stage.
    backbone_blocks_torch = []
    for i, stage_seq in enumerate(pytorch_model.backbone):
        # stage_seq is a nn.Sequential e.g. [VGGBlock, VGGBlock, MaxPool?]
        stage_blocks = []
        for layer in stage_seq:
            if isinstance(layer, VGGBlockTorch):
                stage_blocks.append(layer)
        backbone_blocks_torch.append(stage_blocks)

    # Flax: model.backbone is nnx.Sequential([...]) each item is
    # => nnx.Sequential(vgg1, vgg2, optional pool)
    # We'll do similarly:
    backbone_blocks_flax = []
    for i, stage_seq in enumerate(flax_model.backbone.layers):
        stage_blocks = []
        for submod in stage_seq.layers:
            # submod could be a VGGBlock or a function (pool).
            if isinstance(submod, nnx.Sequential):
                for subsubmod in submod.layers:
                    if isinstance(subsubmod, VGGBlockNNX):
                        stage_blocks.append(subsubmod)
        backbone_blocks_flax.append(stage_blocks)

    # 2) DETECTOR blocks
    # PyTorch: model.detector => nn.Sequential([block1, block2])
    # Flax:    model.detector => nnx.Sequential([block1, block2])
    det_blocks_torch = []
    for layer in pytorch_model.detector:
        if isinstance(layer, VGGBlockTorch):
            det_blocks_torch.append(layer)

    det_blocks_flax = []
    for submod in flax_model.detector.layers:
        if isinstance(submod, VGGBlockNNX):
            det_blocks_flax.append(submod)

    # 3) DESCRIPTOR blocks
    desc_blocks_torch = []
    for layer in pytorch_model.descriptor:
        if isinstance(layer, VGGBlockTorch):
            desc_blocks_torch.append(layer)

    desc_blocks_flax = []
    for submod in flax_model.descriptor.layers:
        if isinstance(submod, VGGBlockNNX):
            desc_blocks_flax.append(submod)

    # Now we have:
    #   backbone_blocks_torch[i][j], backbone_blocks_flax[i][j]
    #   det_blocks_torch[j], det_blocks_flax[j]
    #   desc_blocks_torch[j], desc_blocks_flax[j]
    #
    # For each pair, we set the conv/bn parameters.

    # Helper to load a single block
    def load_vgg_block(pytorch_block, flax_block):
        # PyTorch block => (conv, activation, bn)
        # Flax block => (conv, bn, relu)
        conv_torch, bn_torch = get_vgg_block_torch(pytorch_block)
        conv_flax, bn_flax   = get_vgg_block_flax(flax_block)

        # set conv
        set_conv_params(conv_flax, conv_torch)
        # set bn
        set_bn_params(bn_flax, bn_torch)

    # (A) BACKBONE
    for i, (stage_pt, stage_fx) in enumerate(zip(backbone_blocks_torch, backbone_blocks_flax)):

        for j, (block_pt, block_fx) in enumerate(zip(stage_pt, stage_fx)):
            load_vgg_block(block_pt, block_fx)

    # (B) DETECTOR
    for (pt_block, fx_block) in zip(det_blocks_torch, det_blocks_flax):
        load_vgg_block(pt_block, fx_block)

    # (C) DESCRIPTOR
    for (pt_block, fx_block) in zip(desc_blocks_torch, desc_blocks_flax):
        load_vgg_block(pt_block, fx_block)
    return flax_model
