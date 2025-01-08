from flax import nnx 

import jax
import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates
from jax import random
from typing import Optional, List, Dict, Any
from types import SimpleNamespace

class VGGBlock(nnx.Module):
    c_in: int
    c_out: int
    kernel_size: int
    relu: bool = True
    conv: nnx.Conv
    activation: nnx.Module
    bn: nnx.BatchNorm

    def __init__(self, c_in: int, c_out: int, kernel_size: int, relu: bool = True):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nnx.Conv(
            in_channels=c_in,
            out_channels=c_out,
            kernel_size=(kernel_size, kernel_size),
            strides=(1, 1),
            padding=((padding, padding), (padding, padding)),
            use_bias=False,
        )
        self.activation = nnx.ReLU() if relu else nnx.Identity()
        self.bn = nnx.BatchNorm(
            num_features=c_out,
            eps=0.001,
            momentum=0.9,
            use_running_average=False,
        )

    def __call__(self, x, *, training: bool):
        x = self.conv(x)
        x = self.activation(x)
        x = self.bn(x, use_running_average=not training)
        return x
    

class SuperPoint(nnx.Module):
    default_conf = {
        "nms_radius": 4,
        "max_num_keypoints": None,
        "detection_threshold": 0.005,
        "remove_borders": 4,
        "descriptor_dim": 256,
        "channels": [64, 64, 128, 128, 256],
    }

    def __init__(self, **conf):
        super().__init__()
        conf = {**self.default_conf, **conf}
        self.conf = SimpleNamespace(**conf)
        self.stride = 2 ** (len(self.conf.channels) - 2)
        channels = [1, *self.conf.channels[:-1]]

        # Backbone
        self.backbone = []
        for i, c in enumerate(channels[1:], 1):
            layers = [
                VGGBlock(channels[i - 1], c, 3, rngs=nnx.Rngs(params=0)),
                VGGBlock(c, c, 3, rngs=nnx.Rngs(params=0))
            ]
            if i < len(channels) - 1:
                layers.append(nnx.MaxPool2D(kernel_size=(2, 2), strides=(2, 2)))
            self.backbone.append(nnx.Sequential(layers))

        # Detector Head
        c = self.conf.channels[-1]
        self.detector = nnx.Sequential([
            VGGBlock(channels[-1], c, 3, rngs=nnx.Rngs(params=0)),
            VGGBlock(c, self.stride**2 + 1, 1, relu=False, rngs=nnx.Rngs(params=0))
        ])

        # Descriptor Head
        self.descriptor = nnx.Sequential([
            VGGBlock(channels[-1], c, 3, rngs=nnx.Rngs(params=0)),
            VGGBlock(c, self.conf.descriptor_dim, 1, relu=False, rngs=nnx.Rngs(params=0))
        ])

    def __call__(self, data: Dict[str, jax.Array], *, training: bool) -> Dict[str, Any]:
        image = data["image"]
        if image.shape[1] == 3:  # RGB to gray
            scale = jnp.array([0.299, 0.587, 0.114]).reshape(1, 3, 1, 1)
            image = jnp.sum(image * scale, axis=1, keepdims=True)

        # Backbone
        features = image
        for layer in self.backbone:
            features = layer(features, training=training)

        # Descriptor
        descriptors_dense = self.descriptor(features, training=training)
        descriptors_dense = jax.nn.l2_normalize(descriptors_dense, axis=1)

        # Detector
        scores = self.detector(features, training=training)
        scores = jax.nn.softmax(scores, axis=1)[:, :-1]
        b, _, h, w = scores.shape
        scores = scores.transpose(0, 2, 3, 1).reshape(b, h, w, self.stride, self.stride)
        scores = scores.transpose(0, 1, 3, 2, 4).reshape(b, h * self.stride, w * self.stride)
        scores = self.batched_nms(scores, self.conf.nms_radius)

        # Discard keypoints near the image borders
        if self.conf.remove_borders:
            pad = self.conf.remove_borders
            scores = scores.at[:, :pad].set(-1)
            scores = scores.at[:, :, :pad].set(-1)
            scores = scores.at[:, -pad:].set(-1)
            scores = scores.at[:, :, -pad:].set(-1)

        # Extract keypoints
        keypoints, keypoint_scores, descriptors = self.extract_keypoints_and_descriptors(
            scores, descriptors_dense
        )

        return {
            "keypoints": keypoints,
            "keypoint_scores": keypoint_scores,
            "descriptors": descriptors,
        }
    

def sample_descriptors(keypoints, descriptors, s: int = 8):
    """Interpolate descriptors at keypoint locations."""
    b, c, h, w = descriptors.shape
    # Normalize keypoints to the range [-1, 1]
    keypoints = (keypoints + 0.5) / (jnp.array([w, h]) * s)
    keypoints = keypoints * 2 - 1

    # Convert normalized coordinates to image coordinates
    keypoints = (keypoints + 1) * jnp.array([w - 1, h - 1]) / 2

    def interpolate_single_image(descriptors, keypoints):
        # Separate x and y coordinates
        x = keypoints[:, 0]
        y = keypoints[:, 1]
        # Interpolate descriptors at keypoint locations
        interpolated = map_coordinates(descriptors, [y, x], order=1, mode='nearest')
        return interpolated

    # Apply interpolation for each image in the batch
    descriptors = jax.vmap(interpolate_single_image, in_axes=(0, 0))(descriptors, keypoints)
    # Normalize descriptors
    descriptors = jax.nn.l2_normalize(descriptors, axis=1)
    return descriptors

def batched_nms(scores, nms_radius: int):
    assert nms_radius >= 0

    def max_pool(x):
        return jax.lax.reduce_window(x, -jnp.inf, jax.lax.max, (nms_radius * 2 + 1, nms_radius * 2 + 1), (1, 1), 'SAME')

    zeros = jnp.zeros_like(scores)
    max_mask = scores == max_pool(scores)

    def suppress(scores, max_mask):
        supp_mask = max_pool(max_mask.astype(jnp.float32)) > 0
        supp_scores = jnp.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        return jnp.where(max_mask | (new_max_mask & (~supp_mask)), scores, zeros)

    for _ in range(2):
        scores = suppress(scores, max_mask)
    return scores


def select_top_k_keypoints(keypoints, scores, k):
    if k >= keypoints.shape[0]:
        return keypoints, scores
    top_scores, indices = jax.lax.top_k(scores, k)
    top_keypoints = keypoints[indices]
    return top_keypoints, top_scores