"""FLAX.NNX implementation of the SuperPoint model,
   derived from the PyTorch re-implementation.

   https://github.com/rpautrat/SuperPoint/blob/master/superpoint_pytorch.py
"""
import jax
import jax.numpy as jnp
from flax.linen.pooling import max_pool
from functools import partial
from flax import nnx
from time import time

@partial(jax.jit, static_argnames=['max_k', 'stride'])
def _extract_keypoints_padded(scores_nms: jnp.ndarray,
                              desc_dense: jnp.ndarray,
                              detection_threshold: float,
                              max_k: int,
                              stride: int):
    """Extracts padded keypoints, scores, descriptors, valid mask, and count.

    Returns:
        tuple: A tuple containing:
            - kpts_yx_b (jnp.ndarray): Padded keypoint coordinates, shape (B, max_k, 2).
            - scores_b (jnp.ndarray): Padded keypoint scores, shape (B, max_k).
            - desc_b (jnp.ndarray): Padded descriptors, shape (B, max_k, C').
            - mask_b (jnp.ndarray): Boolean mask indicating valid keypoints, shape (B, max_k).
            - count_b (jnp.ndarray): Number of valid keypoints per image, shape (B,).
    """
    B, HH, WW = scores_nms.shape

    def process_single(scores_map, desc_map):
        """Processes a single image's score map and dense descriptors.

        Returns:
            tuple: A tuple containing:
                - top_yx (jnp.ndarray): Keypoint coordinates (y, x), shape (max_k, 2).
                - top_scores (jnp.ndarray): Keypoint scores, shape (max_k,).
                - desc_sampled (jnp.ndarray): Sampled descriptors, shape (max_k, C').
                - valid_mask (jnp.ndarray): Boolean mask of valid keypoints, shape (max_k,).
                - valid_count (int): Number of valid keypoints.
        """
        scores_flat = scores_map.ravel()
        masked_scores = jnp.where(scores_flat > detection_threshold, scores_flat, -jnp.inf)
        top_scores, top_indices = jax.lax.top_k(masked_scores, max_k)
        
        top_y = top_indices // WW
        top_x = top_indices % WW
        top_yx = jnp.stack([top_y, top_x], axis=-1)
        xy = top_yx[:, ::-1]

        coords_batched = xy[None]
        desc_map_batched = desc_map[None]
        desc_sampled = sample_descriptors_jax(coords_batched, desc_map_batched, s=stride)[0].T
        
        valid_mask = top_scores > -jnp.inf
        valid_count = jnp.sum(valid_mask)

        return top_yx, top_scores, desc_sampled, valid_mask, valid_count

    kpts_yx_b, scores_b, desc_b, mask_b, count_b = jax.vmap(process_single)(scores_nms, desc_dense)

    return kpts_yx_b, scores_b, desc_b, mask_b, count_b

def extract_keypoints_and_descriptors(scores_nms: jnp.ndarray,
                                      desc_dense: jnp.ndarray,
                                      detection_threshold: float,
                                      max_k: int,
                                      stride: int):
    """Extracts keypoints, scores, and descriptors from NMS scores and dense descriptors.

    Returns:
        dict: A dictionary with keys:
            - "keypoints": (jnp.ndarray) shape (B, max_k, 2)
            - "scores": (jnp.ndarray) shape (B, max_k)
            - "descriptors": (jnp.ndarray) shape (B, max_k, C')
            - "valid_counts": (jnp.ndarray) shape (B,)
    """
    kpts_yx_b, scores_b, desc_b, valid_mask_b, valid_counts_b = _extract_keypoints_padded(
        scores_nms, desc_dense, detection_threshold, max_k, stride)
    
    kpts_yx_b_masked = jnp.where(valid_mask_b[..., None], kpts_yx_b, 0.0)
    scores_b_masked  = jnp.where(valid_mask_b, scores_b, 0.0)
    desc_b_masked    = jnp.where(valid_mask_b[..., None], desc_b, 0.0)

    return {
        "keypoints": kpts_yx_b_masked,
        "scores": scores_b_masked,
        "descriptors": desc_b_masked,
        "valid_counts": valid_counts_b,
    }

def bilinear_grid_sample_jax(images: jnp.ndarray, coords: jnp.ndarray) -> jnp.ndarray:
    """Bilinearly samples from images using normalized coordinates.

    Args:
        images (jnp.ndarray): Input images of shape (B, C, H, W).
        coords (jnp.ndarray): Coordinates of shape (B, N, 2) in [-1, 1].

    Returns:
        jnp.ndarray: Sampled values of shape (B, C, N).
    """
    _, _, H, W = images.shape

    x = (coords[..., 0] + 1.0) * 0.5 * (W - 1)
    y = (coords[..., 1] + 1.0) * 0.5 * (H - 1)

    x0 = jnp.floor(x).astype(int)
    x1 = x0 + 1
    y0 = jnp.floor(y).astype(int)
    y1 = y0 + 1

    x0c = jnp.clip(x0, 0, W - 1)
    x1c = jnp.clip(x1, 0, W - 1)
    y0c = jnp.clip(y0, 0, H - 1)
    y1c = jnp.clip(y1, 0, H - 1)

    wx = x - x0
    wy = y - y0

    w_tl = (1 - wx) * (1 - wy)
    w_tr = wx * (1 - wy)
    w_bl = (1 - wx) * wy
    w_br = wx * wy

    def sample_single_image(images_b, x0c_b, x1c_b, y0c_b, y1c_b,
                            w_tl_b, w_tr_b, w_bl_b, w_br_b):
        """Samples a single image with the provided indices and weights.

        Returns:
            jnp.ndarray: Sampled values of shape (C, N).
        """
        top_left = images_b[:, y0c_b, x0c_b]
        top_right = images_b[:, y0c_b, x1c_b]
        bottom_left = images_b[:, y1c_b, x0c_b]
        bottom_right = images_b[:, y1c_b, x1c_b]

        return top_left * w_tl_b + top_right * w_tr_b + bottom_left * w_bl_b + bottom_right * w_br_b

    out = jax.vmap(sample_single_image, in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0))(
        images, x0c, x1c, y0c, y1c, w_tl, w_tr, w_bl, w_br)
    return out

@partial(jax.jit, static_argnames=["s"])
def sample_descriptors_jax(keypoints: jnp.ndarray, descriptors: jnp.ndarray, s: int = 8) -> jnp.ndarray:
    """Samples and L2-normalizes descriptors at keypoint locations.

    Args:
        keypoints (jnp.ndarray): Pixel coordinates, shape (B, N, 2).
        descriptors (jnp.ndarray): Dense descriptor map, shape (B, C, H, W).
        s (int): Scaling factor. Default: 8.

    Returns:
        jnp.ndarray: Normalized descriptors of shape (B, C, N).
    """
    _, _, H, W = descriptors.shape

    wh = jnp.array([W, H], dtype=keypoints.dtype)
    
    coords = (keypoints + 0.5) / (wh * s)
    coords = coords * 2.0 - 1.0

    sampled = bilinear_grid_sample_jax(descriptors, coords)

    eps = 1e-8
    norm = jnp.sqrt(jnp.sum(sampled ** 2, axis=1, keepdims=True) + eps)

    return sampled / norm

def max_pool_2d(x: jnp.ndarray, nms_radius: int) -> jnp.ndarray:
    """Applies max pooling on a (B, H, W) input by adding a channel dimension.

    Args:
        x (jnp.ndarray): Input array of shape (B, H, W).
        nms_radius (int): Pooling radius.

    Returns:
        jnp.ndarray: Pooled output of shape (B, H, W).
    """
    x = x[..., None]

    window_shape = (2 * nms_radius + 1, 2 * nms_radius + 1)
    pooled = max_pool(x, window_shape=window_shape, strides=(1, 1),
                      padding=((nms_radius, nms_radius), (nms_radius, nms_radius)))
    
    return pooled[..., 0]

@partial(jax.jit, static_argnums=(1,))
def batched_nms_jax(scores: jnp.ndarray, nms_radius: int) -> jnp.ndarray:
    """Performs batched non-maximum suppression on score maps.

    Args:
        scores (jnp.ndarray): Score map of shape (B, H, W).
        nms_radius (int): NMS radius.

    Returns:
        jnp.ndarray: Suppressed score map of shape (B, H, W).
    """
    assert nms_radius >= 0

    zeros = jnp.zeros_like(scores)
    max_mask = scores == max_pool_2d(scores, nms_radius)

    for _ in range(2):
        supp_mask = max_pool_2d(max_mask.astype(scores.dtype), nms_radius) > 0
        supp_scores = jnp.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool_2d(supp_scores, nms_radius)
        max_mask = max_mask | (new_max_mask & jnp.logical_not(supp_mask))

    return jnp.where(max_mask, scores, zeros)

@partial(jax.jit, static_argnums=(2,))
def select_top_k_keypoints_jax(keypoints: jnp.ndarray, scores: jnp.ndarray, k: int):
    """Selects the top-k keypoints based on scores.

    Args:
        keypoints (jnp.ndarray): Array of keypoints.
        scores (jnp.ndarray): Array of scores.
        k (int): Number of keypoints to select.

    Returns:
        tuple: A tuple containing:
            - top_kpts (jnp.ndarray): Selected keypoints, shape (k, 2).
            - top_scores (jnp.ndarray): Selected scores, shape (k,).
    """
    top_scores, top_indices = jax.lax.top_k(scores, k)
    top_kpts = jnp.take(keypoints, top_indices, axis=0)

    return top_kpts, top_scores

class VGGBlockNNX(nnx.Module):
    """VGGBlockNNX module: Conv -> ReLU -> (optional) BN."""
    def __init__(self, in_features: int, out_features: int, kernel_size: int = 3, relu: bool = True, rngs: nnx.Rngs = nnx.Rngs(0)):
        padding = (kernel_size - 1) // 2
        self.conv = nnx.Conv(
            in_features=in_features,
            out_features=out_features,
            kernel_size=(kernel_size, kernel_size),
            strides=1,
            padding=(padding, padding),
            use_bias=True,
            rngs=rngs
        )
        self.bn = nnx.BatchNorm(
            num_features=out_features,
            epsilon=1e-3,
            rngs=rngs,
        )
        self.relu = relu

    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        if training:
            self.bn.train()
        else:
            self.bn.eval()
        x = self.conv(x)
        if self.relu:
            x = nnx.relu(x)
        return self.bn(x)

class SuperPointJAX(nnx.Module):
    """
    SuperPointJAX model implemented in Flax NNX.

    Args:
        nms_radius (int): NMS radius.
        max_num_keypoints (int): Maximum number of keypoints.
        detection_threshold (float): Detection threshold.
        remove_borders (int): Border removal margin.
        descriptor_dim (int): Descriptor dimension.
        channels (list): List of channel sizes.
        rngs (nnx.Rngs): Random number generator.
    """
    def __init__(self, nms_radius: int = 4, max_num_keypoints: int = None, detection_threshold: float = 0.005,
                 remove_borders: int = 4, descriptor_dim: int = 256, channels: list = [64, 64, 128, 128, 256],
                 rngs: nnx.Rngs = nnx.Rngs(0)):
        self.nms_radius = nms_radius
        self.max_num_keypoints = max_num_keypoints
        self.detection_threshold = detection_threshold
        self.remove_borders = remove_borders
        self.descriptor_dim = descriptor_dim
        self.channels = channels
        self.stride = 2 ** (len(self.channels) - 2)

        backbone_layers = []
        in_ch = 1
        for i, out_ch in enumerate(self.channels[:-1]):
            vgg1 = VGGBlockNNX(in_features=in_ch, out_features=out_ch, rngs=rngs)
            vgg2 = VGGBlockNNX(in_features=out_ch, out_features=out_ch, rngs=rngs)
            block = nnx.Sequential(vgg1, vgg2)
            in_ch = out_ch
            if i < len(self.channels) - 2:
                pool_func = partial(nnx.max_pool, window_shape=(2, 2), strides=(2, 2))
                block = nnx.Sequential(block, pool_func)
            else:
                block = nnx.Sequential(block)
            backbone_layers.append(block)
        self.backbone = nnx.Sequential(*backbone_layers)

        c = self.channels[-1]
        det1 = VGGBlockNNX(self.channels[-2], c, kernel_size=3, rngs=rngs)
        det2 = VGGBlockNNX(c, self.stride**2 + 1, kernel_size=1, relu=False, rngs=rngs)
        self.detector = nnx.Sequential(det1, det2)

        desc1 = VGGBlockNNX(self.channels[-2], c, kernel_size=3, rngs=rngs)
        desc2 = VGGBlockNNX(c, descriptor_dim, kernel_size=1, relu=False, rngs=rngs)
        self.descriptor = nnx.Sequential(desc1, desc2)

    def __call__(self, image: jnp.ndarray, training: bool = False):
        """Forward pass of the SuperPointJAX model.

        Args:
            image (jnp.ndarray): Input image, expected in NHWC format.
            training (bool): Whether to use training mode. Default: False.

        Returns:
            dict: Dictionary containing keypoints, scores, and descriptors.
        """
        B, H, W, C = image.shape
        if C == 3:
            scale = jnp.array([0.299, 0.587, 0.114], dtype=image.dtype)
            image = jnp.sum(image * scale[None, None, None, :], axis=-1, keepdims=True)
        features, scores, desc_dense = self.forward(image, training=training)
        
        scores = jnp.reshape(scores, (B, features.shape[1], features.shape[2], self.stride, self.stride))
        scores = jnp.transpose(scores, (0, 1, 3, 2, 4))
        B2, Hp, R1, Wp, R2 = scores.shape
        scores = jnp.reshape(scores, (B2, Hp * R1, Wp * R2))
        
        scores_nms = batched_nms_jax(scores, self.nms_radius)


        @partial(jax.jit, static_argnames=["pad"])
        def remove_borders(scores_nms, pad):
            scores_nms = scores_nms.at[:, :pad, :].set(-1)
            scores_nms = scores_nms.at[:, -pad:, :].set(-1)
            scores_nms = scores_nms.at[:, :, :pad].set(-1)
            scores_nms = scores_nms.at[:, :, -pad:].set(-1)
            return scores_nms

        if self.remove_borders > 0:
            scores_nms = remove_borders(scores_nms, self.remove_borders)
        _, HH, WW = scores_nms.shape

        max_k = HH * WW if self.max_num_keypoints is None else self.max_num_keypoints
        return extract_keypoints_and_descriptors(
            scores_nms=scores_nms,
            desc_dense=desc_dense,
            detection_threshold=self.detection_threshold,
            max_k=max_k,
            stride=self.stride,
        )

    @partial(nnx.jit, static_argnames=["training"])
    def forward(self, image: jnp.ndarray, training: bool = False):
        """Computes the forward pass of the SuperPointJAX model.

        Args:
            image (jnp.ndarray): Input image.
            training (bool): Training mode flag.

        Returns:
            tuple: A tuple containing features, detector scores, and dense descriptors.
        """
        
        if training:
            self.backbone.train()
        else:
            self.backbone.eval()
        features = self.backbone(image, training=training)
        
        if training:
            self.descriptor.train()
        else:
            self.descriptor.eval()
        desc_dense = self.descriptor(features, training=training)
        
        eps = 1e-8
        norm = jnp.sqrt(jnp.sum(desc_dense ** 2, axis=3, keepdims=True) + eps)
        desc_dense = desc_dense / norm
        
        if training:
            self.detector.train()
        else:
            self.detector.eval()
        scores = self.detector(features, training=training)

        scores = jax.nn.softmax(scores, axis=3)[..., :-1]

        return features, scores, desc_dense
