"""FLAX.NNX implementation of the SuperPoint model,
   derived from the PyTorch re-implementation.

   https://github.com/rpautrat/SuperPoint/blob/master/superpoint_pytorch.py
"""

import jax
import jax.numpy as jnp
from flax.linen.pooling import max_pool
from functools import partial
from flax import nnx


def bilinear_grid_sample_jax(images: jnp.ndarray, coords: jnp.ndarray) -> jnp.ndarray:
    """
    images: (B, C, H, W)
    coords: (B, N, 2) in [-1, 1], (x, y) => x ~ width dimension, y ~ height dimension
    Returns: (B, C, N)
    """
    _, _, H, W = images.shape

    # Convert normalized coords [-1,1] -> [0, W-1] or [0, H-1]
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
        """
        images_b: (C, H, W)
        returns: (C, N)
        """
        top_left     = images_b[:, y0c_b, x0c_b]     # (C, N)
        top_right    = images_b[:, y0c_b, x1c_b]
        bottom_left  = images_b[:, y1c_b, x0c_b]
        bottom_right = images_b[:, y1c_b, x1c_b]
        return (top_left     * w_tl_b +
                top_right    * w_tr_b +
                bottom_left  * w_bl_b +
                bottom_right * w_br_b)

    out = jax.vmap(
        sample_single_image,
        in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0)
    )(images, x0c, x1c, y0c, y1c, w_tl, w_tr, w_bl, w_br)
    return out  # (B, C, N)


def sample_descriptors_jax(keypoints: jnp.ndarray, descriptors: jnp.ndarray, s: int = 8) -> jnp.ndarray:
    """
    keypoints: (B, N, 2) pixel coordinates
    descriptors: (B, C, H, W)
    """
    _, _, H, W = descriptors.shape
    # Convert pixel coords to [-1,1]
    wh = jnp.array([W, H], dtype=keypoints.dtype)
    coords = (keypoints + 0.5) / (wh * s)
    coords = coords * 2.0 - 1.0  # (B, N, 2)

    # Bilinear sampling -> (B, C, N)
    sampled = bilinear_grid_sample_jax(descriptors, coords)

    # L2 normalize
    eps = 1e-8
    norm = jnp.sqrt((sampled ** 2).sum(axis=1, keepdims=True) + eps)
    descriptors_norm = sampled / norm
    return descriptors_norm


def max_pool_2d(x: jnp.ndarray, nms_radius: int) -> jnp.ndarray:
    """
    Wrap flax.linen.pooling.max_pool for our (B, H, W) inputs
    by temporarily adding a single "channel" dimension.
    """
    # Insert a channel dimension: shape becomes (B, H, W, 1)
    x = x[..., None]

    # Window is (2*nms_radius+1) x (2*nms_radius+1), stride=1, and
    # padding=(nms_radius, nms_radius) for each spatial dimension
    window_shape = (2 * nms_radius + 1, 2 * nms_radius + 1)
    pooled = max_pool(
        x,
        window_shape=window_shape,
        strides=(1, 1),
        padding=((nms_radius, nms_radius), (nms_radius, nms_radius)),
    )

    # Drop the channel dimension to return shape (B, H, W)
    return pooled[..., 0]

@partial(jax.jit, static_argnums=(1,))
def batched_nms_jax(scores: jnp.ndarray, nms_radius: int) -> jnp.ndarray:
    """
    JAX/Flax version of the PyTorch batched_nms, operating on
    scores of shape (B, H, W).
    """
    assert nms_radius >= 0

    zeros = jnp.zeros_like(scores)
    # Compare each pixel to its local max to create an initial mask
    max_mask = (scores == max_pool_2d(scores, nms_radius))

    # Perform the iterative suppression
    for _ in range(2):
        # Expand max_mask so that all points in the local neighborhood
        # are suppressed, then compute new maxima in the remaining scores
        supp_mask = max_pool_2d(max_mask.astype(scores.dtype), nms_radius) > 0
        supp_scores = jnp.where(supp_mask, zeros, scores)
        new_max_mask = (supp_scores == max_pool_2d(supp_scores, nms_radius))
        max_mask = max_mask | (new_max_mask & jnp.logical_not(supp_mask))

    return jnp.where(max_mask, scores, zeros)

@partial(jax.jit, static_argnums=(2,))
def select_top_k_keypoints_jax(keypoints: jnp.ndarray,
                               scores: jnp.ndarray,
                               k: int):
    """
    JAX version of top-k. Must receive a static `k`.
    Returns (k,2) keypoints and (k,) scores in descending order.
    """
    top_scores, top_indices = jax.lax.top_k(scores, k)
    top_kpts = jnp.take(keypoints, top_indices, axis=0)
    return top_kpts, top_scores

class VGGBlockNNX(nnx.Module):
    """A small block of Conv -> ReLU -> (optional) BN in NNX style."""
    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: int = 3,
        relu: bool = True,
        rngs: nnx.Rngs = nnx.Rngs(0)
    ):
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
        x = self.bn(x)         
        return x

class SuperPointJAX(nnx.Module):
    """
    Flax NNX re-implementation of the SuperPoint model.
    By default, expects inputs in NHWC format: (B, H, W, 1 or 3).
    """
    def __init__(
        self,
        nms_radius: int = 4,
        max_num_keypoints: int = None,
        detection_threshold: float = 0.005,
        remove_borders: int = 4,
        descriptor_dim: int = 256,
        channels: list = [64, 64, 128, 128, 256],
        rngs: nnx.Rngs = nnx.Rngs(0),
    ):
        self.nms_radius = nms_radius
        self.max_num_keypoints = max_num_keypoints
        self.detection_threshold = detection_threshold
        self.remove_borders = remove_borders
        self.descriptor_dim = descriptor_dim
        self.channels = channels

        self.stride = 2 ** (len(self.channels) - 2)

        # ---------------------------
        # Build the backbone
        # ---------------------------
        backbone_layers = []
        in_ch = 1
    

        # For each stage:
        for i, out_ch in enumerate(self.channels[:-1]):
            vgg1 = VGGBlockNNX(
                in_features=in_ch,
                out_features=out_ch,
                rngs=rngs
            )

            vgg2 = VGGBlockNNX(
                in_features=out_ch,
                out_features=out_ch,
                rngs=rngs
            )

            block = nnx.Sequential(vgg1, vgg2)
            in_ch = out_ch

            if i < len(self.channels)-2:
                pool_func = partial(nnx.max_pool, window_shape=(2, 2), strides=(2, 2))
                block = nnx.Sequential(block, pool_func)
            else:
                block = nnx.Sequential(block)

            backbone_layers.append(block)

        self.backbone = nnx.Sequential(*backbone_layers)

        # ---------------------------
        # Detector head
        # ---------------------------
        c = self.channels[-1]
        det1 = VGGBlockNNX(self.channels[-2], c, kernel_size=3, rngs=rngs)
        det2 = VGGBlockNNX(c, self.stride**2 + 1, kernel_size=1, relu=False, rngs=rngs)
        self.detector = nnx.Sequential(det1, det2)

        # ---------------------------
        # Descriptor head
        # ---------------------------
        desc1 = VGGBlockNNX(self.channels[-2], c, kernel_size=3, rngs=rngs)
        desc2 = VGGBlockNNX(c, descriptor_dim, kernel_size=1, relu=False, rngs=rngs)
        self.descriptor = nnx.Sequential(desc1, desc2)

    def __call__(self, image: jnp.ndarray, training: bool = False):
        """
        Forward pass for SuperPoint in Flax NNX, matching the original PyTorch logic.

        Returns:
        {
            "keypoints": list of length B, each an array of shape (N_i, 2)
            "keypoint_scores": list of length B, each an array of shape (N_i,)
            "descriptors": list of length B, each an array of shape (N_i, descriptor_dim)
        }
        """
        B, _, _, C = image.shape
        temp_out = {}
        # Convert to grayscale if input has 3 channels.
        if C == 3:
            scale = jnp.array([0.299, 0.587, 0.114], dtype=image.dtype)
            image = jnp.sum(image * scale[None, None, None, :], axis=-1, keepdims=True)

        # Backbone features
        if training:
            self.backbone.train()
        else:
            self.backbone.eval()
        features = self.backbone(image, training=training)  

        # Descriptor head
        if training:
            self.descriptor.train()
        else:
            self.descriptor.eval()

        desc_dense = self.descriptor(features, training=training)
        eps = 1e-8
        norm = jnp.sqrt(jnp.sum(desc_dense**2, axis=3, keepdims=True) + eps)
        desc_dense = desc_dense / norm  


        # Detector head
        if training:
            self.detector.train()
        else:
            self.detector.eval()

        scores = self.detector(features, training=training)  # (B, H', W', stride^2+1)

        # softmax over last channel => discard last channel => (B, H', W', stride^2)
        scores = jax.nn.softmax(scores, axis=3)[..., :-1]

        # reshape => (B, H'*stride, W'*stride)
        scores = jnp.reshape(scores, (B, features.shape[1], features.shape[2], self.stride, self.stride))
        scores = jnp.transpose(scores, (0, 1, 3, 2, 4))  # => (B, H', R, W', R)
        B2, Hp, R1, Wp, R2 = scores.shape
        scores = jnp.reshape(scores, (B2, Hp * R1, Wp * R2))


        # NMS
        scores_nms = batched_nms_jax(scores, self.nms_radius)  # shape (B, HH, WW)


        # Remove borders
        if self.remove_borders > 0:
            pad = self.remove_borders
            # scores_nms has shape (B, HH, WW)
            # Match PyTorch logic by setting each border region to -1
            scores_nms = scores_nms.at[:, :pad, :].set(-1)    # top
            scores_nms = scores_nms.at[:, -pad:, :].set(-1)   # bottom
            scores_nms = scores_nms.at[:, :, :pad].set(-1)    # left
            scores_nms = scores_nms.at[:, :, -pad:].set(-1)   # right

        _, _, WW = scores_nms.shape

        # Threshold => gather keypoints => sample descriptors
        results = {
            "keypoints": [],
            "keypoint_scores": [],
            "descriptors": [],
        }

        def gather_keypoints(scores_map):
            # scores_map => (HH, WW), single-batch
            # flatten
            scores_flat = scores_map.ravel()  # shape (HH*WW,)
            idxs = jnp.where(scores_flat > self.detection_threshold)[0]
            # => positions in [0..HH*WW)
            # convert idx -> (row, col) => (y, x)
            yx = jnp.stack((idxs // WW, idxs % WW), axis=-1)
            s = scores_flat[idxs]
            return yx, s

        # shape => (B, HH, WW)
        for b_idx in range(B):
            scores_map = scores_nms[b_idx]
            yx, s = gather_keypoints(scores_map)  # shape => (N,2), (N,)

            # If specified, apply top-k
            if self.max_num_keypoints is not None:
                yx, s = select_top_k_keypoints_jax(yx, s, self.max_num_keypoints)

            # sample descriptors, using the (x,y) convention
            # The above 'yx' is (row, col) => (y, x). Let's flip => (x,y)
            xy = jnp.flip(yx, axis=-1)  # shape => (N,2)
            coords_b = xy[None]         # shape => (1,N,2)
            desc_b   = desc_dense[b_idx:b_idx+1]  # shape => (1,H',W',C')
            desc_sampled = sample_descriptors_jax(coords_b, desc_b, s=self.stride)
            # desc_sampled => (1, C', N)
            desc_sampled = desc_sampled[0]  # => (C', N)

            # store results
            results["keypoints"].append(yx)        # (N,2) in (y,x)
            results["keypoint_scores"].append(s)   # (N,)
            # transpose descriptors => shape => (N, C')
            results["descriptors"].append(jnp.swapaxes(desc_sampled, 0, 1))

        return results
