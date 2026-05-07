"""Frozen MCR (Manipulation-Centric Robotic Representation) encoder.

Drop-in replacement for ``mjlab.rl.spatial_softmax.SpatialSoftmaxCNNModel``.
Uses a frozen ResNet-50 pretrained on the DROID dataset with MCR's
action-prediction + time-contrastive + dynamics-alignment objectives
(Jiang et al. 2024, arXiv:2410.22325).  See ``docs/sim2real/plan_v2.md``.

Key facts the rest of the pipeline relies on:

  * Each camera obs group arrives as ``(B, C, H, W)`` with ``H = W = 64`` and
    ``C ∈ {3, 6}``.  The vision env (``pick_cube_vision_osc.py``) concatenates
    wrist + d455 channel-wise into a single ``(B, 6, 64, 64)`` "camera"
    group, so this encoder splits the 6 channels into two 3-channel images,
    forwards them independently through the backbone, and concatenates the
    pooled features.
  * MCR was trained at 224×224 ImageNet input.  We bilinearly upsample 64→224
    inside the encoder so the env keeps its cheap 64×64 render path.
  * Output is a flat 2048-dim feature per camera (post global-avg-pool, FC
    dropped).  ``output_dim`` returned to ``CNNModel`` is ``2048 *
    n_cameras_in_this_group`` (so 4096 for the wrist+d455 concat).
  * Backbone is frozen — every weight has ``requires_grad=False`` and the
    module stays in ``eval()`` mode.  The learnable parameters of the student
    visual pathway live entirely in the downstream MLP head.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from rsl_rl.models.cnn_model import CNNModel
from rsl_rl.models.mlp_model import MLPModel
from tensordict import TensorDict
from torchvision.models import resnet50

# ImageNet normalization stats.  ``camera_rgb`` already returns float in
# [0, 1] (see mjlab/.../manipulation/mdp/observations.py:101) so we only
# subtract mean / divide std here — no extra /255.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_MCR_INPUT_HW = 224  # native resolution MCR was pretrained at


def _build_resnet50_trunk() -> tuple[nn.Module, int]:
    """Return a torchvision ResNet-50 with the FC head stripped.

    The trunk emits a (B, 2048) feature after global-avg-pool.
    """
    m = resnet50(weights=None)
    feat_dim = m.fc.in_features  # 2048
    m.fc = nn.Identity()
    return m, feat_dim


class _SpatialSoftmax(nn.Module):
    """Spatial soft-argmax. (B, C, H, W) -> (B, C*2)."""

    def __init__(self, height: int, width: int, temperature: float = 1.0) -> None:
        super().__init__()
        pos_x, pos_y = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height),
            torch.linspace(-1.0, 1.0, width),
            indexing="ij",
        )
        self.register_buffer("pos_x", pos_x.reshape(1, 1, -1), persistent=False)
        self.register_buffer("pos_y", pos_y.reshape(1, 1, -1), persistent=False)
        self.temperature = temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        feat = x.reshape(B, C, -1)
        w = torch.softmax(feat / self.temperature, dim=-1)
        ex = (w * self.pos_x).sum(dim=-1)
        ey = (w * self.pos_y).sum(dim=-1)
        return torch.stack([ex, ey], dim=-1).reshape(B, C * 2)


class _ResNet50Layer4Trunk(nn.Module):
    """Run a torchvision ResNet-50 up to the `layer4` output (no avg-pool).

    Output shape for 224×224 input is (B, 2048, 7, 7).
    """

    def __init__(self, full_resnet: nn.Module) -> None:
        super().__init__()
        # Stitch the ResNet's stem + four residual stages.
        self.conv1 = full_resnet.conv1
        self.bn1 = full_resnet.bn1
        self.relu = full_resnet.relu
        self.maxpool = full_resnet.maxpool
        self.layer1 = full_resnet.layer1
        self.layer2 = full_resnet.layer2
        self.layer3 = full_resnet.layer3
        self.layer4 = full_resnet.layer4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.layer4(x)  # (B, 2048, 7, 7) for 224×224 input


def _load_mcr_state_dict(weights_path: str | Path) -> dict[str, torch.Tensor]:
    """Load MCR weights from ``weights_path``.

    Per the upstream loader (`mcr/__init__.py:load_mcr`), keys map 1:1 onto
    ``torchvision.models.resnet50()`` with FC stripped — no prefix or
    container wrapper.  Download from
    ``https://huggingface.co/GqJiang/robots-pretrain-robots/resolve/main/mcr_resnet50.pth``.
    """
    raw = torch.load(weights_path, map_location="cpu", weights_only=True)
    if not isinstance(raw, dict):
        raise RuntimeError(
            f"MCR checkpoint at {weights_path} did not load as a dict; got "
            f"{type(raw).__name__}.  Expected an OrderedDict of ResNet-50 keys."
        )
    return raw


class FrozenMCREncoder(nn.Module):
    """Per-camera-group frozen MCR encoder.

    Shape contract:
      * Input:  ``(B, C, H, W)`` with ``C`` divisible by 3.  The ``C // 3``
        sub-images are forwarded independently and their features
        concatenated along the channel dim of the output.
      * Output: ``(B, 2048 * n_cameras)`` flat tensor.

    Exposes ``output_dim`` (int) and ``output_channels`` (``None``) to mimic
    ``SpatialSoftmaxCNN``'s interface — ``CNNModel`` reads these to size its
    MLP head.
    """

    def __init__(
        self,
        input_dim: tuple[int, int],
        input_channels: int,
        weights_path: str | Path,
        target_hw: int = _MCR_INPUT_HW,
        feature_layernorm: bool = True,
        pool: str = "avg",
        spatial_softmax_temperature: float = 1.0,
        unfreeze_layers: tuple[str, ...] = (),
    ) -> None:
        super().__init__()
        if input_channels % 3 != 0:
            raise ValueError(
                f"FrozenMCREncoder expects input_channels divisible by 3 "
                f"(one RGB image per camera); got {input_channels}."
            )
        if pool not in ("avg", "spatial_softmax"):
            raise ValueError(f"pool must be 'avg' or 'spatial_softmax', got {pool!r}")
        self._n_cameras = input_channels // 3
        self._target_hw = int(target_hw)
        self._input_hw = tuple(int(x) for x in input_dim)
        self._feature_layernorm = bool(feature_layernorm)
        self._pool = pool

        weights_path = Path(weights_path).expanduser().resolve()
        if not weights_path.exists():
            raise FileNotFoundError(
                f"MCR weights not found at {weights_path}.  Download per "
                f"docs/sim2real/plan_v2.md §3.1 and place at this path."
            )

        # Always build the full ResNet first so we can load the state_dict
        # with strict=True semantics (ignoring fc.*).
        full_resnet, feat_dim = _build_resnet50_trunk()
        state_dict = _load_mcr_state_dict(weights_path)
        missing, unexpected = full_resnet.load_state_dict(state_dict, strict=False)
        leftover_missing = [m for m in missing if not m.startswith("fc.")]
        if leftover_missing or unexpected:
            raise RuntimeError(
                f"MCR weight load failed: missing(non-fc)={leftover_missing[:5]}, "
                f"unexpected={unexpected[:5]}.  Did you download "
                f"mcr_resnet50.pth from "
                f"https://huggingface.co/GqJiang/robots-pretrain-robots ?"
            )
        print(
            f"[MCR] loaded {len(state_dict)} weight tensors from {weights_path} "
            f"(pool={pool})"
        )

        if self._pool == "avg":
            # Standard MCR usage: full ResNet → 2048-D pooled feature.
            backbone = full_resnet
        else:
            # Spatial path: cut at layer4 → (B, 2048, 7, 7), then spatial_softmax.
            backbone = _ResNet50Layer4Trunk(full_resnet)

        backbone.eval()
        for p in backbone.parameters():
            p.requires_grad = False
        # Selective unfreezing: re-enable grads on named submodules.
        # Common targets: ("layer4",) or ("layer3", "layer4").  Module
        # stays in eval() mode so BatchNorm running stats stay fixed —
        # only the conv/bn weights themselves train.
        self._unfrozen_layers = tuple(unfreeze_layers)
        for name in self._unfrozen_layers:
            sub = dict(backbone.named_children()).get(name)
            if sub is None:
                raise ValueError(
                    f"unfreeze_layers entry {name!r} not a child of "
                    f"backbone (children: "
                    f"{list(dict(backbone.named_children()).keys())})"
                )
            for p in sub.parameters():
                p.requires_grad = True
        self.backbone = backbone

        # ImageNet normalization buffers — registered so they ride device moves.
        self.register_buffer(
            "_in_mean",
            torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_in_std",
            torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1),
            persistent=False,
        )

        if self._pool == "avg":
            per_cam_dim = int(feat_dim)
            self.spatial_softmax: nn.Module | None = None
        else:
            # ResNet-50 layer4 output is (2048, 7, 7) for 224×224 input.
            self.spatial_softmax = _SpatialSoftmax(
                height=7, width=7, temperature=spatial_softmax_temperature
            )
            per_cam_dim = int(feat_dim) * 2  # x,y per channel

        self._output_dim = per_cam_dim * self._n_cameras

        # Optional LayerNorm on the per-camera feature.  Avg-pooled ResNet
        # features have unbounded magnitudes; the spatial-softmax keypoint
        # output is already in [-1, 1] so LayerNorm is mostly cosmetic
        # there — leave it on for both for consistency unless ablated.
        self.feature_norm = (
            nn.LayerNorm(per_cam_dim) if self._feature_layernorm else nn.Identity()
        )

    @property
    def output_dim(self) -> int:
        """Flattened output dim (matches ``SpatialSoftmaxCNN.output_dim``)."""
        return self._output_dim

    @property
    def output_channels(self) -> None:
        """Always ``None`` — output is flat (matches ``CNNModel`` contract)."""
        return None

    def train(self, mode: bool = True):  # type: ignore[override]
        """Keep the frozen backbone in eval mode regardless of ``model.train()``."""
        super().train(mode)
        self.backbone.eval()
        return self

    def _featurize_one_camera(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W) in [0, 1].  Resize to 224×224 + ImageNet normalize.
        # If everything is frozen we wrap the backbone forward in no_grad
        # for free memory savings.  If any layer is unfrozen, autograd has
        # to be on.  In either case downstream LayerNorm + spatial-softmax
        # (in spatial mode) remain in the autograd graph.
        if x.shape[-2:] != (self._target_hw, self._target_hw):
            x = F.interpolate(
                x,
                size=(self._target_hw, self._target_hw),
                mode="bilinear",
                align_corners=False,
            )
        x = (x - self._in_mean) / self._in_std
        if self._unfrozen_layers:
            feat = self.backbone(x)
        else:
            with torch.no_grad():
                feat = self.backbone(x)
        if self.spatial_softmax is not None:
            feat = self.spatial_softmax(feat)
        return self.feature_norm(feat)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3*n_cams, H, W).  Split, featurize per camera, concat.
        if x.dim() != 4:
            raise ValueError(f"FrozenMCREncoder expects 4D input, got {x.shape}")
        if x.shape[1] != 3 * self._n_cameras:
            raise ValueError(
                f"FrozenMCREncoder configured for {self._n_cameras} cameras "
                f"(C={3 * self._n_cameras}); got C={x.shape[1]}."
            )
        feats = [
            self._featurize_one_camera(x[:, 3 * i : 3 * (i + 1)])
            for i in range(self._n_cameras)
        ]
        return torch.cat(feats, dim=-1)  # (B, 2048 * n_cams)


class MCRCNNModel(CNNModel):
    """``CNNModel`` that uses ``FrozenMCREncoder`` per 2D obs group.

    Mirrors ``mjlab.rl.spatial_softmax.SpatialSoftmaxCNNModel``'s structure
    so it slots into ``RslRlModelCfg`` via ``class_name`` without changes
    elsewhere in the runner.

    ``cnn_cfg`` is the dict that arrives from the runner config.  Recognized
    keys (everything else is ignored):

      * ``weights_path`` (required): filesystem path to the MCR ResNet-50
        ``.pt`` checkpoint.  Resolved relative to the current working dir.
        Can also be set via the ``MCR_WEIGHTS_PATH`` env var, which takes
        precedence (handy for sweeps that don't want to thread paths
        through tyro).
      * ``target_hw`` (optional, default 224): resolution to upsample to
        before the backbone.  Don't touch this unless you know why.
      * ``feature_layernorm`` (optional, default True): apply a per-camera
        LayerNorm to the 2048-D feature so the freshly-initialized MLP head
        sees a unit-scale input.  Disable to ablate.
      * ``pool`` (optional, default ``"avg"``): how to reduce the ResNet-50
        feature map to a per-camera vector.
          - ``"avg"``: standard MCR usage — global avg-pool to ``(B, 2048)``.
          - ``"spatial_softmax"``: cut the ResNet at ``layer4`` and run
            spatial-softmax over the ``(2048, 7, 7)`` map → ``(B, 4096)``
            keypoint coordinates.  Preserves spatial location information.
      * ``spatial_softmax_temperature`` (optional, default 1.0): only used
        when ``pool="spatial_softmax"``.
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        cnn_cfg: dict[str, dict] | dict[str, Any],
        cnns: nn.ModuleDict | None = None,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict[str, Any] | None = None,
    ) -> None:
        # Populate self.obs_groups_2d / obs_dims_2d / obs_channels_2d.
        self._get_obs_dim(obs, obs_groups, obs_set)

        if cnns is not None:
            if set(cnns.keys()) != set(self.obs_groups_2d):
                raise ValueError(
                    "The 2D observations must be identical for all models "
                    "sharing CNN encoders."
                )
            print(
                "Sharing MCR encoders between models, the CNN configurations "
                "of the receiving model are ignored."
            )
            _cnns: dict[str, nn.Module] = dict(cnns)
        else:
            # Expand a single flat config to per-group configs.
            if not all(isinstance(v, dict) for v in cnn_cfg.values()):
                cnn_cfg = {group: cnn_cfg for group in self.obs_groups_2d}
            assert len(cnn_cfg) == len(self.obs_groups_2d), (
                "The number of MCR configurations must match the number of "
                "2D observation groups."
            )
            _cnns = {}
            for idx, obs_group in enumerate(self.obs_groups_2d):
                group_cfg = dict(cnn_cfg[obs_group])
                env_path = os.environ.get("MCR_WEIGHTS_PATH")
                weights_path = env_path or group_cfg.pop("weights_path", None)
                if weights_path is None:
                    raise ValueError(
                        f"MCRCNNModel needs a 'weights_path' in cnn_cfg "
                        f"(or MCR_WEIGHTS_PATH env var) for group "
                        f"{obs_group!r}."
                    )
                target_hw = group_cfg.pop("target_hw", _MCR_INPUT_HW)
                feature_layernorm = group_cfg.pop("feature_layernorm", True)
                pool = group_cfg.pop("pool", "avg")
                ss_temp = group_cfg.pop("spatial_softmax_temperature", 1.0)
                unfreeze_layers = tuple(group_cfg.pop("unfreeze_layers", ()))
                # Anything left in group_cfg is silently ignored — this lets
                # the same task cfg coexist with the spatial-softmax CNN
                # without a separate config plumbing path.
                _cnns[obs_group] = FrozenMCREncoder(
                    input_dim=self.obs_dims_2d[idx],
                    input_channels=self.obs_channels_2d[idx],
                    weights_path=weights_path,
                    target_hw=target_hw,
                    feature_layernorm=feature_layernorm,
                    pool=pool,
                    spatial_softmax_temperature=ss_temp,
                    unfreeze_layers=unfreeze_layers,
                )

        # Mirror SpatialSoftmaxCNNModel: compute cnn_latent_dim, then go
        # through MLPModel.__init__ rather than CNNModel.__init__ (which
        # would try to construct fresh CNN encoders from cnn_cfg).
        self.cnn_latent_dim = 0
        for cnn in _cnns.values():
            if cnn.output_channels is not None:
                raise ValueError(
                    "The output of the MCR encoder must be flattened before "
                    "passing it to the MLP."
                )
            self.cnn_latent_dim += int(cnn.output_dim)

        MLPModel.__init__(
            self,
            obs=obs,
            obs_groups=obs_groups,
            obs_set=obs_set,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            obs_normalization=obs_normalization,
            distribution_cfg=distribution_cfg,
        )

        if isinstance(_cnns, nn.ModuleDict):
            self.cnns = _cnns
        else:
            self.cnns = nn.ModuleDict(_cnns)
