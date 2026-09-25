"""Shared base classes for the UNSB Schrödinger-Bridge (SB) model.

The model is ported from the CUT-lineage UNSB repository, so its
training dynamics (four networks G/F/D/E, D -> E -> (G, F) update order,
PatchNCE losses, SB energy loss) differ substantially from the BBDM-lineage
``BaseDiffusionStrategy``. They therefore use their own strategy base while
reusing :class:`BaseDiffusionModel` / :class:`BaseDiffusionProcessor` for the
model/inference contract.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn as nn

from ..cut_family.networks.cut_networks import PatchSampleF, init_net
from ..loss.base import CompositeLoss, StainLoss
from ..loss.gan import GANLoss
from ..loss.patchnce import PatchNCELoss
from ..networks.utils import get_norm_layer
from ..strategy.base import TrainingStrategy
from .base import BaseDiffusionModel
from .networks.sb_networks import NLayerDiscriminator_ncsn, ResnetGenerator_ncsn

if TYPE_CHECKING:
    from ...configs.schema import PuzzleStainConfig

logger = logging.getLogger(__name__)


class BaseSBModel(BaseDiffusionModel):
    """Common Schrödinger-Bridge model wrapper.

    Networks (mirroring the original ``SBModel``):
      - ``denoise_fn`` (``netG``): ``ResnetGenerator_ncsn`` transition kernel,
        also used as the PatchNCE feature extractor.
      - ``netF``: ``PatchSampleF`` projection head (lazy MLPs).
      - ``netD``: time-conditioned PatchGAN discriminator.
      - ``netE``: time-conditioned energy net over paired (x_t, x_{t+1}).

    Subclasses set the hyperparameters as attributes before calling
    ``super().__init__()`` and are registered under distinct names in
    :class:`~puzzlestain.models.registry.ModelRegistry`.
    """

    def _build_networks(self) -> None:
        norm_layer = get_norm_layer(self.norm)
        # CUT-family convention: networks stay on CPU unless gpu_ids is
        # non-empty, so netF's lazily-created MLPs always land on the same
        # device as the rest of the model.
        gpu_ids = self.gpu_ids or []
        self.denoise_fn = init_net(
            ResnetGenerator_ncsn(
                self.input_nc,
                self.output_nc,
                self.ngf,
                norm_layer=norm_layer,
                use_dropout=self.use_dropout,
                n_blocks=9,
                n_mlp=self.n_mlp,
            ),
            init_type=self.init_type,
            init_gain=self.init_gain,
            gpu_ids=gpu_ids,
        )
        self.netF = PatchSampleF(
            use_mlp=True,
            init_type=self.init_type,
            init_gain=self.init_gain,
            gpu_ids=gpu_ids,
            nc=self.netF_nc,
        )
        if not self._no_d:
            self.netD = init_net(
                self._build_discriminator(self.netD_name, self.output_nc),
                init_type=self.init_type,
                init_gain=self.init_gain,
                gpu_ids=gpu_ids,
            )
            self.netE = init_net(
                self._build_discriminator(self.netE_name, self.output_nc * 4),
                init_type=self.init_type,
                init_gain=self.init_gain,
                gpu_ids=gpu_ids,
            )

    def _build_discriminator(self, name: str, input_nc: int) -> nn.Module:
        """Build a time-conditioned discriminator / energy net by name."""
        norm_layer = get_norm_layer(self.norm)
        if name == "basic_cond":
            return NLayerDiscriminator_ncsn(
                input_nc, self.ndf, self.n_layers_D, norm_layer=norm_layer
            )
        raise NotImplementedError(
            f"Discriminator [{name}] is not supported for SB models"
        )

    def __init__(
        self,
        input_nc: int = 3,
        output_nc: int = 3,
        ngf: int = 64,
        ndf: int = 64,
        netD: str = "basic_cond",
        netE: str = "basic_cond",
        n_layers_D: int = 3,
        norm: str = "instance",
        init_type: str = "xavier",
        init_gain: float = 0.02,
        use_dropout: bool = False,
        n_mlp: int = 3,
        num_timesteps: int = 5,
        tau: float = 0.01,
        lambda_GAN: float = 1.0,
        lambda_SB: float = 1.0,
        lambda_NCE: float = 1.0,
        nce_idt: bool = True,
        nce_layers: str = "0,4,8,12,16",
        num_patches: int = 256,
        netF_nc: int = 256,
        gpu_ids: list[int] | None = None,
        no_d: bool = False,
    ) -> None:
        self.ngf = ngf
        self.ndf = ndf
        self.netD_name = netD
        self.netE_name = netE
        self.n_layers_D = n_layers_D
        self.norm = norm
        self.init_type = init_type
        self.init_gain = init_gain
        self.use_dropout = use_dropout
        self.n_mlp = n_mlp
        self.num_timesteps = num_timesteps
        self.tau = tau
        self.lambda_GAN = lambda_GAN
        self.lambda_SB = lambda_SB
        self.lambda_NCE = lambda_NCE
        self.nce_idt = nce_idt
        self.nce_layers = [int(i) for i in str(nce_layers).split(",")]
        self.num_patches = num_patches
        self.netF_nc = netF_nc
        self.gpu_ids = gpu_ids
        self._no_d = no_d
        self.netD: nn.Module | None = None
        self.netE: nn.Module | None = None

        super().__init__(input_nc=input_nc, output_nc=output_nc, no_d=no_d)

        # Fixed non-uniform time grid (ported exactly from the original repo).
        # For T=5 this yields [0, 0.5, 0.74, 0.86, 0.94, 1.0]. Registered after
        # super().__init__() because nn.Module must be initialized first.
        incs = np.array([0] + [1 / (i + 1) for i in range(num_timesteps - 1)])
        times = np.cumsum(incs)
        times = times / times[-1]
        times = 0.5 * times[-1] + 0.5 * times
        times = np.concatenate([np.zeros(1), times])
        self.register_buffer("times", torch.tensor(times).float())

    def denoise(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor | None,
    ) -> torch.Tensor:
        """Run one transition-kernel evaluation ``G(x_t, t, z)``.

        ``context`` carries the latent code ``z``; a fresh one is drawn when
        it is None.
        """
        bs = x_t.shape[0]
        if not torch.is_tensor(t):
            t = torch.full((bs,), int(t), dtype=torch.long, device=x_t.device)
        t = t.to(x_t.device).long()
        if t.numel() == 1:
            t = t.expand(bs)
        z = (
            context
            if context is not None
            else torch.randn(bs, 4 * self.ngf, device=x_t.device)
        )
        return self.denoise_fn(x_t, t, z)

    @torch.no_grad()
    def sample(self, y: torch.Tensor) -> torch.Tensor:
        """Run the full T-step Markovian bridge rollout from ``y``.

        Mirrors the original repo's test-time forward: every step blends the
        current state with the generator output and injects Gaussian noise.
        """
        device = y.device
        bs = y.shape[0]
        times = self.times
        Xt = y
        Xt_1 = y
        for t in range(self.num_timesteps):
            if t > 0:
                delta = times[t] - times[t - 1]
                denom = times[-1] - times[t - 1]
                inter = (delta / denom).reshape(-1, 1, 1, 1)
                scale = (delta * (1 - delta / denom)).reshape(-1, 1, 1, 1)
                Xt = (
                    (1 - inter) * Xt
                    + inter * Xt_1.detach()
                    + (scale * self.tau).sqrt() * torch.randn_like(Xt)
                )
            time_idx = t * torch.ones(bs, dtype=torch.long, device=device)
            z = torch.randn(bs, 4 * self.ngf, device=device)
            Xt_1 = self.denoise_fn(Xt, time_idx, z)
        return Xt_1

    def sample_timesteps(
        self, batch_size: int, device: torch.device
    ) -> torch.Tensor:
        """Draw a random integer timestep index in ``[0, num_timesteps)``."""
        return torch.randint(self.num_timesteps, size=(batch_size,), device=device)

    def get_trainable_params(self) -> list[nn.Parameter]:
        """Return parameters of G, F and (when present) D and E."""
        params = list(self.denoise_fn.parameters()) + list(self.netF.parameters())
        if self.netD is not None:
            params += list(self.netD.parameters())
        if self.netE is not None:
            params += list(self.netE.parameters())
        return params


class BaseSBStrategy(TrainingStrategy):
    """Common training strategy for Schrödinger-Bridge models.

    Mirrors the original ``SBModel.optimize_parameters``: one forward pass,
    then a D update, an E update, and a joint G/F update with D/E frozen.
    The energy net requires a second, independent batch as its reference
    stream; the original repo zips two independently-shuffled dataloaders,
    while here the previous shuffled batch plays that role (the current batch
    is reused at step 0 / right after resume, when the cache is empty).

    Args:
        model: A :class:`BaseSBModel` subclass.
        processor: A :class:`BaseDiffusionProcessor`.
        loss_fn: Composite loss containing ``gan`` and ``patchnce`` components.
        config: Resolved experiment config.
    """

    def __init__(
        self,
        model: BaseSBModel,
        processor: Any,
        loss_fn: StainLoss,
        config: "PuzzleStainConfig",
    ) -> None:
        self.model = model
        self.processor = processor
        self.loss_fn = loss_fn
        self.config = config
        self._prev_batch: tuple[torch.Tensor, torch.Tensor] | None = None

    # ------------------------------------------------------------------
    # Networks / optimizers / schedulers
    # ------------------------------------------------------------------
    def get_networks(self) -> dict[str, nn.Module]:
        networks: dict[str, nn.Module] = {
            "G": self.model.netG,
            "F": self.model.netF,
        }
        if self.model.netD is not None:
            networks["D"] = self.model.netD
        if self.model.netE is not None:
            networks["E"] = self.model.netE
        return networks

    def get_optimizers(
        self, config: "PuzzleStainConfig"
    ) -> dict[str, torch.optim.Optimizer]:
        opt_cfg = config.training
        weight_decay = getattr(opt_cfg, "weight_decay", 0.0)
        optimizers: dict[str, torch.optim.Optimizer] = {
            "G": torch.optim.Adam(
                self.model.netG.parameters(),
                lr=opt_cfg.lr,
                betas=(opt_cfg.beta1, opt_cfg.beta2),
                weight_decay=weight_decay,
            ),
        }
        if self.model.netD is not None and self.model.netE is not None:
            optimizers["D"] = torch.optim.Adam(
                self.model.netD.parameters(),
                lr=opt_cfg.lr,
                betas=(opt_cfg.beta1, opt_cfg.beta2),
                weight_decay=weight_decay,
            )
            optimizers["E"] = torch.optim.Adam(
                self.model.netE.parameters(),
                lr=opt_cfg.lr,
                betas=(opt_cfg.beta1, opt_cfg.beta2),
                weight_decay=weight_decay,
            )
        return optimizers

    def get_schedulers(
        self,
        optimizers: dict[str, torch.optim.Optimizer],
        config: "PuzzleStainConfig",
    ) -> dict[str, Any]:
        """Create LR schedulers matching the original linear policy.

        Constant for ``n_epochs``, then linear decay to 0 over
        ``n_epochs_decay`` (the same rule used by the CUT/CycleGAN family).
        """
        opt_cfg = config.training
        policy = getattr(opt_cfg, "lr_policy", "linear")
        if policy != "linear":
            return {name: None for name in optimizers}

        max_epochs = opt_cfg.max_epochs
        n_epochs = getattr(opt_cfg, "n_epochs", max_epochs // 2)
        n_epochs_decay = getattr(opt_cfg, "n_epochs_decay", max_epochs - n_epochs)

        def lambda_rule(epoch: int) -> float:
            return 1.0 - max(0, epoch + 1 - n_epochs) / float(n_epochs_decay + 1)

        return {
            name: torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda_rule)
            for name, opt in optimizers.items()
        }

    # ------------------------------------------------------------------
    # Lazy initialization (netF MLPs)
    # ------------------------------------------------------------------
    def data_dependent_initialize(
        self,
        batch: dict[str, torch.Tensor],
        optimizers: dict[str, torch.optim.Optimizer],
    ) -> None:
        """Warmup forward/backward that materializes netF's lazy MLPs.

        Mirrors the original repo's ``data_dependent_initialize``: one
        forward pass followed by D/E/G loss backwards with no optimizer step.
        The previous-batch cache is empty here, so the current batch is reused
        as the reference stream (the same fallback as step 0).
        """
        device = next(self.model.parameters()).device
        inputs = self.processor.preprocess(batch, device=device)
        real_A = inputs["source_image"]
        real_B = inputs["target_image"]

        out = self._sb_forward(real_A, real_B, real_A)

        if "D" in optimizers:
            loss_D, _, _ = self._compute_d_loss(out, real_B)
            optimizers["D"].zero_grad()
            loss_D.backward()
            optimizers["D"].zero_grad()

            loss_E = self._compute_e_loss(out)
            optimizers["E"].zero_grad()
            loss_E.backward()
            optimizers["E"].zero_grad()

        loss_G, _ = self._compute_g_loss(out, real_A, real_B)
        optimizers["G"].zero_grad()
        loss_G.backward()
        optimizers["G"].zero_grad()

        # Create the F optimizer now that netF's MLPs exist, matching the
        # original repo where setup() runs after data_dependent_initialize.
        if "F" not in optimizers:
            opt_cfg = self.config.training
            optimizers["F"] = torch.optim.Adam(
                self.model.netF.parameters(),
                lr=opt_cfg.lr,
                betas=(opt_cfg.beta1, opt_cfg.beta2),
                weight_decay=getattr(opt_cfg, "weight_decay", 0.0),
            )

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------
    def training_step(
        self,
        batch: dict[str, torch.Tensor],
        global_step: int,
        optimizers: dict[str, torch.optim.Optimizer],
    ) -> dict[str, float]:
        device = next(self.model.parameters()).device
        inputs = self.processor.preprocess(batch, device=device)
        real_A = inputs["source_image"]
        real_B = inputs["target_image"]

        # Second independent batch (reference stream for the energy net).
        if self._prev_batch is None:
            real_A2, _ = real_A, real_B
        else:
            real_A2, _ = self._prev_batch
        self._prev_batch = (real_A, real_B)

        out = self._sb_forward(real_A, real_B, real_A2)
        losses: dict[str, float] = {}

        # ---- Update D ----
        opt_D = optimizers["D"]
        opt_D.zero_grad()
        loss_D, loss_D_real, loss_D_fake = self._compute_d_loss(out, real_B)
        loss_D.backward()
        opt_D.step()
        losses["loss_D"] = loss_D.item()
        losses["loss_D_real"] = loss_D_real.item()
        losses["loss_D_fake"] = loss_D_fake.item()

        # ---- Update E ----
        opt_E = optimizers["E"]
        opt_E.zero_grad()
        loss_E = self._compute_e_loss(out)
        loss_E.backward()
        opt_E.step()
        losses["loss_E"] = loss_E.item()

        # ---- Update G and F (D/E frozen) ----
        opt_G = optimizers["G"]
        self._set_requires_grad([self.model.netD, self.model.netE], False)
        opt_G.zero_grad()
        if "F" in optimizers:
            optimizers["F"].zero_grad()
        loss_G, g_losses = self._compute_g_loss(out, real_A, real_B)
        loss_G.backward()
        opt_G.step()
        if "F" in optimizers:
            optimizers["F"].step()
        self._set_requires_grad([self.model.netD, self.model.netE], True)

        for name, value in g_losses.items():
            losses[name] = value.item()
        return losses

    def sample_step(
        self,
        batch: dict[str, torch.Tensor],
        global_step: int,
    ) -> dict[str, torch.Tensor] | None:
        device = next(self.model.parameters()).device
        inputs = self.processor.preprocess(batch, device=device)
        real_A = inputs["source_image"]
        real_B = inputs["target_image"]
        with torch.no_grad():
            fake_B = self.model(source_image=real_A).pred_image
        return {
            "source": real_A,
            "pred": fake_B,
            "target": real_B,
        }

    def replace_networks(self, networks: dict[str, nn.Module]) -> None:
        """Swap prepared networks back into the model wrapper."""
        if "model" in networks:
            self.model = networks["model"]
        else:
            if "G" in networks:
                self.model.denoise_fn = networks["G"]
            if "F" in networks:
                self.model.netF = networks["F"]
            if "D" in networks and self.model.netD is not None:
                self.model.netD = networks["D"]
            if "E" in networks and self.model.netE is not None:
                self.model.netE = networks["E"]

    # ------------------------------------------------------------------
    # Forward / losses (ported from SBModel.forward / compute_*_loss)
    # ------------------------------------------------------------------
    def _sb_forward(
        self,
        real_A: torch.Tensor,
        real_B: torch.Tensor,
        real_A2: torch.Tensor,
    ) -> dict[str, Any]:
        """Run the SB training forward pass.

        Rolls out the no-grad Markov bridge from t=0 to a randomly sampled
        ``time_idx`` for the main stream, the reference stream and (when
        ``nce_idt``) the identity stream, then produces the trainable
        one-step predictions.
        """
        model = self.model
        bs = real_A.size(0)
        device = real_A.device
        times = model.times
        tau = model.tau
        ngf = model.ngf

        time_idx = torch.randint(model.num_timesteps, size=[1], device=device).long()

        with torch.no_grad():
            model.netG.eval()
            Xt = real_A
            Xt2 = real_A2
            XtB = real_B if model.nce_idt else None
            Xt_1 = Xt_12 = Xt_1B = None
            for t in range(int(time_idx.item()) + 1):
                if t > 0:
                    delta = times[t] - times[t - 1]
                    denom = times[-1] - times[t - 1]
                    inter = (delta / denom).reshape(-1, 1, 1, 1)
                    scale = (delta * (1 - delta / denom)).reshape(-1, 1, 1, 1)
                    Xt = (
                        (1 - inter) * Xt
                        + inter * Xt_1.detach()
                        + (scale * tau).sqrt() * torch.randn_like(Xt)
                    )
                    Xt2 = (
                        (1 - inter) * Xt2
                        + inter * Xt_12.detach()
                        + (scale * tau).sqrt() * torch.randn_like(Xt2)
                    )
                    if model.nce_idt:
                        XtB = (
                            (1 - inter) * XtB
                            + inter * Xt_1B.detach()
                            + (scale * tau).sqrt() * torch.randn_like(XtB)
                        )
                t_idx = t * torch.ones(bs, dtype=torch.long, device=device)
                z = torch.randn(bs, 4 * ngf, device=device)
                Xt_1 = model.netG(Xt, t_idx, z)
                z = torch.randn(bs, 4 * ngf, device=device)
                Xt_12 = model.netG(Xt2, t_idx, z)
                if model.nce_idt:
                    z = torch.randn(bs, 4 * ngf, device=device)
                    Xt_1B = model.netG(XtB, t_idx, z)
            model.netG.train()
            real_A_noisy = Xt.detach()
            real_A_noisy2 = Xt2.detach()
            XtB_out = XtB.detach() if model.nce_idt else None

        if model.nce_idt:
            z_in = torch.randn(2 * bs, 4 * ngf, device=device)
            realt = torch.cat((real_A_noisy, XtB_out), dim=0)
        else:
            z_in = torch.randn(bs, 4 * ngf, device=device)
            realt = real_A_noisy
        z_in2 = torch.randn(bs, 4 * ngf, device=device)

        fake = model.netG(realt, time_idx, z_in)
        fake_B = fake[:bs]
        idt_B = fake[bs:] if model.nce_idt else None
        fake_B2 = model.netG(real_A_noisy2, time_idx, z_in2)

        return {
            "time_idx": time_idx,
            "real_A_noisy": real_A_noisy,
            "real_A_noisy2": real_A_noisy2,
            "fake_B": fake_B,
            "idt_B": idt_B,
            "fake_B2": fake_B2,
        }

    def _compute_d_loss(
        self, out: dict[str, Any], real_B: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """LSGAN PatchGAN loss over real_B vs. detached fake_B."""
        gan_loss = self._find_gan_loss()
        time_idx = out["time_idx"]
        pred_fake = self.model.netD(out["fake_B"].detach(), time_idx)
        loss_D_fake = gan_loss(prediction=pred_fake, target_is_real=False)
        pred_real = self.model.netD(real_B, time_idx)
        loss_D_real = gan_loss(prediction=pred_real, target_is_real=True)
        loss_D = (loss_D_fake + loss_D_real) * 0.5
        return loss_D, loss_D_real, loss_D_fake

    def _compute_e_loss(self, out: dict[str, Any]) -> torch.Tensor:
        """SB energy loss: same-pair logits vs. cross-pair logsumexp."""
        model = self.model
        time_idx = out["time_idx"]
        XtXt_1 = torch.cat([out["real_A_noisy"], out["fake_B"].detach()], dim=1)
        XtXt_2 = torch.cat([out["real_A_noisy2"], out["fake_B2"].detach()], dim=1)
        temp = torch.logsumexp(
            model.netE(XtXt_1, time_idx, XtXt_2).reshape(-1), dim=0
        ).mean()
        loss_E = (
            -model.netE(XtXt_1, time_idx, XtXt_1).mean() + temp + temp**2
        )
        return loss_E

    def _compute_g_loss(
        self,
        out: dict[str, Any],
        real_A: torch.Tensor,
        real_B: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Generator loss: GAN + SB coupling + PatchNCE (+ identity NCE)."""
        model = self.model
        gan_loss = self._find_gan_loss()
        time_idx = out["time_idx"]
        fake_B = out["fake_B"]
        fake_B2 = out["fake_B2"]
        losses: dict[str, torch.Tensor] = {}

        loss_G_GAN = (
            gan_loss(prediction=model.netD(fake_B, time_idx), target_is_real=True)
            * model.lambda_GAN
        )
        losses["loss_G_GAN"] = loss_G_GAN

        XtXt_1 = torch.cat([out["real_A_noisy"], fake_B], dim=1)
        XtXt_2 = torch.cat([out["real_A_noisy2"], fake_B2], dim=1)
        ET_XY = model.netE(XtXt_1, time_idx, XtXt_1).mean() - torch.logsumexp(
            model.netE(XtXt_1, time_idx, XtXt_2).reshape(-1), dim=0
        )
        loss_SB = (
            -(model.num_timesteps - time_idx[0])
            / model.num_timesteps
            * model.tau
            * ET_XY
        )
        loss_SB = loss_SB + model.tau * torch.mean(
            (out["real_A_noisy"] - fake_B) ** 2
        )
        losses["loss_SB"] = loss_SB

        loss_NCE = self._calculate_nce_loss(real_A, fake_B, time_idx)
        losses["loss_NCE"] = loss_NCE
        if model.nce_idt and model.lambda_NCE > 0.0:
            loss_NCE_Y = self._calculate_nce_loss(real_B, out["idt_B"], time_idx)
            losses["loss_NCE_Y"] = loss_NCE_Y
            loss_NCE_both = (loss_NCE + loss_NCE_Y) * 0.5
        else:
            loss_NCE_both = loss_NCE

        loss_G = (
            loss_G_GAN
            + model.lambda_SB * loss_SB
            + model.lambda_NCE * loss_NCE_both
        )
        losses["loss_G"] = loss_G
        return loss_G, losses

    def _calculate_nce_loss(
        self, src: torch.Tensor, tgt: torch.Tensor, time_idx: torch.Tensor
    ) -> torch.Tensor:
        """Multi-layer PatchNCE loss between ``src`` and ``tgt`` features.

        Features are extracted from the generator encoder at time index 0
        with a shared latent code, as in the original repo.
        """
        model = self.model
        bs = src.size(0)
        n_layers = len(model.nce_layers)
        z = torch.randn(bs, 4 * model.ngf, device=src.device)
        feat_q = model.netG(tgt, time_idx * 0, z, model.nce_layers, encode_only=True)
        feat_k = model.netG(src, time_idx * 0, z, model.nce_layers, encode_only=True)
        feat_k_pool, sample_ids = model.netF(feat_k, model.num_patches, None)
        feat_q_pool, _ = model.netF(feat_q, model.num_patches, sample_ids)

        crit = self._find_patchnce_loss()
        total_nce_loss: torch.Tensor | float = 0.0
        for f_q, f_k in zip(feat_q_pool, feat_k_pool):
            # NOTE: the original repo multiplies each layer's loss by
            # lambda_NCE here AND multiplies by lambda_NCE again when summing
            # into loss_G (effective weight lambda_NCE ** 2). We apply
            # lambda_NCE only once, in loss_G; this comment documents the fix.
            total_nce_loss = total_nce_loss + crit(f_q, f_k)
        return total_nce_loss / n_layers

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _set_requires_grad(
        nets: list[nn.Module | None], requires_grad: bool = False
    ) -> None:
        """Toggle ``requires_grad`` on the given networks."""
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def _find_gan_loss(self) -> GANLoss:
        """Extract the GANLoss component from the configured composite loss."""
        if isinstance(self.loss_fn, GANLoss):
            return self.loss_fn
        if isinstance(self.loss_fn, CompositeLoss):
            for loss_fn, _ in self.loss_fn.losses:
                if isinstance(loss_fn, GANLoss):
                    return loss_fn
        raise RuntimeError("No GANLoss found in the configured loss function")

    def _find_patchnce_loss(self) -> PatchNCELoss:
        """Extract the PatchNCELoss component from the composite loss."""
        if isinstance(self.loss_fn, PatchNCELoss):
            return self.loss_fn
        if isinstance(self.loss_fn, CompositeLoss):
            for loss_fn, _ in self.loss_fn.losses:
                if isinstance(loss_fn, PatchNCELoss):
                    return loss_fn
        raise RuntimeError("No PatchNCELoss found in the configured loss function")


__all__ = ["BaseSBModel", "BaseSBStrategy"]
