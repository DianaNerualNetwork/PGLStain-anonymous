# Attribution and third-party notices

This anonymous artifact retains the source framework's Apache-2.0 license
text in `LICENSE`. That license does not override the terms of adapted
third-party code or external pretrained weights. First-party identifying
metadata and development history are omitted for review; upstream attribution
is intentionally retained.

The following provenance is carried forward from the source implementation.
It is not a claim that every method has the same license or that all upstream
repositories have been independently re-audited for this export.

| Component | Upstream source |
| --- | --- |
| CUT and shared contrastive networks | [CUT / FastCUT](https://github.com/taesungp/contrastive-unpaired-translation) |
| CycleGAN and shared translation networks | [CycleGAN / pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) |
| PyramidP2P | [PyramidPix2pix / BCI](https://github.com/bupt-ai-cz/BCI) |
| CPT and ASP | [Adaptive Supervised PatchNCE](https://github.com/lifangda01/AdaptiveSupervisedPatchNCE) |
| PSPStain | [PSPStain](https://github.com/ccitachi/PSPStain) |
| M2PL-GAN | [M2PL-GAN](https://github.com/Pikachu-one/M2PL-GAN) |
| PPT | [PPT, MICCAI 2024](https://link.springer.com/chapter/10.1007/978-3-031-72083-3_17) |
| MDCL | [Mix-Domain Contrastive Learning](https://github.com/SSongWang/Mix-DomainContrastiveLearning) |
| SIMGAN | [SIM-GAN](https://github.com/xianchaoguan/SIM-GAN) |
| USIGAN | [USIGAN](https://github.com/MIXAILAB/USIGAN) |
| UNSB | See the original provenance and paper reference in `puzzlestain/models/diffusion_family/unsb.py` |
| Extracted image replay buffer | [NVIDIA pix2pixHD](https://github.com/NVIDIA/pix2pixHD), retained through the source framework |

The source notices identify USIGAN as non-commercial (CC BY-NC-SA 4.0).
Consult its upstream terms, and the terms of each other method, before
redistribution or use. Upstream license/copyright notices in source headers
have not been removed. CUT/CycleGAN lineage includes the work of Taesung Park
and Jun-Yan Zhu; the replay-buffer lineage includes NVIDIA's pix2pixHD code.

No datasets or third-party pretrained checkpoints are distributed. The two
PGLStain Full generators are provided for inference; see docs/CHECKPOINTS.md.
Downloads initiated by torchvision or other libraries remain subject to the
providers' terms.
The optional nuclei preprocessor derives from TDKStain and uses Cellpose;
it is not required by the supplied comparison recipes.
