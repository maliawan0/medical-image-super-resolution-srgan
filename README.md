# Medical Image Super-Resolution — SRGAN Baseline

An SRGAN implementation for super-resolving multi-modality medical images. This is the **adversarial baseline** for my Final Year Project, built before moving to the diffusion approach in [medical-image-super-resolution-diffusion](https://github.com/maliawan0/medical-image-super-resolution-diffusion).

## Why it's here

SRGAN produces sharp output quickly, which is exactly why it's a risky fit for medical imaging: the adversarial loss rewards *plausible* texture, and plausible texture in a scan can mean detail that was never in the patient. Training is also unstable at the resolutions this project needed. Both observations pushed the FYP toward diffusion — this repo is kept as the documented comparison point rather than deleted.

## Components

| File | Role |
|---|---|
| [`generator.py`](generator.py) | Residual-block generator that upsamples the low-resolution input |
| [`discriminator.py`](discriminator.py) | Convolutional discriminator scoring real vs generated |
| [`feature_extractor.py`](feature_extractor.py) | Pretrained VGG feature space for perceptual loss |
| [`dataset.py`](dataset.py) | Paired high-/low-resolution loader for the multi-modality dataset |
| [`trainer.py`](trainer.py) | Adversarial training loop |

## Loss

The generator is trained against a combination of:

- **Content loss** — MSE in VGG feature space rather than pixel space, so the objective tracks perceptual similarity
- **Adversarial loss** — from the discriminator

## Usage

```bash
python trainer.py
```

Point the dataset loader at your paired HR/LR directories before running.

## Related

- [medical-image-super-resolution-diffusion](https://github.com/maliawan0/medical-image-super-resolution-diffusion) — the diffusion model this baseline was compared against, and the approach the FYP settled on
