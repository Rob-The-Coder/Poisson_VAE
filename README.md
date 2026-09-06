# Poisson VAE — Beyond Gaussian Priors for Variational Autoencoders

A Variational Autoencoder for face generation (CelebA) whose latent space is modeled as a **discrete Poisson distribution** parameterized by a vector of non-negative intensity rates λ instead of the conventional continuous Gaussian. A Streamlit interface is included for training, generation and latent-space analysis.

This repository implements the experiments from my Master's thesis, *"Beyond Gaussian Priors: A Poisson Latent Space for Variational Autoencoders"*, University of Pavia, Dept. of Electrical, Computer and Biomedical Engineering (Supervisor: Prof. Claudio Cusano, Co-Supervisor: Dr. Greta Cabassa).

---

## Motivation

Camera sensors capture images by counting discrete photons, a physical process governed by Poisson statistics rather than a Gaussian one. This project asks whether replacing the VAE's standard Gaussian prior with a Poisson prior can still support a well-structured, semantically meaningful latent space.

Since sampling from a Poisson distribution is discrete and **non-differentiable**, it doesn't admit the standard reparametrization trick used to train VAEs. Two alternative gradient estimation strategies are implemented and evaluated against a Gaussian baseline:

- **PGA - Straight-Through Estimator (Poisson Gradient Approximation)**: samples `z ~ Poisson(λ)` in the forward pass, then backpropagates through a Gaussian surrogate gradient, exploiting the fact that a Poisson distribution is asymptotically well-approximated by `N(λ, λ)`.
- **RLT - Score Function Estimator**: treats sampling as a stochastic policy and estimates gradients via REINFORCE, using multiple Monte Carlo samples per input and a leave-one-out baseline for variance reduction.
- **GRT - Gaussian Reparametrization Trick**: the standard VAE baseline, included for comparison.

## Architecture

An **asymmetric encoder–decoder** design, since inference and generation are treated as fundamentally different problems:

- **Encoder**: inspired by **EfficientNet**, using downsampling blocks followed by Mobile Inverted Bottleneck (MBConv) blocks with Squeeze-and-Excitation attention, compressing a 64×64 face into a vector of strictly positive Poisson rates λ (enforced via an exponential output activation).
- **Decoder**: inspired by **SRGAN**, replacing transposed convolutions with sub-pixel (pixel-shuffle) upsampling to avoid checkerboard artifacts and conditioned on the latent code at every resolution scale via **AdaIN-based style modulation** (StyleGAN-style), rather than injecting it only once at the input.
- Three parameter-count configurations are compared: **36M**, **53M** and **60M**, obtained by scaling channel widths rather than topology.

## Training

- **Objective**: L1 reconstruction loss + analytical KL divergence between two Poisson distributions (or the standard Gaussian KL for the GRT baseline).
- **Posterior collapse mitigation**: a sigmoidal KL-annealing schedule (first ~60% of training) combined with a free-bits floor per latent dimension.
- **Optimizer**: AdamW, lr = 1e-4, batch size 128, trained for 100–600 epochs depending on configuration.
- **Dataset**: [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), center-cropped and resized to 64×64, official train/validation split (~162k / ~20k images).
- **Hardware**: single NVIDIA Titan XP (University of Pavia lab workstation).

## Results

Generation quality was measured via **Fréchet Inception Distance (FID)** on 10k real vs. 10k generated images:

| Method | Config | FID (best) |
|---|---|---|
| GRT (Gaussian baseline) | 60M | **75.5** |
| PGA (straight-through, Poisson) | 36M | 124.6 |
| RLT (score function, Poisson) | 60M | 385.3 |

**Key findings:**
- The straight-through estimator (PGA) achieves **competitive generation quality** with a discrete Poisson latent space, producing sharper facial details than the Gaussian baseline at the cost of some sample diversity.
- The score function estimator (RLT) consistently **underperforms**, hampered by high gradient variance and slower convergence, a known limitation of REINFORCE-style estimators.
- The optimal KL-rescaling factor differs by **four orders of magnitude** between GRT (1e-6) and PGA (1e-2), showing that the discrete Poisson prior fundamentally changes the optimization landscape.
- **Generation quality is bottlenecked by encoder capacity, not total parameter count**: the 53M model (encoder ~7M / decoder ~46M) underperforms the smaller, more balanced 36M model.
- Latent traversals, UMAP clustering and attribute-direction arithmetic (e.g. combining *Smiling + Young*, or *Male + No_Beard*) show that the Poisson latent space supports **structured, semantically meaningful and linearly composable** representations, though with a distinct and more irregular geometry compared to the Gaussian latent space.

## Tech stack

Python · PyTorch (custom `autograd.Function` gradient estimators, AMP) · Streamlit · UMAP · torcheval (FID) · pandas · Rich

## Project structure

```
Poisson_Gradient_Approximation/
├── vae/
│   ├── vae.py            → VAE model, dispatches forward/loss/generation logic per sampling strategy
│   ├── trainer.py         → training loop, checkpointing, optimizer handling
│   ├── encoders/          → EfficientNet-inspired encoders (36M / 53M / 60M, PGA & GRT variants)
│   └── decoders/          → SRGAN + AdaIN-inspired decoders
├── core/
│   ├── model_factory.py   → maps (sampling, model size) → encoder/decoder pair
│   ├── model_args.py      → paths & filenames for persistence
│   └── vae_output.py      → structured model output
├── utils/
│   ├── dataset.py         → CelebA dataset & dataloaders
│   ├── sampling.py        → PGA / GRT / RLT gradient estimators
│   └── loss.py            → matching ELBO losses (incl. KL annealing, free bits)
├── gui/                   → Streamlit pages for training & generation
├── train_vae.py           → CLI entry point for training
└── generate_faces.py      → CLI entry point for generation, FID & latent-space analysis
```

## Usage

**Training**
```bash
python train_vae.py --images_dir /path/to/celeba --project_dir . --vae_filename VAE.pt --vae_checkpoint VAE_checkpoint.pt --sampling PGA --type 36M --epochs 300
```

**Generation & analysis**
```bash
python generate_faces.py --images_dir /path/to/celeba --project_dir . --vae_filename VAE.pt --num_faces 36
```

**Interactive GUI**
```bash
streamlit run gui.py
```

## Author

Roberto Schifano. Master's Thesis, University of Pavia, A.Y. 2025/2026
