# DD-CVAE — Design of Double-Double Laminates Using Multi-Material VAE

> Generation and Optimization of Composite Laminates `[a/-a/b/-b]ₓ` guided by a
> **Conditional Variational Autoencoder**.

---

## CVAE Training

### Two-Phase Training Strategy

**Phase 1 — Warm-up (epochs 1 to 150) :**
- beta increases linearly from 0 to 0.2
- λ_phys = 0 (no penalty, Miki)
- The model learns the reconstruction and the latent structure

**Phase 2 — Penalty Activation (Epochs 151–300) :**
- beta = 0.2 (stable)
- λ_phys increases from 0 to 10 over 50 epochs
- Reset patience and learning rate (5×10⁻⁴)
- The model learns to generate geometrically valid LP

### Training Tracking

```
[  10/300] beta=0.01 λr=0.00 λp=0.00  train 0.04521 (r=0.0421 kl=0.0234 p=0.0000)  val 0.04318  lr=1.0e-03  t=42s <= best
[ 150/300] beta=0.20 λr=0.00 λp=0.00  train 0.01823 ...
[ 151/300] beta=0.20 λr=0.00 λp=0.20  train 0.02104 ...  <= Activation pénalités
[ 300/300] beta=0.20 λr=0.00 λp=10.0  train 0.01654 ...
```

---


