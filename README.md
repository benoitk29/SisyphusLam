# DD-CVAE — Conception de Stratifiés Double-Double par CVAE Multi-Matériaux

> Génération et optimisation de stratifiés composites `[a/-a/b/-b]ₓ` guidées par un
> **Variational Auto-Encoder Conditionnel**.

---

## Entraînement du CVAE

### Stratégie d'entraînement en deux phases

**Phase 1 — Warm-up (epochs 1 à 150) :**
- beta croît linéairement de 0 à 0.2
- λ_phys = 0 (pas de pénalité Miki)
- Le modèle apprend la reconstruction et la structure latente

**Phase 2 — Activation des pénalités (epochs 151 à 300) :**
- beta = 0.2 (stable)
- λ_phys monte de 0 à 10 sur 50 epochs
- Reset de la patience et du learning rate (5×10⁻⁴)
- Le modèle apprend à générer des LP géométriquement valides

### Suivi de l'entraînement

```
[  10/300] beta=0.01 λr=0.00 λp=0.00  train 0.04521 (r=0.0421 kl=0.0234 p=0.0000)  val 0.04318  lr=1.0e-03  t=42s <= best
[ 150/300] beta=0.20 λr=0.00 λp=0.00  train 0.01823 ...
[ 151/300] beta=0.20 λr=0.00 λp=0.20  train 0.02104 ...  <= Activation pénalités
[ 300/300] beta=0.20 λr=0.00 λp=10.0  train 0.01654 ...
```

---


