"""
train.py :  CVAE multi-material drive belt
KL annealing + λ_reg schedule + early stopping + checkpointing
"""
from __future__ import annotations
import csv, time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as sched
from torch.utils.data import DataLoader, random_split

from config import Config
from cvae import CVAE, cvae_loss, count_parameters, miki_penalty
from dataset_dd import DDLaminateDataset, load_dataset

def _run_epoch(model, loader, optimizer, beta, cfg, device,
               lam_reg=0., lam_gen=0., lam_phys=0., mat_props=None):
    train = optimizer is not None
    model.train(train)
    ctx = torch.enable_grad() if train else torch.no_grad()

    tots = dict(loss=0., recon=0., kl=0., reg=0., gen=0., phys=0.)
    n = 0

    with ctx:
        for cond, feat in loader:
            cond, feat = cond.to(device), feat.to(device)
            x_hat, mu, log_var = model(feat, cond)

            # Generation pass : noise in the purely latent component
            mu_gen = x_gen = x_hat_gen = None
            # Generation pass : ONLY for the physical aspects of the new designs
            x_hat_gen = None
            if lam_phys > 0 and train:
                with torch.no_grad():
                    z_gen = torch.randn(feat.size(0), cfg.latent_dim, device=device)
                x_hat_gen = model.decode(z_gen, cond)

            # Calculation of the physical penalty (Miki)
            phys_pen = None
            if lam_phys > 0:
                phys_pen = miki_penalty(x_hat)
                if x_hat_gen is not None:
                    # Force the decoder to generate valid LPs from noise
                    phys_pen += miki_penalty(x_hat_gen)# Force the generation to be valid !
            
            # (Optional) Add buckling calculations here using mat_props and cond
            # if lam_phys > 0 and mat_props is not None:
            #     phys_pen += flambement_loss(x_hat, cond, mat_props)

            loss, recon, kl, reg, gen, phys = cvae_loss(
                x_hat, feat, mu, log_var, beta=beta,
                lam_reg=lam_reg, lam_gen=lam_gen, lam_phys=lam_phys,
                x_gen_target=x_gen, mu_gen=mu_gen,
                phys_penalty=phys_pen, feat_dim=cfg.feat_dim,
            )
            
            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            bs = feat.size(0)
            tots["loss"]  += loss.item()  * bs
            tots["recon"] += recon.item() * bs
            tots["kl"]    += kl.item()    * bs
            tots["reg"]   += reg.item()   * bs
            tots["gen"]   += gen.item()   * bs
            tots["phys"]  += phys.item()  * bs
            n += bs

    return {k: v / n for k, v in tots.items()}

def train(cfg: Config, mat_props=None) -> CVAE:
    Path(cfg.model_path).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.log_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    df = load_dataset(cfg.data_path)
    ds = DDLaminateDataset(df)
    n_val   = max(1, int(len(ds) * cfg.val_fraction))
    n_train = len(ds) - n_val
    ds_tr, ds_val = random_split(ds, [n_train, n_val],
                                generator=torch.Generator().manual_seed(cfg.seed))
    tr_ld = DataLoader(ds_tr,  batch_size=cfg.batch_size, shuffle=True,
                    num_workers=0, pin_memory=False)
    va_ld = DataLoader(ds_val, batch_size=cfg.batch_size * 2, shuffle=False,
                    num_workers=0, pin_memory=False)
    print(f"Train: {n_train}  |  Val: {n_val}")

    model = CVAE(feat_dim=cfg.feat_dim, cond_dim=cfg.cond_dim,
                latent_dim=cfg.latent_dim, hidden_dims=cfg.hidden_dims,
                dropout=cfg.dropout).to(device)
    print(f"CVAE parameters : {count_parameters(model):,}")

    # You mentioned wanting to increase the learning rate. You can force it here if necessary:
    # cfg.lr = 1e-3 or adjust it directly in your config.py file.
    opt = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched_lr = sched.ReduceLROnPlateau(opt, factor=0.5, patience=15, min_lr=1e-6)

    log_path = Path(cfg.log_dir) / "training_log.csv"
    fields   = ["epoch","beta","lam_reg", "lam_phys", "lr",
                 "train_loss","train_recon","train_kl","train_reg","train_gen", "train_phys",
                 "val_loss","val_recon","val_kl","val_reg", "val_phys"]
    lf  = log_path.open("w", newline="")
    wr  = csv.DictWriter(lf, fieldnames=fields); wr.writeheader()

    best_val = np.inf; patience_cnt = 0; PATIENCE = 300; t0 = time.time()

    for epoch in range(1, cfg.n_epochs + 1):
        # A reset is essential to prevent the system from shutting down when penalties are triggered
        if epoch == cfg.kl_anneal_epochs + 1:
            best_val = np.inf
            patience_cnt = 0
            print("\n--- End of Warm-up: Penalties activated (Generation & Physical) and Patience reset ---\n")
            # ---We're giving the learning rate another boost ---
            for param_group in opt.param_groups:
                param_group['lr'] = 5e-4

            sched_lr = sched.ReduceLROnPlateau(opt, factor=0.5, patience=15, min_lr=1e-6)
            print("\n--- End of Warm-up : Reset of the Patience AND the Learning Rate ---\n")

        beta    = min(cfg.beta_kl_max, cfg.beta_kl_max * epoch / max(1, cfg.kl_anneal_epochs))
        # lam_reg = min(0.05, 0.05 * epoch / max(1, cfg.kl_anneal_epochs))
        lam_reg=0.001 
        # Penalties take effect after the warm-up
        if epoch <= cfg.kl_anneal_epochs:
            lam_gen = 0.0
            lam_phys = 0.0
        else:
            # Gradually increase from 0% to 100% over 50 epochs
            progress = min(1.0, (epoch - cfg.kl_anneal_epochs) / 50.0)
            # lam_gen = 0.5 * progress
            lam_gen = 0.001
            lam_phys = 10.0 * progress

        tr  = _run_epoch(model, tr_ld, opt, beta, cfg, device, lam_reg, lam_gen, lam_phys, mat_props)
        val = _run_epoch(model, va_ld, None, beta, cfg, device, lam_reg, 0., lam_phys, mat_props) # not necessary for the validation
        
        sched_lr.step(val["loss"])
        lr_cur = opt.param_groups[0]["lr"]

        if val["loss"] < best_val - 1e-6:
            best_val = val["loss"]
            patience_cnt = 0
            torch.save({"epoch": epoch, "state_dict": model.state_dict(),
                        "val_loss": best_val, "cfg_dict": cfg.__dict__},
                       cfg.model_path)
        else:
            patience_cnt += 1

        wr.writerow(dict(epoch=epoch, beta=round(beta,4), lam_reg=round(lam_reg,4), lam_phys=round(lam_phys,4),
                        lr=lr_cur,
                        train_loss=round(tr["loss"],6), train_recon=round(tr["recon"],6),
                        train_kl=round(tr["kl"],6),     train_reg=round(tr["reg"],6),
                        train_gen=round(tr["gen"],6),   train_phys=round(tr["phys"],6),
                        val_loss=round(val["loss"],6),  val_recon=round(val["recon"],6),
                        val_kl=round(val["kl"],6),      val_reg=round(val["reg"],6),
                        val_phys=round(val["phys"],6)))

        if epoch % 10 == 0 or epoch == 1 or epoch == cfg.kl_anneal_epochs + 1:
            print(f"[{epoch:4d}/{cfg.n_epochs}] β={beta:.2f} λr={lam_reg:.2f} λp={lam_phys:.2f} "
                f"train {tr['loss']:.5f} (r={tr['recon']:.4f} "
                f"kl={tr['kl']:.4f} p={tr['phys']:.4f}) "
                f"val {val['loss']:.5f}  lr={lr_cur:.1e} "
                f"t={time.time()-t0:.0f}s"
                + ("  ← best" if patience_cnt == 0 else ""))

        if patience_cnt >= PATIENCE:
            print(f"Early stopping triggered at the epoch {epoch}.")
            break

    lf.close()
    ckpt = torch.load(cfg.model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    print(f"\nBest model : epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.6f}")
    return model

def load_model(path: str, cfg: Config, device: Optional[torch.device] = None) -> CVAE:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt  = torch.load(path, map_location=device, weights_only=False)
    model = CVAE(feat_dim=cfg.feat_dim, cond_dim=cfg.cond_dim,
                 latent_dim=cfg.latent_dim, hidden_dims=cfg.hidden_dims,
                 dropout=0.0).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"Best model loaded ← {path}  (epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.6f})")
    return model