import json
from contextlib import nullcontext

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import torch
from tqdm import tqdm
from pathlib import Path

from src.training.ema import EMAScheduler


class Trainer:
    def __init__(self, model, optimizer, criterion, scheduler, device, cfg):
        self.model     = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device    = device
        self.cfg       = cfg
        self.history   = []
        self.ema       = None
        if cfg["training"].get("use_ema", False):
            self.ema = EMAScheduler(
                self.model,
                decay=cfg["training"].get("ema_decay", 0.999),
                device=device,
            )
        Path(cfg["paths"]["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)
        Path(cfg["paths"]["reports_dir"]).mkdir(parents=True, exist_ok=True)

    # ── Single epoch ──────────────────────────────────────────────────────────
    def train_epoch(self, loader):
        self.model.train()
        total_loss, correct = 0.0, 0

        for imgs, labels in tqdm(loader, desc="  Train", leave=False):
            imgs, labels = imgs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(imgs)
            loss   = self.criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            if self.ema is not None:
                self.ema.update()

            total_loss += loss.item()
            correct    += (logits.argmax(1) == labels).sum().item()

        return total_loss / len(loader), correct / len(loader.dataset)

    @torch.no_grad()
    def eval_epoch(self, loader):
        context = self.ema if self.ema is not None else nullcontext()
        with context:
            self.model.eval()
            total_loss, correct = 0.0, 0

            for imgs, labels in tqdm(loader, desc="  Val  ", leave=False):
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                logits = self.model(imgs)
                total_loss += self.criterion(logits, labels).item()
                correct    += (logits.argmax(1) == labels).sum().item()

        return total_loss / len(loader), correct / len(loader.dataset)

    # ── Full training loop ────────────────────────────────────────────────────
    def fit(self, train_loader, val_loader):
        best_val_acc     = 0.0
        patience_counter = 0
        patience         = self.cfg["training"]["early_stopping_patience"]
        run_name         = self.cfg["training"].get("run_name", self.cfg["model"]["name"])
        safe_run_name    = run_name.replace("/", "_").replace("\\", "_")
        ckpt_path        = Path(self.cfg["paths"]["checkpoint_dir"]) / f"best_{safe_run_name}.pt"

        mlflow.set_tracking_uri(self.cfg["paths"]["mlflow_uri"])

        with mlflow.start_run():
            mlflow.log_params({
                "model":    self.cfg["model"]["name"],
                "epochs":   self.cfg["training"]["epochs"],
                "lr":       self.cfg["training"]["learning_rate"],
                "batch":    self.cfg["training"]["batch_size"],
                "run_name": run_name,
                "train_csv": self.cfg["data"]["train_csv"],
                "val_csv": self.cfg["data"]["val_csv"],
                "use_ema": self.cfg["training"].get("use_ema", False),
                "ema_decay": self.cfg["training"].get("ema_decay", None),
            })

            for epoch in range(self.cfg["training"]["epochs"]):
                print(f"\nEpoch [{epoch+1}/{self.cfg['training']['epochs']}]")

                train_loss, train_acc = self.train_epoch(train_loader)
                val_loss,   val_acc   = self.eval_epoch(val_loader)

                if self.scheduler.__class__.__name__ == "ReduceLROnPlateau":
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

                mlflow.log_metrics({
                    "train_loss": train_loss, "train_acc": train_acc,
                    "val_loss":   val_loss,   "val_acc":   val_acc,
                }, step=epoch)

                print(
                    f"  train_loss={train_loss:.4f}  train_acc={train_acc:.4f}"
                    f"  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
                )

                self.history.append({
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                })

                if val_acc > best_val_acc:
                    best_val_acc     = val_acc
                    patience_counter = 0
                    if self.ema is not None:
                        with self.ema:
                            torch.save(self.model.state_dict(), ckpt_path)
                    else:
                        torch.save(self.model.state_dict(), ckpt_path)
                    print(f"  New best saved -> {ckpt_path}")
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"\n  Early stopping at epoch {epoch+1}")
                        break

        self._save_training_artifacts(safe_run_name)
        print(f"\nTraining complete. Best val_acc = {best_val_acc:.4f}")
        return best_val_acc

    def _save_training_artifacts(self, run_name: str):
        if not self.history:
            return

        reports_dir = Path(self.cfg["paths"]["reports_dir"])
        history_df = pd.DataFrame(self.history)

        history_path = reports_dir / f"training_history_{run_name}.csv"
        history_df.to_csv(history_path, index=False)

        best_row = history_df.loc[history_df["val_acc"].idxmax()]
        last_row = history_df.iloc[-1]
        summary = {
            "run_name": run_name,
            "best_epoch": int(best_row["epoch"]),
            "best_val_acc": float(best_row["val_acc"]),
            "best_train_acc": float(best_row["train_acc"]),
            "best_acc_gap": float(best_row["train_acc"] - best_row["val_acc"]),
            "last_epoch": int(last_row["epoch"]),
            "last_train_acc": float(last_row["train_acc"]),
            "last_val_acc": float(last_row["val_acc"]),
            "last_acc_gap": float(last_row["train_acc"] - last_row["val_acc"]),
        }

        summary_path = reports_dir / f"training_summary_{run_name}.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 5))
        ax_loss.plot(history_df["epoch"], history_df["train_loss"], label="train")
        ax_loss.plot(history_df["epoch"], history_df["val_loss"], label="val")
        ax_loss.set_title("Loss")
        ax_loss.set_xlabel("Epoch")
        ax_loss.legend()

        ax_acc.plot(history_df["epoch"], history_df["train_acc"], label="train")
        ax_acc.plot(history_df["epoch"], history_df["val_acc"], label="val")
        ax_acc.set_title("Accuracy")
        ax_acc.set_xlabel("Epoch")
        ax_acc.set_ylim(0, 1.05)
        ax_acc.legend()

        plt.suptitle(f"Training Curves - {run_name}")
        plt.tight_layout()
        curves_path = reports_dir / f"training_curves_{run_name}.png"
        plt.savefig(curves_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"[Trainer] History saved -> {history_path}")
        print(f"[Trainer] Summary saved -> {summary_path}")
        print(f"[Trainer] Curves saved -> {curves_path}")
