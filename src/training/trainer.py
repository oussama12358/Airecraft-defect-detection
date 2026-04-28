import torch
import mlflow
from tqdm import tqdm
from pathlib import Path


class Trainer:
    def __init__(self, model, optimizer, criterion, scheduler, device, cfg):
        self.model     = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device    = device
        self.cfg       = cfg
        Path(cfg["paths"]["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)

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

            total_loss += loss.item()
            correct    += (logits.argmax(1) == labels).sum().item()

        return total_loss / len(loader), correct / len(loader.dataset)

    @torch.no_grad()
    def eval_epoch(self, loader):
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
        ckpt_path        = f"{self.cfg['paths']['checkpoint_dir']}/best_{self.cfg['model']['name']}.pt"

        mlflow.set_tracking_uri(self.cfg["paths"]["mlflow_uri"])

        with mlflow.start_run():
            mlflow.log_params({
                "model":    self.cfg["model"]["name"],
                "epochs":   self.cfg["training"]["epochs"],
                "lr":       self.cfg["training"]["learning_rate"],
                "batch":    self.cfg["training"]["batch_size"],
            })

            for epoch in range(self.cfg["training"]["epochs"]):
                print(f"\nEpoch [{epoch+1}/{self.cfg['training']['epochs']}]")

                train_loss, train_acc = self.train_epoch(train_loader)
                val_loss,   val_acc   = self.eval_epoch(val_loader)

                self.scheduler.step(val_loss)

                mlflow.log_metrics({
                    "train_loss": train_loss, "train_acc": train_acc,
                    "val_loss":   val_loss,   "val_acc":   val_acc,
                }, step=epoch)

                print(
                    f"  train_loss={train_loss:.4f}  train_acc={train_acc:.4f}"
                    f"  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
                )

                if val_acc > best_val_acc:
                    best_val_acc     = val_acc
                    patience_counter = 0
                    torch.save(self.model.state_dict(), ckpt_path)
                    print(f"  ✓ New best saved → {ckpt_path}")
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"\n  Early stopping at epoch {epoch+1}")
                        break

        print(f"\nTraining complete. Best val_acc = {best_val_acc:.4f}")
        return best_val_acc