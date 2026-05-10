import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def parse_fold_index(fold_dir: Path) -> int:
    return int(fold_dir.name.split("_")[-1])


def discover_folds(folds_dir: str, selected_folds: list[int] | None) -> list[Path]:
    root = Path(folds_dir)
    fold_dirs = sorted(root.glob("fold_*"), key=parse_fold_index)

    if selected_folds is not None:
        wanted = set(selected_folds)
        fold_dirs = [p for p in fold_dirs if parse_fold_index(p) in wanted]

    if not fold_dirs:
        raise RuntimeError(
            f"No fold directories found in {root}. "
            "Run scripts/prepare_kfold_splits.py first."
        )

    return fold_dirs


def run_command(cmd: list[str], dry_run: bool) -> None:
    print("[KFold] " + " ".join(cmd))

    if not dry_run:
        subprocess.run(cmd, check=True)


def collect_summary(model: str, folds: list[Path], reports_dir: str) -> pd.DataFrame:
    rows = []

    for fold_dir in folds:
        fold = parse_fold_index(fold_dir)
        run_name = f"{model}_fold_{fold}"
        report_path = Path(reports_dir) / f"{run_name}_report.json"

        if not report_path.exists():
            continue

        with open(report_path, "r") as f:
            report = json.load(f)

        metrics = report.get("metrics", {})

        rows.append({
            "fold": fold,
            "run_name": run_name,
            "accuracy": metrics.get("accuracy"),
            "test_csv": metrics.get("test_csv"),
        })

    summary = pd.DataFrame(rows)

    if summary.empty:
        return summary

    summary.loc["mean"] = {
        "fold": "mean",
        "run_name": "",
        "accuracy": summary["accuracy"].mean(),
        "test_csv": "",
    }

    summary.loc["std"] = {
        "fold": "std",
        "run_name": "",
        "accuracy": summary["accuracy"].std(ddof=0),
        "test_csv": "",
    }

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate one model across prepared K-fold splits."
    ) 

    parser.add_argument(
        "--folds_dir",
        default="data/splits/kfold"
    )

    parser.add_argument(
        "--model",
        choices=["baseline_cnn", "resnet50", "efficientnet_b3"],
        default="resnet50"
    )

    parser.add_argument(
        "--fold",
        type=int,
        nargs="*",
        default=None,
        help="Optional subset of fold numbers to run"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None
    )

    parser.add_argument(
        "--use_ema",
        action="store_true"
    )

    parser.add_argument(
        "--ema_decay",
        type=float,
        default=None
    )

    parser.add_argument(
        "--reports_dir",
        default="reports"
    )

    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without training"
    )

    args = parser.parse_args()

    folds = discover_folds(args.folds_dir, args.fold)

    for fold_dir in folds:

        fold = parse_fold_index(fold_dir)
        run_name = f"{args.model}_fold_{fold}"

        checkpoint = Path("checkpoints") / f"best_{run_name}.pt"

        # Skip completed folds
        if checkpoint.exists():
            print(f"[KFold] Skipping fold {fold} (checkpoint exists)")
            continue

        train_cmd = [
            sys.executable,
            "train.py",
            "--model", args.model,
            "--train_csv", str(fold_dir / "train.csv"),
            "--val_csv", str(fold_dir / "val.csv"),
            "--run_name", run_name,
        ]

        if args.epochs is not None:
            train_cmd.extend(["--epochs", str(args.epochs)])

        if args.use_ema:
            train_cmd.append("--use_ema")

        if args.ema_decay is not None:
            train_cmd.extend(["--ema_decay", str(args.ema_decay)])

        eval_cmd = [
            sys.executable,
            "evaluate.py",
            "--checkpoint", str(checkpoint),
            "--model", args.model,
            "--test_csv", str(fold_dir / "test.csv"),
            "--report_name", run_name,
        ]

        run_command(train_cmd, args.dry_run)
        run_command(eval_cmd, args.dry_run)

    if args.dry_run:
        return

    summary = collect_summary(args.model, folds, args.reports_dir)

    if summary.empty:
        print("[KFold] No fold reports found yet.")
        return

    summary_path = Path(args.reports_dir) / f"kfold_summary_{args.model}.csv"

    summary.to_csv(summary_path, index=False)

    print(f"[KFold] Summary saved to {summary_path}")


if __name__ == "__main__":
    main()