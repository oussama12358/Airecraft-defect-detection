import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


DEFAULT_SOURCE_CSVS = [
    "data/splits/train.csv",
    "data/splits/val.csv",
    "data/splits/test.csv",
]


def load_source_records(source_csvs: list[str]) -> pd.DataFrame:
    frames = []
    for csv_path in source_csvs:
        path = Path(csv_path)
        if not path.exists():
            print(f"[KFold] Skipping missing source CSV: {path}")
            continue

        df = pd.read_csv(path)
        missing_cols = {"filename", "label"} - set(df.columns)
        if missing_cols:
            raise ValueError(f"{path} is missing required columns: {sorted(missing_cols)}")
        frames.append(df[["filename", "label"]].copy())

    if not frames:
        raise RuntimeError("No source CSV files were found.")

    df = pd.concat(frames, ignore_index=True)
    conflicts = df.groupby("filename")["label"].nunique()
    conflicts = conflicts[conflicts > 1]
    if not conflicts.empty:
        raise ValueError(
            "Found filenames with multiple labels. Examples: "
            f"{conflicts.index[:5].tolist()}"
        )

    return df.drop_duplicates(subset=["filename"]).reset_index(drop=True)


def split_train_val(
    train_val_df: pd.DataFrame,
    val_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if val_ratio <= 0:
        return train_val_df.reset_index(drop=True), train_val_df.iloc[0:0].copy()

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=val_ratio,
        random_state=seed,
    )
    train_idx, val_idx = next(
        splitter.split(train_val_df["filename"], train_val_df["label"])
    )
    return (
        train_val_df.iloc[train_idx].reset_index(drop=True),
        train_val_df.iloc[val_idx].reset_index(drop=True),
    )


def add_summary_rows(rows: list[dict], fold: int, split: str, df: pd.DataFrame) -> None:
    counts = df["label"].value_counts().sort_index().to_dict()
    row = {"fold": fold, "split": split, "samples": len(df)}
    row.update(counts)
    rows.append(row)


def prepare_kfold_splits(
    source_csvs: list[str],
    output_dir: str,
    n_splits: int,
    val_ratio: float,
    seed: int,
) -> None:
    df = load_source_records(source_csvs)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    summary_rows = []

    for fold, (train_val_idx, test_idx) in enumerate(
        splitter.split(df["filename"], df["label"])
    ):
        fold_dir = output_root / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        train_val_df = df.iloc[train_val_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)
        train_df, val_df = split_train_val(
            train_val_df,
            val_ratio=val_ratio,
            seed=seed + fold,
        )

        train_df.to_csv(fold_dir / "train.csv", index=False)
        val_df.to_csv(fold_dir / "val.csv", index=False)
        test_df.to_csv(fold_dir / "test.csv", index=False)

        add_summary_rows(summary_rows, fold, "train", train_df)
        add_summary_rows(summary_rows, fold, "val", val_df)
        add_summary_rows(summary_rows, fold, "test", test_df)

        print(
            f"[KFold] fold_{fold}: "
            f"train={len(train_df)} | val={len(val_df)} | test={len(test_df)}"
        )

    summary = pd.DataFrame(summary_rows).fillna(0)
    summary_path = output_root / "summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"[KFold] Summary saved to {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create stratified K-fold CSV splits without modifying images."
    )
    parser.add_argument(
        "--source_csv",
        nargs="*",
        default=DEFAULT_SOURCE_CSVS,
        help="Source CSV files to combine before K-fold splitting",
    )
    parser.add_argument("--output_dir", default="data/splits/kfold")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    prepare_kfold_splits(
        source_csvs=args.source_csv,
        output_dir=args.output_dir,
        n_splits=args.n_splits,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
