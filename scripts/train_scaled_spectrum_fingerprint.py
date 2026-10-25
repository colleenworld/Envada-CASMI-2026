from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy import sparse
import torch
from torch import nn


OUTPUT_ROOT = Path(
    "data/processed/spectrum_fingerprint"
)

SEED = 42
DEFAULT_BATCH_SIZE = 256
DEFAULT_MAX_EPOCHS = 12
DEFAULT_PATIENCE = 3
LEARNING_RATE = 1e-3


class SpectrumFingerprintMLP(
    nn.Module
):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
    ) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                input_dim,
                1024,
            ),
            nn.ReLU(),
            nn.Dropout(
                0.15
            ),
            nn.Linear(
                1024,
                1024,
            ),
            nn.ReLU(),
            nn.Dropout(
                0.15
            ),
            nn.Linear(
                1024,
                output_dim,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.network(
            x
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--name",
        default="scaled150k",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )

    parser.add_argument(
        "--max-epochs",
        type=int,
        default=DEFAULT_MAX_EPOCHS,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=DEFAULT_PATIENCE,
    )

    args = parser.parse_args()

    if args.batch_size <= 0:
        parser.error(
            "--batch-size must be positive"
        )

    if args.max_epochs <= 0:
        parser.error(
            "--max-epochs must be positive"
        )

    if args.patience <= 0:
        parser.error(
            "--patience must be positive"
        )

    return args


def soft_jaccard(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    intersection = (
        probabilities
        * targets
    ).sum(
        dim=1
    )

    union = (
        probabilities
        + targets
        - probabilities
        * targets
    ).sum(
        dim=1
    )

    return (
        intersection
        / union.clamp_min(
            1e-8
        )
    )


def batches(
    row_count: int,
    batch_size: int,
    *,
    shuffle: bool,
    rng: np.random.Generator,
):
    indices = np.arange(
        row_count,
        dtype=np.int64,
    )

    if shuffle:
        rng.shuffle(
            indices
        )

    for start in range(
        0,
        row_count,
        batch_size,
    ):
        yield indices[
            start:
            start
            + batch_size
        ]


def dense_batch(
    matrix: sparse.csr_matrix,
    indices: np.ndarray,
) -> torch.Tensor:
    array = (
        matrix[
            indices
        ]
        .toarray()
        .astype(
            np.float32,
            copy=False,
        )
    )

    return torch.from_numpy(
        array
    )


def target_batch(
    targets: np.ndarray,
    indices: np.ndarray,
) -> torch.Tensor:
    array = np.asarray(
        targets[
            indices
        ],
        dtype=np.float32,
    )

    return torch.from_numpy(
        array
    )


def evaluate(
    model: nn.Module,
    x: sparse.csr_matrix,
    y: np.ndarray,
    criterion: nn.Module,
    device: torch.device,
    batch_size: int,
) -> tuple[
    float,
    float,
    float,
]:
    model.eval()

    total_loss = 0.0
    total_rows = 0
    jaccard_sum = 0.0
    bit_accuracy_sum = 0.0

    rng = np.random.default_rng(
        SEED
    )

    with torch.no_grad():
        for indices in batches(
            len(y),
            batch_size,
            shuffle=False,
            rng=rng,
        ):
            batch_x = dense_batch(
                x,
                indices,
            ).to(
                device
            )

            batch_y = target_batch(
                y,
                indices,
            ).to(
                device
            )

            logits = model(
                batch_x
            )

            loss = criterion(
                logits,
                batch_y,
            )

            probabilities = torch.sigmoid(
                logits
            )

            row_count = len(
                indices
            )

            total_loss += (
                float(
                    loss.item()
                )
                * row_count
            )

            total_rows += row_count

            jaccard_sum += float(
                soft_jaccard(
                    probabilities,
                    batch_y,
                ).sum()
            )

            predicted_bits = (
                probabilities
                >= 0.5
            )

            bit_accuracy_sum += float(
                (
                    predicted_bits
                    == batch_y.bool()
                )
                .float()
                .mean(
                    dim=1
                )
                .sum()
            )

    return (
        total_loss
        / max(
            total_rows,
            1,
        ),
        jaccard_sum
        / max(
            total_rows,
            1,
        ),
        bit_accuracy_sum
        / max(
            total_rows,
            1,
        ),
    )


def main() -> None:
    args = parse_args()

    root = (
        OUTPUT_ROOT
        / args.name
    )

    train_vectors_path = (
        root
        / "train_vectors.npz"
    )

    train_targets_path = (
        root
        / "train_targets.npy"
    )

    dev_vectors_path = (
        root
        / "dev_vectors.npz"
    )

    dev_targets_path = (
        root
        / "dev_targets.npy"
    )

    model_path = (
        root
        / "spectrum_to_morgan2048.pt"
    )

    for path in [
        train_vectors_path,
        train_targets_path,
        dev_vectors_path,
        dev_targets_path,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing scaled dataset artifact: {path}"
            )

    torch.manual_seed(
        SEED
    )

    np.random.seed(
        SEED
    )

    train_x = sparse.load_npz(
        train_vectors_path
    ).tocsr()

    dev_x = sparse.load_npz(
        dev_vectors_path
    ).tocsr()

    train_y = np.load(
        train_targets_path,
        mmap_mode="r",
        allow_pickle=False,
    )

    dev_y = np.load(
        dev_targets_path,
        mmap_mode="r",
        allow_pickle=False,
    )

    if (
        train_x.shape[
            0
        ]
        != len(
            train_y
        )
        or dev_x.shape[
            0
        ]
        != len(
            dev_y
        )
    ):
        raise RuntimeError(
            "Vector/target row counts do not match"
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device,
    )
    print(
        "Training rows:",
        f"{len(train_y):,}",
    )
    print(
        "Development rows:",
        f"{len(dev_y):,}",
    )
    print(
        "Input dimensions:",
        train_x.shape[
            1
        ],
    )
    print(
        "Fingerprint bits:",
        train_y.shape[
            1
        ],
    )

    model = SpectrumFingerprintMLP(
        input_dim=train_x.shape[
            1
        ],
        output_dim=train_y.shape[
            1
        ],
    ).to(
        device
    )

    positive_rate = float(
        np.asarray(
            train_y,
            dtype=np.float32,
        ).mean()
    )

    pos_weight_value = (
        (1.0 - positive_rate)
        / max(
            positive_rate,
            1e-8,
        )
    )

    pos_weight_value = min(
        pos_weight_value,
        20.0,
    )

    print(
        "Fingerprint positive rate:",
        f"{positive_rate:.4%}",
    )
    print(
        "BCE positive weight:",
        f"{pos_weight_value:.3f}",
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.full(
            (
                train_y.shape[
                    1
                ],
            ),
            pos_weight_value,
            dtype=torch.float32,
            device=device,
        )
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )

    best_dev_loss = np.inf
    best_epoch = 0
    best_soft_jaccard = 0.0
    epochs_without_improvement = 0

    rng = np.random.default_rng(
        SEED
    )

    for epoch in range(
        1,
        args.max_epochs
        + 1,
    ):
        model.train()

        running_loss = 0.0
        rows = 0

        for indices in batches(
            len(
                train_y
            ),
            args.batch_size,
            shuffle=True,
            rng=rng,
        ):
            batch_x = dense_batch(
                train_x,
                indices,
            ).to(
                device
            )

            batch_y = target_batch(
                train_y,
                indices,
            ).to(
                device
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(
                batch_x
            )

            loss = criterion(
                logits,
                batch_y,
            )

            loss.backward()

            optimizer.step()

            row_count = len(
                indices
            )

            running_loss += (
                float(
                    loss.item()
                )
                * row_count
            )

            rows += row_count

        (
            dev_loss,
            dev_soft_jaccard,
            dev_bit_accuracy,
        ) = evaluate(
            model,
            dev_x,
            dev_y,
            criterion,
            device,
            args.batch_size,
        )

        train_loss = (
            running_loss
            / max(
                rows,
                1,
            )
        )

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.5f} | "
            f"dev_loss={dev_loss:.5f} | "
            f"dev_soft_jaccard={dev_soft_jaccard:.4f} | "
            f"dev_bit_accuracy={dev_bit_accuracy:.4f}"
        )

        if dev_loss < (
            best_dev_loss
            - 1e-5
        ):
            best_dev_loss = (
                dev_loss
            )
            best_epoch = epoch
            best_soft_jaccard = (
                dev_soft_jaccard
            )
            epochs_without_improvement = 0

            torch.save(
                {
                    "state_dict":
                        model.state_dict(),
                    "input_dim":
                        train_x.shape[
                            1
                        ],
                    "output_dim":
                        train_y.shape[
                            1
                        ],
                    "fingerprint_radius":
                        2,
                    "fingerprint_bits":
                        train_y.shape[
                            1
                        ],
                    "best_epoch":
                        best_epoch,
                    "best_dev_loss":
                        best_dev_loss,
                    "best_dev_soft_jaccard":
                        best_soft_jaccard,
                    "dataset_name":
                        args.name,
                },
                model_path,
            )
        else:
            epochs_without_improvement += 1

            if (
                epochs_without_improvement
                >= args.patience
            ):
                print(
                    "Early stopping."
                )
                break

    print()
    print(
        "Scaled spectrum-to-fingerprint model"
    )
    print(
        "=" * 72
    )
    print(
        "Best epoch:",
        best_epoch,
    )
    print(
        "Best dev loss:",
        f"{best_dev_loss:.6f}",
    )
    print(
        "Best dev soft Jaccard:",
        f"{best_soft_jaccard:.4f}",
    )
    print(
        "Model:",
        model_path,
    )


if __name__ == "__main__":
    main()
