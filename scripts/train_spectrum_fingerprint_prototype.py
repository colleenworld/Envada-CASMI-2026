from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import sparse
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(
    "data/processed/spectrum_fingerprint/prototype"
)

TRAIN_VECTORS_PATH = (
    ROOT
    / "train_vectors.npz"
)

TRAIN_TARGETS_PATH = (
    ROOT
    / "train_targets.npy"
)

DEV_VECTORS_PATH = (
    ROOT
    / "dev_vectors.npz"
)

DEV_TARGETS_PATH = (
    ROOT
    / "dev_targets.npy"
)

MODEL_PATH = (
    ROOT
    / "spectrum_to_morgan2048.pt"
)

SEED = 42
BATCH_SIZE = 256
MAX_EPOCHS = 12
PATIENCE = 3
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


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[
    float,
    float,
    float,
]:
    model.eval()

    total_loss = 0.0
    total_rows = 0
    jaccards = []
    bit_accuracies = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(
                device,
                non_blocking=True,
            )
            y = y.to(
                device,
                non_blocking=True,
            )

            logits = model(
                x
            )

            loss = criterion(
                logits,
                y
            )

            probabilities = torch.sigmoid(
                logits
            )

            total_loss += (
                float(
                    loss.item()
                )
                * len(
                    x
                )
            )

            total_rows += len(
                x
            )

            jaccards.append(
                soft_jaccard(
                    probabilities,
                    y,
                )
                .detach()
                .cpu()
            )

            predicted_bits = (
                probabilities
                >= 0.5
            )

            bit_accuracies.append(
                (
                    predicted_bits
                    == y.bool()
                )
                .float()
                .mean(
                    dim=1
                )
                .detach()
                .cpu()
            )

    return (
        total_loss
        / max(
            total_rows,
            1,
        ),
        float(
            torch.cat(
                jaccards
            ).mean()
        ),
        float(
            torch.cat(
                bit_accuracies
            ).mean()
        ),
    )


def main() -> None:
    for path in [
        TRAIN_VECTORS_PATH,
        TRAIN_TARGETS_PATH,
        DEV_VECTORS_PATH,
        DEV_TARGETS_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing prototype dataset artifact: {path}"
            )

    torch.manual_seed(
        SEED
    )

    np.random.seed(
        SEED
    )

    train_x = sparse.load_npz(
        TRAIN_VECTORS_PATH
    ).toarray().astype(
        np.float32,
        copy=False,
    )

    train_y = np.load(
        TRAIN_TARGETS_PATH,
        allow_pickle=False,
    ).astype(
        np.float32,
        copy=False,
    )

    dev_x = sparse.load_npz(
        DEV_VECTORS_PATH
    ).toarray().astype(
        np.float32,
        copy=False,
    )

    dev_y = np.load(
        DEV_TARGETS_PATH,
        allow_pickle=False,
    ).astype(
        np.float32,
        copy=False,
    )

    if (
        len(
            train_x
        )
        != len(
            train_y
        )
        or len(
            dev_x
        )
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
        f"{len(train_x):,}",
    )
    print(
        "Development rows:",
        f"{len(dev_x):,}",
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

    train_dataset = TensorDataset(
        torch.from_numpy(
            train_x
        ),
        torch.from_numpy(
            train_y
        ),
    )

    dev_dataset = TensorDataset(
        torch.from_numpy(
            dev_x
        ),
        torch.from_numpy(
            dev_y
        ),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(
            device.type
            == "cuda"
        ),
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(
            device.type
            == "cuda"
        ),
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
        train_y.mean()
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

    criterion = (
        nn.BCEWithLogitsLoss(
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
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )

    best_dev_loss = np.inf
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(
        1,
        MAX_EPOCHS
        + 1,
    ):
        model.train()

        running_loss = 0.0
        rows = 0

        for x, y in train_loader:
            x = x.to(
                device,
                non_blocking=True,
            )
            y = y.to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(
                x
            )

            loss = criterion(
                logits,
                y
            )

            loss.backward()

            optimizer.step()

            running_loss += (
                float(
                    loss.item()
                )
                * len(
                    x
                )
            )

            rows += len(
                x
            )

        (
            dev_loss,
            dev_soft_jaccard,
            dev_bit_accuracy,
        ) = evaluate(
            model,
            dev_loader,
            criterion,
            device,
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
                },
                MODEL_PATH,
            )
        else:
            epochs_without_improvement += 1

            if (
                epochs_without_improvement
                >= PATIENCE
            ):
                print(
                    "Early stopping."
                )
                break

    print()
    print(
        "Spectrum-to-fingerprint prototype"
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
        "Model:",
        MODEL_PATH,
    )


if __name__ == "__main__":
    main()
