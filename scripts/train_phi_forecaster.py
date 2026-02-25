"""Train PhiForecaster on synthetic data — local smoke test.

Generates 8 scenario types x 10 instances each, trains for 20 epochs on CPU,
and verifies the loss decreases. Saves checkpoint to checkpoints/.
"""
import numpy as np
import torch

from src.forecasting.engine import train_phi_epoch, validate_phi_epoch
from src.forecasting.model import PhiForecaster
from src.forecasting.pipeline import PhiPipeline
from src.forecasting.synthetic import PhiScenarioGenerator
from src.inference.welfare_scoring import ALL_CONSTRUCTS


def main():
    seed = 42
    pred_len = 10
    seq_len = 50
    hidden_size = 64
    epochs = 20
    batch_size = 16
    lr = 1e-3

    torch.manual_seed(seed)
    np.random.seed(seed)

    print("Generating synthetic data...")
    gen = PhiScenarioGenerator(seed=seed)
    dataset = gen.generate_dataset(scenarios_per_type=10, length=200)
    print(f"Generated {len(dataset)} scenario sequences")

    pipe = PhiPipeline(seq_len=seq_len)

    all_X, all_y_phi, all_y_construct = [], [], []

    for idx, df in enumerate(dataset):
        if idx == 0:
            X = pipe.fit_transform(df)
        else:
            X = pipe.transform(df)
        n_features = X.shape[1]

        for i in range(len(X) - seq_len - pred_len):
            all_X.append(X[i : i + seq_len])

            phi_target = df["phi"].values[i + seq_len : i + seq_len + pred_len]
            if len(phi_target) < pred_len:
                continue
            all_y_phi.append(phi_target.reshape(-1, 1))

            construct_target = np.stack(
                [df[c].values[i + seq_len : i + seq_len + pred_len] for c in ALL_CONSTRUCTS],
                axis=-1,
            )
            all_y_construct.append(construct_target)

    X_all = torch.tensor(np.array(all_X), dtype=torch.float32)
    y_phi = torch.tensor(np.array(all_y_phi), dtype=torch.float32)
    y_construct = torch.tensor(np.array(all_y_construct), dtype=torch.float32)

    print(f"Training data: {X_all.shape[0]} sequences, {n_features} features")

    # Train/val split
    n = len(X_all)
    perm = torch.randperm(n)
    split = int(0.8 * n)
    train_idx, val_idx = perm[:split], perm[split:]

    train_ds = torch.utils.data.TensorDataset(
        X_all[train_idx], y_phi[train_idx], y_construct[train_idx]
    )
    val_ds = torch.utils.data.TensorDataset(
        X_all[val_idx], y_phi[val_idx], y_construct[val_idx]
    )

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size)

    device = torch.device("cpu")
    model = PhiForecaster(
        input_size=n_features, hidden_size=hidden_size, n_layers=1, pred_len=pred_len
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"Model: {model.num_parameters:,} parameters")
    print(f"Training for {epochs} epochs...\n")

    first_loss = None
    for epoch in range(1, epochs + 1):
        train_metrics = train_phi_epoch(
            model, train_loader, optimizer, device, use_amp=False
        )
        val_metrics = validate_phi_epoch(model, val_loader, device)
        if first_loss is None:
            first_loss = train_metrics["loss"]
        print(
            f"  Epoch {epoch:2d}/{epochs}"
            f"  train_loss={train_metrics['loss']:.4f}"
            f"  val_loss={val_metrics['loss']:.4f}"
        )

    final_loss = train_metrics["loss"]
    print(f"\nLoss: {first_loss:.4f} -> {final_loss:.4f}")
    if final_loss < first_loss:
        print("PASS: Loss decreased — model is learning.")
    else:
        print("WARN: Loss did not decrease.")

    import os
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/phi_forecaster_smoke.pt")
    print("Saved to checkpoints/phi_forecaster_smoke.pt")


if __name__ == "__main__":
    main()
