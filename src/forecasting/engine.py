"""Training engine for PhiForecaster."""
import torch
import torch.nn as nn
from tqdm import tqdm


def train_phi_epoch(model, dataloader, optimizer, device, use_amp=True, grad_clip=1.0):
    """Train for one epoch. Expects (X, y_phi, y_construct) batches."""
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)
    for batch in tqdm(dataloader, desc="Training", leave=False):
        x, y_phi, y_construct = [b.to(device) for b in batch]
        optimizer.zero_grad()
        phi_pred, construct_pred, _ = model(x)
        loss = model.compute_loss(phi_pred, y_phi, construct_pred, y_construct)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item()
    return {"loss": total_loss / max(num_batches, 1)}


def validate_phi_epoch(model, dataloader, device):
    """Validate for one epoch."""
    model.eval()
    total_loss = 0.0
    num_batches = len(dataloader)
    with torch.no_grad():
        for batch in dataloader:
            x, y_phi, y_construct = [b.to(device) for b in batch]
            phi_pred, construct_pred, _ = model(x)
            loss = model.compute_loss(phi_pred, y_phi, construct_pred, y_construct)
            total_loss += loss.item()
    return {"loss": total_loss / max(num_batches, 1)}
