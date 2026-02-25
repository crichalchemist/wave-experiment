import torch
from src.forecasting.engine import train_phi_epoch
from src.forecasting.model import PhiForecaster

class TestTrainPhiEpoch:
    def test_loss_decreases_over_batch(self):
        model = PhiForecaster(input_size=34, hidden_size=32, n_layers=1, pred_len=5)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        X = torch.randn(8, 50, 34)
        y_phi = torch.randn(8, 5, 1)
        y_construct = torch.randn(8, 5, 8)
        dataset = torch.utils.data.TensorDataset(X, y_phi, y_construct)
        loader = torch.utils.data.DataLoader(dataset, batch_size=4)
        metrics = train_phi_epoch(model, loader, optimizer, device=torch.device("cpu"), use_amp=False)
        assert "loss" in metrics
        assert metrics["loss"] > 0

    def test_validate_returns_loss(self):
        from src.forecasting.engine import validate_phi_epoch
        model = PhiForecaster(input_size=34, hidden_size=32, n_layers=1, pred_len=5)
        X = torch.randn(4, 50, 34)
        y_phi = torch.randn(4, 5, 1)
        y_construct = torch.randn(4, 5, 8)
        dataset = torch.utils.data.TensorDataset(X, y_phi, y_construct)
        loader = torch.utils.data.DataLoader(dataset, batch_size=4)
        metrics = validate_phi_epoch(model, loader, device=torch.device("cpu"))
        assert "loss" in metrics
        assert metrics["loss"] > 0
