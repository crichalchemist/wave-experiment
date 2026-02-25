from src.forecasting.phi_trajectory import PhiTrajectoryForecaster

class TestPhiTrajectoryForecaster:
    def test_forecast_returns_dict(self):
        forecaster = PhiTrajectoryForecaster(hidden_size=32, n_layers=1, pred_len=5, seq_len=50)
        result = forecaster.forecast_from_scenario("stable_community", history_len=100)
        assert "phi_predicted" in result
        assert "phi_actual" in result
        assert len(result["phi_predicted"]) == 5
        assert len(result["phi_actual"]) == 5

    def test_forecast_bounded(self):
        forecaster = PhiTrajectoryForecaster(hidden_size=32, n_layers=1, pred_len=5, seq_len=50)
        result = forecaster.forecast_from_scenario("capitalism_suppresses_love", history_len=100)
        assert all(abs(v) < 100 for v in result["phi_predicted"])

    def test_forecast_scenario_name(self):
        forecaster = PhiTrajectoryForecaster(hidden_size=32, n_layers=1, pred_len=5, seq_len=50)
        result = forecaster.forecast_from_scenario("surveillance_state", history_len=100)
        assert result["scenario"] == "surveillance_state"
