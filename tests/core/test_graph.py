"""Tests for HybridGraphLayer and edge scorer."""
import types
from unittest.mock import MagicMock, patch
from importlib import reload


def _make_mock_nn():
    class _MockModule:
        def __init__(self): pass
        def __init_subclass__(cls, **kw): pass
    return types.SimpleNamespace(Module=_MockModule)


def _instantiate_layer_with_mocks():
    # NOTE: reload() inside a patch context is idempotent while torch is absent
    # (ImportError re-runs, all three names stay None). If torch ever becomes
    # available in this test runner, the reload will import the real packages
    # and leave the module in a mixed state after the patch reverts. At that
    # point, refactor to use persistent patchers instead of reload().
    import src.core.graph as graph_module
    mock_nn = _make_mock_nn()
    mock_torch = MagicMock()
    mock_gat_cls = MagicMock(return_value=MagicMock())

    with (
        patch("src.core.graph.nn", mock_nn),
        patch("src.core.graph.torch", mock_torch),
        patch("src.core.graph.GATv2Conv", mock_gat_cls),
    ):
        reload(graph_module)
        # reload() re-executes `import torch_geometric` which succeeds when
        # torch_geometric is installed, overwriting all three patches.
        # Re-apply so __init__ reads the mocks via sys.modules[__name__].
        graph_module.torch = mock_torch
        graph_module.nn = mock_nn
        graph_module.GATv2Conv = mock_gat_cls
        layer = graph_module.HybridGraphLayer.__new__(graph_module.HybridGraphLayer)
        graph_module.HybridGraphLayer.__init__(layer)

    return layer, mock_gat_cls


def test_graph_constants():
    from src.core.graph import DEFAULT_HIDDEN_DIM, DEFAULT_HEADS, _FALLBACK_EDGE_SCORE
    assert DEFAULT_HIDDEN_DIM == 64
    assert DEFAULT_HEADS == 4
    assert _FALLBACK_EDGE_SCORE == 0.5


def test_score_edge_valid_float():
    from src.core.graph import score_edge
    from src.core.providers import MockProvider
    provider = MockProvider(response="0.85")
    score = score_edge("Entity A", "Entity B", "both active 2013", provider)
    assert score == 0.85


def test_score_edge_clamps_above_one():
    from src.core.graph import score_edge
    from src.core.providers import MockProvider
    provider = MockProvider(response="1.5")
    assert score_edge("A", "B", "ctx", provider) == 1.0


def test_score_edge_clamps_below_zero():
    from src.core.graph import score_edge
    from src.core.providers import MockProvider
    provider = MockProvider(response="-0.3")
    assert score_edge("A", "B", "ctx", provider) == 0.0


def test_score_edge_fallback_on_unparseable():
    from src.core.graph import score_edge, _FALLBACK_EDGE_SCORE
    from src.core.providers import MockProvider
    provider = MockProvider(response="not a number")
    assert score_edge("A", "B", "ctx", provider) == _FALLBACK_EDGE_SCORE


def test_score_edge_prompt_includes_source_and_target():
    from src.core.graph import score_edge
    prompts = []
    class _Cap:
        def complete(self, p, **_):
            prompts.append(p)
            return "0.7"
        def embed(self, t): return []
    score_edge("UNIQUE_SOURCE", "UNIQUE_TARGET", "ctx", _Cap())
    assert "UNIQUE_SOURCE" in prompts[0]
    assert "UNIQUE_TARGET" in prompts[0]


def test_hybrid_layer_has_gat_attribute():
    layer, _ = _instantiate_layer_with_mocks()
    assert hasattr(layer, "gat")


def test_hybrid_layer_gat_constructed_with_edge_dim():
    """GATv2Conv must be called with edge_dim=1."""
    layer, mock_gat_cls = _instantiate_layer_with_mocks()
    call_kwargs = mock_gat_cls.call_args.kwargs
    assert call_kwargs.get("edge_dim") == 1


def test_forward_calls_gat_and_returns_result():
    import src.core.graph as graph_module
    mock_nn = _make_mock_nn()
    mock_torch = MagicMock()
    mock_gat_cls = MagicMock(return_value=MagicMock())

    with (
        patch("src.core.graph.nn", mock_nn),
        patch("src.core.graph.torch", mock_torch),
        patch("src.core.graph.GATv2Conv", mock_gat_cls),
    ):
        reload(graph_module)
        graph_module.torch = mock_torch
        graph_module.nn = mock_nn
        graph_module.GATv2Conv = mock_gat_cls
        layer = graph_module.HybridGraphLayer.__new__(graph_module.HybridGraphLayer)
        graph_module.HybridGraphLayer.__init__(layer)

        mock_x = MagicMock()
        mock_edge_index = MagicMock()
        mock_weights = MagicMock()
        result = layer.forward(mock_x, mock_edge_index, mock_weights)

    layer.gat.assert_called_once_with(mock_x, mock_edge_index, edge_attr=mock_weights)
    assert result is layer.gat.return_value
