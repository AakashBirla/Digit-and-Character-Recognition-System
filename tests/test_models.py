"""DigitCNN: construction, output shapes, logits vs. probabilities, parameter count, config round-trip."""

from __future__ import annotations

import json

import pytest
import torch
from torch import nn

from src.models import DigitCNN, architecture_summary


def expected_parameter_count(num_classes: int, c1: int = 32, c2: int = 64, hidden: int = 128,
                             batch_norm: bool = True) -> int:
    """Parameter count worked out by hand from the architecture in the spec."""
    convs = 1 * c1 * 9 + c1 * c2 * 9  # 3x3 kernels
    convs += 2 * c1 + 2 * c2 if batch_norm else c1 + c2  # BN weight+bias, or conv biases
    fully_connected = (c2 * 7 * 7 * hidden + hidden) + (hidden * num_classes + num_classes)
    return convs + fully_connected


@pytest.fixture
def seeded() -> None:
    torch.manual_seed(0)


@pytest.mark.parametrize("num_classes", [10, 62])
def test_model_initialises_with_requested_number_of_classes(num_classes: int) -> None:
    model = DigitCNN(num_classes=num_classes)
    assert model.num_classes == num_classes
    assert model.classifier[-1].out_features == num_classes
    assert all(p.requires_grad for p in model.parameters())


@pytest.mark.parametrize("num_classes", [10, 62])
@pytest.mark.parametrize("batch", [1, 5])
def test_output_shape_is_batch_by_num_classes(seeded: None, num_classes: int, batch: int) -> None:
    model = DigitCNN(num_classes=num_classes).eval()
    with torch.inference_mode():
        logits = model(torch.randn(batch, 1, 28, 28))
    assert logits.shape == (batch, num_classes)
    assert logits.dtype == torch.float32
    assert torch.isfinite(logits).all()


@pytest.mark.parametrize("num_classes", [10, 62])
def test_softmax_of_logits_is_finite_and_sums_to_one(seeded: None, num_classes: int) -> None:
    model = DigitCNN(num_classes=num_classes).eval()
    with torch.inference_mode():
        probabilities = torch.softmax(model(torch.randn(8, 1, 28, 28)), dim=1)
    assert torch.isfinite(probabilities).all()
    assert (probabilities >= 0).all() and (probabilities <= 1).all()
    torch.testing.assert_close(probabilities.sum(dim=1), torch.ones(8), atol=1e-5, rtol=0)


def test_forward_returns_raw_logits_without_softmax(seeded: None) -> None:
    model = DigitCNN(num_classes=62).eval()
    assert not any(isinstance(m, (nn.Softmax, nn.LogSoftmax)) for m in model.modules())
    with torch.inference_mode():
        logits = model(torch.randn(4, 1, 28, 28))
    # Probabilities would be non-negative and sum to 1 per row; random-init logits are neither.
    assert (logits < 0).any()
    assert not torch.allclose(logits.sum(dim=1), torch.ones(4))


@pytest.mark.parametrize("num_classes, expected", [(10, 421_738), (62, 428_446)])
def test_parameter_count_matches_the_architecture(num_classes: int, expected: int) -> None:
    model = DigitCNN(num_classes=num_classes)
    assert model.count_parameters() == expected == expected_parameter_count(num_classes)


def test_batch_norm_convolutions_have_no_bias() -> None:
    with_bn = DigitCNN(num_classes=10, batch_norm=True)
    convs = [m for m in with_bn.modules() if isinstance(m, nn.Conv2d)]
    assert len(convs) == 2 and all(conv.bias is None for conv in convs)
    assert sum(isinstance(m, nn.BatchNorm2d) for m in with_bn.modules()) == 2

    without_bn = DigitCNN(num_classes=10, batch_norm=False)
    assert all(m.bias is not None for m in without_bn.modules() if isinstance(m, nn.Conv2d))
    assert not any(isinstance(m, nn.BatchNorm2d) for m in without_bn.modules())
    assert without_bn.count_parameters() == expected_parameter_count(10, batch_norm=False)


def test_cpu_inference_is_deterministic_in_eval_mode(seeded: None) -> None:
    model = DigitCNN(num_classes=10)
    assert next(model.parameters()).device.type == "cpu"
    x = torch.rand(3, 1, 28, 28)

    model.eval()
    with torch.inference_mode():
        first, second = model(x), model(x)
    assert torch.equal(first, second)

    model.train()  # dropout is active in training mode
    assert not torch.equal(model(x), model(x))


def test_gradients_reach_every_parameter(seeded: None) -> None:
    model = DigitCNN(num_classes=62)
    loss = nn.functional.cross_entropy(model(torch.randn(4, 1, 28, 28)), torch.tensor([0, 9, 10, 61]))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_get_config_round_trips_through_json_and_from_config() -> None:
    model = DigitCNN(num_classes=62, conv_channels=(16, 32), hidden_units=64, dropout=0.25, batch_norm=False)
    config = model.get_config()
    assert config == {
        "architecture": "DigitCNN",
        "num_classes": 62,
        "conv_channels": [16, 32],
        "hidden_units": 64,
        "dropout": 0.25,
        "batch_norm": False,
    }
    rebuilt = DigitCNN.from_config(json.loads(json.dumps(config)))
    assert rebuilt.get_config() == config
    rebuilt.load_state_dict(model.state_dict(), strict=True)  # identical parameter layout


def test_from_config_rejects_another_architecture() -> None:
    with pytest.raises(ValueError, match="architecture"):
        DigitCNN.from_config({"architecture": "ResNet18", "num_classes": 10})


@pytest.mark.parametrize("kwargs", [{"num_classes": 0}, {"dropout": 1.0}, {"hidden_units": 0},
                                    {"conv_channels": (0, 64)}])
def test_invalid_constructor_arguments_raise(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        DigitCNN(**kwargs)


def test_architecture_summary_describes_the_network() -> None:
    expected = "Conv(32,3x3)+BN+ReLU+MaxPool → Conv(64,3x3)+BN+ReLU+MaxPool → FC(128)+ReLU+Dropout(0.3) → FC(62)"
    assert architecture_summary(DigitCNN(num_classes=62).get_config()) == expected
    assert architecture_summary(DigitCNN(num_classes=10)).endswith("→ FC(10)")
