"""Check the weighted neural objective against an independent actuarial metric."""
import numpy as np
import pytest
from sklearn.metrics import mean_tweedie_deviance

torch = pytest.importorskip("torch", reason="Install the optional neural requirements to test MLPs")
from nonlife.neural import PricingMLP, predict, tweedie_objective


def test_neural_objective_difference_equals_weighted_tweedie_deviance_difference():
    y = np.array([0., 100., 3000., 0.])
    exposure = np.array([.1, 1., .25, .5])
    scale = 150.
    first, second = np.array([30., 150., 500., 90.]), np.array([60., 90., 350., 250.])
    losses = []
    for mu in [first, second]:
        losses.append(tweedie_objective(torch.tensor(np.log(mu / scale)), torch.tensor(y / scale),
                                      torch.tensor(exposure / exposure.mean())).item())
    actual = 2 * scale ** .1 * (losses[0] - losses[1])
    expected = mean_tweedie_deviance(y, first, power=1.9, sample_weight=exposure) - mean_tweedie_deviance(
        y, second, power=1.9, sample_weight=exposure)
    assert actual == pytest.approx(expected, abs=1e-11)


def test_initial_network_is_positive_portfolio_mean_and_reproducible():
    torch.manual_seed(7)
    first = PricingMLP(3, [4, 2])
    torch.manual_seed(7)
    second = PricingMLP(3, [4, 2])
    for a, b in zip(first.parameters(), second.parameters()):
        assert torch.equal(a, b)
    x = np.array([[0, 1, 2], [100, 0, -1]], dtype=np.float32)
    np.testing.assert_allclose(predict(first, x, 150.), [150., 150.])


def test_zero_claims_have_finite_loss_and_downward_gradient():
    log_rate = torch.zeros(4, requires_grad=True)
    loss = tweedie_objective(log_rate, torch.zeros(4), torch.ones(4))
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.all(log_rate.grad > 0)
