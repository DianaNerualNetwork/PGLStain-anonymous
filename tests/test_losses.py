"""Synthetic CPU checks for PECC and MRSA; no datasets or weights required."""

import pytest
import torch

from puzzlestain.models.loss.mrsa import MRSAMarginalLoss, MRSARelationalLoss
from puzzlestain.models.loss.pecc import PECCCorrespondenceLoss, PECCStructureLoss


@pytest.fixture(autouse=True)
def reproducible_cpu_randomness():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(31415)
        yield


def _image(*, size=16, requires_grad=False):
    return (torch.rand(1, 3, size, size) * 1.6 - 0.8).requires_grad_(requires_grad)


def _assert_finite_gradient(tensor):
    assert tensor.grad is not None
    assert torch.isfinite(tensor.grad).all()
    assert tensor.grad.abs().sum() > 0


def _assert_scalar_finite(loss):
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss >= 0


def _mrsa_loss(name):
    if name == "marginal":
        return MRSAMarginalLoss(patch_size=4, num_quantiles=16, min_tissue_patches=4)
    return MRSARelationalLoss(
        patch_size=4,
        max_nodes=16,
        min_nodes=4,
        mass_penalty=2.0,
        outer_iterations=3,
        sinkhorn_iterations=12,
    )


@pytest.mark.parametrize("name", ["marginal", "relational"])
def test_mrsa_backpropagates_to_prediction_only(name):
    prediction = _image(requires_grad=True)
    reference = _image(requires_grad=True)
    loss = _mrsa_loss(name)(image_s=prediction, image_t=reference)
    _assert_scalar_finite(loss)
    loss.backward()
    _assert_finite_gradient(prediction)
    assert reference.grad is None, "The reference image is a detached anchor"


@pytest.mark.parametrize("name", ["marginal", "relational"])
def test_mrsa_is_invariant_to_independent_patch_reordering(name):
    """Horizontal reflection permutes pooled nodes without changing their sets."""
    prediction = _image()
    reference = _image()
    criterion = _mrsa_loss(name)
    expected = criterion(image_s=prediction, image_t=reference)
    reordered = criterion(image_s=prediction.flip(-1), image_t=reference.flip(-2))
    torch.testing.assert_close(reordered, expected, rtol=1e-4, atol=1e-6)


def test_mrsa_marginal_identical_distributions_have_zero_loss():
    image = _image()
    loss = _mrsa_loss("marginal")(image_s=image, image_t=image.flip(-1))
    torch.testing.assert_close(loss, torch.zeros_like(loss), atol=1e-6, rtol=0)


def test_pecc_correspondence_has_finite_source_gradients_and_detached_anchor():
    criterion = PECCCorrespondenceLoss(nc=8, num_hop=2, pairing="index_softgate")
    source_features = torch.randn(8, 8, requires_grad=True)
    target_features = torch.randn(8, 8, requires_grad=True)
    prediction = _image(requires_grad=True)
    reference = _image(requires_grad=True)
    loss = criterion(
        feat_s=[source_features],
        feat_t=[target_features],
        image_s=prediction,
        image_t=reference,
        patch_ids=[torch.arange(8)],
        feat_map_sizes=[(4, 4)],
    )
    _assert_scalar_finite(loss)
    loss.backward()
    _assert_finite_gradient(source_features)
    assert target_features.grad is None
    assert reference.grad is None
    # The gate is detached and graph construction is discrete; gradients
    # reach the generator through features, not by modifying gate inputs.
    assert prediction.grad is None


def test_pecc_structure_has_finite_image_gradients():
    prediction = _image(requires_grad=True)
    reference = _image()
    loss = PECCStructureLoss(num_patches=16)(image_s=prediction, image_t=reference)
    _assert_scalar_finite(loss)
    loss.backward()
    _assert_finite_gradient(prediction)


def test_pecc_structure_is_invariant_to_node_order_with_full_sampling():
    prediction = _image(size=8)
    reference = _image(size=8)
    criterion = PECCStructureLoss(num_patches=64)
    expected = criterion(image_s=prediction, image_t=reference)
    reordered = criterion(image_s=prediction.flip(-1), image_t=reference.flip(-2))
    torch.testing.assert_close(reordered, expected, rtol=1e-4, atol=1e-6)
