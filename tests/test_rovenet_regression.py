"""Regression checks for the fixed RTOR+A3 network and legacy checkpoints."""

import importlib.util
import os
from pathlib import Path

import pytest
import torch

from rovenet.models import ROVENet


_EXPERIMENT = Path(__file__).resolve().parents[1] / 'experiments/geotransformer.p2p_liver'


def _configuration():
    spec = importlib.util.spec_from_file_location('rovenet_p2p_cfg', _EXPERIMENT / 'config.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.make_cfg(interaction_profile='cooperative')


def test_fixed_network_parameter_layout():
    cfg = _configuration()
    assert 'ablation' not in cfg
    assert 'dual_encoder' not in cfg.model
    model = ROVENet(cfg)
    names = list(model.state_dict())
    assert len(names) == 307
    assert sum(parameter.numel() for parameter in model.parameters()) == 10_621_188
    assert any(name.startswith('topology_overlap_refiner.') for name in names)
    assert any(name.startswith('fine_local_refiner.') for name in names)
    assert not hasattr(model, 'rtor_enabled')
    assert not hasattr(model, 'a3_enabled')
    assert not hasattr(model, 'fine_backbone')
    assert not any(name.startswith('fine_backbone.') for name in names)


def test_legacy_complete_checkpoint_strict_load():
    checkpoint_path = os.environ.get('ROVENET_LEGACY_CHECKPOINT')
    if not checkpoint_path:
        pytest.skip('Set ROVENET_LEGACY_CHECKPOINT to validate an existing complete RTOR+A3 checkpoint')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    protocol = checkpoint.get('metadata', {}).get('p2p_protocol', {})
    assert protocol.get('architecture') == 'rtor_a3'
    model = ROVENet(_configuration())
    expected = model.state_dict()
    actual = checkpoint['model']
    assert list(expected) == list(actual)
    assert {key: tuple(value.shape) for key, value in expected.items()} == {
        key: tuple(value.shape) for key, value in actual.items()
    }
    model.load_state_dict(actual, strict=True)
