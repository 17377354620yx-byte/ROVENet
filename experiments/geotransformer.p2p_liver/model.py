"""RTOR+A3 model entry point for the P2P liver experiment."""

from rovenet.models.registration import ROVENet


def create_model(config):
    """Build the checkpoint-compatible P2P registration model."""
    return ROVENet(config)


__all__ = ['ROVENet', 'create_model']
