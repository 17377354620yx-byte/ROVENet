"""RTOR+A3 model entry point for the 3Dircadb dataset_v3 experiment."""

from rovenet.models.registration import ROVENet


def create_model(config):
    """Build the RTOR+A3 rigid registration model."""
    return ROVENet(config)


__all__ = ['ROVENet', 'create_model']
