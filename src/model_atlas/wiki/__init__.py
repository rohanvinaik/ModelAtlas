"""Wiki materializer — deterministic documentation generation with provenance."""

from .config import load_config, WikiConfig
from .drift import check_drift, DriftReport
from .manifest import load_manifest, Manifest
from .renderer import materialize

__all__ = [
    "load_config",
    "WikiConfig",
    "check_drift",
    "DriftReport",
    "load_manifest",
    "Manifest",
    "materialize",
]
