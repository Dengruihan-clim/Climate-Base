"""Explicit, lazy adaptations of declared monthly pressure products."""

from .api import adapt_product, adapter_source_time, validate_adapter_options

__all__ = ["adapt_product", "adapter_source_time", "validate_adapter_options"]
