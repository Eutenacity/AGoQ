# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Exposes the interface of MS-AMP common utilities."""

from megatron.core.msamp.common.utils.logging import MsAmpLogger
from megatron.core.msamp.common.utils.lazy_import import LazyImport
from megatron.core.msamp.common.utils.dist import DistUtil
from megatron.core.msamp.common.utils.device import Device

TransformerEngineWrapper = LazyImport('msamp.common.utils.transformer_engine_wrapper', 'TransformerEngineWrapper')

__all__ = ['MsAmpLogger', 'TransformerEngineWrapper', 'DistUtil', 'Device']
