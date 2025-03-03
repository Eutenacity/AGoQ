# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Expose the interface of MS-AMP tensor package."""

from megatron.core.msamp.common.tensor.cast import TypeCast
from megatron.core.msamp.common.tensor.hook import HookManager
from megatron.core.msamp.common.tensor.meta import ScalingMeta
from megatron.core.msamp.common.tensor.tensor import ScalingTensor
from megatron.core.msamp.common.tensor.tensor_dist import TensorDist

__all__ = ['TypeCast', 'HookManager', 'ScalingMeta', 'ScalingTensor', 'TensorDist']
