# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from typing import List, Optional

import torch
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

from .. import parallel_state
from ..transformer.transformer_config import TransformerConfig
from ..utils import get_attr_wrapped_model, get_model_config

from ..msamp.common.tensor import ScalingTensor, ScalingMeta
from ..msamp.common.dtype import Dtypes, Floating
from ..msamp.operators.dist_op import DistOp

import math 

def scale_fp8e4m3_tensor(fp8_tensor: torch.Tensor, scale: float) -> torch.Tensor:
        """
        对 FP8 E4M3 格式的 uint8 张量进行缩放（直接操作指数部分）
        
        Args:
            fp8_tensor (torch.Tensor): uint8 张量，数据为 FP8 E4M3 格式
            scale (float): 缩放因子，必须是2的整数次幂（如 2, 0.5, 4, 等）
        
        Returns:
            torch.Tensor: 缩放后的 uint8 张量（FP8 E4M3 格式）
        """
        # 验证 scale 是2的整数次幂且为正数
        assert scale > 0, "Scale must be positive"
        log2_scale = math.log2(scale)
        # print(log2_scale)
        assert math.isclose(log2_scale, round(log2_scale), rel_tol=1e-5), \
            "Scale must be an exact power of 2, scale:{}".format(scale)
        k = int(round(log2_scale))

        # 提取各部分位信息（无需转浮点数）
        sign = (fp8_tensor >> 7) & 0x1          # [0,1]
        exponent = (fp8_tensor >> 3) & 0xF      # [0,15]
        mantissa = fp8_tensor & 0x7             # [0,7]

        # 调整指数并限制范围 [0,15])
        exponent = exponent.to(torch.int8)
        new_exponent = torch.clamp(exponent + k, 0, 15)

        # 重组为新的 FP8 E4M3 格式
        scaled_fp8 = (sign << 7) | (new_exponent << 3) | mantissa

        return scaled_fp8.to(fp8_tensor.dtype)

def _allreduce_word_embedding_grads(model: List[torch.nn.Module], config: TransformerConfig):
    """
    All-reduce word embedding grads.

    Reduce grads across first and last stages to ensure that word_embeddings parameters stay in
    sync. This should only run for models that support pipelined model parallelism (BERT and GPT).
    """

    if (
        parallel_state.is_rank_in_embedding_group(ignore_virtual=True)
        and parallel_state.get_pipeline_model_parallel_world_size() > 1
    ):
        if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
            model_module = model[0]
        elif parallel_state.is_pipeline_last_stage(ignore_virtual=True):
            model_module = model[-1]
        else:  # We do not support the interleaved schedule for T5 yet.
            model_module = model[0]

        # Look for module with 'pre_process' attribute to get around the fact that DDP and
        # other wrapper classes inherit from non-core MegatronModule that has
        # 'share_embeddings_and_output_weights' and 'shared_embedding_or_output_weight'
        # attributes already, causing get_attr_wrapped_model() to not unwrap anything here.
        # TODO: Clean this up once the wrapper classes inherit from core MegatronModule.
        model_module = get_attr_wrapped_model(model_module, 'pre_process', return_model_obj=True)
        if model_module.share_embeddings_and_output_weights:
            weight = model_module.shared_embedding_or_output_weight()
            grad = weight.main_grad
            torch.distributed.all_reduce(grad, group=parallel_state.get_embedding_group())


def _allreduce_position_embedding_grads(model: List[torch.nn.Module], config: TransformerConfig):
    """
    All-reduce position_embeddings grad across first (encoder) and split (decoder) stages to
    ensure that position embeddings parameters stay in sync. This should only run for T5 models
    with pipeline parallelism.
    """
    if (
        parallel_state.is_rank_in_position_embedding_group()
        and parallel_state.get_pipeline_model_parallel_world_size() > 1
        and config.pipeline_model_parallel_split_rank is not None
    ):
        model_module = model[0]
        grad = get_attr_wrapped_model(
            model_module, 'language_model.embedding.position_embeddings.weight.main_grad'
        )
        torch.distributed.all_reduce(grad, group=parallel_state.get_position_embedding_group())


def _allreduce_embedding_grads(model: List[torch.nn.Module], config: TransformerConfig):
    """
    All-reduce both word and position embeddings.
    """
    _allreduce_word_embedding_grads(model, config)
    _allreduce_position_embedding_grads(model, config)


def _allreduce_layernorm_grads(model: List[torch.nn.Module], config: TransformerConfig):
    """
    All-reduce layernorm grads (for sequence parallelism).
    """

    # All-reduce layernorm parameters across model parallel nodes
    # when sequence parallelism is used
    if parallel_state.get_tensor_model_parallel_world_size() > 1 and (
        config.sequence_parallel or config.qk_layernorm
    ):  
        if config.grad_reduce_in_fp8:
            print(config.grad_reduce_in_fp8)
            grads = []
            for model_chunk in model:
                for name, param in get_attr_wrapped_model(model_chunk, 'named_parameters')():
                    if (
                        param.requires_grad
                        and getattr(param, 'sequence_parallel', False)
                        or 'q_layernorm' in name
                        or 'k_layernorm' in name
                    ):
                        grad = param.main_grad.value
                        wgrad_qtype = param.main_grad.meta.qtype
            
                        amax_tensor = param.main_grad.meta.amax
                            # amax_tensor = torch.tensor([amax], device=device)
                        amax_tensor.nan_to_num_(nan=torch.inf, posinf=torch.inf)  # 处理异常值
                        torch.distributed.all_reduce(amax_tensor, op=torch.distributed.ReduceOp.MAX)
                        global_amax = amax_tensor[0].clamp(min=1e-12)
                        # print(self.grad_data.meta.scale)

                        fp_max = Floating.qfp_max[wgrad_qtype]
                        new_scale = ScalingMeta.compute_scaling_factor(
                            global_amax, 
                            param.main_grad.meta.scale, 
                            fp_max, 
                            margin=0
                        ) #???

                        fp8_scale = new_scale.div(param.main_grad.meta.scale)

                        param.main_grad.value = scale_fp8e4m3_tensor(param.main_grad.value, fp8_scale)

                        param.main_grad.meta.amax[0] = global_amax
                        param.main_grad.meta.scale.copy_(new_scale)
                        param.main_grad.meta.scale_inv.copy_(torch.reciprocal(new_scale))

                        DistOp.all_reduce(
                            param.main_grad.value,  # 访问底层int8数据
                            wgrad_qtype,
                            torch.distributed.ReduceOp.SUM,
                            group=torch.distributed.group.WORLD,
                            async_op=False
                        )
        else:
            grads = []
            for model_chunk in model:
                for name, param in get_attr_wrapped_model(model_chunk, 'named_parameters')():
                    if (
                        param.requires_grad
                        and getattr(param, 'sequence_parallel', False)
                        or 'q_layernorm' in name
                        or 'k_layernorm' in name
                    ):
                        grad = param.main_grad
                        grads.append(grad.data)
            if grads:
                coalesced = _flatten_dense_tensors(grads)
                torch.distributed.all_reduce(
                    coalesced, group=parallel_state.get_tensor_model_parallel_group()
                )
                for buf, synced in zip(grads, _unflatten_dense_tensors(coalesced, grads)):
                    buf.copy_(synced)


def finalize_model_grads(model: List[torch.nn.Module], num_tokens: Optional[torch.Tensor] = None):
    """
    All-reduce all model grads across DP replicas, layernorm grads for sequence parallelism,
    embedding grads across first and last pipeline stages (if not tied),
    scale gradients by `num_tokens`.
    """

    config = get_model_config(model[0])

    # All-reduce / reduce-scatter across DP replicas.
    if config.timers is not None:
        config.timers('all-grads-sync', log_level=1).start(barrier=config.barrier_with_L1_time)
    for model_chunk in model:
        model_chunk.finish_grad_sync()
    if config.timers is not None:
        config.timers('all-grads-sync').stop()

    # from ..utils import get_attr_wrapped_model
    # for model_chunk in model:
    #     for param in get_attr_wrapped_model(model_chunk, 'named_parameters')() :
    #             grad = param[1].main_grad.value
    #             print('model_chunk')
    #             print(grad.dtype) 

    # All-reduce layer-norm grads (for sequence parallelism).
    if config.timers is not None:
        config.timers('layernorm-grads-all-reduce', log_level=1).start(
            barrier=config.barrier_with_L1_time
        )
    _allreduce_layernorm_grads(model, config)
    if config.timers is not None:
        config.timers('layernorm-grads-all-reduce').stop()

    # All-reduce embedding grads (for pipeline parallelism).
    if config.timers is not None:
        config.timers('embedding-grads-all-reduce', log_level=1).start(
            barrier=config.barrier_with_L1_time
        )
    _allreduce_embedding_grads(model, config)
    if config.timers is not None:
        config.timers('embedding-grads-all-reduce').stop()

    # normalize gradients for per-token loss normalization.
    # if we are using by the number of tokens, then we use that as a divisor. this number
    # will be the total number of non-padded tokens in the global batch.
    if num_tokens is not None:
        # the number of tokens is only present on the last stage, so broadcast it
        # to the other ranks in the pipeline parallel group.
        torch.distributed.broadcast(
            num_tokens,
            src=parallel_state.get_pipeline_model_parallel_last_rank(),
            group=parallel_state.get_pipeline_model_parallel_group(),
        )
        # all-reduce across DP ranks.
        torch.distributed.all_reduce(num_tokens, group=parallel_state.get_data_parallel_group())
        for model_chunk in model:
            if num_tokens > 0:
                scaling = 1.0 / num_tokens
                model_chunk.scale_gradients(scaling)
