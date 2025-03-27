# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from typing import List, Optional

import torch
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

from .. import parallel_state
from ..transformer.transformer_config import TransformerConfig
from ..utils import get_attr_wrapped_model, get_model_config

import bitsandbytes.functional as B_F
import math 

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

def a2a_ag(inp, quant_state, group = None):
    world_size = torch.distributed.get_world_size(group)
    out = torch.empty_like(inp)
    if inp.numel() % world_size != 0 or (inp.numel() // world_size) % quant_state.blocksize != 0:
        fp_grad = torch.empty_like(inp, dtype=torch.bfloat16)
        B_F.dequantize_blockwise(inp, quant_state, out=fp_grad , blocksize=quant_state.blocksize)
                    
        torch.distributed.all_reduce(
            fp_grad,
            group=group,
            async_op=False, # false
        )
        
        B_F.quantize_blockwise(fp_grad, code=quant_state.code, absmax=quant_state.absmax, out=out, blocksize=quant_state.blocksize)
        return out.view(inp.shape)

    assert ((inp.numel() % world_size == 0) and (inp.numel() // world_size) % quant_state.blocksize == 0), 'input size mod blocksize must be 0'
    # inp_shape = inp.shape

    
    absmax = quant_state.absmax
    
    out_absmax = torch.empty_like(absmax)

    torch.distributed.all_to_all_single(out, inp, group = group)
    torch.distributed.all_to_all_single(out_absmax, absmax, group = group)

    out = out.view(-1)
    out_absmax = out_absmax.view(-1)

    ag_in = out.view([world_size,out.shape[0]//world_size])
    ag_absmax = out_absmax.view([world_size,out_absmax.shape[0]//world_size])
    
    # print(ag_in.shape)

    ag_in_fp32 = []
    for i in range(world_size):
        row1 = ag_in[i]        # 获取 tensor1 的第 i 行，形状为 (1000,)
        row2 = ag_absmax[i]     # 获取 tensor2 的第 i 行，形状为 (1,)
        result_row = torch.empty_like(row1, dtype=torch.bfloat16)
        B_F.dequantize_blockwise(row1, absmax=row2, out=result_row , blocksize=quant_state.blocksize)
        # print(result_row)
        ag_in_fp32.append(result_row)
        
        absmax_numel = row2.numel()
    ag_in_fp32= torch.stack(ag_in_fp32, dim=0).view([world_size,out.shape[0]//world_size]).sum(0) 
    
    absmax_local = torch.zeros(absmax_numel, dtype=torch.float32, device=ag_in_fp32.device)
    ag_local = torch.zeros(ag_in_fp32.numel(), dtype=torch.uint8, device=ag_in_fp32.device)
    B_F.quantize_blockwise(ag_in_fp32, code=quant_state.code, absmax=absmax_local, out=ag_local, blocksize=quant_state.blocksize)

    # print(out.shape, ag_local.shape, quant_state.absmax.shape, absmax_local.shape)
    
    torch.distributed._all_gather_base(out, ag_local,group = group)
    torch.distributed._all_gather_base(quant_state.absmax, absmax_local,group = group)
    # print("测试", out, meta.scale_inv)
    # quant_state.absmax.copy_(out_absmax)
    
    return out.view(inp.shape)


def _allreduce_layernorm_grads(model: List[torch.nn.Module], config: TransformerConfig):
    """
    All-reduce layernorm grads (for sequence parallelism).
    """

    # All-reduce layernorm parameters across model parallel nodes
    # when sequence parallelism is used
    if parallel_state.get_tensor_model_parallel_world_size() > 1 and (
        config.sequence_parallel or config.qk_layernorm
    ):  
        if config.accumulate_allreduce_grads_in_fp8:
            grads = []
            for model_chunk in model:
                for name, param in get_attr_wrapped_model(model_chunk, 'named_parameters')():
                    if (
                        param.requires_grad
                        and getattr(param, 'sequence_parallel', False)
                        or 'q_layernorm' in name
                        or 'k_layernorm' in name
                    ):
                        import bitsandbytes.functional as B_F
                        # param.main_grad.value.copy_(a2a_ag(
                        #     param.main_grad.value,  # 访问底层uint8数据
                        #     param.main_grad.meta,
                        #     group=parallel_state.get_tensor_model_parallel_group(),
                        # ))
                        
                        fp_grad = torch.empty_like(param.main_grad.value, dtype=torch.bfloat16)
                        B_F.dequantize_blockwise(param.main_grad.value, param.main_grad.quant_state, out=fp_grad , blocksize=param.main_grad.quant_state.blocksize)
                        
                        
                        torch.distributed.all_reduce(
                            fp_grad,
                            group=parallel_state.get_tensor_model_parallel_group(),
                            async_op=False, # false
                        )
                        
                        B_F.quantize_blockwise(fp_grad, code=param.main_grad.quant_state.code, absmax=param.main_grad.quant_state.absmax, out=param.main_grad.value, blocksize=param.main_grad.quant_state.blocksize)

                        
                        
                        # grad = param.main_grad.value
                        # wgrad_qtype = param.main_grad.meta.qtype
            
                        # amax_tensor = param.main_grad.meta.amax
                        #     # amax_tensor = torch.tensor([amax], device=device)
                        # amax_tensor.nan_to_num_(nan=torch.inf, posinf=torch.inf)  # 处理异常值
                        # torch.distributed.all_reduce(amax_tensor, op=torch.distributed.ReduceOp.MAX)
                        # global_amax = amax_tensor[0].clamp(min=1e-12)
                        # # print(param.main_grad.meta.scale)

                        # fp_max = Floating.qfp_max[wgrad_qtype]
                        # new_scale = ScalingMeta.compute_scaling_factor(
                        #     global_amax, 
                        #     param.main_grad.meta.scale, 
                        #     fp_max, 
                        #     margin=0
                        # ) #???

                        # fp8_scale = new_scale.div(param.main_grad.meta.scale)

                        # param.main_grad.value = scale_fp8e4m3_tensor(param.main_grad.value, fp8_scale)

                        # param.main_grad.meta.amax[0] = global_amax
                        # param.main_grad.meta.scale.copy_(new_scale)
                        # param.main_grad.meta.scale_inv.copy_(torch.reciprocal(new_scale))

                        # DistOp.all_reduce(
                        #     param.main_grad.value,  # 访问底层int8数据
                        #     wgrad_qtype,
                        #     torch.distributed.ReduceOp.SUM,
                        #     group=torch.distributed.group.WORLD,
                        #     async_op=False
                        # )
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
