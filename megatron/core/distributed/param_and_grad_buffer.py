# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import math
import os
from enum import Enum
from logging import getLogger
from typing import Dict, List, Optional

import torch
import torch.distributed
from memory_profiler import profile
import bitsandbytes.functional as B_F
from .. import parallel_state
from .distributed_data_parallel_config import DistributedDataParallelConfig

logger = getLogger(__name__)


class BufferType(Enum):
    PARAM = 1
    GRAD = 2


def shard_buffer(buffer: torch.Tensor, data_parallel_world_size: int):
    """
    Shard buffer into data_parallel_world_size chunks of equal size.
    """
    assert buffer.numel() % data_parallel_world_size == 0
    shard_size = buffer.numel() // data_parallel_world_size
    sharded_buffer = [
        buffer[(r * shard_size) : ((r + 1) * shard_size)] for r in range(data_parallel_world_size)
    ]
    return sharded_buffer


class Bucket:
    """
    Bucket to keep track of a subset of the model's gradients. Provides functionality to register
    when params in the bucket have grads ready to be synced; an asynchronous communication call
    is automatically launched when _all_ params in the bucket have grads ready.

    Args:
        ddp_config: DistributedDataParallel config object.
        params: List of parameters whose gradients are collated in this bucket.
        param_data: View in larger ParamAndGradBuffer.param_data that this bucket is responsible for.
        grad_data: View in larger ParamAndGradBuffer.grad_data that this bucket is responsible for.
        offset: Offset of this bucket's view in the larger ParamAndGradBuffer.
        numel_unpadded: Number of unpadded elements in bucket.
        data_parallel_group: Data-parallel process group.
        data_parallel_world_size: World size using the data-parallel group group.
        gradient_scaling_factor: This factor is utilized to scale gradients prior to their
            communication. Its application is twofold: it facilitates the averaging of gradients
            and the scaling of gradients in the context of the Mixture of Experts (MoE) model.
    """

    def __init__(
        self,
        ddp_config: DistributedDataParallelConfig,
        params: List[torch.nn.Parameter],
        param_data: Optional[torch.Tensor],
        grad_data: torch.Tensor,
        offset: int,
        numel_unpadded: int,
        data_parallel_group: torch.distributed.ProcessGroup,
        data_parallel_world_size: int,
        gradient_scaling_factor: float,
    ):
        self.ddp_config = ddp_config

        # State for bookkeeping: params is the set of parameters this bucket is
        # responsible for, params_with_grad is the set of parameters with grads
        # available. When overlap_grad_reduce is True, communication (all-reduce
        # or reduce-scatter) is issued when params_with_grad equals params.
        self.params_list = params
        self.params = set(params)
        self.params_with_grad = set()
        self.param_data = param_data
        self.grad_data = grad_data
        # The distributed optimizer needs to keep track of this bucket's offset
        # within the full grad_buffer.
        self.offset = offset
        self.numel_unpadded = numel_unpadded
        self.data_parallel_group = data_parallel_group
        self.data_parallel_world_size = data_parallel_world_size
        self.data_parallel_rank = torch.distributed.get_rank(group=data_parallel_group)
        self.gradient_scaling_factor = gradient_scaling_factor

        self.reset()

    def reset(self):
        """
        Reset metadata in bucket in preparation for the next iteration of training.
        """
        self.params_with_grad = set()
        self.communication_handle = None
        self.communication_issued = False

    def start_grad_sync(self):
        """
        Initiates grad sync (all-reduce or reduce-scatter) communication operation
        for this bucket.

        When overlap_grad_reduce is set to True, dispatches an asynchronous
        communication call. When overlap_grad_reduce is set to False, makes
        synchronous call.
        """
        assert (
            self.communication_handle is None and not self.communication_issued
        ), 'Should not have multiple communication calls in flight at once'

        # Make sure norm of grads in bucket are not NaN
        # prior to data-parallel all-reduce / reduce-scatter.
        if self.ddp_config.check_for_nan_in_grad:
            global_rank = torch.distributed.get_rank()
            norm = self.grad_data.norm(p=2)
            assert not norm.isnan(), (
                f'Rank {global_rank}: found NaN in local grad norm in '
                f'backward pass before data-parallel communication collective. '
                f'Device: {torch.cuda.current_device()}, node: {os.uname()[1]}'
            )
            
        if self.gradient_scaling_factor != 1.0: 
            self.grad_data *= self.gradient_scaling_factor
        # Use async_op only when overlap_grad_reduce is True.
        if self.ddp_config.use_distributed_optimizer:
            local_data_view = shard_buffer(self.grad_data, self.data_parallel_world_size)[
                self.data_parallel_rank
            ]
            self.communication_handle = torch.distributed._reduce_scatter_base(
                local_data_view,
                self.grad_data,
                group=self.data_parallel_group,
                async_op=self.ddp_config.overlap_grad_reduce,
            )
        else:
            # print('rank:', torch.distributed.get_rank(), self.grad_data.shape)
            # self.grad_data.copy_(self.a2a_ag_fp16(
            #     self.grad_data,  # 访问底层uint8数据
            #     group=self.data_parallel_group,
            # ))

            self.communication_handle = torch.distributed.all_reduce(
                self.grad_data,
                group=self.data_parallel_group,
                async_op=self.ddp_config.overlap_grad_reduce, # false
            )
            # print((grad_data - self.grad_data).sum())
            
            
        self.communication_issued = True


    # def quantize_e4m3(self,tensor, scale):
    #     """
    #     将输入的高精度张量量化为 FP8 E4M3 格式。

    #     参数：
    #     tensor: 输入张量（例如 torch.float32）
    #     scale: 缩放因子，用于调整输入数值的范围

    #     返回：
    #     一个 torch.uint8 张量，每个元素为 FP8 格式（E4M3）的编码

    #     FP8 E4M3 格式解释：
    #     - 1 位符号位（bit7）：0 表示正，1 表示负
    #     - 4 位指数位（bits 6-3）：带偏置，通常偏置为 7
    #     - 3 位尾数位（bits 2-0）：表示小数部分（近似精度）
    #     """
    #     device = tensor.device
    #     # 1. 缩放输入数值
    #     scaled = tensor * scale

    #     # 2. 提取符号位：0 表示正，1 表示负
    #     sign = (scaled < 0).to(torch.int32)

    #     # 3. 取绝对值（后续计算均以正数为基础）
    #     abs_val = torch.abs(scaled)

    #     # 4. 处理零值和极小值
    #     eps = 1e-8
    #     abs_val = torch.where(abs_val < eps, torch.tensor(eps, device=device, dtype=abs_val.dtype), abs_val)

    #     # 5. 计算指数部分：利用 log2 求取值所在数量级，并向下取整
    #     exponent = torch.floor(torch.log2(abs_val))

    #     # 6. 根据公式，计算尾数部分
    #     # 对于一个正数 x，可表示为 x = 2^(exponent) * (1 + mantissa/8)
    #     # 因此，mantissa = (x/2^(exponent) - 1) * 8
    #     mantissa = (abs_val / (2 ** exponent) - 1.0) * 8.0
    #     mantissa_int = torch.round(mantissa).to(torch.int32)

    #     # 7. 处理尾数溢出：如果 mantissa_int == 8，则设置为 0 并进位（即 exponent + 1）
    #     carry_mask = (mantissa_int == 8)
    #     mantissa_int = torch.where(carry_mask, torch.tensor(0, device=device, dtype=torch.int32), mantissa_int)
    #     exponent = torch.where(carry_mask, exponent + 1, exponent)

    #     # 8. 指数需要加上偏置（对于 4 位指数，常用偏置为 7）
    #     bias = 7
    #     exponent_int = (exponent + bias).to(torch.int32)

    #     # 9. 裁剪指数和尾数到合法范围：
    #     # 指数：4 位表示，范围 0 ~ 15
    #     # 尾数：3 位表示，范围 0 ~ 7
    #     exponent_int = torch.clamp(exponent_int, 0, 15)
    #     mantissa_int = torch.clamp(mantissa_int, 0, 7)

    #     # 10. 打包：FP8 的 8 位中，最高位为符号位，接下来的 4 位为指数，最低 3 位为尾数
    #     fp8 = (sign << 7) | (exponent_int << 3) | mantissa_int

    #     # 11. 处理非正规化数（Subnormal Numbers）：当指数全为0时，使用尾数表示
    #     subnormal_mask = (exponent_int == 0)
    #     fp8 = torch.where(subnormal_mask, (sign << 7) | mantissa_int, fp8)

    #     # 12. 处理零值
    #     zero_mask = (abs_val < eps)
    #     fp8 = torch.where(zero_mask, torch.tensor(0, dtype=torch.uint8, device=device), fp8)

    #     # 13. 处理 NaN 和 Inf
    #     nan_mask = torch.isnan(scaled)
    #     inf_mask = torch.isinf(scaled)
    #     fp8 = torch.where(nan_mask, torch.tensor(0b11111111, dtype=torch.uint8, device=device), fp8)
    #     fp8 = torch.where(inf_mask, torch.tensor(0b11111000, dtype=torch.uint8, device=device), fp8)

    #     return fp8.to(torch.uint8)


    # def dequantize_e4m3(self,fp8_tensor, scale_inv):
    #     """
    #     将 FP8 E4M3 格式张量反量化为 FP32 数值（完整处理所有情况）
    #     """
    #     # 提取位信息
    #     sign = (fp8_tensor >> 7) & 0x1
    #     exponent = (fp8_tensor >> 3) & 0xF
    #     mantissa = fp8_tensor & 0x7

    #     # 配置参数
    #     bias = 7
    #     fp8_max = 448.0  # E4M3最大正规化数 (1.1111b * 2^7)

    #     # 符号因子
    #     sign_factor = torch.where(sign == 0, 1.0, -1.0)

    #     # 分类处理不同情况
    #     is_zero = (exponent == 0) & (mantissa == 0)
    #     is_subnormal = (exponent == 0) & (mantissa != 0)
    #     is_normal = exponent != 0

    #     # 计算指数和尾数缩放因子
    #     exponent_normalized = torch.zeros_like(exponent, dtype=torch.float32)
    #     mantissa_scale = torch.zeros_like(exponent, dtype=torch.float32)

    #     # 1. 处理正规化数
    #     exponent_normalized[is_normal] = (exponent[is_normal].float() - bias)
    #     mantissa_scale[is_normal] = 1.0 + mantissa[is_normal] / 8.0

    #     # 2. 处理非正规化数
    #     exponent_normalized[is_subnormal] = float((-bias + 1))  # -6 for E4M3
    #     mantissa_scale[is_subnormal] = mantissa[is_subnormal] / 8.0

    #     # 3. 处理零值
    #     exponent_normalized[is_zero] = 0.0
    #     mantissa_scale[is_zero] = 0.0

    #     # 计算数值
    #     value = sign_factor * (2.0 ** exponent_normalized) * mantissa_scale
    #     # 应用逆缩放并限制范围
    #     value = value * scale_inv
    #     value = torch.clamp(value, -fp8_max*scale_inv, fp8_max*scale_inv)

    #     return value
    
    # def quantize_e4m3(self,tensor, scale):
    #     wgrad_qtype = Dtypes.kfloat8_e4m3
    #     dummy_amax = torch.empty((1,), device=tensor.device)  # 占位参数
    #     fp8_grad = TransformerEngineWrapper.cast_to_fp8(
    #             tensor.view(1, -1),
    #             scale,
    #             dummy_amax,
    #             torch.reciprocal(scale),
    #             wgrad_qtype
    #         ).view_as(tensor)
    #     return fp8_grad


    # def dequantize_e4m3(self,tensor, scale_inv):
    #     wgrad_qtype = Dtypes.kfloat8_e4m3
    #     fp32_grad = TransformerEngineWrapper.cast_from_fp8(
    #             tensor.view(1, -1),
    #             scale_inv,  # 补偿预缩放
    #             wgrad_qtype,
    #             Dtypes.kfloat32
    #         ).view_as(tensor)
    #     return fp32_grad

    def get_split_sizes(self, total_dim, num_splits):
        if num_splits <= 0:
            raise ValueError("num_splits must be a positive integer")
        if num_splits == 1:
            return [total_dim]
        
        chunk_base = 2048
        num_prev_splits = num_splits - 1
        max_multiple = total_dim // (chunk_base * num_prev_splits)
        sum_prev = max_multiple * chunk_base * num_prev_splits
        last_chunk = total_dim - sum_prev
        
        # 前num_prev_splits个分块大小均为max_multiple * 4096
        input_split_sizes = [max_multiple * chunk_base] * num_prev_splits
        input_split_sizes.append(last_chunk)
        
        return input_split_sizes
    
    def a2a_ag(self, inp, quant_state, group = None):
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
    
    def a2a_ag_fp16(self,inp, group = None):
        world_size = torch.distributed.get_world_size(group)
        out = torch.empty_like(inp)

        torch.distributed.all_to_all_single(out, inp, group = group)
        out = out.view(-1)
        ag_in = out.view([world_size,out.shape[0]//world_size]).sum(0)
        torch.distributed._all_gather_base(out, ag_in, group = group)
        return out.view(inp.shape)
    
    # @profile
    def start_grad8bit_sync(self):
        """
        Initiates grad sync (all-reduce or reduce-scatter) communication operation
        for this bucket.

        When overlap_grad_reduce is set to True, dispatches an asynchronous
        communication call. When overlap_grad_reduce is set to False, makes
        synchronous call.
        """
        assert (
            self.communication_handle is None and not self.communication_issued
        ), 'Should not have multiple communication calls in flight at once'
        # print(self.grad_data.value)
        # Make sure norm of grads in bucket are not NaN
        # prior to data-parallel all-reduce / reduce-scatter.
    # if self.ddp_config.check_for_nan_in_grad:
    #     global_rank = torch.distributed.get_rank()
    #     fp_grad = torch.empty_like(self.grad_data.value, dtype=torch.bfloat16)
    #     B_F.dequantize_blockwise(self.grad_data.value, self.grad_data.quant_state, out=fp_grad , blocksize=self.grad_data.quant_state.blocksize)
    #     # fp_grad= B_F.dequantize_blockwise(self.grad_data.value, self.grad_data.quant_state, blocksize=self.grad_data.quant_state.block_size)
    #     norm = fp_grad.norm(p=2)
    #     # norm = self.grad_data.norm(p=2)
    #     assert not norm.isnan(), (
    #         f'Rank {global_rank}: found NaN in local grad norm in '
    #         f'backward pass before data-parallel communication collective. '
    #         f'Device: {torch.cuda.current_device()}, node: {os.uname()[1]}'
    #     )   
    #     del fp_grad, norm
        # def print_memory_stats(prefix):
        #     torch.cuda.synchronize()
        #     allocated = torch.cuda.memory_allocated() / (1024 * 1024 * 1024)
        #     reserved = torch.cuda.memory_reserved() / (1024 * 1024 * 1024)
        #     max_allocated = torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024)
        #     print(f"{prefix}:")
        #     print(f"  Allocated: {allocated:.2f} GB")
        #     print(f"  Reserved: {reserved:.2f} GB")
        #     print(f"  Max Allocated: {max_allocated:.2f} GB")
        
        # print_memory_stats("Before fp8 conversion")
    
            # import transformer_engine_torch as tex
            
            # rank = torch.distributed.get_rank()
            # local_rank = rank % torch.cuda.device_count()
            # torch.cuda.set_device(local_rank)
            # device = torch.device(f'cuda:{local_rank}')
            
        # wgrad_qtype = self.grad_data.meta.qtype
            
        # amax_tensor = self.grad_data.meta.amax
        #     # amax_tensor = torch.tensor([amax], device=device)
        # amax_tensor.nan_to_num_(nan=torch.inf, posinf=torch.inf)  # 处理异常值
        # torch.distributed.all_reduce(amax_tensor, op=torch.distributed.ReduceOp.MAX)
        # global_amax = amax_tensor[0].clamp(min=1e-12)
        # # print(self.grad_data.meta.scale)

        # fp_max = Floating.qfp_max[wgrad_qtype]
        # new_scale = ScalingMeta.compute_scaling_factor(
        #     global_amax, 
        #     self.grad_data.meta.scale, 
        #     fp_max, 
        #     margin=0
        # ) #???

        # fp8_scale = new_scale.div(self.grad_data.meta.scale)

        # self.grad_data.value = self.scale_fp8e4m3_tensor(self.grad_data.value, fp8_scale)

        # self.grad_data.meta.amax[0] = global_amax
        # self.grad_data.meta.scale.copy_(new_scale)
        # # self.grad_data.meta.scale_inv.copy_(torch.reciprocal(new_scale))
            
        if self.gradient_scaling_factor != 1.0: # 1
            # print(self.gradient_scaling_factor)
            # fp_grad = self.dequantize_e4m3(self.grad_data.value, self.grad_data.meta.scale_inv)
            for param in self.params_list:
                fp_grad = torch.empty_like(param.main_grad.value, dtype=torch.bfloat16)
                B_F.dequantize_blockwise(param.main_grad.value, param.main_grad.quant_state, out=fp_grad , blocksize=param.main_grad.quant_state.blocksize)
                # fp_grad= B_F.dequantize_blockwise(self.grad_data.value, self.grad_data.quant_state, blocksize=self.grad_data.quant_state.block_size)
                fp_grad *= self.gradient_scaling_factor
            # amax_tensor = fp_grad.max()
            # amax_tensor.nan_to_num_(nan=torch.inf, posinf=torch.inf)  # 处理异常值
            # global_amax = amax_tensor.clamp(min=1e-12)
            
            # fp_max = Floating.qfp_max[self.grad_data.meta.qtype]
            # global_scale = ScalingMeta.compute_scaling_factor(
            #     global_amax, 
            #     self.grad_data.meta.scale, 
            #     fp_max, 
            #     margin=0
            # ) #???
            
            # # print(ag_in_fp32)
            # self.grad_data.value = self.quantize_e4m3(fp_grad, global_scale)
                B_F.quantize_blockwise(fp_grad, code=param.main_grad.quant_state.code, absmax=param.main_grad.quant_state.absmax, out=param.main_grad.value, blocksize=param.main_grad.quant_state.blocksize)
            # value, quant_state = B_F.quantize_blockwise(fp_grad, code=self.grad_data.quant_state.code, blocksize=self.grad_data.quant_state.block_size)
            # self.grad_data.value.copy_(value)
            # self.grad_data.quant_state.copy_(quant_state)
            # self.scale_fp8e4m3_tensor(self.grad_data.value, self.gradient_scaling_factor)
            # self.grad_data *= self.gradient_scaling_factor
        # Use async_op only when overlap_grad_reduce is True.
        if self.ddp_config.use_distributed_optimizer:
            local_data_view = shard_buffer(self.grad_data, self.data_parallel_world_size)[
                self.data_parallel_rank
            ]
            self.communication_handle = torch.distributed._reduce_scatter_base(
                local_data_view,
                self.grad_data,
                group=self.data_parallel_group,
                async_op=self.ddp_config.overlap_grad_reduce,
            )
        else:
            # self.communication_handle = DistOp.all_reduce(
            #     self.grad_data.value,  # 访问底层int8数据
            #     wgrad_qtype,
            #     torch.distributed.ReduceOp.SUM,
            #     group=torch.distributed.group.WORLD,
            #     async_op=self.ddp_config.overlap_grad_reduce
            # )

            # self.communication_handle = self.a2a_ag(
            #     self.grad_data.value,  # 访问底层uint8数据
            #     new_scale,
            #     group=torch.distributed.group.WORLD,
            # )
            # if torch.distributed.get_rank()==0:
            #     print(self.grad_data.value.shape)
            # self.a2a_ag(
            #     self.grad_data.value,  # 访问底层uint8数据
            #     self.grad_data.quant_state,
            #     group=self.data_parallel_group,
            # )
            for param in self.params_list:
                param.main_grad.value.copy_(self.a2a_ag(
                    param.main_grad.value,  # 访问底层uint8数据
                    param.main_grad.quant_state,
                    group=self.data_parallel_group,
                ))
                
                # fp_grad = torch.empty_like(param.main_grad.value, dtype=torch.bfloat16)
                # B_F.dequantize_blockwise(param.main_grad.value, param.main_grad.quant_state, out=fp_grad , blocksize=param.main_grad.quant_state.blocksize)
                
                
                # self.communication_handle = torch.distributed.all_reduce(
                #     fp_grad,
                #     group=self.data_parallel_group,
                #     async_op=self.ddp_config.overlap_grad_reduce, # false
                # )
                
                # B_F.quantize_blockwise(fp_grad, code=param.main_grad.quant_state.code, absmax=param.main_grad.quant_state.absmax, out=param.main_grad.value, blocksize=param.main_grad.quant_state.blocksize)

        self.communication_issued = True

    def finish_grad_sync(self):
        """
        Finishes grad sync (all-reduce or reduce-scatter) communication operation
        for this bucket.

        When overlap_grad_reduce is set to True, waits for asynchronous communication
        call to complete. When overlap_grad_reduce is set to False, makes synchronous call.
        """
        # If overlap_grad_reduce is False, start (and finish) synchronous communication call here.
        if not self.ddp_config.overlap_grad_reduce:
            if self.ddp_config.grad_reduce_in_fp8:
                self.start_grad8bit_sync()
            else:
                self.start_grad_sync()
            return
        assert self.communication_handle is not None and self.communication_issued, (
            f'Communication call has not been issued for this bucket '
            f'({len(self.params_with_grad)}/{len(self.params)} params have grad available)'
        )
        self.communication_handle.wait()

    def register_grad_ready(self, param: torch.nn.Parameter):
        """
        Registers grads for the passed-in param to be "ready" for grad sync.

        When the number of microbatches is greater than 1, we only want to register
        grads as ready when processing the last microbatch and overlap_grad_reduce is True.
        """
        assert param in self.params, 'Param is not in the bucket'
        assert param not in self.params_with_grad, 'Cannot set grad twice'
        assert (
            self.ddp_config.overlap_grad_reduce
        ), 'register_grad_ready() should be called only when overlapping grad reduce'
        self.params_with_grad.add(param)
        # If all params in bucket have grads available, issue communication call.
        if len(self.params_with_grad) == len(self.params):
            if self.ddp_config.grad_reduce_in_fp8:
                self.start_grad8bit_sync()
            else:
                self.start_grad_sync()


class ParamAndGradBuffer:
    """
    Groups parameters and gradients into a contiguous buffer, and then breaks the buffer into
    buckets with roughly `bucket_size` parameters each.

    Args:
        ddp_config: DistributedDataParallel config object.
        param_dtype: Type of param tensor.
        grad_dtype: Type of grad tensor.
        params: List of parameters whose parameters and gradients are collated in the underlying
            tensor.
        data_parallel_group: Data-parallel process group.
        bucket_size: The rough size of each bucket in terms of number of parameters.
        param_to_name: Mapping from `torch.nn.Parameter` to name (for logging purposes).
        gradient_scaling_factor: This factor is utilized to scale gradients prior to their
            communication. Its application is twofold: it facilitates the averaging of gradients
            and the scaling of gradients in the context of the Mixture of Experts (MoE) model.
    """

    def __init__(
        self,
        ddp_config: DistributedDataParallelConfig,
        param_dtype: torch.dtype,
        grad_dtype: torch.dtype,
        params: List[torch.nn.Parameter],
        data_parallel_group: torch.distributed.ProcessGroup,
        bucket_size: int,
        param_to_name: Dict[torch.nn.Parameter, str],
        gradient_scaling_factor: float,
    ):
        self.ddp_config = ddp_config

        # Check that params are unique.
        unique_params = set()
        for param in params:
            assert param not in unique_params
            unique_params.add(param)
        del unique_params

        # Store attributes that will be needed later.
        self.param_dtype = param_dtype
        self.grad_dtype = grad_dtype
        self.data_parallel_group = data_parallel_group
        self.data_parallel_world_size = torch.distributed.get_world_size(
            group=self.data_parallel_group
        )
        self.gradient_scaling_factor = gradient_scaling_factor
        self.is_last_microbatch = True

        # Data structures to store underlying buckets and relevant indexing data.
        self.buckets = []
        self.param_to_bucket = {}  # Param -> bucket mapping.
        self.param_index_map = {}  # Param -> location in buffer mapping (used in dist. optimizer).

        def _pad(number_to_be_padded: int, divisor: int) -> int:
            return int(math.ceil(number_to_be_padded / divisor) * divisor)

        def _pad_if_needed(data_index: int) -> int:
            """
            Pads data indices if using distributed optimizer (to ensure uniform sharding).
            """
            if self.ddp_config.use_distributed_optimizer:
                # Workaround for TE bug causing cuBLAS to pick an incompatible algorithm.
                # This also helps cuBLAS pick more efficient algorithms for GEMMs.
                # We now ensure that all buckets start at a memory address that is 256-byte
                # aligned (128 values since params and grads use >= 16-bit precision).
                return _pad(data_index, math.lcm(self.data_parallel_world_size, 128))
            return data_index

        # First, figure out how many elements should be in the underlying buffer storage.
        # Note that if we need to split the buffer into smaller buckets, each of these
        # might need to be padded as well (if using the distributed optimizer).
        data_start_index = 0
        bucket_data_start_index = data_start_index
        # bucket_params = set()
        bucket_params = []
        self.bucket_indices = []
        per_bucket_numel_unpadded = []
        bucket_id = 0

        def _create_new_bucket(data_end_index: int) -> int:
            """
            Create the bucket_id'th bucket with collected bucket_params, starting at
            bucket_data_start_index.
            """
            nonlocal bucket_data_start_index, bucket_params, bucket_id
            per_bucket_numel_unpadded.append(data_end_index - bucket_data_start_index)
            data_end_index = _pad_if_needed(data_end_index)
            # Update bucket metadata.
            self.bucket_indices.append((bucket_data_start_index, data_end_index))
            bucket_data_start_index = data_end_index
            # Re-set bucket_params and increment bucket_id for next bucket.
            # bucket_params = set()
            bucket_params = []
            bucket_id += 1
            # Return the potentially padded data_end_index.
            return data_end_index

        for param in params[::-1]:
            # Iterate through parameters in reverse order to roughly follow backprop order,
            # and skip parameters that don't require gradients.
            if not param.requires_grad:
                continue
            this_numel = param.data.nelement()
            data_end_index = data_start_index + this_numel

            def _does_param_require_new_bucket(param):
                """
                Split shared embedding parameters into separate bucket if using distributed
                optimizer that makes use of reduce-scatters instead of all-reduces.
                This ensures that the first and last pipeline stage partition optimizer state
                for the shared embedding parameters the same way across DP replicas, allowing
                the DP reduce-scatter to be before the embedding all-reduce.
                """
                return (
                    getattr(param, "shared_embedding", False)
                    and self.ddp_config.use_distributed_optimizer
                )

            # Create bucket with already collected parameters if current param needs its own bucket.
            if _does_param_require_new_bucket(param) and len(bucket_params) > 0:
                # We are creating a bucket for the already accumulated parameters, whose params
                # end at the current data_start_index.
                if self.ddp_config.use_distributed_optimizer:
                    # data_start_index should already be padded.
                    assert data_start_index % self.data_parallel_world_size == 0
                _create_new_bucket(data_start_index)

            self.param_index_map[param] = (
                data_start_index,
                data_end_index,
                bucket_id,
            )
            # bucket_params.add(param)
            bucket_params.append(param)

            # If we have enough elements already or the current param is part of the shared embedding
            # layer and needs a separate bucket, form a new bucket.
            if (
                bucket_size is not None
                and (data_end_index - bucket_data_start_index) >= bucket_size
            ) or _does_param_require_new_bucket(param):
                data_end_index = _create_new_bucket(data_end_index)
            data_start_index = data_end_index

        # Add remaining params to a new bucket.
        if len(bucket_params) > 0:
            data_end_index = _create_new_bucket(data_end_index)

        # Next, create underlying storage for buffer (with numel elements that includes
        # padding as necessary).
        self.numel = data_end_index
        self.numel_unpadded = sum(per_bucket_numel_unpadded)
        assert self.numel_unpadded <= self.numel
        if self.ddp_config.use_distributed_optimizer:
            assert self.numel % self.data_parallel_world_size == 0
        else:
            assert self.numel == self.numel_unpadded

        self.param_data = None
        # Only re-map param tensors if using distributed optimizer.
        if self.ddp_config.use_distributed_optimizer:
            self.param_data = torch.zeros(
                self.numel,
                dtype=self.param_dtype,
                device=torch.cuda.current_device(),
                requires_grad=False,
            )
        if self.ddp_config.grad_reduce_in_fp8:
            self.grad_data = torch.zeros(
                self.numel,
                dtype=torch.uint8,
                device=torch.cuda.current_device(),
                requires_grad=False,
            )
        else:
            self.grad_data = torch.zeros(
                self.numel,
                dtype=self.grad_dtype,
                device=torch.cuda.current_device(),
                requires_grad=False,
            )

        # Finally, map param.data and param.main_grad fields to buffers.
        # bucket_params = set()
        bucket_params = []
        bucket_data_start_index = 0
        cur_bucket_id = 0
        # num_params = len(params)
        # window_size = 1
        # t = 0
        # pre_scale = 1.0 / math.sqrt(self.data_parallel_world_size)

        # scales = torch.ones((num_params, ), device='cuda')
        # scale_invs = torch.ones((num_params, ), device='cuda')
        # amaxs = torch.zeros((num_params, window_size), device='cuda')
        
        import bitsandbytes.functional as B_F
        code = B_F.create_dynamic_map(signed=True)  # 创建动态量化映射表
        blocksize = 256
        class GradWrapper:
            def __init__(self, value, quant_state):
                self.value = value
                self.quant_state = quant_state
                self.quant_state.code = self.quant_state.code.to(value.device)
        
        
        for param in params[::-1]:
            if not param.requires_grad:
                continue
            data_start_index, data_end_index, bucket_id = self.param_index_map[param]

            # Assign param.data to appropriate segment of self.param_data.
            if self.param_data is not None:
                old_param_data = param.data
                param.data = self._get(
                    param.data.shape, data_start_index, buffer_type=BufferType.PARAM
                )
                assert old_param_data._base is None
                # Copy tensor values (from initialization or checkpoint).
                param.data.detach().copy_(old_param_data)
                del old_param_data
            # param.main_grad = self._get(
            #     param.data.shape, data_start_index, buffer_type=BufferType.GRAD
            # )
            if self.ddp_config.grad_reduce_in_fp8:
                # self.wgrad_qtype = Dtypes.kfloat8_e4m3
                grad_buffer = self._get(param.data.shape, data_start_index, buffer_type=BufferType.GRAD)
                _, quant_state= B_F.quantize_blockwise(torch.empty_like(grad_buffer, dtype=self.param_dtype).fill_(0),
                                                        code=code, 
                                                        blocksize=blocksize)
                # quant_state = B_F.QuantState(code=code,blocksize=blocksize)
                # t += 1
                param.main_grad = GradWrapper(grad_buffer, quant_state)
            else:
                param.main_grad = self._get(
                    param.data.shape, data_start_index, buffer_type=BufferType.GRAD
                )
            if bucket_id != cur_bucket_id:
                bucket_data_end_index = _pad_if_needed(data_start_index)
                self._set_bucket(
                    bucket_params=bucket_params,
                    start_index=bucket_data_start_index,
                    end_index=bucket_data_end_index,
                    numel_unpadded=per_bucket_numel_unpadded[cur_bucket_id],
                    bucket_id=cur_bucket_id,
                )
                bucket_data_start_index = bucket_data_end_index
                # bucket_params = set()
                bucket_params = []
                assert cur_bucket_id + 1 == len(self.buckets)
                assert bucket_id == cur_bucket_id + 1
                cur_bucket_id = bucket_id
            # bucket_params.add(param)
            bucket_params.append(param)

        # Add remaining params to a new bucket.
        if len(bucket_params) > 0:
            bucket_data_end_index = _pad_if_needed(data_end_index)
            self._set_bucket(
                bucket_params=bucket_params,
                start_index=bucket_data_start_index,
                end_index=bucket_data_end_index,
                numel_unpadded=per_bucket_numel_unpadded[cur_bucket_id],
                bucket_id=cur_bucket_id,
            )

        # Log buckets for all PP stages.
        if (
            parallel_state.get_data_parallel_rank(with_context_parallel=True) == 0
            and parallel_state.get_tensor_model_parallel_rank() == 0
        ):
            logger.info(
                f'Number of buckets for gradient all-reduce / reduce-scatter: {len(self.buckets)}'
            )
            for index, bucket in enumerate(self.buckets):
                numel = 0
                for param in bucket.params:
                    numel += param.data.nelement()
                logger.info(f'Params for bucket {index+1} ({numel} elements):')
                for param in bucket.params:
                    logger.info(f'    {param_to_name[param]}')

    def scale_gradients(self, scaling_factor: float) -> None:
        """Scale the gradient data by `scaling_factor`."""
        self.grad_data *= scaling_factor

    def _get(self, shape: torch.Size, start_index: int, buffer_type: BufferType) -> torch.Tensor:
        """
        Return a tensor with the input `shape` as a view into the 1-D data starting at
        `start_index`.
        """
        end_index = start_index + shape.numel()
        assert end_index <= self.numel, 'Requested tensor is out of buffer range'
        if buffer_type == BufferType.PARAM:
            assert self.param_data is not None
            buffer_tensor = self.param_data[start_index:end_index]
        elif buffer_type == BufferType.GRAD:
            buffer_tensor = self.grad_data[start_index:end_index]
        else:
            raise Exception("Illegal buffer type provided to GradBuffer._get() function")
        buffer_tensor = buffer_tensor.view(shape)
        return buffer_tensor

    def _set_bucket(
        self,
        bucket_params: List[torch.nn.Parameter],
        start_index: int,
        end_index: int,
        numel_unpadded: int,
        bucket_id: int,
    ):
        """
        Helper function to create new bucket, add it to list of buckets, and
        also update param->bucket mapping.
        """

        # Assert that indices are correctly padded (if needed), and that bucket
        # position is same as originally computed.
        if self.ddp_config.use_distributed_optimizer:
            assert start_index % self.data_parallel_world_size == 0
            assert end_index % self.data_parallel_world_size == 0
        assert (start_index, end_index) == self.bucket_indices[bucket_id]

        # Get appropriate view into global ParamAndGradBuffer.
        bucketed_param_data = None
        if self.param_data is not None:
            bucketed_param_data = self._get(
                torch.Size([end_index - start_index]), start_index, buffer_type=BufferType.PARAM
            )
        bucketed_grad_data = self._get(
            torch.Size([end_index - start_index]), start_index, buffer_type=BufferType.GRAD
        )
        bucket = Bucket(
            ddp_config=self.ddp_config,
            params=bucket_params,
            param_data=bucketed_param_data,
            grad_data=bucketed_grad_data,
            offset=start_index,
            numel_unpadded=numel_unpadded,
            data_parallel_group=self.data_parallel_group,
            data_parallel_world_size=self.data_parallel_world_size,
            gradient_scaling_factor=self.gradient_scaling_factor,            
        )
        self.buckets.append(bucket)
        for bucket_param in bucket_params:
            assert bucket_param not in self.param_to_bucket
            self.param_to_bucket[bucket_param] = bucket

    def reset(self):
        """
        Zero out the underlying grad_buffer and reset all buckets in preparation for the next
        iteration of training.
        """
        self.grad_data.zero_()
        for bucket in self.buckets:
            bucket.reset()
        self.is_last_microbatch = True

    def start_grad_sync(self):
        """
        Initiates grad sync (all-reduce or reduce-scatter) communication operations
        for all buckets in the grad buffer.

        When overlap_grad_reduce is set to True, dispatches asynchronous communication
        calls. When overlap_grad_reduce is set to False, calls synchronous
        communication ops.
        """
        for bucket in self.buckets:
            if self.ddp_config.grad_reduce_in_fp8:
                bucket.start_grad8bit_sync()
            else:
                bucket.start_grad_sync()

    def finish_grad_sync(self):
        """
        Finishes grad sync (all-reduce or reduce-scatter) communication operations
        for all buckets in the grad buffer.

        When overlap_grad_reduce is set to True, waits for asynchronous communication
        calls to complete. When overlap_grad_reduce is set to False, calls synchronous
        communication ops.
        """
        for bucket in self.buckets:
            bucket.finish_grad_sync()

    def register_grad_ready(self, param: torch.nn.Parameter):
        """
        Registers grads for the passed-in param to be "ready" for grad sync.

        When the number of microbatches is greater than 1, we only want to register
        grads as ready when processing the last microbatch and overlap_grad_reduce is True.
        """
        assert (
            self.ddp_config.overlap_grad_reduce
        ), 'register_grad_ready() should only be called when overlap_grad_reduce is True'
        if self.is_last_microbatch:
            bucket = self.param_to_bucket[param]
            bucket.register_grad_ready(param)
