from bitsandbytes.functional import quantize_4bit,dequantize_4bit,quantize_blockwise,dequantize_blockwise
from gact.ops import op_quantize, op_dequantize
from .geneNF import NF5_code,NF8_code
from typing import TYPE_CHECKING, List, Optional, Union
import torch
import os

from .split_4group import optimized_four_grouped_4bit_quantize,optimized_four_grouped_4bit_dequantize

per_layers=32/4

activation_quantization_type = os.getenv('ACTIVATION_QUANTIZATION_TYPE', 'fp16')
# import traceback
# cishu=0
# import time
# quan_timelist=[]
# dequan_timelist=[]

from .tensor_saver import TensorSaver,TensorSaverSync
saverbool=False
# saver = TensorSaverSync(
#         save_dir="/workspace/data/activation_tensors/llama2_small",
#         log_file="/workspace/ModelLink2/logs/train_llama2_7b_small_model.log",
#         # iter_list=[1,499,999 ,1499,1999],
#         iter_list=[1,50],
#     )
saver = TensorSaverSync(
        save_dir="/workspace/data/activation_tensors/llama2_7b_tp4dp2",
        log_file="/workspace/github/ModelLink/logs/202504/train_llama2_7b_node8_bs512_202504_tp4dp2_488.log",
        iter_list=[1,499,999 ,1499,1999],
        # iter_list=[1,50],
    )


def __act_quan_fp16(inputTensor:torch.Tensor):
    tensors=(inputTensor,)
    arguments=()
    return tensors,arguments

def __act_dequan_fp16(tensors:tuple[torch.tensor],arguments:tuple):

    return tensors[0]


def __act_quan_gact(inputTensor:torch.Tensor,qbit:int=4,seed:int=1):
    q_input, q_bit, q_scale, q_min = op_quantize(inputTensor, qbit, seed)
    tensors=(q_input,q_scale)
    arguments=(q_bit,q_min,inputTensor.shape)
    return tensors,arguments

def __act_dequan_gact(tensors:tuple[torch.tensor],arguments:tuple):
    q_input,q_scale=tensors
    (q_bit,q_min,inputTensor_shape)=arguments
    q_inputs = [q_input, q_bit, q_scale, q_min]
    return op_dequantize(q_inputs, inputTensor_shape)


def __act_quan_bnb(
        inputTensor:torch.Tensor,
        blocksize: int = 64,
        quant_type="nf4",
        compress_statistics=False,
        quant_storage=torch.uint8,):
    q_input,q_scale=quantize_4bit(
        inputTensor,
        quant_type=quant_type,
        blocksize=blocksize,
        compress_statistics=compress_statistics,
        quant_storage=quant_storage)
    tensors=(q_input,q_scale)
    arguments=(quant_type,blocksize,compress_statistics,quant_storage)
    return tensors,arguments

def __act_dequan_bnb(tensors:tuple[torch.tensor],arguments:tuple):
    (q_input,q_scale)=tensors
    (quant_type,blocksize,compress_statistics,quant_storage)=arguments
    return dequantize_4bit(q_input,q_scale,q_scale.absmax,out=None,blocksize=blocksize,quant_type=quant_type,)

def __act_quan_bnb_8bit(
        inputTensor:torch.Tensor,
        code: Optional[torch.Tensor] = None,
        blocksize: int = 4096,
        nested=False):
    q_input,q_scale=quantize_blockwise(
        A=inputTensor,
        code=code,
        blocksize=blocksize,
        nested=nested
        )
    tensors=(q_input,q_scale)
    arguments=(blocksize,nested)
    return tensors,arguments

def __act_dequan_bnb_8bit(tensors:tuple[torch.tensor],arguments:tuple):
    (q_input,q_scale)=tensors
    (blocksize,nested)=arguments
    return dequantize_blockwise(q_input,quant_state=q_scale,absmax=q_scale.absmax,blocksize=blocksize,nested=nested)

# @profile
def split_tensor(input_tensor, r):
    # 记录原始形状并展平
    original_shape = input_tensor.shape
    input_1d = input_tensor.flatten()
    #计算均值？这里没减去均值，测试时再用
    mean_value = input_1d.mean()
    input_1d=input_1d-mean_value
    # 计算标准差
    sita = input_1d.std()
    
    # 生成布尔掩码
    mask_bool = (input_1d.abs() > r * sita)
    # 根据掩码分类元素
    group_out = input_1d[mask_bool]
    group_in = input_1d[~mask_bool]
    
    # 将布尔掩码打包为uint8格式的正确方式
    n = mask_bool.numel()
    num_bytes = (n + 7) // 8
    padded_mask = torch.zeros(num_bytes * 8, dtype=torch.bool, device=mask_bool.device)
    padded_mask[:n] = mask_bool
    padded_mask = padded_mask.view(num_bytes, 8)
    mask_uint8 = (padded_mask.to(torch.uint8) << torch.arange(8, device=mask_bool.device)).sum(dim=1)
    
    # #mask压缩2
    # uint8_tensor = mask_bool.to(torch.uint8)
    # original_length = uint8_tensor.size(0)
    # pad = (8 - (original_length % 8)) % 8
    
    # # 填充至8的倍数并重塑为二维张量
    # padded = torch.cat([uint8_tensor, torch.zeros(pad, dtype=torch.uint8, device=input_tensor.device)]) if pad else uint8_tensor
    # packed = padded.view(-1, 8)
    
    # # 位权矩阵（2^0到2^7）并计算压缩值
    # weights = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.uint8, device=input_tensor.device)
    # mask_uint8 = (packed * weights).sum(dim=1, dtype=torch.uint8)

    return group_in, group_out, mask_uint8,original_shape,sita,mean_value

# @profile
def merge_tensor(group_in, group_out,  mask_uint8,original_shape,mean):
    # 计算原始张量的总元素数
    total_elements = torch.prod(torch.tensor(original_shape, device=mask_uint8.device)).item()
    
    # # 从uint8掩码重建布尔掩码的正确方式
    # bool_mask = (mask_uint8.unsqueeze(-1) & (1 << torch.arange(8, device=mask_uint8.device))).bool()
    # bool_mask = bool_mask.view(-1)[:total_elements]
    
    #测试解压2
    # 生成位掩码并扩展维度以支持广播
    mask = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.uint8, device=mask_uint8.device)
    expanded = mask_uint8.unsqueeze(-1)  # 形状变为 (M, 1)
    
    # 逐位提取布尔值并截断至原始长度
    bits = (expanded & mask) > 0
    bool_mask = bits.view(-1)[:total_elements]

    # # 验证分组元素数量与掩码是否匹配
    # if bool_mask.sum() != len(group_out) or (~bool_mask).sum() != len(group_in):
    #     raise ValueError("分组元素数量与掩码不匹配")
    
    # 重建原始张量
    restored = torch.empty(total_elements, dtype=group_in.dtype if group_in.numel() else group_out.dtype,
                          device=mask_uint8.device)
    restored[bool_mask] = group_out
    restored[~bool_mask] = group_in
    #加上均值
    restored=restored+mean
    return restored.reshape(original_shape)

# @profile
def split_tensor_double_scale(input_tensor, r):
    # 记录原始形状并展平
    original_shape = input_tensor.shape
    input_1d = input_tensor.flatten()
    #计算均值？这里没减去均值，测试时再用
    mean_value = input_1d.mean()
    input_1d=input_1d-mean_value
    # 计算标准差
    sita = input_1d.std()
    
    # 生成布尔掩码
    mask_bool = (input_1d.abs() > r * sita)
    
    # 根据掩码分类元素
    group_out = input_1d*mask_bool
    # group_in = input_1d*(~mask_bool)
    group_in=input_1d-group_out

    # 将布尔掩码打包为uint8格式的正确方式
    n = mask_bool.numel()
    num_bytes = (n + 7) // 8
    padded_mask = torch.zeros(num_bytes * 8, dtype=torch.bool, device=mask_bool.device)
    padded_mask[:n] = mask_bool
    padded_mask = padded_mask.view(num_bytes, 8)
    mask_uint8 = (padded_mask.to(torch.uint8) << torch.arange(8, device=mask_bool.device)).sum(dim=1)

    # #mask压缩2
    # uint8_tensor = mask_bool.to(torch.uint8)
    # original_length = uint8_tensor.size(0)
    # pad = (8 - (original_length % 8)) % 8
    
    # # 填充至8的倍数并重塑为二维张量
    # padded = torch.cat([uint8_tensor, torch.zeros(pad, dtype=torch.uint8, device=input_tensor.device)]) if pad else uint8_tensor
    # packed = padded.view(-1, 8)
    
    # # 位权矩阵（2^0到2^7）并计算压缩值
    # weights = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.uint8, device=input_tensor.device)
    # mask_uint8 = (packed * weights).sum(dim=1, dtype=torch.uint8)
    
    return group_in, group_out, mask_uint8,original_shape,sita,mean_value
# @profile
def merge_tensor_double_scale(group_in, group_out,  mask_uint8,original_shape,mean):
    # 计算原始张量的总元素数
    total_elements = group_in.numel()
    
    # # 从uint8掩码重建布尔掩码的正确方式
    # bool_mask = (mask_uint8.unsqueeze(-1) & (1 << torch.arange(8, device=mask_uint8.device))).bool()
    # bool_mask = bool_mask.view(-1)[:total_elements]

    #测试解压2
    # 生成位掩码并扩展维度以支持广播
    mask = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.uint8, device=mask_uint8.device)
    expanded = mask_uint8.unsqueeze(-1)  # 形状变为 (M, 1)
    
    # 逐位提取布尔值并截断至原始长度
    bits = (expanded & mask) > 0
    bool_mask = bits.view(-1)[:total_elements]
    
    # # 验证分组元素数量与掩码是否匹配
    # if bool_mask.sum() != len(group_out) or (~bool_mask).sum() != len(group_in):
    #     raise ValueError("分组元素数量与掩码不匹配")
    
    # 重建原始张量
    restored=torch.where(bool_mask,group_out,group_in)
    #加上均值
    restored=restored+mean
    return restored.reshape(original_shape)


def split_tensor_outer(input_tensor, r):#分类方法，仅分离出异常组
    # 记录原始形状并展平
    original_shape = input_tensor.shape
    input_1d = input_tensor.flatten()
    #计算均值？这里没减去均值，测试时再用
    mean_value = input_1d.mean()
    input_1d=input_1d-mean_value
    # 计算标准差
    sita = input_1d.std()
    
    # 生成布尔掩码
    mask_bool = (input_1d.abs() > r * sita)
    
    # 根据掩码分类元素
    group_out = input_1d[mask_bool]
    group_in = input_1d
    
    # 将布尔掩码打包为uint8格式的正确方式
    n = mask_bool.numel()
    num_bytes = (n + 7) // 8
    padded_mask = torch.zeros(num_bytes * 8, dtype=torch.bool, device=mask_bool.device)
    padded_mask[:n] = mask_bool
    padded_mask = padded_mask.view(num_bytes, 8)
    mask_uint8 = (padded_mask.to(torch.uint8) << torch.arange(8, device=mask_bool.device)).sum(dim=1)
    
    return group_in, group_out, mask_uint8,original_shape,sita,mean_value

def merge_tensor_outer(group_in, group_out,  mask_uint8,original_shape,mean):
    # 计算原始张量的总元素数
    total_elements = torch.prod(torch.tensor(original_shape, device=mask_uint8.device)).item()
    
    # 从uint8掩码重建布尔掩码的正确方式
    bool_mask = (mask_uint8.unsqueeze(-1) & (1 << torch.arange(8, device=mask_uint8.device))).bool()
    bool_mask = bool_mask.view(-1)[:total_elements]
    
    # 验证分组元素数量与掩码是否匹配
    if bool_mask.sum() != len(group_out) or len(bool_mask) != len(group_in):
        raise ValueError("分组元素数量与掩码不匹配")
    
    # 重建原始张量
    restored = torch.empty(total_elements, dtype=group_in.dtype if group_in.numel() else group_out.dtype,
                          device=mask_uint8.device)
    restored = group_in
    restored[bool_mask] = group_out
    
    #加上均值
    restored=restored+mean
    return restored.reshape(original_shape)

def split_tensor_3(input_tensor, r1, r2):
    # 确保r1 <= r2
    if r1 > r2:
        r1, r2 = r2, r1
    
    # 记录原始形状并展平
    original_shape = input_tensor.shape
    input_1d = input_tensor.flatten()
    
    # 计算标准差
    sita = input_1d.std()
    
    # 计算绝对值并确定阈值
    abs_vals = input_1d.abs()
    threshold1 = r1 * sita
    threshold2 = r2 * sita
    
    # 生成三层布尔掩码
    mask1_bool = abs_vals <= threshold1
    mask2_bool = (abs_vals > threshold1) & (abs_vals <= threshold2)
    mask3_bool = abs_vals > threshold2
    
    # 根据掩码分类元素
    group1 = input_1d[mask1_bool]
    group2 = input_1d[mask2_bool]
    group3 = input_1d[mask3_bool]
    
    # 定义掩码打包函数
    def pack_mask(mask_bool):
        n = mask_bool.numel()
        num_bytes = (n + 7) // 8
        padded_mask = torch.zeros(num_bytes * 8, dtype=torch.bool, device=mask_bool.device)
        padded_mask[:n] = mask_bool
        padded_mask = padded_mask.view(num_bytes, 8)
        mask_uint8 = (padded_mask.to(torch.uint8) << torch.arange(8, device=mask_bool.device)).sum(dim=1)
        return mask_uint8
    
    # 打包三个掩码
    # mask1_uint8 = pack_mask(mask1_bool)
    # mask2_uint8 = pack_mask(mask2_bool)
    # mask3_uint8 = pack_mask(mask3_bool)
    mask1_uint8 = mask1_bool
    mask2_uint8 = mask2_bool
    mask3_uint8 = mask3_bool
    
    return group1, mask1_uint8, group2, mask2_uint8, group3, mask3_uint8, original_shape, sita
    
def merge_tensor_3(group1, mask1_uint8, group2, mask2_uint8, group3, mask3_uint8, original_shape):
    # 计算原始张量的总元素数
    total_elements = torch.prod(torch.tensor(original_shape)).item()
    
    # 定义掩码解包函数
    def unpack_mask(mask_uint8):
        bool_mask = (mask_uint8.unsqueeze(-1) & (1 << torch.arange(8, device=mask_uint8.device))).bool()
        bool_mask = bool_mask.view(-1)[:total_elements]
        return bool_mask
    
    # 解包三个布尔掩码
    mask1_bool = unpack_mask(mask1_uint8)
    mask2_bool = unpack_mask(mask2_uint8)
    mask3_bool = unpack_mask(mask3_uint8)

    # 验证掩码完整性
    total_mask = mask1_bool | mask2_bool | mask3_bool
    if total_mask.sum() != total_elements or not torch.all(total_mask):
        raise ValueError("掩码未完整覆盖所有元素")
    if (mask1_bool & mask2_bool).any() or (mask1_bool & mask3_bool).any() or (mask2_bool & mask3_bool).any():
        raise ValueError("掩码存在区域重叠")

    # 验证分组元素数量匹配
    if mask1_bool.sum().item() != len(group1):
        raise ValueError(f"group1元素数量不匹配: 掩码{len(group1)} vs 实际{mask1_bool.sum().item()}")
    if mask2_bool.sum().item() != len(group2):
        raise ValueError(f"group2元素数量不匹配: 掩码{len(group2)} vs 实际{mask2_bool.sum().item()}")
    if mask3_bool.sum().item() != len(group3):
        raise ValueError(f"group3元素数量不匹配: 掩码{len(group3)} vs 实际{mask3_bool.sum().item()}")

    # 确定最终数据类型
    final_dtype = group3.dtype if group3.numel() > 0 else (
                    group2.dtype if group2.numel() > 0 else group1.dtype)
    
    # 重建原始张量
    restored = torch.empty(total_elements, 
                          dtype=final_dtype,
                          device=mask1_uint8.device)
    
    # 按优先级填充（group3 > group2 > group1）
    restored[mask3_bool] = group3.to(final_dtype)
    restored[mask2_bool] = group2.to(final_dtype)
    restored[mask1_bool] = group1.to(final_dtype)

    return restored.reshape(original_shape)
# # @profile
def __act_quan_split(
        inputTensor:torch.Tensor,
        group_size_in:int=64,
        group_size_out:int=64,
        std_dev:float=1.5,
        yu:float=0.8,
        quant_type_in="nf4",
        quant_type_out="nf4",
        compress_statistics=False,
        quant_storage=torch.uint8,
        split_tensor=split_tensor,
        merge_tensor=merge_tensor):
    std_dev=abs(std_dev)
    group_in, group_out, mask,shape,sita,mean = split_tensor(inputTensor, std_dev)
    threshold=std_dev*sita*yu
    group_out=torch.where(group_out > 0, group_out - threshold, group_out + threshold)
    group_in_q_input,group_in_q_scale=quantize_4bit(
        group_in,
        quant_type=quant_type_in,
        blocksize=group_size_in,
        compress_statistics=compress_statistics,
        quant_storage=quant_storage
    )
    group_out_q_input,group_out_q_scale=quantize_4bit(
        group_out,
        quant_type=quant_type_out,
        blocksize=group_size_out,
        compress_statistics=compress_statistics,
        quant_storage=quant_storage
    )
    tensors=(group_in_q_input,group_in_q_scale,group_out_q_input,group_out_q_scale,mask)
    arguments=(shape,threshold,group_size_in,group_size_out,quant_type_in,quant_type_out,mean,merge_tensor)
    return tensors,arguments  

def __act_dequan_split(tensors:tuple[torch.tensor],arguments:tuple):
    (group_in_q_input,group_in_q_scale,group_out_q_input,group_out_q_scale,mask)=tensors
    (shape,threshold,group_size_in,group_size_out,quant_type_in,quant_type_out,mean,merge_tensor)=arguments
    group_in=dequantize_4bit(group_in_q_input,group_in_q_scale,group_in_q_scale.absmax,out=None,blocksize=group_size_in,quant_type=quant_type_in)
    group_out=dequantize_4bit(group_out_q_input,group_out_q_scale,group_out_q_scale.absmax,out=None,blocksize=group_size_out,quant_type=quant_type_out)
    group_out=torch.where(group_out > 0, group_out + threshold, group_out - threshold)
    output=merge_tensor(group_in=group_in,group_out=group_out,mask_uint8=mask,original_shape=shape,mean=mean)
    return output

def __act_quan_split_gact(
        inputTensor:torch.Tensor,
        group_size_in:int=128,
        group_size_out:int=128,
        std_dev:float=1.5,
        quant_type="gact",
        compress_statistics=False,
        quant_storage=torch.uint8,):
    std_dev=abs(std_dev)
    group_in, group_out, mask,shape,sita,mean = split_tensor(inputTensor, std_dev)
    threshold=std_dev*sita*0.80
    group_out=torch.where(group_out > 0, group_out - threshold, group_out + threshold)

    group_in_tensors,group_in_arguments=__act_quan_gact(group_in)
    group_out_tensors,group_out_arguments=__act_quan_gact(group_out)  
    
    tensors=( group_in_tensors,group_in_arguments,group_out_tensors,group_out_arguments,mask)
    arguments=(shape,threshold,group_size_in,group_size_out,quant_type,mean)
    return tensors,arguments  



def __act_dequan_split_gact(tensors:tuple[torch.tensor],arguments:tuple):
    (group_in_tensors,group_in_arguments,group_out_tensors,group_out_arguments,mask)=tensors
    (shape,threshold,group_size_in,group_size_out,quan_type,mean)=arguments
    group_in=__act_dequan_gact(group_in_tensors,group_in_arguments)
    group_out=__act_dequan_gact(group_out_tensors,group_out_arguments)
    group_out=torch.where(group_out > 0, group_out + threshold, group_out - threshold)
    return merge_tensor(group_in=group_in,group_out=group_out,mask_uint8=mask,original_shape=shape,mean=mean)


def __act_quan_split_e( #这里直接将里面的部分设置为0
        inputTensor:torch.Tensor,
        group_size_in:int=64,
        group_size_out:int=64,
        std_dev:float=1.5,
        quant_type_in="nf4",
        quant_type_out="nf4",
        compress_statistics=False,
        quant_storage=torch.uint8,
        split_tensor=split_tensor,
        merge_tensor=merge_tensor):
    std_dev=abs(std_dev)
    group_in, group_out, mask,shape,sita,mean = split_tensor(inputTensor, std_dev)
    threshold=std_dev*sita*0.80
    group_out=torch.where(group_out > 0, group_out - threshold, group_out + threshold)

    group_in_q_input,group_in_q_scale=quantize_4bit(
        group_in,
        quant_type=quant_type_in,
        blocksize=group_size_in,
        compress_statistics=compress_statistics,
        quant_storage=quant_storage
    )
    group_out_q_input,group_out_q_scale=quantize_4bit(
        group_out,
        quant_type=quant_type_out,
        blocksize=group_size_out,
        compress_statistics=compress_statistics,
        quant_storage=quant_storage
    )
    
    tensors=(group_in_q_input,group_in_q_scale,group_out_q_input,group_out_q_scale,mask)
    arguments=(shape,threshold,group_size_in,group_size_out,quant_type_in,quant_type_out,mean,merge_tensor)
    return tensors,arguments  


def __act_dequan_split_e(tensors:tuple[torch.tensor],arguments:tuple):
    (group_in_q_input,group_in_q_scale,group_out_q_input,group_out_q_scale,mask)=tensors
    (shape,threshold,group_size_in,group_size_out,quant_type_in,quant_type_out,mean,merge_tensor)=arguments
    group_in=dequantize_4bit(group_in_q_input,group_in_q_scale,group_in_q_scale.absmax,out=None,blocksize=group_size_in,quant_type=quant_type_in)
    #TODO 测试用，正式版这里需要修改
    group_in=torch.zeros_like(group_in).to(group_in.device)
    group_out=dequantize_4bit(group_out_q_input,group_out_q_scale,group_out_q_scale.absmax,out=None,blocksize=group_size_out,quant_type=quant_type_out)
    group_out=torch.where(group_out > 0, group_out + threshold, group_out - threshold)
    return merge_tensor(group_in=group_in,group_out=group_out,mask_uint8=mask,original_shape=shape,mean=mean)




# def __act_quan_split_extreme(inputTensor:torch.Tensor,
#         group_size:int=64,
#         std_dev:float=1.5,
#         quant_type="nf4",
#         compress_statistics=False,
#         quant_storage=torch.uint8,):
#     std_dev=abs(std_dev)
#     # 记录原始形状并展平
#     original_shape = inputTensor.shape
#     input_1d = inputTensor.flatten()
#     #计算均值？这里没减去均值，测试时再用
#     mean_value = input_1d.mean()
#     input_1d=input_1d-mean_value
#     # 计算标准差
#     sita = input_1d.std()
    
#     # 生成布尔掩码
#     mask_bool = (input_1d.abs() > std_dev * sita)

#     # 根据掩码加减元素
#     threshold=std_dev*sita*0.80
#     group = input_1d
#     group[mask_bool] = torch.where(group[mask_bool] > 0, group[mask_bool] - threshold, group[mask_bool] + threshold)
#     # # 根据掩码分类元素
#     group_out = input_1d[mask_bool]
#     group_in = input_1d[~mask_bool]
    
#     # 将布尔掩码打包为uint8格式的正确方式
#     n = mask_bool.numel()
#     num_bytes = (n + 7) // 8
#     padded_mask = torch.zeros(num_bytes * 8, dtype=torch.bool, device=mask_bool.device)
#     padded_mask[:n] = mask_bool
#     padded_mask = padded_mask.view(num_bytes, 8)
#     mask_uint8 = (padded_mask.to(torch.uint8) << torch.arange(8, device=mask_bool.device)).sum(dim=1)
    


#     return group_in, group_out, mask_uint8,original_shape,sita,mean_value


def __act_quan_4group(
        inputTensor:torch.Tensor,
        blocksize: int = 128,
        ):
    originalshape=inputTensor.shape
    output, thresholds, absmax, block_size=optimized_four_grouped_4bit_quantize(
        inputTensor.contiguous(),
        blocksize)
    tensors=(output, thresholds, absmax)
    arguments=(block_size,originalshape)
    return tensors,arguments

def __act_dequan_4group(tensors:tuple[torch.tensor],arguments:tuple):
    (output, thresholds, absmax)=tensors
    (block_size,originalshape)=arguments
    return optimized_four_grouped_4bit_dequantize(
        input=output,
        thresholds=thresholds,
        abs_max=absmax,
        block_size=block_size,
        original_shape=originalshape
        )

def __act_quan_sub_group(
        inputTensor:torch.Tensor,
        blocksize: int = 64,
        group_size:int=32,
        group_scale_type: str = "mean",
        group_scale_bits: int = 8,
        quant_type="nf4",
        compress_statistics=False,
        quant_storage=torch.uint8,):
    """
    Args:
      input_tensor: 任意形状，只要求最后两个维度是 (H, W)
      group_size:    每多少个“行”分为一组
    Returns:
      residual:    torch.Tensor of shape (N, H, W)
      group_scale: torch.Tensor of shape (B, H, W), B = N // group_size
      orig_shape:  tuple, input_tensor.shape
    """
    orig_shape = inputTensor.shape
    H, W = orig_shape[-2], orig_shape[-1]

    N = inputTensor.numel() // (H * W)
    x3d = inputTensor.contiguous().view(N, H, W)
    if group_scale_type=="mean":
        B = N // group_size
        assert N % group_size == 0, "N must be divisible by group_size"
        x_group = x3d.view(B, group_size, H, W)
        group_scale = x_group.mean(dim=1)                             # (B, H, W)
    else:
        B = N // group_size
        head_idx = torch.arange(B, device=x3d.device) * group_size      # [0, g, 2g, ...]
        group_scale = x3d[head_idx]                                     # (B, H, W)
    if group_scale_bits == 4:
        group_tensors,group_arguments,q_group_type=activation_quantize(
            group_scale,
            quan_type="nf4"
        )
    elif group_scale_bits == 8:
        group_tensors,group_arguments,q_group_type=activation_quantize(
            group_scale,
            quan_type="bnb-8bit"
        )
    else:
        group_tensors,group_arguments,q_group_type=activation_quantize(
            group_scale,
            quan_type="fp16"
        )
    block_of = torch.arange(N, device=x3d.device) // group_size     # (N,)
    scale_per_row = group_scale[block_of]                           # (N, H, W)
    residual = x3d - scale_per_row                                   # (N, H, W)
    q_input,q_scale=quantize_4bit(
        residual,
        quant_type=quant_type,
        blocksize=blocksize,
        compress_statistics=compress_statistics,
        quant_storage=quant_storage)
    tensors=(q_input,q_scale,group_tensors,group_arguments)
    arguments=(quant_type,blocksize,compress_statistics,quant_storage,orig_shape,group_size,group_scale_type,q_group_type)
    return tensors,arguments

def __act_dequan_sub_group(tensors:tuple[torch.tensor],arguments:tuple):
    (q_input,q_scale,group_tensors,group_arguments)=tensors
    (quant_type,blocksize,compress_statistics,quant_storage,orig_shape,group_size,group_scale_type,q_group_type)=arguments
    group_scale=activation_dequantize(group_tensors,group_arguments,q_group_type)
    residual= dequantize_4bit(q_input,q_scale,q_scale.absmax,out=None,blocksize=blocksize,quant_type=quant_type,)
    N, H, W = residual.shape
    # 1. 恢复每行对应的 head
    block_of = torch.arange(N, device=residual.device) // group_size  # (N,)
    scale_per_row = group_scale[block_of]                             # (N, H, W)

    # 2. 还原
    x3d = residual + scale_per_row                                     # (N, H, W)

    # 3. reshape 回原形状
    return x3d.view(orig_shape)

def activation_quantize(
    inputTensor:torch.Tensor,
    quan_type=activation_quantization_type,
    config: Optional[Union[str, dict]] = {},
    **kargs
    ):
    # global cishu
    # import json
    # with open("/workspace/test/traceback/act_quan.json","a") as file:
    #     if cishu<10:
    #         json.dump(traceback.format_stack(),file, indent=4)
    #         print("\n")
    #         cishu+=1
    if saverbool:
        idx=config.get("idx",333)
        module=config.get("module","no")
        saver.check_and_save(inputTensor,idx,module)
    quan_type=quan_type.lower()
    if quan_type=="nf45816":
        idx=config.get("idx",4)
        if idx<=1*per_layers:
            quan_type="nf4"
        elif idx<=2*per_layers:
            quan_type="nf5"
        elif idx<=3*per_layers:
            quan_type="nf8"
        else: 
            quan_type="fp16"
    if quan_type == "sub_group":
        module=config.get("module","no")
        idx=config.get("idx",333)
        if not (("RowParallelLinear-LN_ACT_RE" in module) or ("ColumnParallelLinear" in module and idx<=16 and idx>0)):
            quan_type="nf4"
    if quan_type=="fp16":
        return __act_quan_fp16(inputTensor)+(quan_type,)
    elif quan_type=="gact":
        bit=config.get("bit",4)
        return __act_quan_gact(inputTensor,qbit=bit)+(quan_type,)
    elif quan_type=="nf4":
        blocksize=config.get("blocksize",64)
        quant_storage=config.get('quant_storage',torch.uint8)
        compress_statistics=config.get('compress_statistics',False)
        return __act_quan_bnb(
            inputTensor,
            blocksize=blocksize,
            quant_type="nf4",
            compress_statistics=compress_statistics,
            quant_storage=quant_storage
            )+(quan_type,)
    elif quan_type=="fp4":
        blocksize=config.get("blocksize",64)
        quant_storage=config.get('quant_storage',torch.uint8)
        compress_statistics=config.get('compress_statistics',False)
        return __act_quan_bnb(
            inputTensor,
            blocksize=blocksize,
            quant_type="fp4",
            compress_statistics=compress_statistics,
            quant_storage=quant_storage
            )+(quan_type,)
    elif quan_type=="sub_group":
        group_size=config.get("groupsize",32)
        group_scale_bits=config.get("group_scale_bits",8)
        group_scale_type=config.get("group_scale_type","mean")
        blocksize=config.get("blocksize",64)
        quant_storage=config.get('quant_storage',torch.uint8)
        compress_statistics=config.get('compress_statistics',False)
        return __act_quan_sub_group(
            inputTensor,
            group_size=group_size,
            group_scale_type=group_scale_type,
            group_scale_bits=group_scale_bits,
            blocksize=blocksize,
            quant_type="fp4",
            compress_statistics=compress_statistics,
            quant_storage=quant_storage
            )+(quan_type,)
    elif quan_type=="fp3":
        blocksize=config.get("blocksize",64)
        quant_storage=config.get('quant_storage',torch.uint8)
        compress_statistics=config.get('compress_statistics',False)
        return __act_quan_bnb(
            inputTensor,
            blocksize=blocksize,
            quant_type="fp3",
            compress_statistics=compress_statistics,
            quant_storage=quant_storage
            )+(quan_type,)
    elif quan_type=="bnb-8bit":
        blocksize=config.get("blocksize",4096)
        nested=config.get('nested',False)
        return __act_quan_bnb_8bit(
            inputTensor,
            blocksize=blocksize,
            nested=nested
            )+(quan_type,)
    elif quan_type=="nf5":
        blocksize=config.get("blocksize",64)
        nested=config.get('nested',False)
        return __act_quan_bnb_8bit(
            inputTensor,
            code=NF5_code,
            blocksize=blocksize,
            nested=nested
            )+(quan_type,)
    elif quan_type=="nf8":
        blocksize=config.get("blocksize",128)
        nested=config.get('nested',False)
        return __act_quan_bnb_8bit(
            inputTensor,
            code=NF8_code,
            blocksize=blocksize,
            nested=nested
            )+(quan_type,)
    elif quan_type=="4group":
        blocksize=config.get("blocksize",128)
        return __act_quan_4group(
            inputTensor,
            blocksize=blocksize,
            )+(quan_type,)
    elif quan_type=="fp2":
        blocksize=config.get("blocksize",64)
        quant_storage=config.get('quant_storage',torch.uint8)
        compress_statistics=config.get('compress_statistics',False)
        return __act_quan_bnb(
            inputTensor,
            blocksize=blocksize,
            quant_type="fp2",
            compress_statistics=compress_statistics,
            quant_storage=quant_storage
            )+(quan_type,)
    elif quan_type=="split":
        bit=config.get("bit",4)
        if bit==2:
            quant_type_in="fp2"
            quant_type_out="fp2"
        elif bit==3:
            quant_type_in="fp3"
            quant_type_out="fp3"
        else:
            quant_type_in="nf4"
            quant_type_out="nf4"
        group_size_in=config.get("group_size_in",64)
        group_size_out=config.get("group_size_out",64)
        std_dev=config.get("std_dev",1.5)
        yu=abs(config.get("yu",0.8))
        return __act_quan_split(inputTensor,
                                group_size_in=group_size_in,
                                group_size_out=group_size_out,
                                std_dev=std_dev,
                                yu=yu,
                                quant_type_in=quant_type_in,
                                quant_type_out=quant_type_out,                                
                                )+(quan_type,)
    elif quan_type=="split_dev":
        bit=config.get("bit",2)
        if bit==2:
            quant_type_in="fp2"
            quant_type_out="fp2"
        else:
            quant_type_in="nf4"
            quant_type_out="nf4"
        group_size_in=config.get("group_size_in",64)
        group_size_out=config.get("group_size_out",64)
        std_dev=config.get("std_dev",1.5)
        yu=abs(config.get("yu",0.8))
        return __act_quan_split(inputTensor,
                                group_size_in=group_size_in,
                                group_size_out=group_size_out,
                                std_dev=std_dev,
                                yu=yu,
                                quant_type_in=quant_type_in,
                                quant_type_out=quant_type_out,
                                split_tensor=split_tensor_outer,
                                merge_tensor=merge_tensor_outer                                
                                )+(quan_type,)
    elif quan_type=="split_double_scale":
        bit=config.get("bit",2)
        if bit==2:
            quant_type_in="fp2"
            quant_type_out="fp2"
        elif bit==3:
            quant_type_in="fp3"
            quant_type_out="fp3"
        else:
            quant_type_in="nf4"
            quant_type_out="nf4"
        group_size_in=config.get("group_size_in",64)
        group_size_out=config.get("group_size_out",64)
        std_dev=config.get("std_dev",1.5)
        yu=abs(config.get("yu",0.8))
        return __act_quan_split(inputTensor,
                                group_size_in=group_size_in,
                                group_size_out=group_size_out,
                                std_dev=std_dev,
                                yu=yu,
                                quant_type_in=quant_type_in,
                                quant_type_out=quant_type_out,
                                split_tensor=split_tensor_double_scale,
                                merge_tensor=merge_tensor_double_scale,                                
                                )+(quan_type,)
    elif quan_type=="split_dev_in":
        bit=config.get("bit",2)
        if bit==2:
            quant_type_in="nf4"
            quant_type_out="fp2"
        else:
            quant_type_in="nf4"
            quant_type_out="nf4"
        group_size_in=config.get("group_size_in",64)
        group_size_out=config.get("group_size_out",64)
        std_dev=config.get("std_dev",1.5)
        return __act_quan_split(inputTensor,
                                group_size_in=group_size_in,
                                group_size_out=group_size_out,
                                std_dev=std_dev,
                                quant_type_in=quant_type_in,
                                quant_type_out=quant_type_out,
                                split_tensor=split_tensor_outer,
                                merge_tensor=merge_tensor_outer                                
                                )+(quan_type,)
    elif quan_type=="split_dev_e":
        bit=config.get("bit",2)
        if bit==2:
            quant_type_in="nf4"
            quant_type_out="fp2"
        else:
            quant_type_in="nf4"
            quant_type_out="nf4"
        group_size_in=config.get("group_size_in",64)
        group_size_out=config.get("group_size_out",64)
        std_dev=config.get("std_dev",1.5)
        return __act_quan_split(inputTensor,
                                group_size_in=group_size_in,
                                group_size_out=group_size_out,
                                std_dev=std_dev,
                                quant_type_in=quant_type_in,
                                quant_type_out=quant_type_out,
                                split_tensor=split_tensor_outer,
                                merge_tensor=merge_tensor_outer                                
                                )+(quan_type,)
    elif quan_type=="split_gact":
        return __act_quan_split_gact(inputTensor)+(quan_type,)
    else:
        return __act_quan_fp16(inputTensor)+(quan_type,)

def activation_dequantize(tensors:tuple[torch.tensor],arguments:tuple,quan_type=activation_quantization_type,dtype=torch.bfloat16,**config):
    quan_type=quan_type.lower()
    if quan_type=="fp16":
        return __act_dequan_fp16(tensors,arguments).to(dtype)
    elif quan_type=="gact":
        return __act_dequan_gact(tensors,arguments).to(dtype)
    elif quan_type=="nf4":
        return __act_dequan_bnb(tensors,arguments).to(dtype)
    elif quan_type=="fp4":
        return __act_dequan_bnb(tensors,arguments).to(dtype)
    elif quan_type=="sub_group":
        return __act_dequan_sub_group(tensors,arguments).to(dtype)
    elif quan_type=="bnb-8bit":
        return __act_dequan_bnb_8bit(tensors,arguments).to(dtype)
    elif quan_type=="nf5":
        return __act_dequan_bnb_8bit(tensors,arguments).to(dtype)
    elif quan_type=="nf8":
        return __act_dequan_bnb_8bit(tensors,arguments).to(dtype)
    elif quan_type=="4group":
        return __act_dequan_4group(tensors,arguments).to(dtype)
    elif quan_type=="fp2":
        return __act_dequan_bnb(tensors,arguments).to(dtype)
    elif quan_type=="fp3":
        return __act_dequan_bnb(tensors,arguments).to(dtype)
    elif quan_type=="split":
        return __act_dequan_split(tensors,arguments).to(dtype)
    elif quan_type=="split_dev":
        return __act_dequan_split(tensors,arguments).to(dtype)
    elif quan_type=="split_double_scale":
        return __act_dequan_split(tensors,arguments).to(dtype)
    elif quan_type=="split_dev_in":
        return __act_dequan_split(tensors,arguments).to(dtype)
    elif quan_type=="split_dev_e":
        return __act_dequan_split_e(tensors,arguments).to(dtype)
    elif quan_type=="split_gact":
        return __act_dequan_split_gact(tensors,arguments).to(dtype)
    else:
        return __act_dequan_fp16(tensors,arguments).to(dtype)


if __name__ == "__main__":
    
    # torch.manual_seed(42)
    # input_tensor = torch.randn(8).to("cuda") * 2
    # tensors,argument,quantype=activation_quantize(input_tensor,"fp2")
    # print(input_tensor)
    # print(tensors,argument,quantype)
    # output=activation_dequantize(tensors=tensors,arguments=argument,quan_type=quantype)
    # print(output)
    # 测试1，分割合并测试
    # 生成测试数据
    # torch.manual_seed(42)
    # input_tensor = torch.randn(16,256,4096).to("cuda") * 2
    # r1,r2 = 0.75,1.5
    
    # # 拆分测试
    # # group_in, group_out, mask,shape,sita = split_tensor(input_tensor, r)
    # group1, mask1_uint8, group2, mask2_uint8, group3, mask3_uint8, original_shape, sita=split_tensor_3(input_tensor,r1,r2)
    # print(f"Original shape: {original_shape}")
    # # print(f"Mask.device (uint8): {mask.device}")
    # # print(f"group_in.device: {group_in.device}")
    # # print(f"group_out.device: {group_out.device}")
    
    # # 合并测试
    # try:
    #     restored = merge_tensor_3(group1, mask1_uint8, group2, mask2_uint8, group3, mask3_uint8, original_shape)
    #     print(f"restored.device: {restored.device}")
    #     print("Restoration succeeded:", torch.allclose(input_tensor.flatten(), restored.flatten()))
    # except ValueError as e:
    #     print("Restoration failed:", e)

    

    # py测试
    # import numpy as np
    
    # input_tensor = torch.randn(16,4096,4096).to("cuda") * 2
    # # mini=999
    # # minvalue=9999
    # # for i in np.arange(0.6,1,0.01):
    # #     quan_input=__act_quan_split (input_tensor,py=i)
    # #     output_tensor=activation_dequantize(quan_input,"split")
    # #     value=torch.max(torch.abs(input_tensor-output_tensor)).item()
    # #     if value<minvalue:
    # #         mini=i
    # #         minvalue=value
    # # print(f"mini:{mini}")
    # # print(f"minvalue:{minvalue}")       
    # quan_input=activation_quantize(input_tensor)
    # # print(quan_input)
    # output_tensor=activation_dequantize(quan_input)



    # print(torch.max(torch.abs(input_tensor-output_tensor)) )


    group_result={}
    torch.manual_seed(13)
    # quan_types=["split","split_dev"]
    quanlist=[
        # ("fp2","fp2",{}),
        # ("gact-2bit","gact",{"bit":2}),
        # ("split-2bit","split",{"bit":2}),
        # ("split-dscale-2bit","split_double_scale",{"bit":2}),
        # ("split-2bit-nondev","split",{"bit":2,"yu":0}),
        # ("split-2bit单组提取","split_dev",{"bit":2}),
        # ("split-2bit-nondev单组提取","split_dev",{"bit":2,"yu":0}),
        # ("fp3","fp3",{}),
        # ("split-3bit-1std","split",{"bit":3,"yu":0,"std_dev":1}),
        # ("split-3bit-1.5std","split",{"bit":3,"yu":0,"std_dev":1.5}),
        # ("split-3bit-2std","split",{"bit":3,"yu":0,"std_dev":2}),
        # ("split-dscale-3bit","split_double_scale",{"bit":3,"yu":0}),
        # ("split-dscale-3bit","split_double_scale",{"bit":3,"yu":0}),
        # ("fp4","fp4",{}),
        # ("fp4","fp4",{}),
        ("nf4","nf4",{}),
        ("nf5","nf5",{}),
        ("nf8","nf8",{}),
        # ("split-4bit","split",{"bit":4,"yu":0}),
        # ("split-dscale-4bit","split_double_scale",{"bit":4,"yu":0}),
        # ("bnb-8bit","bnb-8bit",{}),
        # ("gact-8bit","gact",{"bit":8}),
        # ("bnb-8bit","bnb-8bit",{}),
        # ("group_quant","split",{}),
        # ("group_only_out","split_dev",{})
    ]
    # quan_types=['fp16','fp4','nf4']
    inputtensor=torch.randn(16*2048*4096,dtype=torch.float16).to("cuda")
    r1,r2=0.75,1.5
    group1, mask1_uint8, group2, mask2_uint8, group3, mask3_uint8, original_shape, sita=split_tensor_3(inputtensor,r1,r2)

    def compute_deviation(group,sita,blocksize=128,r=0,py=0.8):
        threshold=sita*py*r
        # group=torch.where(group > 0, group - threshold, group + threshold)
        # tensors,arguments,quan_type=activation_quantize(group,"gact")
        tensors,arguments=__act_quan_bnb(group,blocksize=blocksize,quant_type="fp4")
        group_out=__act_dequan_bnb(tensors,arguments)
        # group_out=activation_dequantize(tensors,arguments,quan_type)
        # group_out=torch.where(group_out > 0, group_out + threshold, group_out - threshold)
        avg_deviation=torch.mean(torch.abs(group-group_out))
        max_deviation=torch.max(torch.abs(group-group_out))
        return avg_deviation.item(),max_deviation.item()
    avg_deviation_1,max_deviation_1=compute_deviation(group1,sita,r=0)
    avg_deviation_2,max_deviation_2=compute_deviation(group2,sita,r=0.75)
    avg_deviation_3,max_deviation_3=compute_deviation(group3,sita,r=1.5)
    # group_result["split_3"]={
    #     "group1(avg)":avg_deviation_1,
    #     "group1(max)":max_deviation_1,
    #     "group2(avg)":avg_deviation_2,
    #     "group2(max)":max_deviation_2,
    #     "group3(avg)":avg_deviation_3,
    #     "group3(max)":max_deviation_3,
    # }
    def split_mask_deviation(tensor,tensor_act,mask):
        group=tensor[mask]
        group_out=tensor_act[mask]
        avg_deviation=torch.mean(torch.abs(group-group_out))
        max_deviation=torch.max(torch.abs(group-group_out))
        return avg_deviation.item(),max_deviation.item()
    for quan_name,quan_type,config in quanlist:
        tensors,arguments,quan_type=activation_quantize(inputtensor.clone(),quan_type=quan_type,config=config)
        # print(quan_input)
        output_tensor=activation_dequantize(tensors,arguments,quan_type,dtype=torch.float16)
        avg_deviation_1,max_deviation_1=split_mask_deviation(inputtensor,output_tensor,mask1_uint8)
        avg_deviation_2,max_deviation_2=split_mask_deviation(inputtensor,output_tensor,mask2_uint8)
        avg_deviation_3,max_deviation_3=split_mask_deviation(inputtensor,output_tensor,mask3_uint8)
        group_result[quan_name]={
            "group1(avg)":avg_deviation_1,
            "group1(max)":max_deviation_1,
            "group2(avg)":avg_deviation_2,
            "group2(max)":max_deviation_2,
            "group3(avg)":avg_deviation_3,
            "group3(max)":max_deviation_3,
        }
        # print(f"{quan_type}最大偏差：{torch.max(torch.abs(inputtensor-output_tensor)) }")
        # print(f"{quan_type}平均偏差：{torch.mean(torch.abs(inputtensor-output_tensor)) }")
        # del quan_input
        # del output_tensor
    print(group_result)

    # tensor_size=range(1,10)
    # quanlist=[
    #     ("fp2","fp2",{}),
    #     # ("fp4","fp4",{}),
    #     # ("nf4","nf4",{}),
    #     # # ("gact-4bit","gact",{"bit":4}),
    #     # # ("gact-8bit","gact",{"bit":8}),
    #     # ("bnb-8bit","bnb-8bit",{}),
    #     # ("group_quant","split",{}),
    #     # ("group_only_out","split_dev",{})
    #     # ("gact-2bit","gact",{"bit":2}),
    #     ("split-2bit","split",{"bit":2}),
    #     ("split-dscale-2bit","split_double_scale",{"bit":2}),
    #     # ("split-2bit-nondev","split",{"bit":2,"yu":0}),
    #     # ("split-2bit单组提取","split_dev",{"bit":2}),
    #     # ("split-2bit-nondev单组提取","split_dev",{"bit":2,"yu":0}),
    #     # ("fp4","fp4",{}),
    #     ("nf4","nf4",{}),
    #     ("split-4bit","split",{"bit":4}),
    #     ("split-dscale-4bit","split_double_scale",{"bit":4}),
    # ]
    
    # quan_names = [f"{q[0]}" for q in quanlist]

    # # 1. Output header in Markdown format
    # header = f"|size|type|{'|'.join(quan_names)}|"
    # separator = f"|{'|'.join(['---']*(len(quan_names)+2))}|"
    # print("Markdown Table Header:")
    # print(header)
    # print(separator)
    # print("\n")

    # result_quant=[]
    # temp_list_quant={}
    # result_dequant=[]
    # temp_list_dequant={}
    # n=10 #实验多次取平均值 
    # import time

    # for size in tensor_size:
    #     size_K=2**size #tensor大小
    #     for quan_name,_,__ in quanlist:
    #         temp_list_quant[quan_name]=0
    #         temp_list_dequant[quan_name]=0
    #     if size<10:
    #         size_str=f"{size_K:.0f}K"
    #     else:
    #         size_str=f"{size_K/1024:.0f}M"
    #     #构造tensor
    #     # 新增的GPU warmup逻辑
    #     warmup_iters = 5  # 预热迭代次数
    #     warmuptensor=torch.randn(16*256*4096).to("cuda")
    #     for _ in range(warmup_iters):           
    #         for quan_name, quan_type, config in quanlist:
    #             # 执行量化和反量化但不记录时间
    #             tensors, arguments, _ = activation_quantize(warmuptensor, quan_type=quan_type, config=config)
    #             dequan_out = activation_dequantize(tensors, arguments, quan_type=quan_type)
    #         torch.cuda.synchronize()  # 确保GPU操作完成
    #     del warmuptensor
    #     for i in range(n):
    #         torch.manual_seed(i)
    #         inputtensor=torch.randn(1024*size_K).to("cuda")
    #         for quan_name,quan_type,config in quanlist:
    #             start_time=time.time()
    #             tensors,arguments,_=activation_quantize(inputtensor,quan_type=quan_type,config=config)
    #             quant_time=time.time()
    #             dequan_out=activation_dequantize(tensors,arguments,quan_type=quan_type)
    #             end_time=time.time()
    #             temp_list_quant[quan_name]+=(quant_time-start_time)
    #             temp_list_dequant[quan_name]+=(end_time-quant_time)
    #     for quan_name,_,__ in quanlist:
    #         temp_list_quant[quan_name]=temp_list_quant[quan_name]*1000000/n #ms
    #         temp_list_dequant[quan_name]=temp_list_dequant[quan_name]*1000000/n
    #     result_quant.append((size_str,temp_list_quant.copy()))
    #     result_dequant.append((size_str,temp_list_dequant.copy()))
    #     for (size_str, quant_times), (_, dequant_times) in zip(result_quant, result_dequant):
    #         # Quant row
    #         quant_values = '|'.join([f"{quant_times[name]:.2f}" for name in quan_names])
    #         print(f"|{size_str}|quant|{quant_values}|")
            
    #         # Dequant row 
    #         dequant_values = '|'.join([f"{dequant_times[name]:.2f}" for name in quan_names])
    #         print(f"|{size_str}|dequant|{dequant_values}|")
    #         print()


    #     filename="/workspace/test/quant_time/quant_results_split.md"
    #     with open(filename, 'w+') as f:
    #         f.write("### 量化时间表(单位：us)\n")
    #         q_type="quant"
    #         f.write(f"{header}\n")
    #         f.write(f"{separator}\n")
    #         for size_str, times in result_quant:
    #             values = '|'.join([f"{times[name]:.2f}" for name in quan_names])
    #             f.write(f"|{size_str}|{q_type}|{values}|\n")
    #         f.write("### 反量化时间表(单位：us)\n")
    #         q_type="dequant"
    #         f.write(f"{header}\n")
    #         f.write(f"{separator}\n")
    #         for size_str, times in result_dequant:
    #             values = '|'.join([f"{times[name]:.2f}" for name in quan_names])
    #             f.write(f"|{size_str}|{q_type}|{values}|\n")

    # quan_types=['nf4','split','split_dev']
    # inputtensor=torch.randn(16,2048,4096).to("cuda")
    # # input_dict=torch.randn(16,2048,4096).to("cuda")
    # # input_dict=activation_quantize(inputtensor,quan_type='gact')
    # # q_inputs = [input_dict.get("q_input"), input_dict.get("q_scale"), input_dict.get("q_min")]
    # # for tensor in q_inputs:
    # #     print(tensor.shape)
    # # dequan_out=activation_dequantize(input_dict,quan_type='gact')
    # # print(dequan_out.shape)
    # for quan_type in quan_types:
    #     starttime=time.time()
    #     for i in range(100):
    #         tensors,arguments,_=activation_quantize(inputtensor,quan_type=quan_type)
    #         # print(quan_out.get("q_input"))
    #         dequan_out=activation_dequantize(tensors,arguments,quan_type=quan_type)
    #         del tensors
    #         del dequan_out
    #     endtime=time.time()
    #     elapsed_time = endtime - starttime  # 计算运行时间
    #     print(f"{quan_type}运行时间: {elapsed_time:.6f}秒")

    # #测试每个步骤时间
    # tensor_size=range(1,6)
    # quanlist=[
    #     ("fp2","fp2",{}),
    #     ("fp4","fp4",{}),
    #     ("nf4","nf4",{}),
    #     ("gact-4bit","gact",{"bit":4}),
    #     ("gact-8bit","gact",{"bit":8}),
    #     ("bnb-8bit","bnb-8bit",{}),
    #     ("group_quant","split",{}),
    #     ("group_only_out","split_dev",{})

    # ]
    
    # quan_names = [f"{q[0]}" for q in quanlist]

    # # 1. Output header in Markdown format
    # header = f"|size|split|quant|dequant|merge|"
    # separator = f"|---|---|---|---|---|"
    # print("Markdown Table Header:")
    # print(header)
    # print(separator)
    # print("\n")

    # result_quant=[]
    # temp_list_quant={}
    # result_dequant=[]
    # temp_list_dequant={}
    # n=20 #实验多次取平均值 
    # import time

    # for size in tensor_size:
    #     size_K=2**size #tensor大小
    #     for quan_name,_,__ in quanlist:
    #         temp_list_quant[quan_name]=0
    #         temp_list_dequant[quan_name]=0
    #     if size<10:
    #         size_str=f"{size_K:.0f}K"
    #     else:
    #         size_str=f"{size_K/1024:.0f}M"
    #     #构造tensor
    #     # 新增的GPU warmup逻辑
    #     warmup_iters = 20  # 预热迭代次数
    #     warmuptensor=torch.randn(16*256*4096).to("cuda")
    #     for _ in range(warmup_iters):           
    #         tensors, arguments, _ = activation_quantize(warmuptensor, quan_type="nf4")
    #         dequan_out = activation_dequantize(tensors, arguments, quan_type="nf4")
    #         torch.cuda.synchronize()  # 确保GPU操作完成
    #     del warmuptensor

    #     torch.manual_seed(2)
    #     inputtensor=torch.randn(1024*size_K).to("cuda")
    #     tensors, arguments, _ = activation_quantize(inputtensor, quan_type="split")
    #     dequan_out = activation_dequantize(tensors, arguments, quan_type="split")
    #     # time_1=time.time()
    #     # tensors, arguments, _ = activation_quantize(inputtensor, quan_type="nf4")
    #     # time_2=time.time()
    #     # dequan_out = activation_dequantize(tensors, arguments, quan_type="nf4")
    #     # time_3=time.time()
    #     # quan_time=time_2-time_1
    #     # dequan_time=time_3-time_2
    #     # quan_timelist.append((inputtensor.numel(),0,quan_time*1000000))
    #     # dequan_timelist.append((inputtensor.numel(),dequan_time*1000000,0))
    # import math
    # for (size1,splittime,quanttime),(size,dequanttime,mergetime) in zip(quan_timelist,dequan_timelist):
    #     print(f"|{math.log2(size)}|{splittime:.2f}|{quanttime:.2f}|{dequanttime:.2f}|{mergetime:.2f}|")
    #warmup
    # inputtensor=torch.randn(16*1024*4096).to("cuda")
    # for i in range(10):
    #     input,argument,qt=activation_quantize(inputTensor=inputtensor,quan_type="nf4")
    #     ouput=activation_dequantize(input,argument,qt)

    # for i in range(100):
    #     torch.manual_seed(i)
    #     # inputtensor2=torch.randn(16*1024*4096).to("cuda")
    #     # input,argument,qt=activation_quantize(inputTensor=inputtensor2,quan_type="split")
    #     # ouput=activation_dequantize(input,argument,qt)
    #     # torch.manual_seed(12)
    #     inputtensor=torch.randn(16*1024*4096).to("cuda")
    #     input1,argument1,qt1=activation_quantize(inputTensor=inputtensor,quan_type="split")
    #     ouput1=activation_dequantize(input1,argument1,qt1)
    
