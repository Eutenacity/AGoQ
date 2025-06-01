from bitsandbytes.functional import quantize_4bit,dequantize_4bit,quantize_blockwise,dequantize_blockwise
from gact.ops import op_quantize, op_dequantize
from typing import TYPE_CHECKING, List, Optional, Union
import torch
import os
activation_quantization_type = os.getenv('ACTIVATION_QUANTIZATION_TYPE', 'fp16')

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
        blocksize: int = 4096,
        nested=False):
    q_input,q_scale=quantize_blockwise(
        A=inputTensor,
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
    
    return group_in, group_out, mask_uint8,original_shape,sita,mean_value

def merge_tensor(group_in, group_out,  mask_uint8,original_shape,mean):
    # 计算原始张量的总元素数
    total_elements = torch.prod(torch.tensor(original_shape, device=mask_uint8.device)).item()
    
    # 从uint8掩码重建布尔掩码的正确方式
    bool_mask = (mask_uint8.unsqueeze(-1) & (1 << torch.arange(8, device=mask_uint8.device))).bool()
    bool_mask = bool_mask.view(-1)[:total_elements]
    
    # 验证分组元素数量与掩码是否匹配
    if bool_mask.sum() != len(group_out) or (~bool_mask).sum() != len(group_in):
        raise ValueError("分组元素数量与掩码不匹配")
    
    # 重建原始张量
    restored = torch.empty(total_elements, dtype=group_in.dtype if group_in.numel() else group_out.dtype,
                          device=mask_uint8.device)
    restored[bool_mask] = group_out
    restored[~bool_mask] = group_in
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
    mask1_uint8 = pack_mask(mask1_bool)
    mask2_uint8 = pack_mask(mask2_bool)
    mask3_uint8 = pack_mask(mask3_bool)
    
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

def activation_quantize(
    inputTensor:torch.Tensor,
    quan_type=activation_quantization_type,
    config: Optional[Union[str, dict]] = {},
    **kargs
    ):
    quan_type=quan_type.lower()
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
    elif quan_type=="bnb-8bit":
        blocksize=config.get("blocksize",4096)
        nested=config.get('nested',False)
        return __act_quan_bnb_8bit(
            inputTensor,
            blocksize=blocksize,
            nested=nested
            )+(quan_type,)
    # elif quan_type=="fp2":
    #     blocksize=config.get("blocksize",64)
    #     quant_storage=config.get('quant_storage',torch.uint8)
    #     compress_statistics=config.get('compress_statistics',False)
    #     return __act_quan_bnb(
    #         inputTensor,
    #         blocksize=blocksize,
    #         quant_type="fp2",
    #         compress_statistics=compress_statistics,
    #         quant_storage=quant_storage
    #         )+(quan_type,)
    elif quan_type=="split":
        bit=config.get("bit",4)
        # if bit==2:
        #     quant_type_in="fp2"
        #     quant_type_out="fp2"
        # else:
        #     quant_type_in="nf4"
        #     quant_type_out="nf4"
        #暂时屏蔽fp2
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
    # elif quan_type=="split_dev":
    #     bit=config.get("bit",2)
    #     if bit==2:
    #         quant_type_in="fp2"
    #         quant_type_out="fp2"
    #     else:
    #         quant_type_in="nf4"
    #         quant_type_out="nf4"
    #     group_size_in=config.get("group_size_in",64)
    #     group_size_out=config.get("group_size_out",64)
    #     std_dev=config.get("std_dev",1.5)
    #     yu=abs(config.get("yu",0.8))
    #     return __act_quan_split(inputTensor,
    #                             group_size_in=group_size_in,
    #                             group_size_out=group_size_out,
    #                             std_dev=std_dev,
    #                             yu=yu,
    #                             quant_type_in=quant_type_in,
    #                             quant_type_out=quant_type_out,
    #                             split_tensor=split_tensor_outer,
    #                             merge_tensor=merge_tensor_outer                                
    #                             )+(quan_type,)
    # elif quan_type=="split_dev_in":
    #     bit=config.get("bit",2)
    #     if bit==2:
    #         quant_type_in="nf4"
    #         quant_type_out="fp2"
    #     else:
    #         quant_type_in="nf4"
    #         quant_type_out="nf4"
    #     group_size_in=config.get("group_size_in",64)
    #     group_size_out=config.get("group_size_out",64)
    #     std_dev=config.get("std_dev",1.5)
    #     return __act_quan_split(inputTensor,
    #                             group_size_in=group_size_in,
    #                             group_size_out=group_size_out,
    #                             std_dev=std_dev,
    #                             quant_type_in=quant_type_in,
    #                             quant_type_out=quant_type_out,
    #                             split_tensor=split_tensor_outer,
    #                             merge_tensor=merge_tensor_outer                                
    #                             )+(quan_type,)
    # elif quan_type=="split_dev_e":
        # bit=config.get("bit",2)
        # if bit==2:
        #     quant_type_in="nf4"
        #     quant_type_out="fp2"
        # else:
        #     quant_type_in="nf4"
        #     quant_type_out="nf4"
        # group_size_in=config.get("group_size_in",64)
        # group_size_out=config.get("group_size_out",64)
        # std_dev=config.get("std_dev",1.5)
        # return __act_quan_split(inputTensor,
        #                         group_size_in=group_size_in,
        #                         group_size_out=group_size_out,
        #                         std_dev=std_dev,
        #                         quant_type_in=quant_type_in,
        #                         quant_type_out=quant_type_out,
        #                         split_tensor=split_tensor_outer,
        #                         merge_tensor=merge_tensor_outer                                
        #                         )+(quan_type,)
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
    elif quan_type=="bnb-8bit":
        return __act_dequan_bnb_8bit(tensors,arguments).to(dtype)
    # elif quan_type=="fp2":
    #     return __act_dequan_bnb(tensors,arguments).to(dtype)
    elif quan_type=="split":
        return __act_dequan_split(tensors,arguments).to(dtype)
    # elif quan_type=="split_dev":
    #     return __act_dequan_split(tensors,arguments).to(dtype)
    # elif quan_type=="split_dev_in":
    #     return __act_dequan_split(tensors,arguments).to(dtype)
    # elif quan_type=="split_dev_e":
    #     return __act_dequan_split_e(tensors,arguments).to(dtype)
    elif quan_type=="split_gact":
        return __act_dequan_split_gact(tensors,arguments).to(dtype)
    else:
        return __act_dequan_fp16(tensors,arguments).to(dtype)


if __name__ == "__main__":
    pass


