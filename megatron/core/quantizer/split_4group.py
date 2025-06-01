
import torch
import triton
import triton.language as tl
from torch.cuda.amp import custom_bwd, custom_fwd


import numpy as np
from scipy.stats import norm

# from eight_group_q import q_dq_eight_group
# 2组 对应3bit + 1bit

def get_thr_two_group(b):
    # 离散化 t
    gap = 10240
    k_values = np.arange(gap + 1)  # 0 到 256
    t_values = b * k_values / gap #切分256块
    # 计算概率
    phi_t = norm.cdf(t_values)
    probabilities = phi_t - 0.5
    minv = 9999
    ans = None
    for i in range(gap):
        l = probabilities[i] * (i) + (probabilities[gap] - probabilities[i])* (gap-i)
        if minv > l :
            minv = l
            ans = i / gap * b
    return ans
# 4组 对应2bit + 2bit
def get_thr_four_group(b):
    # 离散化 t
    k_values = np.arange(257)  # 0 到 256
    t_values = b * k_values / 256 #切分256块
    # 计算概率
    phi_t = norm.cdf(t_values)
    probabilities = phi_t - 0.5
    minv = 9999
    ans = None
    for i in range(256):
        for j in range(i+1,256):
            for k in range(j+1,256):
        # print(f"k = {k:3d}, t = {t_values[k]:.4f}, P = {probabilities[k]:.4f}")
                l = probabilities[i] * (i) + (probabilities[j] - probabilities[i]) * (j-i) + (probabilities[k] - probabilities[j]) * (k-j) + (probabilities[256] - probabilities[k]) * (256-k) 
                if minv > l :
                    minv = l
                    ans = [i / 256 * b ,j / 256 * b ,k / 256 * b ]
    return ans
# for i in range(2,100):
#     print(get_thr_two_group(i))
# print(get_thr_four_group(2.7))


@triton.jit
def optimized_4bit_quant_kernel(
    input_ptr,          # 输入张量指针
    output_ptr,         # 输出量化张量指针(打包后)
    absmax_ptr,          # absmax指针
    block_size,        # 分块大小
    n_elements,        # 输入元素总数
    BLOCK_SIZE: tl.constexpr,
):
    
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    # 前半块偏移 (0 ~ BLOCK_SIZE//2 - 1)
    offsets_lo = block_start + tl.arange(0, BLOCK_SIZE // 2)
    mask_lo = offsets_lo < n_elements
    
    # 后半块偏移 (BLOCK_SIZE//2 ~ BLOCK_SIZE-1)
    offsets_hi = block_start + BLOCK_SIZE // 2 + tl.arange(0, BLOCK_SIZE // 2)
    mask_hi = offsets_hi < n_elements
    
    # 加载前后半块数据
    x_lo = tl.load(input_ptr + offsets_lo, mask=mask_lo)
    x_hi = tl.load(input_ptr + offsets_hi, mask=mask_hi)
    
    # 计算块ID
    block_id_lo = offsets_lo // block_size  # 前后半块属于同一个block
    block_id_hi = offsets_hi // block_size  # 前后半块属于同一个block
    
  
    
    # 量化前半块
    abs_x_lo = tl.abs(x_lo)
    sign_bit_lo = (x_lo < 0).to(tl.int8)
    
    absmax_lo = tl.load(absmax_ptr + block_id_lo, mask=mask_lo)
    scale_lo = (absmax_lo) / 7.0 

    quant_bits_lo = tl.minimum(tl.floor(abs_x_lo / scale_lo + 0.5) , 7).to(tl.int8)
    quant_lo = (sign_bit_lo << 3) | quant_bits_lo
    
    # 量化后半块
    abs_x_hi = tl.abs(x_hi)
    sign_bit_hi = (x_hi < 0).to(tl.int8)
    
    absmax_hi = tl.load(absmax_ptr + block_id_hi, mask=mask_hi)
    scale_hi = (absmax_hi) / 7.0 
   
    quant_bits_hi = tl.minimum(tl.floor(abs_x_hi / scale_hi + 0.5) , 7).to(tl.int8)
    quant_hi = (sign_bit_hi << 3)| quant_bits_hi

    
    # 打包：前半块放在偶数位置，后半块放在奇数位置
    packed_offset = (pid * BLOCK_SIZE // 2) + tl.arange(0, BLOCK_SIZE // 2)
    output_mask = packed_offset < (n_elements + 1) // 2
    
    # 前半块(lo)放在高4位，后半块(hi)放在低4位
    packed = (quant_lo << 4) | quant_hi
    tl.store(output_ptr + packed_offset, packed, mask=output_mask)
@triton.jit
def optimized_4bit_dequant_kernel(
    input_ptr,          # 输入量化张量指针(打包后)
    output_ptr,         # 输出反量化张量指针
    absmax_ptr,    # 高动态范围组的scale数组
    block_size,        # 分块大小
    n_elements,        # 原始元素总数
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    # 计算打包位置
    packed_offset = (pid * BLOCK_SIZE // 2) + tl.arange(0, BLOCK_SIZE // 2)
    packed = tl.load(input_ptr + packed_offset, 
                    mask=packed_offset < (n_elements + 1) // 2)
    
    # 解包：前半块在高4位，后半块在低4位
    quant_lo = (packed >> 4) & 0x0F  # 前半块
    quant_hi = packed & 0x0F          # 后半块
    
    # 计算输出位置
    offsets_lo = block_start + tl.arange(0, BLOCK_SIZE // 2)
    offsets_hi = block_start + BLOCK_SIZE // 2 + tl.arange(0, BLOCK_SIZE // 2)
    mask_lo = offsets_lo < n_elements
    mask_hi = offsets_hi < n_elements
    
   
    
    # 反量化前半块
    block_id_lo = offsets_lo // block_size
    sign_bit_lo = (quant_lo >> 3) & 0x01
    quant_bits_lo = quant_lo & 0x07
    
    absmax_lo = tl.load(absmax_ptr + block_id_lo, mask=mask_lo)
    scale_lo = (absmax_lo) / 7.0 
    
    val_lo = quant_bits_lo * scale_lo 
    dequant_lo =  tl.where(sign_bit_lo == 1, -val_lo, val_lo)
    
    # 反量化后半块
    block_id_hi = offsets_hi // block_size
    sign_bit_hi = (quant_hi >> 3) & 0x01
    quant_bits_hi = quant_hi & 0x07
    
    absmax_hi = tl.load(absmax_ptr + block_id_hi, mask=mask_hi)
    scale_hi = (absmax_hi) / 7.0 
    
    val_hi = quant_bits_hi * scale_hi 
    dequant_hi =  tl.where(sign_bit_hi == 1, -val_hi, val_hi)
    
    # 存储结果
    tl.store(output_ptr + offsets_lo, dequant_lo, mask=mask_lo)
    tl.store(output_ptr + offsets_hi, dequant_hi, mask=mask_hi)

def optimized_4bit_quantize(input: torch.Tensor, block_size: int):
    assert input.is_contiguous(), "输入张量必须是连续的"
    
    # 预处理:计算每块的scale_high
    n_blocks = (input.numel() + block_size - 1) // block_size
    absmax = torch.zeros(n_blocks, device=input.device)
    
    abs_input = torch.abs(input)
    
    input_view = input.view(n_blocks,block_size)
    absmax = input_view.abs().max(dim = 1)[0]
    
    # 分配输出
    output_size = (input.numel() + 1) // 2
    output = torch.zeros(output_size, dtype=torch.int8, device=input.device)
    
    # 启动核函数
    n_elements = input.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    optimized_4bit_quant_kernel[grid](
        input, output, absmax, block_size, n_elements, BLOCK_SIZE=128
    )
    
    return output, absmax, block_size

def optimized_4bit_dequantize(
    input: torch.Tensor,
    abs_max: torch.Tensor,
    block_size: int,
    original_shape: tuple
):
    assert input.is_contiguous(), "输入张量必须是连续的"
    
    # 分配输出
    output = torch.empty(original_shape, dtype=torch.float32, device=input.device)
    n_elements = output.numel()
    
    # 启动核函数
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    optimized_4bit_dequant_kernel[grid](
        input, output, abs_max, 
        block_size, n_elements, BLOCK_SIZE=128
    )
    
    return output

@triton.jit
def four_group_4bit_quant_kernel(
    input_ptr,          # 输入张量指针
    output_ptr,         # 输出量化张量指针(打包后)
    thresholds_ptr,     # 阈值数组指针(包含3个阈值)
    absmax_ptr,         # absmax指针
    block_size,         # 分块大小
    n_elements,         # 输入元素总数
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    # 前半块偏移 (0 ~ BLOCK_SIZE//2 - 1)
    offsets_lo = block_start + tl.arange(0, BLOCK_SIZE // 2)
    mask_lo = offsets_lo < n_elements
    
    # 后半块偏移 (BLOCK_SIZE//2 ~ BLOCK_SIZE-1)
    offsets_hi = block_start + BLOCK_SIZE // 2 + tl.arange(0, BLOCK_SIZE // 2)
    mask_hi = offsets_hi < n_elements
    
    # 加载前后半块数据
    x_lo = tl.load(input_ptr + offsets_lo, mask=mask_lo)
    x_hi = tl.load(input_ptr + offsets_hi, mask=mask_hi)
    
    # 计算块ID
    block_id_lo = offsets_lo // block_size
    block_id_hi = offsets_hi // block_size
    
    # 量化前半块
    abs_x_lo = tl.abs(x_lo)
    # 加载3个阈值(thr0 < thr1 < thr2)
    thr0_lo = tl.load(thresholds_ptr + block_id_lo * 3, mask=mask_lo)
    thr1_lo = tl.load(thresholds_ptr + block_id_lo * 3 + 1, mask=mask_lo)
    thr2_lo = tl.load(thresholds_ptr + block_id_lo * 3 + 2, mask=mask_lo)
    
    # 确定分组(2-bit, 4组)
    group_bits_lo = tl.where(
        abs_x_lo <= thr0_lo, 0,          # 组0: [0, thr0]
        tl.where(
            abs_x_lo <= thr1_lo, 1,      # 组1: (thr0, thr1]
            tl.where(
                abs_x_lo <= thr2_lo, 2,  # 组2: (thr1, thr2]
                3                        # 组3: (thr2, ∞)
            )
        )
    )
    
    sign_bit_lo = (x_lo < 0).to(tl.int8)
    absmax_lo = tl.load(absmax_ptr + block_id_lo, mask=mask_lo)
    
    # FP4类型量化
    scale_lo = tl.where(
        group_bits_lo == 0, thr0_lo / 2.0,
        tl.where(
            group_bits_lo == 1, (thr1_lo - thr0_lo) / 2.0,
            tl.where(
                group_bits_lo == 2, (thr2_lo - thr1_lo) / 2.0,
                (absmax_lo - thr2_lo) / 2.0  # 组3
            )
        )
    )
    
    centered_val_lo = tl.where(
        group_bits_lo == 0, abs_x_lo,
        tl.where(
            group_bits_lo == 1, abs_x_lo - thr0_lo,
            tl.where(
                group_bits_lo == 2, abs_x_lo - thr1_lo,
                abs_x_lo - thr2_lo  # 组3
            )
        )
    )
    
    quant_bits_lo = tl.floor(centered_val_lo / scale_lo).to(tl.int8)
    quant_bits_lo = tl.minimum(quant_bits_lo, 1)  # 1-bit量化(0或1)
    
    # 组合成4-bit: [sign(1)|group(2)|quant(1)]
    packed_4bit_lo = (sign_bit_lo << 3) | (group_bits_lo << 1) | quant_bits_lo
    
    # 量化后半块
    abs_x_hi = tl.abs(x_hi)
    thr0_hi = tl.load(thresholds_ptr + block_id_hi * 3, mask=mask_hi)
    thr1_hi = tl.load(thresholds_ptr + block_id_hi * 3 + 1, mask=mask_hi)
    thr2_hi = tl.load(thresholds_ptr + block_id_hi * 3 + 2, mask=mask_hi)
    
    group_bits_hi = tl.where(
        abs_x_hi <= thr0_hi, 0,
        tl.where(
            abs_x_hi <= thr1_hi, 1,
            tl.where(
                abs_x_hi <= thr2_hi, 2,
                3
            )
        )
    )
    
    sign_bit_hi = (x_hi < 0).to(tl.int8)
    absmax_hi = tl.load(absmax_ptr + block_id_hi, mask=mask_hi)
    
    scale_hi = tl.where(
        group_bits_hi == 0, thr0_hi / 2.0,
        tl.where(
            group_bits_hi == 1, (thr1_hi - thr0_hi) / 2.0,
            tl.where(
                group_bits_hi == 2, (thr2_hi - thr1_hi) / 2.0,
                (absmax_hi - thr2_hi) / 2.0  # 组3
            )
        )
    )
    
    centered_val_hi = tl.where(
        group_bits_hi == 0, abs_x_hi,
        tl.where(
            group_bits_hi == 1, abs_x_hi - thr0_hi,
            tl.where(
                group_bits_hi == 2, abs_x_hi - thr1_hi,
                abs_x_hi - thr2_hi  # 组3
            )
        )
    )
    
    quant_bits_hi = tl.floor(centered_val_hi / scale_hi).to(tl.int8)
    quant_bits_hi = tl.minimum(quant_bits_hi, 1)
    
    packed_4bit_hi = (sign_bit_hi << 3) | (group_bits_hi << 1) | quant_bits_hi
    
    # 打包：前半块放在偶数位置，后半块放在奇数位置
    packed_offset = (pid * BLOCK_SIZE // 2) + tl.arange(0, BLOCK_SIZE // 2)
    output_mask = packed_offset < (n_elements + 1) // 2
    
    # 前半块(lo)放在高4位，后半块(hi)放在低4位
    packed = (packed_4bit_lo << 4) | packed_4bit_hi
    tl.store(output_ptr + packed_offset, packed, mask=output_mask)


@triton.jit
def four_group_4bit_dequant_kernel(
    input_ptr,          # 输入量化张量指针(打包后)
    output_ptr,         # 输出反量化张量指针
    thresholds_ptr,     # 阈值数组指针(包含3个阈值)
    absmax_ptr,         # absmax指针
    block_size,         # 分块大小
    n_elements,         # 原始元素总数
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    # 计算打包位置
    packed_offset = (pid * BLOCK_SIZE // 2) + tl.arange(0, BLOCK_SIZE // 2)
    packed = tl.load(input_ptr + packed_offset, 
                    mask=packed_offset < (n_elements + 1) // 2)
    
    # 解包：前半块在高4位，后半块在低4位
    quant_lo = (packed >> 4) & 0x0F  # 前半块
    quant_hi = packed & 0x0F          # 后半块
    
    # 计算输出位置
    offsets_lo = block_start + tl.arange(0, BLOCK_SIZE // 2)
    offsets_hi = block_start + BLOCK_SIZE // 2 + tl.arange(0, BLOCK_SIZE // 2)
    mask_lo = offsets_lo < n_elements
    mask_hi = offsets_hi < n_elements
    
    # 反量化前半块
    block_id_lo = offsets_lo // block_size
    sign_bit_lo = (quant_lo >> 3) & 0x01
    group_bits_lo = (quant_lo >> 1) & 0x03
    quant_bit_lo = quant_lo & 0x01
    
    thr0_lo = tl.load(thresholds_ptr + block_id_lo * 3, mask=mask_lo)
    thr1_lo = tl.load(thresholds_ptr + block_id_lo * 3 + 1, mask=mask_lo)
    thr2_lo = tl.load(thresholds_ptr + block_id_lo * 3 + 2, mask=mask_lo)
    absmax_lo = tl.load(absmax_ptr + block_id_lo, mask=mask_lo)
    
    scale_lo = tl.where(
        group_bits_lo == 0, thr0_lo / 2.0,
        tl.where(
            group_bits_lo == 1, (thr1_lo - thr0_lo) / 2.0,
            tl.where(
                group_bits_lo == 2, (thr2_lo - thr1_lo) / 2.0,
                (absmax_lo - thr2_lo) / 2.0  # 组3
            )
        )
    )
    
    centered_val_lo = (quant_bit_lo + 0.5) * scale_lo
    
    abs_val_lo = tl.where(
        group_bits_lo == 0, centered_val_lo,
        tl.where(
            group_bits_lo == 1, centered_val_lo + thr0_lo,
            tl.where(
                group_bits_lo == 2, centered_val_lo + thr1_lo,
                centered_val_lo + thr2_lo  # 组3
            )
        )
    )
    
    dequant_lo = tl.where(sign_bit_lo == 1, -abs_val_lo, abs_val_lo)
    
    # 反量化后半块
    block_id_hi = offsets_hi // block_size
    sign_bit_hi = (quant_hi >> 3) & 0x01
    group_bits_hi = (quant_hi >> 1) & 0x03
    quant_bit_hi = quant_hi & 0x01
    
    thr0_hi = tl.load(thresholds_ptr + block_id_hi * 3, mask=mask_hi)
    thr1_hi = tl.load(thresholds_ptr + block_id_hi * 3 + 1, mask=mask_hi)
    thr2_hi = tl.load(thresholds_ptr + block_id_hi * 3 + 2, mask=mask_hi)
    absmax_hi = tl.load(absmax_ptr + block_id_hi, mask=mask_hi)
    
    scale_hi = tl.where(
        group_bits_hi == 0, thr0_hi / 2.0,
        tl.where(
            group_bits_hi == 1, (thr1_hi - thr0_hi) / 2.0,
            tl.where(
                group_bits_hi == 2, (thr2_hi - thr1_hi) / 2.0,
                (absmax_hi - thr2_hi) / 2.0  # 组3
            )
        )
    )
    
    centered_val_hi = (quant_bit_hi + 0.5) * scale_hi
    
    abs_val_hi = tl.where(
        group_bits_hi == 0, centered_val_hi,
        tl.where(
            group_bits_hi == 1, centered_val_hi + thr0_hi,
            tl.where(
                group_bits_hi == 2, centered_val_hi + thr1_hi,
                centered_val_hi + thr2_hi  # 组3
            )
        )
    )
    
    dequant_hi = tl.where(sign_bit_hi == 1, -abs_val_hi, abs_val_hi)
    
    # 存储结果
    tl.store(output_ptr + offsets_lo, dequant_lo, mask=mask_lo)
    tl.store(output_ptr + offsets_hi, dequant_hi, mask=mask_hi)

def get_thr_n(absmax):
    # t= get_thr_four_group(2.7)
    # print(t)
    t=[0.2578125, 0.45703125, 0.67578125]
    thresholds = torch.tensor(t).cuda()
    thresholds = thresholds.unsqueeze(0) * absmax.unsqueeze(-1)
    return thresholds
def optimized_four_grouped_4bit_quantize(input: torch.Tensor, block_size: int):
    assert input.is_contiguous(), "输入张量必须是连续的"
    input_a = input.view(-1,input.shape[-1])
    t,m = input_a.shape
    # 预处理:计算每块的scale_high
    n_blocks = (input.numel() + block_size - 1) // block_size
    absmax = torch.zeros(n_blocks, device=input.device)
    
    abs_input = torch.abs(input)
    
    input_view = input.view(n_blocks,block_size)
    absmax = input_view.abs().max(dim = 1)[0]

    #thr block
    # mean = input.mean().cpu().item()
    # std = input.std().cpu().item()

    # thresholds = torch.empty(n_blocks,3).cuda()
    # absmax_cpu = absmax.cpu()
    # for i in range(n_blocks):
    #     r = ((absmax_cpu[i]-mean) / std)
    #     if r<2.7:

    #         thr = torch.tensor(get_thr_four_group( r.item()))
    #     else:
    #         thr = torch.tensor([.44296875,.928125,1.5609375])
    #     thresholds[i].copy_( mean + thr * std)
    #thr whole
    
    # mean = input_a.mean(dim = -1,keepdim = True).expand(t,n_blocks//t)
    # std = input_a.std(dim = -1,keepdim = True).expand(t,n_blocks//t)
    # thresholds0 = (mean + .44296875 *std).view(n_blocks,1)
    # thresholds1 = (mean + .928125 *std).view(n_blocks,1)
    # thresholds2 = (mean + 1.5609375 *std).view(n_blocks,1)
    # thresholds = torch.cat([thresholds0,thresholds1,thresholds2],dim = 1)
    # print("optimized_four_grouped_4bit_quantizeinto============")
    thresholds = get_thr_n(absmax)
    # print(thresholds)
    # 分配输出
    output_size = (input.numel() + 1) // 2
    output = torch.zeros(output_size, dtype=torch.int8, device=input.device)
    
    # 启动核函数
    n_elements = input.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    four_group_4bit_quant_kernel[grid](
        input, output, thresholds, absmax, block_size, n_elements, BLOCK_SIZE=128
    )
    
    return output, thresholds, absmax, block_size

def optimized_four_grouped_4bit_dequantize(
    input: torch.Tensor,
    thresholds: torch.Tensor,
    abs_max: torch.Tensor,
    block_size: int,
    original_shape: tuple
):
    assert input.is_contiguous(), "输入张量必须是连续的"
    
    # 分配输出
    output = torch.empty(original_shape, dtype=torch.float32, device=input.device)
    n_elements = output.numel()
    
    # 启动核函数
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    four_group_4bit_dequant_kernel[grid](
        input, output, thresholds, abs_max, 
        block_size, n_elements, BLOCK_SIZE=128
    )
    
    return output
@triton.jit
def optimized_grouped_4bit_quant_kernel(
    input_ptr,          # 输入张量指针
    output_ptr,         # 输出量化张量指针(打包后)
    thresholds_ptr,     # 阈值数组指针
    absmax_ptr,          # absmax指针
    block_size,        # 分块大小
    n_elements,        # 输入元素总数
    BLOCK_SIZE: tl.constexpr,
):
    
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    # 前半块偏移 (0 ~ BLOCK_SIZE//2 - 1)
    offsets_lo = block_start + tl.arange(0, BLOCK_SIZE // 2)
    mask_lo = offsets_lo < n_elements
    
    # 后半块偏移 (BLOCK_SIZE//2 ~ BLOCK_SIZE-1)
    offsets_hi = block_start + BLOCK_SIZE // 2 + tl.arange(0, BLOCK_SIZE // 2)
    mask_hi = offsets_hi < n_elements
    
    # 加载前后半块数据
    x_lo = tl.load(input_ptr + offsets_lo, mask=mask_lo)
    x_hi = tl.load(input_ptr + offsets_hi, mask=mask_hi)
    
    # 计算块ID
    block_id_lo = offsets_lo // block_size  # 前后半块属于同一个block
    block_id_hi = offsets_hi // block_size  # 前后半块属于同一个block
    
  
    
    # 量化前半块
    abs_x_lo = tl.abs(x_lo)
    thr_lo = tl.load(thresholds_ptr + block_id_lo, mask=mask_lo)
    group_bit_lo = (abs_x_lo > thr_lo).to(tl.int8)
    sign_bit_lo = (x_lo < 0).to(tl.int8)
    
    absmax_lo = tl.load(absmax_ptr + block_id_lo, mask=mask_lo)
    
    # nf4 type
    # normalized_lo = tl.where(
    #     group_bit_lo == 1,  # 高动态范围组
    #     (abs_x_lo - thr_lo) / (absmax_lo-thr_lo),  # 假设高组范围是[thr, 4*thr]
    #     abs_x_lo / thr_lo       # 低动态范围组[0, thr]
    # )


    # quant_bits_lo = tl.where(
    #     normalized_lo < 0.25, 0,  # 00
    #     tl.where(
    #         normalized_lo < 0.5, 1,  # 01
    #         tl.where(
    #             normalized_lo < 0.75, 2,  # 10
    #             3                     # 11
    #         )
    #     )
    # )
    # fp4 type
    scale_lo = (absmax_lo-thr_lo) / 4.0 
    scale_low_lo = thr_lo / 4.0
    quant_bits_lo = tl.where(
        group_bit_lo == 1,
        tl.floor((abs_x_lo - thr_lo) / scale_lo ),  # 高动态范围组
        tl.floor(abs_x_lo / scale_low_lo )             # 低动态范围组
    ).to(tl.int8)
    quant_bits_lo = tl.minimum(quant_bits_lo, 3)
    quant_lo = (sign_bit_lo << 3) | (group_bit_lo << 2) | quant_bits_lo
    
    # 量化后半块
    abs_x_hi = tl.abs(x_hi)
    thr_hi = tl.load(thresholds_ptr + block_id_hi, mask=mask_hi)
    group_bit_hi = (abs_x_hi > thr_hi).to(tl.int8)
    sign_bit_hi = (x_hi < 0).to(tl.int8)
    
    absmax_hi = tl.load(absmax_ptr + block_id_hi, mask=mask_hi)

    #nf4 type
    # normalized_hi = tl.where(
    #     group_bit_hi == 1,  # 高动态范围组
    #     (abs_x_hi - thr_hi) / (absmax_hi-thr_hi),  # 假设高组范围是[thr, 4*thr]
    #     abs_x_hi / thr_hi       # 低动态范围组[0, thr]
    # )


    # quant_bits_hi = tl.where(
    #     normalized_hi < 0.25, 0,  # 00
    #     tl.where(
    #         normalized_hi < 0.5, 1,  # 01
    #         tl.where(
    #             normalized_hi < 0.75, 2,  # 10
    #             3                     # 11
    #         )
    #     )
    # )

    # fp4 type
    scale_hi = (absmax_hi-thr_hi) / 4.0 
    scale_low_hi = thr_hi / 4.0
    quant_bits_hi = tl.where(
        group_bit_hi == 1,
        tl.floor((abs_x_hi - thr_hi) / scale_hi ),  # 高动态范围组
        tl.floor(abs_x_hi / scale_low_hi )             # 低动态范围组
    ).to(tl.int8)
    quant_bits_hi = tl.minimum(quant_bits_hi, 3)


    quant_hi = (sign_bit_hi << 3) | (group_bit_hi << 2) | quant_bits_hi

    
    # 打包：前半块放在偶数位置，后半块放在奇数位置
    packed_offset = (pid * BLOCK_SIZE // 2) + tl.arange(0, BLOCK_SIZE // 2)
    output_mask = packed_offset < (n_elements + 1) // 2
    
    # 前半块(lo)放在高4位，后半块(hi)放在低4位
    packed = (quant_lo << 4) | quant_hi
    tl.store(output_ptr + packed_offset, packed, mask=output_mask)
@triton.jit
def optimized_grouped_4bit_dequant_kernel(
    input_ptr,          # 输入量化张量指针(打包后)
    output_ptr,         # 输出反量化张量指针
    thresholds_ptr,     # 阈值数组指针
    absmax_ptr,    # 高动态范围组的scale数组
    block_size,        # 分块大小
    n_elements,        # 原始元素总数
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    # 计算打包位置
    packed_offset = (pid * BLOCK_SIZE // 2) + tl.arange(0, BLOCK_SIZE // 2)
    packed = tl.load(input_ptr + packed_offset, 
                    mask=packed_offset < (n_elements + 1) // 2)
    
    # 解包：前半块在高4位，后半块在低4位
    quant_lo = (packed >> 4) & 0x0F  # 前半块
    quant_hi = packed & 0x0F          # 后半块
    
    # 计算输出位置
    offsets_lo = block_start + tl.arange(0, BLOCK_SIZE // 2)
    offsets_hi = block_start + BLOCK_SIZE // 2 + tl.arange(0, BLOCK_SIZE // 2)
    mask_lo = offsets_lo < n_elements
    mask_hi = offsets_hi < n_elements
    
   
    
    # 反量化前半块
    block_id_lo = offsets_lo // block_size
    sign_bit_lo = (quant_lo >> 3) & 0x01
    group_bit_lo = (quant_lo >> 2) & 0x01
    quant_bits_lo = quant_lo & 0x03
    
    thr_lo = tl.load(thresholds_ptr + block_id_lo, mask=mask_lo)
    absmax_lo = tl.load(absmax_ptr + block_id_lo, mask=mask_lo)

    #nf4 type
    # nf4_val_lo = tl.where(
    #     quant_bits_lo == 0, 0.125,  # 00
    #     tl.where(
    #         quant_bits_lo == 1, 0.375,  # 01
    #         tl.where(
    #             quant_bits_lo == 2, 0.625,  # 10
    #             0.875                     # 11
    #         )
    #     )
    # )
    # val_lo = tl.where(group_bit_lo == 1,(nf4_val_lo) * (absmax_lo-thr_lo) + thr_lo, (nf4_val_lo) * thr_lo)

    #fp4 type
    scale_lo = (absmax_lo-thr_lo) / 4.0 
    scale_low_lo = thr_lo / 4.0
    val_lo = tl.where(group_bit_lo == 1,(quant_bits_lo +0.5) * scale_lo + thr_lo, (quant_bits_lo +0.5) * scale_low_lo)



    dequant_lo =  tl.where(sign_bit_lo == 1, -val_lo, val_lo)
    
    # 反量化后半块
    block_id_hi = offsets_hi // block_size
    sign_bit_hi = (quant_hi >> 3) & 0x01
    group_bit_hi = (quant_hi >> 2) & 0x01
    quant_bits_hi = quant_hi & 0x03
    
    thr_hi = tl.load(thresholds_ptr + block_id_hi, mask=mask_hi)
    absmax_hi = tl.load(absmax_ptr + block_id_hi, mask=mask_hi)

    #nf4 type
    # nf4_val_hi = tl.where(
    #     quant_bits_hi == 0, 0.125,  # 00
    #     tl.where(
    #         quant_bits_hi == 1, 0.375,  # 01
    #         tl.where(
    #             quant_bits_hi == 2, 0.625,  # 10
    #             0.875                     # 11
    #         )
    #     )
    # )
    # val_hi = tl.where(group_bit_hi == 1,(nf4_val_hi) * (absmax_hi-thr_hi) + thr_hi, (nf4_val_hi) * thr_hi)

    #fp4 type
    scale_hi = (absmax_hi-thr_hi) / 4.0 
    scale_low_hi = thr_hi / 4.0
    val_hi = tl.where(group_bit_hi == 1,(quant_bits_hi+0.5) * scale_hi + thr_hi, (quant_bits_hi+0.5) * scale_low_hi)


    dequant_hi =  tl.where(sign_bit_hi == 1, -val_hi, val_hi)
    
    # 存储结果
    tl.store(output_ptr + offsets_lo, dequant_lo, mask=mask_lo)
    tl.store(output_ptr + offsets_hi, dequant_hi, mask=mask_hi)



def optimized_grouped_4bit_quantize(input: torch.Tensor, block_size: int):
    assert input.is_contiguous(), "输入张量必须是连续的"
    # 预处理:计算每块的scale_high
    n_blocks = (input.numel() + block_size - 1) // block_size
    absmax = torch.zeros(n_blocks, device=input.device)
    
    abs_input = torch.abs(input)
    
    input_view = input.view(n_blocks,block_size)
    absmax = input_view.abs().max(dim = 1)[0]
    mean = input.mean().expand_as(absmax)
    std = input.std().expand_as(absmax)
    # thresholds = torch.empty_like(absmax)
    # for i in range(n_blocks):
    #     thresholds[i] = get_thr_two_group( ((absmax[i]-mean) / std).cpu().item())
    # thresholds = mean + get_thr_two_group(((absmax.max()-input.mean())/input.std()).cpu().item()) * std
    thresholds = (mean + .97 *std)
    # print(thresholds)
    # 分配输出
    output_size = (input.numel() + 1) // 2
    output = torch.zeros(output_size, dtype=torch.int8, device=input.device)
    
    # 启动核函数
    n_elements = input.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    optimized_grouped_4bit_quant_kernel[grid](
        input, output, thresholds, absmax, block_size, n_elements, BLOCK_SIZE=128
    )
    
    return output, thresholds, absmax, block_size

def optimized_grouped_4bit_dequantize(
    input: torch.Tensor,
    thresholds: torch.Tensor,
    abs_max: torch.Tensor,
    block_size: int,
    original_shape: tuple
):
    assert input.is_contiguous(), "输入张量必须是连续的"
    
    # 分配输出
    output = torch.empty(original_shape, dtype=torch.float32, device=input.device)
    n_elements = output.numel()
    
    # 启动核函数
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    optimized_grouped_4bit_dequant_kernel[grid](
        input, output, thresholds, abs_max, 
        block_size, n_elements, BLOCK_SIZE=128
    )
    
    return output

max_m = 0
mean_m = 0
max_m_nf4 = 0
mean_m_nf4 = 0
from bitsandbytes.functional import quantize_blockwise,dequantize_blockwise,quantize_4bit,dequantize_4bit
def q_dq_nf4(x):
    d,quant_state = quantize_4bit(x,quant_type = "nf4",blocksize = 128,quant_storage = torch.uint8)
    ret = dequantize_4bit(d,quant_state,quant_state.absmax,blocksize = 128,quant_type = "nf4")
    
    diff = (x-ret).abs()
    return diff
def q_dq_no_group(x):
    outputs= optimized_4bit_quantize(x,128)
    output, abs_max, block_size  = outputs
    # print(outputs)
    y=optimized_4bit_dequantize(output, abs_max,  block_size,x.shape)
    diff = (x-y).abs()
  
    # print(y)
    return diff
def q_dq_four_group(x):
    outputs= optimized_four_grouped_4bit_quantize(x,128)
    output, thresholds,abs_max, block_size  = outputs
    # print(outputs)
    y=optimized_four_grouped_4bit_dequantize(output, thresholds,abs_max,  block_size,x.shape)
    diff = (x-y).abs()
  
    # print(y)
    return diff
def q_dq_two_group(x):
    outputs= optimized_grouped_4bit_quantize(x,128)
    output, thresholds, abs_max, block_size  = outputs
    # print(outputs)
    y = optimized_grouped_4bit_dequantize(output, thresholds, abs_max,  block_size,x.shape)
    # print(x)
    # print(y)
    diff = (x-y).abs()
    return diff

import torch.nn.functional as F

def pad_to_multiple_of_128_1d(x):
    """
    将一维数据填充为128的倍数
    参数:
        x: 输入张量，形状为 [L]
    返回:
        填充后的张量
    """
    length = x.size(0)
    pad_size = (128 - (length % 128)) % 128
    if pad_size > 0:
        x = F.pad(x, (0, pad_size),mode='replicate')  # 在末尾填充
    return x

def group_q(x,q_type = "nf4"):
    mean = x.mean()
    std = x.std()
    thr = mean+std
    mask = x<thr
    x0 = x[mask].contiguous()
    x0_len = x0.shape[0]
    x0 = pad_to_multiple_of_128_1d(x0)
    x1 = x[~mask].contiguous()
    x1_len = x1.shape[0]
    x1 = pad_to_multiple_of_128_1d(x1)
    print(x0_len)
    print(x1_len)
    y = torch.empty_like(x)
    if q_type == "nf4" :
        d,quant_state = quantize_4bit(x0,quant_type = "nf4",blocksize = 128,quant_storage = torch.uint8)
        ret = dequantize_4bit(d,quant_state,quant_state.absmax,blocksize = 128,quant_type = "nf4")
        y[mask].copy_(ret[:x0_len])

        d,quant_state = quantize_4bit(x1,quant_type = "nf4",blocksize = 128,quant_storage = torch.uint8)
        ret = dequantize_4bit(d,quant_state,quant_state.absmax,blocksize = 128,quant_type = "nf4")
        y[~mask].copy_(ret[:x1_len])
    print(x)
    print(y)
    return y
# x = torch.randn(1024).cuda()
# y = group_q(x)
# diff = (x-y).abs()
# print(diff.max())
# print(diff.mean())
# iter_times = 10
# import time
# begin = time.time()
# for i in range(iter_times):
#     x = torch.randn(8192,11008)
#     # x[0] = 9
#     # x[-1] = 9
#     x = x.cuda()
    
#     diff = q_dq_four_group(x)
#     max_m += diff.max()
#     mean_m += diff.mean()
#     diff_nf4 = q_dq_nf4(x)
#     max_m_nf4 += diff_nf4.max()
#     mean_m_nf4 += diff_nf4.mean()
# print(time.time()-begin)
# # print(diff.max())
# # print(diff.mean())
# # print(x[diff.argmax()])
# # print(y[diff.argmax()])
# print(max_m/iter_times)
# print(mean_m/iter_times)
# print(max_m_nf4/iter_times)
# print(mean_m_nf4/iter_times)