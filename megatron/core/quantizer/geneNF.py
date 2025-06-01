# import torch
# from scipy.stats import norm

# # NF5
# nf5_offset = 1 / 2 * (1.0 / 32 + 1.0 / 30)

# f = norm.ppf(torch.linspace(nf5_offset, 0.5, 16)[:-1])
# z = norm.ppf(torch.linspace(0.5, 1 - nf5_offset, 17))

# NF5_code = torch.cat((f, z))

# # 所有值除以绝对值最大值
# absmax = NF5_code.abs().max()
# NF5_code = NF5_code / absmax

# # 扩展为 5 倍长度再排序
# NF5_code = NF5_code.repeat(5)
# NF5_code, _ = torch.sort(NF5_code)

# # NF8
# nf8_offset = 1 / 2 * (1.0 / 64 + 1.0 / 62)

# f = norm.ppf(torch.linspace(nf8_offset, 0.5, 2**7)[:-1])
# z = norm.ppf(torch.linspace(0.5, 1 - nf8_offset, 2**7 + 1))

# NF8_code = torch.cat((f, z))
# absmax = NF8_code.abs().max()
# NF8_code = NF8_code / absmax
# NF8_code, _ = torch.sort(NF8_code)
import torch
from scipy.stats import norm

#NF5
nf5_offset = 1 / 2 * (1.0 / 32 + 1.0 / 30)

f = (norm.ppf(torch.linspace(nf5_offset, 0.5, 16)[:-1])).tolist()
z = norm.ppf(torch.linspace(0.5, 1-nf5_offset, 17)).tolist()

NF5_code=f+z
# 所有值除以绝对值最大值
absmax = max(abs(i) for i in NF5_code)
NF5_code = [i / absmax for i in NF5_code]
NF5_code*=5
NF5_code.sort()

#NF8
nf8_offset = 1 / 2 * (1.0 / 64 + 1.0 / 62)

f = (norm.ppf(torch.linspace(nf8_offset, 0.5, 2**7)[:-1])).tolist()
z = norm.ppf(torch.linspace(0.5, 1-nf8_offset, 2**7+1)).tolist()

NF8_code=f+z
# 所有值除以绝对值最大值
absmax = max(abs(i) for i in NF8_code)
NF8_code = [i / absmax for i in NF8_code]
NF8_code.sort()
NF5_code=torch.tensor(NF5_code, dtype=torch.float32)
NF8_code=torch.tensor(NF8_code, dtype=torch.float32)
if __name__ == "__main__":
    print(NF8_code)
    print(len(NF8_code))