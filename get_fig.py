import matplotlib.pyplot as plt
import re
# 读取日志文件
with open('./logs/train_llama3_8b.log', 'r') as file:
    lines = file.readlines()

# 提取 loss 值和对应的迭代次数
iterations = []
loss_values = []

iteration_pattern = re.compile(r"iteration\s+(\d+)")
loss_pattern = re.compile(r"lm loss:\s+([\d.E+-]+)")

for line in lines:
    # 查找迭代次数和损失值
    iteration_match = iteration_pattern.search(line)
    loss_match = loss_pattern.search(line)
    
    if iteration_match and loss_match:
        iteration = int(iteration_match.group(1))  # 提取迭代次数
        loss = float(loss_match.group(1))          # 提取损失值
        
        iterations.append(iteration)
        loss_values.append(loss)
        print(iteration,loss)

# 绘制 loss 曲线
# 绘制 loss 曲线
plt.plot(iterations, loss_values)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss over Iterations')
plt.grid(True)

# 保存图表到当前目录
plt.savefig('./logs/llama3-8b_baseline.png')
