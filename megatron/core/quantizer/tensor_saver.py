import os
import re
import threading
import torch

class TensorSaver:
    def __init__(self, save_dir, log_file, iter_list):
        self.save_dir = save_dir
        self.log_file = log_file
        self.iter_list = set(iter_list)
        self.last_checked_line = 0  # 跟踪已读取的日志行

    def _get_current_iter(self):
        """从日志文件提取最新迭代次数"""
        try:

            with open(self.log_file, 'r') as f:
                lines = f.readlines()[self.last_checked_line:]
                self.last_checked_line += len(lines)

            # 反向查找最新的iteration记录
            for line in reversed(lines):
                match = re.search(r'iteration\s+(\d+)/\s*\d+', line)
                if match:
                    return int(match.group(1))
            return None
        except Exception as e:
            print(f"读取日志失败: {str(e)}")
            return None

    def _async_save(self, tensor, save_path):
        """异步保存张量"""
        def save_task():
            try:
                # 确保目录存在
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                
                # 处理重复文件
                counter = 1
                base_path, ext = os.path.splitext(save_path)
                while os.path.exists(save_path):
                    save_path = f"{base_path}_{counter}{ext}"
                    counter += 1
                
                # 保存张量
                torch.save(tensor.clone().detach().cpu(), save_path,_use_new_zipfile_serialization=True)
            except Exception as e:
                print(f"保存失败: {str(e)}")

        # 启动异步线程
        threading.Thread(target=save_task, daemon=True).start()

    def check_and_save(self, tensor, idx, module):
        """主接口：检查条件并触发保存"""
        current_iter = self._get_current_iter()
        print(current_iter)
        if current_iter is None or current_iter not in self.iter_list:
            return

        # 构建保存路径
        device = str(tensor.device).replace(':', '')  # cuda:0 -> cuda0
        save_path = os.path.join(
            self.save_dir,
            str(current_iter),
            str(idx),
            f"{module}_{device}.pt"
        )

        # 启动异步保存（保留原始设备）
        self._async_save(tensor, save_path)

##########################################
# 使用示例

class TensorSaverSync:
    def __init__(self, save_dir, log_file, iter_list):
        self.save_dir = save_dir
        self.log_file = log_file
        self.iter_list = set(iter_list)
        self.last_checked_line = 0

    def _get_current_iter(self):
        """同步读取日志获取当前迭代次数"""
        try:
            with open(self.log_file, 'r') as f:
                lines = f.readlines()[self.last_checked_line:]
                self.last_checked_line += len(lines)

            for line in reversed(lines):
                match = re.search(r'iteration\s+(\d+)/\s*\d+', line)
                if match:
                    return int(match.group(1))
            return None
        except Exception as e:
            print(f"日志读取错误: {str(e)}")
            return None

    def _sync_save(self, tensor, save_path):
        """同步保存张量（阻塞主线程）"""
        # 创建目录结构
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 处理重复文件名
        counter = 1
        base_path, ext = os.path.splitext(save_path)
        while os.path.exists(save_path):
            save_path = f"{base_path}_{counter}{ext}"
            counter += 1
        
        # 执行保存操作
        try:
            # 保持设备不变（按需求选择是否转CPU）
            torch.save(tensor.clone().detach(), save_path)
        except Exception as e:
            print(f"同步保存失败: {str(e)}")
            raise  # 可根据需要修改错误处理方式

    def check_and_save(self, tensor, idx, module):
        """同步保存入口"""
        current_iter = self._get_current_iter()
        if current_iter is None or current_iter not in self.iter_list:
            return

        # 构建设备字符串
        device_str = str(tensor.device).replace(':', '')
        
        # 生成保存路径
        save_path = os.path.join(
            self.save_dir,
            str(current_iter),
            str(idx),
            f"{module}_{device_str}.pt"
        )

        # 执行同步保存（阻塞主线程）
        self._sync_save(tensor, save_path)

if __name__ == "__main__":
    # 初始化配置
    saver = TensorSaver(
        save_dir="./saved_tensors",
        log_file="./training.log",
        iter_list=[100, 200, 300]
    )

    # 模拟接口调用
    def your_interface(input_tensor, idx, module):
        # 主任务代码...
        
        # 触发保存检查
        saver.check_and_save(input_tensor, idx, module)
        
        # 主任务继续执行...
    
    # 测试数据
    test_tensor = torch.randn(3, 64, device="cuda:0")
    your_interface(test_tensor, idx=5, module="transformer")