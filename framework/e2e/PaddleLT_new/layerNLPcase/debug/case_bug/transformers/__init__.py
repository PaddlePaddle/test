import os
import glob

# 获取当前文件所在目录
current_dir = os.path.dirname(__file__)

# 获取当前目录下所有的文件夹路径（注意：这里不需要尾随的斜杠）  
folders = glob.glob(os.path.join(current_dir, '*'))  
  
# 过滤出文件夹（排除文件）  
folders = [folder for folder in folders if os.path.isdir(folder) and not os.path.basename(folder) == '__pycache__']

# 动态导入所有 .py 文件
for folder in folders:
    # 获取文件名（不含扩展名）
    module_name = os.path.basename(folder)
    # 导入模块
    __import__('layerNLPcase.debug.case_bug.transformers.' + module_name, globals(), locals(), [])
