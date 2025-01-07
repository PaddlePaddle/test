import os
import glob

# 获取当前文件所在目录
current_dir = os.path.dirname(__file__)

# 获取当前目录下所有的 .py 文件路径
py_files = glob.glob(os.path.join(current_dir, "*.py"))

# 动态导入所有 .py 文件
for py_file in py_files:
    # 获取文件名（不含扩展名）
    module_name = os.path.basename(py_file)[:-3]
    # 导入模块
    __import__("layerNLPcase.debug.case_bug.transformers.luke." + module_name, globals(), locals(), [])
