import os
import subprocess
import random
import json
import sys
import shutil
# 初始化变量
def process_init(executed_log_path, model_num=5):

    exit_code = 0

    root_path = os.getenv("root_path", ".")
    log_dir = os.path.join(root_path, "log")
    work_path = os.getcwd()
    # 现在paddlenlp脚本
    # shutil.copy("change_paddlenlp_version.sh", os.path.join(root_path, "PaddleMIX"))

    # 执行 bash prepare.sh 脚本
    # subprocess.run(["bash", "prepare.sh"], check=True)

    # executed_log_path = os.path.join(root_path, "executed_dirs.json")  # 用于记录已执行的目录和轮次
    skip_dirs = {
        "infer/", 
        "ut/", 
        "deploy/", 
        "kandinsky2_2_text_to_image/", 
        "ppdiffusers_example_test/"
    }

    # 获取所有有效的子目录
    all_valid_dirs = [
        f"{subdir}/" for subdir in next(os.walk('.'))[1]
        if f"{subdir}/" not in skip_dirs
    ]

    # 初始化或加载记录

    if os.path.isfile(executed_log_path):
        try:
            with open(executed_log_path, "r") as file:
                record = json.load(file)
                executed_dirs = set(record.get("executed_dirs", []))
                current_epoch = record.get("epoch", 1)
        except Exception as e:
            print(f"Error loading the executed directory record from {executed_log_path}: {str(e)}")
            executed_dirs = set()
            current_epoch = 1
    else:
        executed_dirs = set()
        current_epoch = 1

    # 找到未执行的目录
    if model_num > len(all_valid_dirs):
        model_num = len(all_valid_dirs)
        executed_dirs = set()
        current_epoch = 1
    remaining_dirs = list(set(all_valid_dirs) - executed_dirs)

    # 如果未执行目录为空，开始新的轮次
    if not remaining_dirs:
        print(f"All directories have been covered in epoch {current_epoch}. Starting a new epoch.")
        executed_dirs = set()
        remaining_dirs = list(all_valid_dirs)
        current_epoch += 1

    # 随机选择 5 个目录
    selected_dirs = random.sample(remaining_dirs, min(model_num, len(remaining_dirs)))
    print(f"Epoch {current_epoch}: Selected directories: {selected_dirs}")

    # 更新已执行的目录记录
    executed_dirs.update(selected_dirs)

    for subdir_with_slash in selected_dirs:
        subdir = subdir_with_slash.rstrip("/")
        start_script_path = os.path.join(subdir, "start.sh")
        
        # 检查 start.sh 是否存在
        if os.path.isfile(start_script_path):
            os.chdir(subdir)
            # 执行 start.sh，并累加退出码
            result = subprocess.run(["bash", "start.sh"])
            exit_code += result.returncode
            os.chdir(work_path)

    # 保存更新后的已执行目录和轮次
    with open(executed_log_path, "w") as file:
        json.dump({"executed_dirs": list(executed_dirs), "epoch": current_epoch}, file)
        print(f"Epoch {current_epoch}: Selected directories: {selected_dirs}")
        print(f"Executed models in this epoch {executed_dirs}")

    print(f"Exit code: {exit_code}")

    # 查看结果
    ce_res_log_path = os.path.join(log_dir, "ce_res.log")
    if os.path.isfile(ce_res_log_path):
        with open(ce_res_log_path, "r") as log_file:
            print(log_file.read())

    # 退出脚本
    exit(exit_code)

if __name__ == '__main__':
    try:
        print("Starting script...")
        record_path = sys.argv[1]
        model_num = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        process_init(record_path, model_num)
    except Exception as e:
        print(e)
    