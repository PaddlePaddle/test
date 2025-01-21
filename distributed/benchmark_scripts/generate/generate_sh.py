import yaml
import os
import sys
# 已完成benchmark_common及其中一个case编写，适配generate_sh.py，根据case配置更新params.yaml
# 执行位置在该模型套件的根目录:cd /path/to/repo && python /path/to/generate_sh.py paddle /path/to/params.yaml
def generate_paddle_case(arg_params):
    test_case_template = """
#!/usr/bin/env bash
param="model_item={model_item_temp} "
param+="global_batch_size={global_batch_size_temp} "
param+="fp_item={fp_item_temp} "
param+="run_mode={run_mode_temp} "
param+="device_num={device_num_temp} "
param+="micro_batch_size={micro_batch_size_temp}  "
# 以下为run_benchmark.sh的可选参数
param+="dp_degree={dp_degree_temp} "
param+="mp_degree={mp_degree_temp} "
param+="pp_degree={pp_degree_temp} "
param+="sharding_degree={sharding_degree_temp} "
param+="sharding_stage={sharding_stage_temp} "
param+="level={level_temp} "
param+="local_batch_size={local_batch_size_temp} "
param+="workerlog_id={workerlog_id_temp} "

cd ./benchmarks
# get data
bash {model_item_script_path_temp}/benchmark_common/prepare.sh
# run
bash -c "${{param}} bash {model_item_script_path_temp}/benchmark_common/run_benchmark.sh"
"""

    # 分割参数值并为每个组合生成测试用例
    print(arg_params)
    model_item=arg_params['model_item']
    base_batch_size=arg_params['global_batch_size']
    fp_item=arg_params['fp_item']
    run_mode=arg_params['run_mode']
    device_num=arg_params['device_num']
    test_case = test_case_template.format(
        model_item_temp = model_item,
        global_batch_size_temp = base_batch_size,
        fp_item_temp = fp_item,
        run_mode_temp = run_mode,
        device_num_temp = device_num,
        micro_batch_size_temp=arg_params['micro_batch_size'],
        dp_degree_temp=arg_params['dp_degree'],
        mp_degree_temp=arg_params['mp_degree'],
        pp_degree_temp=arg_params['pp_degree'],
        sharding_degree_temp=arg_params['sharding_degree'],
        sharding_stage_temp=arg_params['sharding_stage'],
        level_temp=arg_params['level'],
        local_batch_size_temp=arg_params['local_batch_size'],
        workerlog_id_temp=arg_params['workerlog_id'],
        model_item_script_path_temp=arg_params['model_item_script_path'],
    )

    # 创建目录
    os.makedirs(os.path.join(arg_params['benchmark_path'], arg_params['model_item_script_path'], device_num), exist_ok=True)
    with open(os.path.join(arg_params['benchmark_path'], arg_params['model_item_script_path'], \
        device_num, f'{model_item}_bs{base_batch_size}_{fp_item}_{run_mode}.sh'), 'w') as f:
        f.write(test_case)
    


def generate_pytorch_case(arg_params):
    test_case_template = """
#!/usr/bin/env bash
model_item={model_item_temp}
bs_item={base_batch_size_temp}
fp_item={fp_item_temp}
run_mode={run_mode_temp}
device_num={device_num_temp}
max_iter={max_iter_temp}
num_workers={num_workers_temp}
# get data
bash prepare.sh
# run
bash run_benchmark.sh ${{model_item}} ${{bs_item}} ${{fp_item}} ${{run_mode}} ${{device_num}} ${{max_iter}} ${{num_workers}} 2>&1;
"""

    # 分割参数值并为每个组合生成测试用例
    print(arg_params)
    model_item = arg_params['model_item']
    base_batch_size = arg_params['base_batch_size']
    fp_item = arg_params['fp_item']
    run_mode = arg_params['run_mode']
    device_num = arg_params['device_num']
    max_iter = arg_params['max_iter']
    num_workers = arg_params['num_workers']
    test_case = test_case_template.format(
        model_item_temp=model_item,
        base_batch_size_temp=base_batch_size,
        fp_item_temp=fp_item,
        run_mode_temp=run_mode,
        device_num_temp=device_num,
        max_iter_temp=max_iter,
        num_workers_temp=num_workers,
        model_item_script_path_temp=arg_params['model_item_script_path'],
    )
    # 创建目录
    os.makedirs(os.path.join(arg_params['model_item_script_path'], model_item, device_num), exist_ok=True)
    with open(os.path.join(arg_params['model_item_script_path'], model_item, device_num, f'{model_item}_bs{base_batch_size}_{fp_item}_{run_mode}.sh'), 'w') as f:
        f.write(test_case)


if __name__ == "__main__":
    # 获取传入的参数
    frame = sys.argv[1]
    config_path = sys.argv[2]
    #mode 读取 YAML 文件中的参数
    with open(config_path, 'r') as file:
        arg_params = yaml.safe_load(file)
    if frame == 'paddle':
        generate_paddle_case(arg_params)
    else:
        generate_pytorch_case(arg_params)
    
