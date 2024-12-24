import random
import os
import shutil
import subprocess
import sys
import json
import pexpect
# 假设我们有一个脚本列表
def infer_process(executed_log_path, model_num):
    scripts = [
        "dual_text_and_image_guided_generation-versatile_diffusion.py",
        "image_guided_image_inpainting-paint_by_example.py",
        "image_inpainting-repaint.py",
        "image_mixing-clip_guided_stable_diffusion.py",
        "image_to_image_text_guided_generation-alt_diffusion.py",
        "image_to_image_text_guided_generation-controlnet.py",
        "image_to_image_text_guided_generation-deepfloyd_if.py",
        "image_to_image_text_guided_generation-kandinsky2_2_controlnet.py",
        "image_to_image_text_guided_generation-kandinsky2_2.py",
        "image_to_image_text_guided_generation-kandinsky.py",
        "image_to_image_text_guided_generation-stable_diffusion_2.py",
        "image_to_image_text_guided_generation-stable_diffusion_3.py",
        "image_to_image_text_guided_generation-stable_diffusion_controlnet.py",
        "image_to_image_text_guided_generation-stable_diffusion.py",
        "image_to_image_text_guided_generation-stable_diffusion_xl.py",
        "image_to_text_generation-unidiffuser.py",
        "image_variation-stable_diffusion.py",
        "image_variation-unidiffuser.py",
        "image_variation-versatile_diffusion.py",
        "instruct_pix2pix-stable_diffusion_xl.py",
        "super_resolution-latent_diffusion.py",
        "text_guided_generation-semantic_stable_diffusion.py",
        "text_guided_image_inpainting-deepfloyd_if.py",
        "text_guided_image_inpainting-kandinsky2_2.py",
        "text_guided_image_inpainting-kandinsky.py",
        "text_guided_image_inpainting-stable_diffusion_2.py",
        "text_guided_image_inpainting-stable_diffusion_controlnet.py",
        "text_guided_image_inpainting-stable_diffusion.py",
        "text_guided_image_inpainting-stable_diffusion_xl.py",
        "text_guided_image_upscaling-stable_diffusion_2.py",
        "text_to_3d_generation-shape_e.py",
        "text_to_3d_generation-shape_e_image2image.py",
        "text_to_audio_generation-audio_ldm.py",
        "text_to_image_generation-alt_diffusion.py",
        "text_to_image_generation-auto.py",
        "text_to_image_generation-consistency_models.py",
        "text_to_image_generation-deepfloyd_if.py",
        "text_to_image_generation-kandinsky2_2_controlnet.py",
        "text_to_image_generation-kandinsky2_2.py",
        "text_to_image_generation-kandinsky.py",
        "text_to_image_generation-latent_diffusion.py",
        "text_to_image_generation_mixture_tiling-stable_diffusion.py",
        "text_to_image_generation-sdxl_base_with_refiner.py",
        "text_to_image_generation-stable_diffusion_2.py",
        "text_to_image_generation-stable_diffusion_3.py",
        "text_to_image_generation-stable_diffusion_controlnet.py",
        "text_to_image_generation-stable_diffusion.py",
        "text_to_image_generation-stable_diffusion_safe.py",
        "text_to_image_generation-stable_diffusion_t2i_adapter.py",
        "text_to_image_generation-stable_diffusion_xl_controlnet.py",
        "text_to_image_generation-stable_diffusion_xl.py",
        "text_to_image_generation-t2i-adapter.py",
        "text_to_image_generation-unclip.py",
        "text_to_image_generation-unidiffuser.py",
        "text_to_image_generation-versatile_diffusion.py",
        "text_to_image_generation-vq_diffusion.py",
        "text_to_video_generation-lvdm.py",
        "text_to_video_generation-synth_img2img.py",
        "text_to_video_generation-synth.py",
        "text_to_video_generation-zero.py",
        "text_variation-unidiffuser.py",
        "unconditional_audio_generation-audio_diffusion.py",
        "unconditional_audio_generation-dance_diffusion.py",
        "unconditional_audio_generation-spectrogram_diffusion.py",
        "unconditional_image_generation-ddim.py",
        "unconditional_image_generation-ddpm.py",
        "unconditional_image_generation-latent_diffusion_uncond.py",
        "unconditional_image_generation-pndm.py",
        "unconditional_image_generation-score_sde_ve.py",
        "unconditional_image_generation-stochastic_karras_ve.py",
        "unconditional_image_text_joint_generation-unidiffuser.py",
        "class_conditional_image_generation-dit.py",
        "class_conditional_image_generation-large_dit_3b.py",
        "class_conditional_image_generation-large_dit_7b.py",
        "image_to_video_generation_image_to_video.py",
        "image_to_video_generation_stable_video_diffusion.py",
        "text_to_audio_generation-audio_ldm2.py",
        "text_to_image_generation-latent_diffusion_uvit_small.py",
        "text_to_image_generation_consistency_model.py",
        "text_to_image_generation_kandinsky3.py",
        "text_to_image_generation_largedit_3b.py",
        "text_to_image_generation_wuerstchen.py",
        "text_to_video_generation-synth.py",
        "text_to_video_generation-synth_img2img.py",
        "text_to_video_generation-zero.py",
        "text_to_video_generation_animediff.py",
        "unconditional_audio_generation-audio_diffusion.py",
        "unconditional_audio_generation-dance_diffusion.py",
        "video_to_video_generation_video_to_video.py",
        "text_to_image_generation-stable_diffusion_3_5.py"
    ]

    # 随机选择要执行的脚本数量（假设选择5个脚本）
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

    remaining_dirs = list(set(scripts) - executed_dirs)

    if not remaining_dirs:
        print(f"All directories have been covered in epoch {current_epoch}. Starting a new epoch.")
        executed_dirs = set()
        remaining_dirs = list(scripts)
        current_epoch += 1

    selected_dirs = random.sample(remaining_dirs, min(model_num, len(remaining_dirs)))
    

    # 更新已执行的目录记录
    executed_dirs.update(selected_dirs)

    # 定义路径
    root_path = os.getenv('root_path')  # 获取root_path环境变量
    work_path = os.path.join(root_path, 'PaddleMIX/ppdiffusers/examples/inference/')
    work_path2 = os.path.join(root_path, 'PaddleMIX/ppdiffusers/')
    work_path3 = os.path.join(root_path, 'PaddleTest/models/PaddleMIX/CE/ppdiffusers')
    log_dir = os.path.join(root_path, 'infer_log')

    # 打印路径
    print(f"Current work path: {work_path}")
    print(f"Secondary work path: {work_path2}")
    print(f"Log directory: {log_dir}")

    # 创建日志目录
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 复制文件
    # 不对paddlenlp的版本进行限制
    # shutil.copy(work_path3 + "/change_paddlenlp_version.sh", os.path.join(root_path, "PaddleMIX"))
    # os.system('chmod +x ${root_path}/PaddleMIX/change_paddlenlp_version.sh')
    command = f"cp -rf ./* {work_path}/"
    subprocess.run(command, shell=True, check=True)

    # 安装依赖
    os.chdir(work_path2)

    subprocess.run(['python', '-m', 'pip', 'install', '--upgrade', 'pip'], check=True)
    subprocess.run(['pip', 'install', '-r', 'requirements.txt'], check=True)
    subprocess.run(['pip', 'install', '-e', '.'], check=True)

    # 执行更改 PaddleNLP 版本的脚本
   
    # subprocess.run([os.path.join(root_path, 'PaddleMIX/change_paddlenlp_version.sh')], shell=True, check=True)

    # 安装其他依赖
    subprocess.run(['pip', 'install', 'pytest', 'safetensors', 'ftfy', 'fastcore', 'opencv-python', 'einops', 'parameterized', 'requests-mock'], check=True)
    subprocess.run(['pip', 'install', 'ligo-segments'], check=True)
    subprocess.run(['pip', 'install', 'fastdeploy-gpu-python', '-f', 'https://www.paddlepaddle.org.cn/whl/fastdeploy.html'], check=True)

    # 返回工作路径
    os.chdir(work_path)

    # 设置环境变量
    os.environ['FLAGS_use_cuda_managed_memory'] = 'true'
    os.environ['FLAGS_allocator_strategy'] = 'auto_growth'
    os.environ['FLAGS_embedding_deterministic'] = '1'
    os.environ['FLAGS_cudnn_deterministic'] = '1'

    # 执行接下来的操作（如模型推理等）
    exit_code = 0
    print(f"Epoch {current_epoch}: Selected directories: {selected_dirs}")

    # 你可以继续根据需要添加其他的 脚本执行或操作
    # 例如运行某些 文件
    # subprocess.run(['python', 'your_script.py'], check=True)

    print(f"Exit code: {exit_code}")


    for script in selected_dirs:
        print(f"******* Running {script} ***********")
        process_log = os.path.join(log_dir, f"{script}.log")
        tmp_exit_code = -1  # 初始化为默认值

        try:
            # 打开日志文件以记录输出
            with open(process_log, "w") as log_process:
                # 启动子进程
                child = pexpect.spawn(f"python {script}", encoding="utf-8", logfile=log_process)

                while True:
                    try:
                        # 尝试读取一行输出
                        line = child.readline().strip()
                        if not line:  # 如果没有新行，子进程可能结束
                            break
                        print(line)  # 实时打印到控制台
                    except pexpect.exceptions.EOF:
                        # 子进程结束
                        break

                # 等待子进程退出并获取退出状态
                child.wait()
                tmp_exit_code = child.exitstatus

        except Exception as e:
            print(f"Error running script {script}: {e}")

        finally:
            # 记录运行结果
            with open(f"{log_dir}/infer_res.log", "a") as log_file:
                if tmp_exit_code == 0:
                    log_file.write(f"{script} run success\n")
                else:
                    log_file.write(f"{script} run fail\n")
            print(f"******* Finished running {script} ***********")
    
    # 保存更新后的已执行目录和轮次
    with open(executed_log_path, "w") as file:
        json.dump({"executed_dirs": list(executed_dirs), "epoch": current_epoch}, file)
        print(f"Epoch {current_epoch}: Selected directories: {selected_dirs}")
        print(f"Executed models in this epoch {executed_dirs}")

    # 输出最终的 exit_code
    print(f"Final exit code: {exit_code}")

     # 查看结果
    ce_res_log_path = os.path.join(log_dir, "infer_res.log")
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
        infer_process(record_path, model_num)
    except Exception as e:
        print(e)