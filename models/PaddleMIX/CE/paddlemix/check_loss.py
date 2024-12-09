#!/bin/env python3
# -*- coding: utf-8 -*-
"""
check loss
"""
import os
import requests
import subprocess
import time
import sys
sys.stdout.reconfigure(line_buffering=True)
def run_cmd(task_cmd, true_flag, wrong_flag):
    """
    运行命令
    """
    traceback_lines = [] 
    capturing_traceback = False  # 标志是否在捕获 Traceback
    capture_output = None
    start_time = None  # 记录开始捕获的时间


    # 启动任务
    process = subprocess.Popen(
        task_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    print("正在执行命令：", task_cmd)

    try:
        # 实时读取 stdout 输出
        while True:
            stdout_line = process.stdout.readline()
            stderr_line = process.stderr.readline()


            # 检查 stdout 无输出的话就退出循环
            if stdout_line == "" and stderr_line == "" and process.poll() is not None:
                break

            # 打印输出 (可选)
            # 实时打印 stdout
            if stdout_line:
                print("[STDOUT]:", stdout_line.strip())
            # 实时打印 stderr
            if stderr_line:
                print("[STDERR]:", stderr_line.strip())

            if stdout_line or stderr_line:
                # 检查是否包含 Traceback 关键字
                if wrong_flag in stdout_line.lower() or wrong_flag in stderr_line.lower():
                    if not capturing_traceback:
                        capturing_traceback = True
                        capture_output = 'traceback'
                        print("捕获到" + wrong_flag)
                        start_time = time.time()  # 记录开始时间
                
                if true_flag in stdout_line.lower() or true_flag in stderr_line.lower():
                    if not capturing_traceback:
                        capturing_traceback = True
                        capture_output = 'loss'
                        print('捕获到' + true_flag)
                        start_time = time.time()  # 记录开始时间

                # 如果已经在捕获 Traceback，继续收集行
                if capturing_traceback:
                    traceback_lines.append(output.strip())

                    # 检查是否超过 20 秒
                    if time.time() - start_time > 20:
                        print("超过 20 秒，停止捕获任务...")
                        process.terminate()  # 结束进程
                        break

        # 等待进程完全结束
        exit_code = process.wait()

        print('exit_code', exit_code)

        # 返回完整的 Traceback 信息
        if capture_output == 'traceback':
            return False
        elif capture_output == 'loss':
            return True
        elif exit_code == 0:
            return True
        else:
            return False
        

    except Exception as e:
        print(f"任务出错: {e}")
        process.terminate()  # 确保在出错时终止进程
        return None

if __name__ == "__main__":
    if len(sys.argv) == 0:
        raise ValueError("Usage: python check_loss.py <task_cmd> <true_flag> <wrong_flag>")

    task_cmd = sys.argv[1]
    print(task_cmd)
    true_flag = 'loss:'
    wrong_flag = 'tracebsack'
    result = run_cmd(task_cmd, true_flag, wrong_flag)
    if result:
        print("任务成功完成")
        sys.exit(0)
    else:
        print("任务失败")
        sys.exit(1)
    