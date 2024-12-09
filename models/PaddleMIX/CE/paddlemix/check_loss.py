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
import pexpect

sys.stdout.reconfigure(line_buffering=True)
def execute_task_with_flag_detection(task_cmd, true_flag, wrong_flag):
    try:
        print("正在执行命令：", task_cmd)
        
        # 启动子进程
        process = pexpect.spawn(task_cmd, timeout=None, encoding='utf-8')

        capture_output = None
        capturing_traceback = False
        start_time = None
        traceback_lines = []

        while True:
            try:
                # 捕获子进程的输出
                output = process.readline().strip()
                if output:
                    print(output)

                    # 检测关键字
                    if wrong_flag in output.lower():
                        if not capturing_traceback:
                            capturing_traceback = True
                            capture_output = "traceback"
                            start_time = time.time()
                            print(f"捕获到错误标志 {wrong_flag}")
                        traceback_lines.append(output)

                    elif true_flag in output.lower():
                        if not capturing_traceback:
                            capturing_traceback = True
                            capture_output = "loss"
                            start_time = time.time()
                            print(f"捕获到成功标志 {true_flag}")
                        traceback_lines.append(output)

                    # 检查是否超时
                    if capturing_traceback and time.time() - start_time > 20:
                        print("超过 20 秒，强制终止任务...")
                        process.terminate()
                        break

                # 如果子进程退出，跳出循环
                if process.eof():
                    break

            except pexpect.exceptions.TIMEOUT:
                print("任务超时...")
                break

        # 获取退出状态码
        exit_code = process.exitstatus or process.signalstatus
        print("任务结束，退出码：", exit_code)

        # 根据捕获情况返回
        if capture_output == "traceback":
            return False
        elif capture_output == "loss" or exit_code == 0:
            return True
        else:
            return False

    except Exception as e:
        print("执行命令时发生错误：", str(e))
        return False

if __name__ == "__main__":
    if len(sys.argv) == 0:
        raise ValueError("Usage: python check_loss.py <task_cmd> <true_flag> <wrong_flag>")

    task_cmd = sys.argv[1]
    print(task_cmd)
    true_flag = 'loss:'
    wrong_flag = 'tracebsack'
    result = execute_task_with_flag_detection(task_cmd, true_flag, wrong_flag)
    if result:
        print("任务成功完成")
        sys.exit(0)
    else:
        print("任务失败")
        sys.exit(1)
    