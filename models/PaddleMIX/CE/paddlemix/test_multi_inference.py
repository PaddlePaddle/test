#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pexpect

def test_interactive_script():
    try:
        # 启动对话式 Python 程序
        script_command = "python3 your_script.py --model_name_or_path THUDM/cogagent-chat"
        child = pexpect.spawn(script_command, encoding="utf-8")

        # 等待提示符，并输入第一步内容
        child.expect("image path >>>>>", timeout=30)  # 等待出现 'image path >>>>>'
        child.sendline("")  # 输入空行，表示跳过图像部分

        # 等待对话提示，输入一个问题
        child.expect("Human:", timeout=30)  # 等待出现 'Human:'
        child.sendline("What is the capital of France?")  # 输入问题

        # 等待 AI 返回结果（确保正常返回即可）
        child.expect("Cog:", timeout=30)  # 等待出现 'Cog:'
        print(f"AI Response: {child.before.strip()}")  # 打印返回内容

        # 模拟结束对话
        child.expect("Human:", timeout=30)
        child.sendline("clear")  # 输入 'clear' 清空对话历史

        # 关闭子进程
        child.close()
        if child.exitstatus == 0:
            print("Test passed: Program executed without errors.")
        else:
            print("Test failed: Non-zero exit status.")
    except pexpect.exceptions.TIMEOUT:
        print("Test failed: Timeout waiting for the program's response.")
    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    test_interactive_script()
