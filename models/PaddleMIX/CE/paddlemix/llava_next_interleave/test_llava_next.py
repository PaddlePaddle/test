import subprocess
import time
import os

def start_model():
    # 启动 LLaVA 模型
    command = [
        "python", "paddlemix/examples/llava_next_interleave/run_siglip_encoder_predict.py",
        "--model-path", "paddlemix/llava_next/llava-next-interleave-qwen-7b",
        "--image-file", "paddlemix/examples/demo_images/twitter3.jpeg", "paddlemix/examples/demo_images/twitter4.jpeg"
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return process

def ask_question(question):
    # 模拟与模型的交互
    process = start_model()
    try:
        # 等待模型启动并准备好
        time.sleep(10)  # 视情况调整
        print(f"提问: {question}")
        
        # 将问题发送到模型输入，假设模型有交互式会话
        process.stdin.write(f"{question}\n".encode())
        process.stdin.flush()

        # 读取模型的回答
        response = process.stdout.readline().decode().strip()
        return response
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # 终止模型进程
        process.terminate()

def validate_response(response, expected):
    # 自动判断输出是否符合预期
    if expected in response:
        print("测试成功!")
    else:
        print("测试失败!")
        print(f"模型回答: {response}")

if __name__ == "__main__":
    # 定义问题和期望回答
    question = "这两张图有什么共同点？"
    expected_answer = "共同点"  # 修改为合适的预期回答部分

    # 获取模型回答
    response = ask_question(question)

    # 验证模型回答
    validate_response(response, expected_answer)
