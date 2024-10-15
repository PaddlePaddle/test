import subprocess
import time

def start_model():
    # 启动 LLaVA 模型，允许通过 stdout 和 stderr 查看启动日志
    command = [
        "python", "paddlemix/examples/llava_next_interleave/run_siglip_encoder_predict.py",
        "--model-path", "paddlemix/llava_next/llava-next-interleave-qwen-7b",
        "--image-file", "paddlemix/demo_images/twitter3.jpeg", "paddlemix/demo_images/twitter4.jpeg"
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return process

def wait_for_stdin_ready(process, max_retries=30, interval=5):
    """ 循环检查直到 stdin 可用或达到最大重试次数 """
    retries = 0
    while process.stdin is None and retries < max_retries:
        print("等待模型的 stdin 可用...")
        time.sleep(interval)
        retries += 1

    if process.stdin is None:
        print("模型进程的 stdin 未正确打开，无法发送问题。")
        return False
    return True

def ask_question(process, question):
    # 向模型发送问题并获取回答
    try:
        # 循环检查 stdin 是否可用
        if not wait_for_stdin_ready(process):
            return None
        
        print(f"提问: {question}")
        # 发送问题到模型的标准输入
        process.stdin.write(question + "\n")
        process.stdin.flush()

        # 等待模型处理并读取回答
        response = process.stdout.readline().strip()
        return response
    except Exception as e:
        print(f"Error: {e}")
        return None

def validate_response(response, expected):
    # 自动判断输出是否符合预期
    if response is None:
        print("未能获取到模型的回答。")
        return

    if expected in response:
        print("测试成功!")
    else:
        print("测试失败!")
        print(f"模型回答: {response}")

if __name__ == "__main__":
    # 定义问题和期望回答
    question = "Please write a twitter blog post with the images."
    expected_answer = "ASSISTANT:"  # 修改为合适的预期回答部分

    # 启动模型进程
    process = start_model()

    # 获取模型回答
    response = ask_question(process, question)

    # 验证模型回答
    validate_response(response, expected_answer)

    # 终止模型进程
    process.terminate()
