import subprocess
import time

def start_model():
    # 启动 LLaVA 模型，允许通过 stdout 查看启动日志
    command = [
        "python", "paddlemix/examples/llava_next_interleave/run_siglip_encoder_predict.py",
        "--model-path", "paddlemix/llava_next/llava-next-interleave-qwen-7b",
        "--image-file", "paddlemix/examples/demo_images/twitter3.jpeg", "paddlemix/examples/demo_images/twitter4.jpeg"
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return process

def wait_for_model_ready(process, ready_indicator="user:"):
    # 持续读取输出，直到检测到模型准备就绪
    while True:
        output = process.stdout.readline().strip()
        print(output)  # 打印启动过程
        if ready_indicator in output:
            print("模型已启动完毕，准备提问...")
            break
        time.sleep(1)

def ask_question(process, question):
    # 向模型发送问题并获取回答
    try:
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
    if expected in response:
        print("测试成功!")
    else:
        print("测试失败!")
        print(f"模型回答: {response}")

if __name__ == "__main__":
    # 定义问题和期望回答
    question = "Please write a twitter blog post with the images."
    expected_answer = "ASSISTANT"  # 修改为合适的预期回答部分

    # 启动模型进程
    process = start_model()

    # 等待模型启动完毕
    wait_for_model_ready(process)

    # 获取模型回答
    response = ask_question(process, question)

    # 验证模型回答
    if response:
        validate_response(response, expected_answer)

    # 终止模型进程
    process.terminate()
