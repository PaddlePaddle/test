from paddlex import PaddleInferenceOption, create_model

def run_predict(model_name, image_path):
    """
    模型推理测试
    """
    # model_name = "PP-LCNet_x1_0"

    # 实例化 PaddleInferenceOption 设置推理配置
    kernel_option = PaddleInferenceOption()
    kernel_option.set_device("gpu")

    model = create_model(model_name=model_name, kernel_option=kernel_option)

    # 预测
    result = model.predict({'input_path': f"{image_path}"})
    return result

def get_clas_result(result):
    """
    获取分类结果
    """
    cls_result = result['cls_result'][0]['class_ids']
    return cls_result

def get_ocr_result(result):
    """
    获取OCR识别结果
    """
    ocr_result = result['rec_text'][0]
    return ocr_result

if __name__ == '__main__':
    print(run_predict('PP-OCRv4_server_rec', 'https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_rec_001.png')['rec_text'][0])