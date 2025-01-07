# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import pytest

import sys
from tools.run_predict import run_predict, get_clas_result, get_ocr_result

def test_ResNet18():
    """
    测试ResNet18模型
    """
    result = run_predict(model_name="ResNet18", image_path="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg")
    cls_result = get_clas_result(result)
    assert 296 in cls_result


if __name__ == "__main__":
    # 使用 pytest 模块运行测试函数
    code = pytest.main(["--alluredir=./allure", sys.argv[0]])
    sys.exit(code)