"""
写入paddle.seed和np.random.seed
"""
import os


def add_seeds_if_string_exists(file_path):
    """读取文件内容并写入seed"""
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    # 标志位，用于标记是否找到了目标字符串
    found_target_string = False

    # 遍历文件内容，查找目标字符串
    for i, line in enumerate(lines):
        if '"""' in line:
            # 检查下一行是否是文档字符串
            next_1_line = lines[i + 1].strip()
            next_2_line = lines[i + 2].strip()
            # print('next_1_line is: ', next_1_line)
            # exit(0)
            if next_1_line == "forward" and next_2_line == '"""':
                found_target_string = True
                break

    # 如果找到了目标字符串，则在适当位置添加代码
    if found_target_string:
        # 定义要添加的代码
        additional_code = ["\n", "        paddle.seed(33)\n", "        np.random.seed(33)\n"]

        # 插入代码到目标位置之后（假设在函数定义之后立即添加）
        lines.insert(i + 3, additional_code[0])  # 插入空行
        lines.insert(i + 4, additional_code[1])  # 插入 paddle.seed(33)
        lines.insert(i + 5, additional_code[2])  # 插入 np.random.seed(33)

        # 写回文件
        with open(file_path, "w", encoding="utf-8") as file:
            file.writelines(lines)
    else:
        print(f"The target string was not found in {file_path}")


## file_path = 'layerApicase/nn_sublayer/conv1d_transpose_0_func.py'
## add_seeds_if_string_exists(file_path)

# 使用示例, 确保文件夹路径下没有子文件夹
nn_list = os.listdir("layerApicase/nn_sublayer")
for nn_file in nn_list:
    file_path = "layerApicase/nn_sublayer/" + nn_file
    print(file_path)
    add_seeds_if_string_exists(file_path)

# # 使用示例, 确保文件夹路径下没有子文件夹
# math_list = os.listdir('layerApicase/math_sublayer')
# for math_file in math_list:
#     file_path = 'layerApicase/math_sublayer/' + math_file
#     print(file_path)
#     add_seeds_if_string_exists(file_path)

# # 使用示例, 确保文件夹路径下没有子文件夹
# inplace_list = os.listdir('layerApicase/inplace_stragtegy')
# for inplace_file in inplace_list:
#     file_path = 'layerApicase/inplace_stragtegy/' + inplace_file
#     print(file_path)
#     add_seeds_if_string_exists(file_path)
