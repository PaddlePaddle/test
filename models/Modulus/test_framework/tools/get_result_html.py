import json
import os
import datetime
import math
def get_html(json_data,html_data,env_json="../env_json.json",model_class="model_class.json"):
    """
    将测试结果以html格式写入文件
    """
    with open(json_data, 'r', encoding='utf-8') as f:
        data = json.load(f)
    with open(env_json, 'r', encoding='utf-8') as f:
        env_data = json.load(f)
    with open(model_class, 'r', encoding='utf-8') as f:
        model_class = json.load(f)
    stringified_data = '<br>'.join([f"{str(key)}: {str(value)}" for key, value in env_data.items()])
    html_content = '<!DOCTYPE html>\n<html lang="zh">\n<head>\n<meta charset="UTF-8">\n<title>测试结果</title>\n</head>\n<body>\n'
    AGILE_JOB_BUILD_ID = os.getenv("AGILE_JOB_BUILD_ID")
    current_date = datetime.datetime.now()
    formatted_date = current_date.strftime("%Y_%m_%d")
    html_content += f'<h2>报告说明</h2>\
                    <p>此邮件为Modulus天级别监控邮件，目前进行动态图、动转静、动转静+组合算子、动转静+组合算子+编译器监控，监控粒度为L2和L4，由于模型运行时间较久，暂未监控L6级别。</p>\
                    <ul>\
                        <li>本次例行模型运行pytest测试报告见：<a href="https://paddle-qa.bj.bcebos.com/Modulus/{AGILE_JOB_BUILD_ID}/{formatted_date}_report.tar.gz">下载链接</a></li>\
                        <li>本次例行模型运行allure测试报告见：<a href="https://xly.bce.baidu.com/ipipe/ipipe-report/report/{AGILE_JOB_BUILD_ID}/test/">下载链接</a></li>\
                        <li>本次例行模型的具体测试日志见：<a href="https://paddle-qa.bj.bcebos.com/Modulus/{AGILE_JOB_BUILD_ID}/{formatted_date}_logs.tar.gz">下载链接</a></li>\
                        <li>本次例行模型的全部文件和产物见：<a href="https://paddle-qa.bj.bcebos.com/Modulus/{AGILE_JOB_BUILD_ID}/{formatted_date}_modulus-sym.tar.gz">下载链接</a></li>\
                    </ul>\
                    <h2>测试环境</h2>\
                    <p>{stringified_data}</p>\
                    <h2>精度对齐数据列表</h2>\
                    <p>*精度对齐列表中红色表示未通过，其中N/A表示功能测试未通过，没有精度数据</p>\
                    <p>QA说明：此报告为天级别例行，有问题请联系suijiaxin</p>'
    html_content += '<table border="1">\n<tr><th></th><th>动态图</th><th></th><th>动转静</th><th></th><th>动转静+组合算子</th><th></th><th>动转静+组合算子+cse</th><th></th><th>动转静+组合算子+编译器</th><th></th><th>动转静+组合算子+编译器+cse</th><th></th><th>编译器是否有精度优化</th><th>模型类别</th></tr>\n'
    html_content += '<tr><td> </td><td>L2</td><td>L4</td><td>L2</td><td>L4</td><td>L2</td><td>L4</td><td>L2</td><td>L4</td><td>L2</td><td>L4</td><td>L2</td><td>L4</td><td></td></tr>\n'
    for model_name in data.keys():
        # 定义编辑器精度优化flag 
        accuracy_flag = ""
        html_content += '<tr><td>' + model_name.replace("^", "/") + '</td>'
        for sub_key in data[model_name].keys():
            for sub_sub_key in data[model_name][sub_key].keys():
                if 'atol' not in data[model_name][sub_key][sub_sub_key] or 'rtol' not in data[model_name][sub_key][sub_sub_key]:
                    if sub_key == "dy2st_prim_cinn_cse":
                        accuracy_flag = "False"
                    if data[model_name][sub_key][sub_sub_key]['status'] == 'Timeout':
                        html_content += '<td style="background-color:rgb(250,234,200);">Timeout</td>'
                    elif sub_key == "dy2st":
                        html_content += '<td style="background-color:rgb(135,141,153)">Skipped</td>'
                    # elif "darcy" in model_name and sub_key != "dynamic":
                    #     html_ctenont += '<td style="background-color:rgb(135,141,153)">Skipped</td>'
                    else:
                        html_content += '<td style="background-color:rgb(254,230,230);">N/A</td>'
                    continue
                atol = data[model_name][sub_key][sub_sub_key]['atol']
                rtol = data[model_name][sub_key][sub_sub_key]['rtol']
                if data[model_name][sub_key][sub_sub_key]['status'] == 'Failed':
                    html_content += '<td style="background-color:rgb(254,230,230);">' + 'atol: ' + str("{:.3e}".format(atol)) + ' \n ' + 'rotl:' + str("{:.3e}".format(rtol)) + '</td>'
                elif data[model_name][sub_key][sub_sub_key]['status'] == 'Success':
                    html_content += '<td style="background-color:rgb(195, 242, 206);">' + 'atol: ' + str("{:.3e}".format(atol)) + ' \n ' + 'rotl:' + str("{:.3e}".format(rtol)) + '</td>'
        # 判断编译器是否有精度优化
        if accuracy_flag == "":
            accuracy_flag = 'True'
            key = 'L4'
            dy2st_prim_cinn_cse_atol_exp = int("{:.3e}".format(data[model_name]['dy2st_prim_cinn_cse'][key]['atol']).split("e")[1])
            dy2st_prim_cinn_cse_rtol_exp = int("{:.3e}".format(data[model_name]['dy2st_prim_cinn_cse'][key]['rtol']).split("e")[1])
            if "atol" in data[model_name]['dy2st_prim_cse'][key].keys() and "atol" in data[model_name]['dynamic'][key].keys():
                dy2st_prim_cse_atol_exp = int("{:.3e}".format(data[model_name]['dy2st_prim_cse'][key]['atol']).split("e")[1])
                dy2st_prim_cse_rtol_exp = int("{:.3e}".format(data[model_name]['dy2st_prim_cse'][key]['rtol']).split("e")[1])
                dynamic_atol_exp = int("{:.3e}".format(data[model_name]['dynamic'][key]['atol']).split("e")[1])
                dynamic_rtol_exp = int("{:.3e}".format(data[model_name]['dynamic'][key]['rtol']).split("e")[1])
                if (dy2st_prim_cse_atol_exp < dy2st_prim_cinn_cse_atol_exp or dy2st_prim_cse_rtol_exp < dy2st_prim_cinn_cse_rtol_exp) and (dynamic_atol_exp < dy2st_prim_cinn_cse_atol_exp or dynamic_rtol_exp < dy2st_prim_cinn_cse_rtol_exp):
                    accuracy_flag = "False"
            elif "atol" in data[model_name]['dy2st_prim_cse'][key].keys():
                dy2st_prim_cse_atol_exp = int("{:.3e}".format(data[model_name]['dy2st_prim_cse'][key]['atol']).split("e")[1])
                dy2st_prim_cse_rtol_exp = int("{:.3e}".format(data[model_name]['dy2st_prim_cse'][key]['rtol']).split("e")[1])
                if dy2st_prim_cse_atol_exp < dy2st_prim_cinn_cse_atol_exp or dy2st_prim_cse_rtol_exp < dy2st_prim_cinn_cse_rtol_exp:
                    accuracy_flag = "False"
            elif "atol" in data[model_name]['dynamic'][key].keys():
                dynamic_atol_exp = int("{:.3e}".format(data[model_name]['dynamic'][key]['atol']).split("e")[1])
                dynamic_rtol_exp = int("{:.3e}".format(data[model_name]['dynamic'][key]['rtol']).split("e")[1])
                if dynamic_atol_exp < dy2st_prim_cinn_cse_atol_exp or dynamic_rtol_exp < dy2st_prim_cinn_cse_rtol_exp:
                    accuracy_flag = "False"
        if accuracy_flag == "True":
            html_content += '<td style="background-color:rgb(195, 242, 206);">True</td>'
        else:
            html_content += '<td style="background-color:rgb(254,230,230);">False</td>'    
        html_content += '<td>' + model_class[model_name] + '</td>'
        html_content += '</tr>\n'

    html_content += '</table>'

    with open(html_data, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print("HTML 表格已生成！")

if __name__ == '__main__':
    json_data = './test_data.json'
    html_data = './index.html'
    get_html(json_data, html_data)
