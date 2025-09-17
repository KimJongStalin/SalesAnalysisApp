# # Cell 3: 创建 Flask API 服务器 app.py

# import uuid
# import json
# from flask import Flask, request, render_template
# from werkzeug.utils import secure_filename
# import os
# # 导入我们项目中的分析引擎
# from analyzer_script import SalesAnalyzer

# # 初始化 Flask 应用
# app = Flask(__name__)

# # 配置上传文件夹
# UPLOAD_FOLDER = 'uploads'
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# @app.route('/', methods=['GET', 'POST'])
# def home_and_upload():
#     if request.method == 'GET':
#         # 当用户访问主页时，显示一个简单的上传表单
#         return """
#             <!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8"><title>VOM 分析工具</title>
#             <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
#             <style>body{padding:2rem;}</style></head>
#             <body><div class="container"><h1 class="mb-4">VOM 销售数据分析</h1>
#             <form method="post" enctype="multipart/form-data"><div class="input-group"><input type="file" class="form-control" name="file" required>
#             <button type="submit" class="btn btn-primary">开始分析</button></div></form>
#             </div></body></html>
#         """

#     if request.method == 'POST':
#         # 当用户上传文件后，处理文件
#         if 'file' not in request.files or request.files['file'].filename == '':
#             return "错误：未选择文件", 400
        
#         file = request.files['file']
#         temp_filepath = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
#         file.save(temp_filepath)
        
#         try:
#             # 准备配置，传入上传文件的路径
#             config = {
#                 "input_filepath": temp_filepath,
#                 "html_report": { "template_path": "sales_analysis_report.html" },
#                 "columns": {
#                     "date": "Date", "sales": "Sales", "type": "类型", "brand": "Brand",
#                     "packsize": "产品支数", "pricerange": "ASP区间", "tiptype": "tiptype", "asin": "ASIN",
#                     "first_listed_date": "上架时间"
#                 }
#             }
            
#             # 调用引擎，获取完整的数据字典
#             analyzer = SalesAnalyzer(config=config)
#             dashboard_data = analyzer.prepare_and_get_data()
            
#             if not dashboard_data:
#                 return "分析失败，未能生成数据。", 500
            
#             # 读取HTML模板文件
#             with open(config['html_report']['template_path'], 'r', encoding='utf-8') as f:
#                 template_str = f.read()

#             # 将数据注入模板，并直接返回最终的HTML页面
#             final_html = template_str.replace('__DATA_PLACEHOLDER__', json.dumps(dashboard_data, default=str))
#             return final_html

#         except Exception as e:
#             import traceback
#             return f"<pre>分析过程中发生严重错误:\n{traceback.format_exc()}</pre>", 500
#         finally:
#             if os.path.exists(temp_filepath):
#                 os.remove(temp_filepath)

# # 这是为 Railway 等平台提供的标准启动入口
# if __name__ == '__main__':
#     # Railway 会通过环境变量 PORT 告诉应用应该监听哪个端口
#     port = int(os.environ.get("PORT", 5000))
#     app.run(host='0.0.0.0', port=port)


# 文件名: app.py
# 描述: Flask API 服务器，用于接收用户上传的文件和自定义分析维度，并返回分析报告。

import json
import os
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

# 导入我们项目中的分析引擎
# 确保 analyzer_script.py 和 app.py 在同一个目录下
from analyzer_script import SalesAnalyzer

# 初始化 Flask 应用
app = Flask(__name__)

# 配置上传文件夹
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def home_and_upload():
    if request.method == 'GET':
        # --- 1. 提供带有输入框的新前端界面 ---
        # 当用户访问主页时，显示一个包含自定义维度输入框的上传表单。
        return """
            <!DOCTYPE html>
            <html lang="zh-CN">
            <head>
                <meta charset="UTF-8">
                <title>VOM分析工具</title>
                <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
                <style>body { padding: 2rem; background-color: #f8f9fa; }</style>
            </head>
            <body>
                <div class="container">
                    <div class="card">
                        <div class="card-body">
                            <h1 class="card-title text-center mb-4">VOM 销售数据分析 (自定义模式)</h1>
                            <form method="post" enctype="multipart/form-data">
                                <div class="mb-3">
                                    <label for="file" class="form-label fw-bold">1. 上传Excel文件</label>
                                    <input type="file" class="form-control" name="file" required>
                                </div>
                                <div class="mb-3">
                                    <label for="single_dims" class="form-label fw-bold">2. 输入【单维度】分析列名 (可选)</label>
                                    <input type="text" class="form-control" name="single_dims" id="single_dims" placeholder="例如: brand,packsize">
                                    <div class="form-text">
                                        请使用您在 `analyzer_script.py` 中配置的【内部键名】，多个请用英文逗号分隔。
                                    </div>
                                </div>
                                <div class="mb-3">
                                    <label for="cross_dims" class="form-label fw-bold">3. 输入【交叉维度】分析列名 (可选)</label>
                                    <input type="text" class="form-control" name="cross_dims" id="cross_dims" placeholder="例如: tiptype&packsize,brand&pricerange">
                                    <div class="form-text">
                                        使用 "&" 连接两个列名，多个组合请用英文逗号分隔。
                                    </div>
                                </div>
                                <button type="submit" class="btn btn-primary w-100 mt-3">开始分析</button>
                            </form>
                        </div>
                    </div>
                </div>
            </body>
            </html>
        """

    if request.method == 'POST':
        # --- 2. 处理文件上传和用户输入的维度 ---
        if 'file' not in request.files or request.files['file'].filename == '':
            return "错误：未选择文件", 400
        
        file = request.files['file']
        temp_filepath = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
        file.save(temp_filepath)
        
        try:
            # 准备基础配置
            config = {
                "input_filepath": temp_filepath,
                "html_report": { "template_path": "sales_analysis_report.html" },
                # 这个 columns 映射关系依然重要，因为它告诉程序内部键名和Excel列名的对应关系
                "columns": {
                    "date": "Date", "sales": "Sales", "type": "类型", "brand": "Brand",
                    "packsize": "产品支数", "pricerange": "ASP区间", "tiptype": "tiptype", "asin": "ASIN",
                    "first_listed_date": "上架时间"
                }
            }
            
            # <-- 关键改动: 从前端表单接收用户输入的维度字符串 -->
            single_dims_str = request.form.get('single_dims', '')
            cross_dims_str = request.form.get('cross_dims', '')
            
            # 将用户选择打包成一个字典
            user_choices = {
                'single': single_dims_str,
                'cross': cross_dims_str
            }

            # 调用引擎，并将用户的选择作为参数传递进去
            analyzer = SalesAnalyzer(config=config)
            dashboard_data = analyzer.prepare_and_get_data(user_choices=user_choices)
            
            if not dashboard_data:
                return "分析失败，未能生成数据。", 500
            
            # 读取HTML模板并注入数据
            with open(config['html_report']['template_path'], 'r', encoding='utf-8') as f:
                template_str = f.read()

            final_html = template_str.replace('__DATA_PLACEHOLDER__', json.dumps(dashboard_data, default=str))
            return final_html

        except Exception as e:
            import traceback
            return f"<pre>分析过程中发生严重错误:\n{traceback.format_exc()}</pre>", 500
        finally:
            # 清理临时文件
            if os.path.exists(temp_filepath):
                os.remove(temp_filepath)

# 这是为 Railway 等平台提供的标准启动入口
if __name__ == '__main__':
    # Railway 会通过环境变量 PORT 告诉应用应该监听哪个端口
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

