import traceback
import json
import os
from werkzeug.utils import secure_filename
from analyzer_script import SalesAnalyzer
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

# 配置上传文件夹为 Railway 挂载的卷路径
UPLOAD_FOLDER = '/mnt/uploads'
ALLOWED_EXTENSIONS = {'xlsx', 'xls'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 检查文件扩展名是否合法
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
            
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            # 新增这行代码，确保目录存在
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

            try:
                # 尝试将文件保存到 /mnt/uploads 目录
                file.save(file_path)
                print(f"✅ 文件已成功保存到: {file_path}")
            except Exception as e:
                print("❌ 文件保存失败！")
                traceback.print_exc()
                return f"❌ 文件保存失败: {e}", 500

            # 配置分析器使用上传的文件
            analysis_config = {
                "input_filepath": file_path,
                "columns": {
                    "date": "Date", "sales": "Sales", "type": "类型", "brand": "Brand",
                    "packsize": "产品支数", "pricerange": "ASP区间", "tiptype": "tiptype", "asin": "ASIN",
                    "first_listed_date": "上架时间"
                }
            }
            
            try:
                # 运行分析并获取报告数据
                analyzer = SalesAnalyzer(analysis_config)
                dashboard_data = analyzer.prepare_html_report_data()
                
                if dashboard_data:
                    data_json = json.dumps(dashboard_data, ensure_ascii=False)
                    return render_template('sales_analysis_report.html', data_json=data_json)
                else:
                    return "❌ 分析失败，请检查上传文件的内容。", 500
            except Exception as e:
                print("❌ 分析脚本发生未预期的错误！")
                traceback.print_exc()
                return f"❌ 内部服务器错误: {e}", 500
                
    return render_template('upload.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
