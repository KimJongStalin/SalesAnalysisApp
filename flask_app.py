from flask import Flask, render_template, request, redirect, url_for
import json
import os
from werkzeug.utils import secure_filename
from analyzer import SalesAnalyzer

app = Flask(__name__)
# 配置上传文件夹和允许的文件扩展名
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'xlsx', 'xls'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 创建上传文件夹
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# 检查文件扩展名是否合法
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # 检查是否有文件部分在请求中
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        # 如果用户没有选择文件，浏览器也会提交一个空的文件部分
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
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
                analyzer = SalesAnalyzer(analysis_config)
                dashboard_data = analyzer.prepare_html_report_data() # 修改为调用 prepare_html_report_data
                if dashboard_data:
                    data_json = json.dumps(dashboard_data, ensure_ascii=False)
                    return render_template('report.html', data_json=data_json)
                else:
                    return "❌ 分析失败，请检查上传文件的内容。", 500
            except Exception as e:
                return f"❌ 内部服务器错误: {e}", 500
                
    return render_template('upload.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
