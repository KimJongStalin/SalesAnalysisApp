import os
from flask import Flask, render_template, request, send_from_directory, flash, redirect, url_for
from werkzeug.utils import secure_filename
from analyzer_script import SalesAnalyzer
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)

# --- 配置 ---
# 【重要】请务必将 'YourPythonAnywhereUsername' 修改为您的 PythonAnywhere 用户名
PYTHONANYWHERE_USERNAME = "YourPythonAnywhereUsername"
BASE_DIR = f"/home/{PYTHONANYWHERE_USERNAME}/SalesAnalysisApp"
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
TEMPLATE_FOLDER = os.path.join(BASE_DIR, 'templates')

# 允许上传的文件类型，已添加 .csv
ALLOWED_EXTENSIONS = {'xlsx', 'csv'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__, template_folder=TEMPLATE_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'a_very_strong_secret_key_for_deployment'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            input_filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            output_filename = f"Report_{os.path.splitext(filename)[0]}.html"
            output_filepath = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
            
            file.save(input_filepath)
            logging.info(f"文件已保存至: {input_filepath}")

            try:
                analysis_config = {
                    "input_filepath": input_filepath,
                    "html_report": {
                        "enabled": True,
                        "template_path": os.path.join(TEMPLATE_FOLDER, "sales_analysis_report.html"),
                        "output_path": output_filepath
                    },
                    "columns": {
                        "date": "Date", "sales": "Sales", "type": "类型", "brand": "Brand",
                        "packsize": "产品支数", "pricerange": "ASP区间", "tiptype": "tiptype", "asin": "ASIN",
                        "first_listed_date": "上架时间"
                    },
                    "header_mappings": { 
                        "brand": "Brand", "packsize": "PackSize", "pricerange": "PriceRange", 
                        "tiptype": "TipType", "tiptype packsize": "TipType & PackSize" 
                    },
                    "time_events": {
                        "Prime Day": "07-16", "Back to School": "08-15",
                        "Prime Fall Event": "10-10", "Black Friday": "11-29", "Christmas": "12-25",
                        "Valentine's Day": "02-14", "Mother's Day": "05-11"
                    }
                }
                analyzer = SalesAnalyzer(config=analysis_config)
                analyzer.run_analysis()
                logging.info(f"分析完成，報告已生成: {output_filepath}")

                return send_from_directory(app.config['UPLOAD_FOLDER'], output_filename, as_attachment=True)

            except Exception as e:
                logging.error(f"分析過程中發生錯誤: {e}", exc_info=True)
                flash(f'分析過程中發生錯誤: {e}')
                return redirect(request.url)
        else:
            flash('Invalid file type. Please upload an .xlsx or .csv file.')
            return redirect(request.url)

    return render_template('upload.html')