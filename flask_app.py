# Cell 3: 创建 Flask API 服务器 app.py

%%writefile app.py

import os
import uuid
import json
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from analyzer import SalesAnalyzer

# --- 初始化和配置 ---
UPLOAD_FOLDER = 'uploads'
REPORT_FOLDER = 'reports'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

app = Flask(__name__)
CORS(app)

# 基础配置
base_config = {
    "html_report": { "template_path": "sales_analysis_report.html" },
    "columns": {
        "date": "Date", "sales": "Sales", "type": "类型", "brand": "Brand",
        "packsize": "产品支数", "pricerange": "ASP区间", "tiptype": "tiptype", "asin": "ASIN",
        "first_listed_date": "上架时间"
    },
    # Add other base configs from your analyzer if needed
}

@app.route('/analyze', methods=['POST'])
def analyze_endpoint():
    if 'file' not in request.files or request.files['file'].filename == '':
        return jsonify({'error': '未找到文件'}), 400
    file = request.files['file']
    
    unique_id = uuid.uuid4().hex
    input_filename = f"{unique_id}_{os.path.basename(file.filename)}"
    input_filepath = os.path.join(UPLOAD_FOLDER, input_filename)
    file.save(input_filepath)

    report_filename = f"report_{unique_id}.html"
    report_filepath = os.path.join(REPORT_FOLDER, report_filename)

    current_config = base_config.copy()
    current_config['input_filepath'] = input_filepath
    
    try:
        analyzer = SalesAnalyzer(config=current_config)
        success = analyzer.run_analysis()

        if success:
            analyzer.export_to_html(output_path=report_filepath)
            return jsonify({ 
                'message': '分析成功！', 
                'report_url': f'/reports/{report_filename}' 
            })
        else:
            return jsonify({'error': '分析引擎未能处理数据'}), 500
            
    except Exception as e:
        import traceback
        return jsonify({'error': f'分析出错: {e}', 'trace': traceback.format_exc()}), 500

@app.route('/reports/<filename>')
def serve_report(filename):
    return send_from_directory(REPORT_FOLDER, filename)

@app.route('/')
def home():
    # This is the user-facing upload page with JavaScript to handle API calls
    return """
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <title>VOM 分析工具 (API版)</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body { display: flex; align-items: center; justify-content: center; min-height: 100vh; background-color: #f8f9fa; }
            .container { max-width: 600px; text-align: center; } .spinner-border { width: 3rem; height: 3rem; }
            #result, #error { margin-top: 2rem; }
        </style>
    </head>
    <body>
        <div class="container p-5 bg-white rounded shadow">
            <div id="upload-form">
                <h1 class="mb-4">VOM 销售数据分析 (API版)</h1>
                <p class="text-muted mb-4">请上传您的销售数据 Excel 文件。</p>
                <form id="analysis-form">
                    <div class="input-group mb-3"><input type="file" class="form-control" id="fileInput" name="file" accept=".xlsx,.xls" required></div>
                    <button type="submit" class="btn btn-primary btn-lg mt-3">开始分析</button>
                </form>
            </div>
            <div id="loading-state" style="display: none;">
                <div class="spinner-border text-primary" role="status"></div>
                <p class="mt-3">正在进行深度分析... 报告生成后会自动跳转。</p>
            </div>
            <div id="result" class="alert alert-success" style="display: none;"></div>
            <div id="error" class="alert alert-danger" style="display: none;"></div>
        </div>
    <script>
        const form = document.getElementById('analysis-form');
        const fileInput = document.getElementById('fileInput');
        const uploadDiv = document.getElementById('upload-form');
        const loadingDiv = document.getElementById('loading-state');
        const resultDiv = document.getElementById('result');
        const errorDiv = document.getElementById('error');

        form.addEventListener('submit', async (event) => {
            event.preventDefault();
            uploadDiv.style.display = 'none';
            loadingDiv.style.display = 'block';
            resultDiv.style.display = 'none';
            errorDiv.style.display = 'none';

            const formData = new FormData();
            formData.append('file', fileInput.files[0]);

            try {
                const response = await fetch('/analyze', { method: 'POST', body: formData });
                const data = await response.json();

                if (response.ok) {
                    resultDiv.innerHTML = `分析成功！将在3秒后跳转到报告页面... <a href="${data.report_url}" target="_blank">立即跳转</a>`;
                    resultDiv.style.display = 'block';
                    setTimeout(() => { window.location.href = data.report_url; }, 3000);
                } else {
                    throw new Error(data.trace || data.error || '未知错误');
                }
            } catch (error) {
                errorDiv.innerHTML = '<b>分析失败:</b><pre style="text-align: left; white-space: pre-wrap;">' + error.message + '</pre>';
                errorDiv.style.display = 'block';
                uploadDiv.style.display = 'block';
            } finally {
                loadingDiv.style.display = 'none';
            }
        });
    </script>
    </body></html>
    """

print("✅ 第三步完成：Flask API 服务器 'app.py' 已创建！")
