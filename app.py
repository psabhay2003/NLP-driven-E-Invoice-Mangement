from flask import Flask, render_template, request, send_file
from werkzeug.utils import secure_filename
import os
import pandas as pd
from field_extractor import extract_fields
from ocr_extract import extract_text
from model_loader import load_model_and_tokenizer

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model and tokenizer once
try:
    model, tokenizer = load_model_and_tokenizer()
except Exception as e:
    print("❌ Error loading model/tokenizer:", str(e))
    model, tokenizer = None, None

@app.route('/', methods=['GET', 'POST'])
def index():
    extracted_data = None

    if request.method == 'POST':
        if model is None or tokenizer is None:
            return render_template('index.html', error='Model not loaded.')

        if 'image' not in request.files:
            return render_template('index.html', error='No file part')

        file = request.files['image']
        if file.filename == '':
            return render_template('index.html', error='No selected file')

        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        try:
            text = extract_text(filepath)
            extracted_data = extract_fields(text, model, tokenizer)

            df = pd.DataFrame([extracted_data])
            csv_path = os.path.join(UPLOAD_FOLDER, 'output.csv')
            df.to_csv(csv_path, index=False)

            return render_template('index.html', data=extracted_data, download_link='/download')

        except Exception as e:
            return render_template('index.html', error=f'Processing failed: {str(e)}')

    return render_template('index.html')

@app.route('/download')
def download_csv():
    path = os.path.join(UPLOAD_FOLDER, 'output.csv')
    if os.path.exists(path):
        return send_file(path, as_attachment=True)
    return "CSV not found", 404

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
