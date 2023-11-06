# app.py
import base64
from flask import Flask, request, render_template, jsonify
from PIL import Image
from io import BytesIO
from predict import predict_sign, prdict_model2

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_image', methods=['POST'])
def process_image():
    print('Processing image...')
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        if request.form['model'] == 'm1':
            # Read the uploaded image and process it (e.g., resize, apply filters, etc.)
            img = Image.open(file)
            img = img.convert('RGB')

            # Predict the sign
            sign = predict_sign(img)

            # Save the processed image to a BytesIO buffer
            processed_buffer = BytesIO()
            img.save(processed_buffer, format='JPEG')
            processed_buffer.seek(0)
            processed_image = base64.b64encode(processed_buffer.read()).decode('utf-8')

            return jsonify({'processed_image': processed_image, 'text': sign})
        else:
            img = Image.open(file)
            img = img.convert('RGB')
            prdict_model2(img)
            processed_image = base64.b64encode(open('temp.jpg', 'rb').read()).decode('utf-8')
            return jsonify({'processed_image': processed_image, 'text': ''})

if __name__ == '__main__':
    app.run(debug=True)
