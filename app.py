import os
import numpy as np
from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Path to training classes (adjust if different)
TRAIN_DIR = 'data/train'

# Load the pre-trained model (safe fallback if missing)
try:
    model = load_model('model/final_model.h5')
except Exception:
    model = None

if os.path.isdir(TRAIN_DIR):
    class_names = sorted(os.listdir(TRAIN_DIR))
else:
    class_names = []

def allow_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif'}

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_result = None
    confidence = None
    image_path = None

    if request.method == 'POST':
       file = request.files.get('file')
       if file and allow_file(file.filename):
           filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
           file.save(filepath)
           image_path = filepath

           # Image preprocessing
           img = load_img(filepath, target_size=(64, 64))
           x = img_to_array(img)
           x = x / 255.0
           x = np.expand_dims(x, axis=0)
       
           # Prediction (only if model is loaded and class names are available)
           if model is not None and class_names:
               preds = model.predict(x)
               class_id = int(np.argmax(preds[0]))
               prediction_result = class_names[class_id]
               confidence = float(preds[0][class_id])
           else:
               prediction_result = 'Model not available'
               confidence = 0.0

           return render_template('index.html', 
                                   prediction=prediction_result, 
                                   confidence=f"{confidence*100:.2f}%", 
                                   image_path=image_path)
       
    return render_template('index.html', prediction=prediction_result, confidence=(f"{confidence*100:.2f}%" if confidence is not None else None), image_path=image_path)

@app.route('/upload/<filename>')
def uploaded_file(filename):
   return redirect(url_for('static', filename='uploads/' + filename))
    
if __name__ == '__main__':
    app.run(host="0.0.0.0" , port=8080 , debug=True)