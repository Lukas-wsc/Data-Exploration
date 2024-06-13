import os
import pandas as pd
from flask import Flask, request, render_template, redirect, url_for, send_file
import pickle
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')

with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

def extract_features(text):
    words = word_tokenize(text)
    features = {word.lower(): True for word in words if word.isalpha()}
    return features

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(url_for('index'))
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect(url_for('index'))
    
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        data = pd.read_csv(filepath)
        
        if 'Text' in data.columns or 'text' in data.columns:
            if 'Text' in data.columns: 
                data['text'] = data['Text']
            data['features'] = data['text'].apply(lambda text: extract_features(str(text)))
            features_list = data['features'].tolist()
            predictions = model.classify_many(features_list)
    
            data['Prediction'] = predictions
            positive_count = sum(1 for p in predictions if p == 'positive')
            negative_count = sum(1 for p in predictions if p == 'negative')
    
            prediction_file = os.path.join(app.config['UPLOAD_FOLDER'], 'predictions_with_data.csv')
            data.to_csv(prediction_file, index=False)
    
            return render_template('results.html', positive_count=positive_count, negative_count=negative_count, filename='predictions_with_data.csv')
        else:
            return "Uploaded file does not contain 'Text' or 'text' column"

@app.route('/download/<filename>')
def download_file(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    return send_file(file_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)