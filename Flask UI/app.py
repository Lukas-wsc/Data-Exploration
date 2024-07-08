import os
import pandas as pd
from flask import Flask, request, render_template, redirect, url_for, send_file
import pickle
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')

# Load the pre-trained models
# Set the base directory to the directory where the script is located
base_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the relative paths to the model files
naive_bayes_model_path = os.path.join(base_dir, '..', 'ML-Based Approach', 'Training', 'naive_bayes', 'naive_bayes_with_stopwords_classifier.pkl')
lgbm_model_path = os.path.join(base_dir, '..', 'ML-Based Approach', 'Training', 'LGBM', 'lgbm_model_with_stopwords.pkl')
tfidf_vectorizer_path = os.path.join(base_dir, '..', 'Data-Preparation', 'sentiment140','tfidf_vectorizer.pkl')

# Ensure the relative paths are correct by printing the absolute paths
print("Naive Bayes Model path:", naive_bayes_model_path)
print("LGBM Model path:", lgbm_model_path)
print("TFIDF Vectorizer path:", tfidf_vectorizer_path)

# Check if the files exist before loading
if not os.path.isfile(naive_bayes_model_path):
    raise FileNotFoundError(f"Naive Bayes model file not found: {naive_bayes_model_path}")

if not os.path.isfile(lgbm_model_path):
    raise FileNotFoundError(f"LGBM model file not found: {lgbm_model_path}")

if not os.path.isfile(tfidf_vectorizer_path):
    raise FileNotFoundError(f"TFIDF vectorizer file not found: {tfidf_vectorizer_path}")

with open(naive_bayes_model_path, 'rb') as nb_model_file:
    nb_model = pickle.load(nb_model_file)

with open(lgbm_model_path, 'rb') as lgbm_model_file:
    lgbm_model = pickle.load(lgbm_model_file)

with open(tfidf_vectorizer_path, 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# List of available models
models = ['Naive Bayes with Stopwords', 'LGBM with TfidfVectorizer']

def extract_features(text):
    words = word_tokenize(text)
    features = {word.lower(): True for word in words if word.isalpha()}
    return features

@app.route('/')
def index():
    return render_template('index.html', models=models)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(url_for('index'))
    
    file = request.files['file']
    model_choice = request.form.get('model')
    
    if file.filename == '':
        return redirect(url_for('index'))
    
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        data = pd.read_csv(filepath)
        
        if 'Text' in data.columns or 'text' in data.columns:
            if 'Text' in data.columns: 
                data.rename(columns={'Text': 'text'}, inplace=True)
            
            if model_choice == 'Naive Bayes with Stopwords':
                data['features'] = data['text'].apply(lambda text: extract_features(str(text)))
                features_list = data['features'].tolist()
                predictions = nb_model.classify_many(features_list)
            elif model_choice == 'LGBM with TfidfVectorizer':
                if 'text' not in data.columns:
                    return "Uploaded file does not contain 'Text' or 'text' column"
                
                # Vectorize the text data
                X_test = vectorizer.transform(data['text'].astype(str))
                predictions = lgbm_model.predict(X_test)
                predictions = ['negative' if pred == 0 else 'positive' for pred in predictions]
            
            data['Prediction'] = predictions
            positive_count = sum(1 for p in predictions if p == 'positive')
            negative_count = sum(1 for p in predictions if p == 'negative')
    
            prediction_file = os.path.join(app.config['UPLOAD_FOLDER'], 'predictions_with_data.csv')
            data.to_csv(prediction_file, index=False)
    
            return render_template('results.html', positive_count=positive_count, negative_count=negative_count, filename='predictions_with_data.csv')
        else:
            return "Uploaded file does not contain 'Text' or 'text' column"

@app.route('/classify_text', methods=['POST'])
def classify_text():
    text = request.form['text']
    model_choice = request.form.get('model')
    
    if model_choice == 'Naive Bayes with Stopwords':
        features = extract_features(text)
        prediction = nb_model.classify(features)
    elif model_choice == 'LGBM with TfidfVectorizer':
        X_test = vectorizer.transform([text])
        prediction = lgbm_model.predict(X_test)[0]
        prediction = 'negative' if prediction == 0 else 'positive'
    
    return render_template('result_text.html', prediction=prediction, text=text)

@app.route('/download/<filename>')
def download_file(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    return send_file(file_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)