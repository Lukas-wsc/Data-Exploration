import subprocess
import sys
import os

def run_notebook(notebook_path):
    """Utility function to run a Jupyter notebook."""
    result = subprocess.run([sys.executable, '-m', 'jupyter', 'nbconvert', '--to', 'notebook', '--execute', '--inplace', notebook_path])
    if result.returncode != 0:
        raise RuntimeError(f"Failed to execute notebook {notebook_path}")

def install_requirements():
    """Install the required packages."""
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])

def data_preparation():
    """Run data preparation notebooks."""
    run_notebook('Data-Preparation/sentiment140/Preprocessing_training.ipynb')
    run_notebook('Data-Preparation/twitter-corpus/Preprocessing_corpus.ipynb')

def train_models():
    """Run training notebooks."""
    run_notebook('ML-Based Approach/Training/naive_bayes/Naive_Bayes_with_stopwords.ipynb')
    run_notebook('ML-Based Approach/Training/naive_bayes/Naive_Bayes_without_stopwords.ipynb')

def test_models():
    """Run testing notebooks."""
    run_notebook('ML-Based Approach/Testing/naive_bayes/Naive_Bayes_with_stopwords.ipynb')
    run_notebook('ML-Based Approach/Testing/naive_bayes/Naive_Bayes_without_stopwords.ipynb')

def validate_models():
    """Run validation notebooks."""
    run_notebook('ML-Based Approach/Validation/naive_bayes_validation.ipynb')

def start_ui():
    """Start the Flask web application."""
    subprocess.run([sys.executable, 'Flesk UI/app.py'])

def main():
    install_requirements()
    data_preparation()
    train_models()
    test_models()
    validate_models()
    start_ui()

if __name__ == '__main__':
    main()
