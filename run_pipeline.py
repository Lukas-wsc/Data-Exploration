import os
import time
import subprocess
import sys

def run_notebook(notebook_path):
    # Libraries get imported here because they are not native to python and require the requirements.txt installation
    import nbformat
    from nbconvert.preprocessors import ExecutePreprocessor
    """Run a Jupyter notebook."""
    print(f"Running notebook: {notebook_path}")
    if not os.path.exists(notebook_path):
        raise FileNotFoundError(f"Notebook not found: {notebook_path}")
    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)
    ep = ExecutePreprocessor(timeout=6000, kernel_name='python3')
    start_time = time.time()
    ep.preprocess(nb, {'metadata': {'path': os.path.dirname(notebook_path)}})
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Completed notebook: {notebook_path} in {elapsed_time:.2f} seconds")

def install_requirements():
    """Install the required packages."""
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
    
def data_preparation():
    """Run data preparation notebooks."""
    base_path = os.path.join('Data-Preparation')
    notebooks = [
        os.path.join(base_path, 'sentiment140', 'Preprocessing_training.ipynb'),
        os.path.join(base_path, 'twitter-corpus', 'Preprocessing_corpus.ipynb')
    ]
    for nb in notebooks:
        run_notebook(nb)

def run_training():
    """Run training notebooks."""
    base_path = os.path.join('ML-Based Approach', 'Training')
    notebooks = [
        os.path.join(base_path, 'LGBM', 'LGBM_with_stopwords.ipynb'),
        os.path.join(base_path, 'LGBM', 'LGBM_without_stopwords.ipynb'),
        os.path.join(base_path, 'naive_bayes', 'Naive_Bayes_with_stopwords.ipynb'),
        os.path.join(base_path, 'naive_bayes', 'Naive_Bayes_without_stopwords.ipynb'),
        os.path.join(base_path, 'XGBoost', 'XGBoost_with_Stopwords.ipynb'),
        os.path.join(base_path, 'XGBoost', 'XGBoost_without_Stopwords.ipynb'),
    ]
    base_path_rule = os.path.join('Rule-Based Approach', 'Training')
    notebooks_rule = [
        os.path.join(base_path_rule, 'sentiwordnet_with_stopwords.ipynb'),
        os.path.join(base_path_rule, 'sentiwordnet_without_stopwords.ipynb'),
        os.path.join(base_path_rule, 'vader_with_stopwords.ipynb'),
        os.path.join(base_path_rule, 'vader_without_stopwords.ipynb')
    ]
    for nb in notebooks + notebooks_rule:
        run_notebook(nb)

def run_testing():
    """Run testing notebooks."""
    base_path = os.path.join('ML-Based Approach', 'Testing')
    notebooks = [
        os.path.join(base_path, 'LGBM', 'LGBM_with_stopwords.ipynb'),
        os.path.join(base_path, 'LGBM', 'LGBM_without_stopwords.ipynb'),
        os.path.join(base_path, 'naive_bayes', 'Naive_Bayes_with_stopwords.ipynb'),
        os.path.join(base_path, 'naive_bayes', 'Naive_Bayes_without_stopwords.ipynb'),
        os.path.join(base_path, 'XGBoost', 'XGBoost_with_Stopwords.ipynb'),
        os.path.join(base_path, 'XGBoost', 'XGBoost_without_Stopwords.ipynb'),
    ]
    base_path_rule = os.path.join('Rule-Based Approach', 'Testing')
    notebooks_rule = [
        os.path.join(base_path_rule, 'sentiwordnet_with_stopwords.ipynb'),
        os.path.join(base_path_rule, 'sentiwordnet_without_stopwords.ipynb'),
        os.path.join(base_path_rule, 'vader_with_stopwords.ipynb'),
        os.path.join(base_path_rule, 'vader_without_stopwords.ipynb')
    ]
    for nb in notebooks + notebooks_rule:
        run_notebook(nb)

def run_validation():
    """Run validation notebooks."""
    base_path = os.path.join('ML-Based Approach', 'Validation')
    notebooks = [
        os.path.join(base_path, 'lgbm_validation.ipynb'),
        os.path.join(base_path, 'naive_bayes_validation.ipynb'),
        os.path.join(base_path, 'xgboost_validation.ipynb'),
    ]
    base_path_rule = os.path.join('Rule-Based Approach', 'Validation')
    notebooks_rule = [
        os.path.join(base_path_rule, 'sentiwordnet_without_stopwords.ipynb'),
        os.path.join(base_path_rule, 'vader_with_stopwords.ipynb'),
    ]
    for nb in notebooks + notebooks_rule:
        run_notebook(nb)

def run_flask_app():
    """Run the Flask app."""
    print("Running Flask app: app.py")
    start_time = time.time()
    os.system('python Flask\ UI/app.py')
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Completed Flask app: app.py in {elapsed_time:.2f} seconds")


def start_pipeline():
    """Start the full workflow pipeline."""
    total_start_time = time.time()
    install_requirements()
    
    data_preparation() 
    run_training() 
    run_testing() 
    run_validation()
    run_flask_app()
    total_end_time = time.time()
    total_elapsed_time = total_end_time - total_start_time
    print(f"Pipeline execution completed in {total_elapsed_time:.2f} seconds.")

if __name__ == "__main__":
    start_pipeline()
