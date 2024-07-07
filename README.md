# Sentiment Analysis Project

This project performs sentiment analysis using multiple machine learning and rule-based approaches. It involves data preparation, training, testing, and validation, with a user interface for visualizing results and allowing user interaction.

### Used Python Version

Python 3.12.3

### Dataset

- Training dataset currently used: ['https://www.kaggle.com/datasets/kazanova/sentiment140'](https://www.kaggle.com/datasets/kazanova/sentiment140)
- The Sentiment140 dataset is large and labeled using emoticons.
- Validation dataset: ([https://github.com/zfz/twitter_corpus](https://github.com/zfz/twitter_corpus/blob/master/full-corpus.csv))
- The Twitter corpus is smaller but manually labeled, making it ideal for validation.
### Data Preprocessing

Data preprocessing takes place in `Preprocessing.ipynb`.
## Execution Guide
- Download Training set Sentiment140 at given link and place it into the project at ```Data Exploration/Github/Data-Exploration/Data-Preparation/sentiment140/training.1600000.processed.noemoticon.csv```
- Download Validation set twitter-corpus at given link and place it into the project at ```Data Exploration/Github/Data-Exploration/Data-Preparation/twitter-corpus/full-corpus.csv```
- The sequence in which the files need to be executed is performed by run_pipeline.py. Execute this file in the root directory by performing 
 ```python run_pipeline.py```

However, this file needs about 488.87 seconds on an M1 Mac Pro, so expect it to be loading for a while. If the file does run into an error, you can also perform the execution of the notebooks step by step for all given models: 
Ensure the correct sequence: Data Preparation -> Training -> Testing -> Validation -> UI. 
  
1. **Install Required Packages**  
First, ensure all required packages are installed. You can install them using the `requirements.txt` file:

    ```sh
    pip install -r requirements.txt
    ```

2. **Data Preparation**  
Prepare the datasets by running the preprocessing notebooks:

    - For training data from Sentiment140:
        ```sh
        jupyter notebook Data-Preparation/sentiment140/Preprocessing_training.ipynb
        ```
    - For the Twitter corpus:
        ```sh
        jupyter notebook Data-Preparation/twitter-corpus/Preprocessing_corpus.ipynb
        ```
        
3. **Training Models**  
Train the models using the training notebooks:

    - For Naive Bayes with stopwords:
        ```sh
        jupyter notebook "ML-Based Approach/Training/naive_bayes/Naive_Bayes_with_stopwords.ipynb"
        ```
    - For Naive Bayes without stopwords:
        ```sh
        jupyter notebook "ML-Based Approach/Training/naive_bayes/Naive_Bayes_without_stopwords.ipynb"
        ```

4. **Testing Models**  
Test the trained models:

    - For Naive Bayes with stopwords:
        ```sh
        jupyter notebook "ML-Based Approach/Testing/naive_bayes/Naive_Bayes_with_stopwords.ipynb"
        ```
    - For Naive Bayes without stopwords:
        ```sh
        jupyter notebook "ML-Based Approach/Testing/naive_bayes/Naive_Bayes_without_stopwords.ipynb"
        ```

5. **Validation**  
Validate the models using the validation notebooks:

    - For Naive Bayes:
        ```sh
        jupyter notebook "ML-Based Approach/Validation/naive_bayes_validation.ipynb"
        ```

6. **User Interface**  
Start the Flask web application to visualize the results and interact with the models:

    ```sh
    python app.py
    ```

    **UI Features**
    - **Upload Datasets:** Users can upload their own datasets and get them labeled by the trained models.
    - **Label Single Sentence:** Users can input a single sentence and get it labeled by different models.



## Project Structure

The project is organized into the following main directories and files:

### 1. Data-Exploration
Contains notebooks and scripts for initial data exploration.

### 2. Data-Preparation
- `sentiment140`
  - `Preprocessing_training.ipynb`: Notebook for preprocessing the Sentiment140 training dataset.
  - `testdata_with_stopwords_preprocessed.csv`: Preprocessed test data with stopwords.
  - `testdata_without_stopwords_preprocessed.csv`: Preprocessed test data without stopwords.
  - `traindata_with_stopwords_preprocessed.csv`: Preprocessed training data with stopwords.
  - `traindata_without_stopwords_preprocessed.csv`: Preprocessed training data without stopwords.
  - `training.1600000.processed.noemoticon.csv`: Raw training data from Sentiment140.
  
- `twitter-corpus`
  - `full_corpus_preprocessed.csv`: Preprocessed full corpus.
  - `full-corpus.csv`: Raw full corpus data.
  - `Preprocessing_corpus.ipynb`: Notebook for preprocessing the Twitter corpus.

### 3. Flet UI
- `static/uploads`
  - Contains files related to UI uploads like predictions and labeled data.
  - `predictions_with_data.csv`: Predictions combined with original data.
  - `predictions.csv`: Predictions from models.
  - `twitter_labelled.csv`: Twitter corpus with manual labels.
  
- `templates`
  - `index.html`: Homepage of the web application.
  - `result_text.html`: Page displaying text results.
  - `results.html`: Page displaying overall results.

- `app.py`: Main application file to run the Flask web server.
- `lgbm_model_with_stopwords.pkl`: Pretrained LGBM model with stopwords.
- `naive_bayes_with_stopwords_classifier.pkl`: Pretrained Naive Bayes classifier with stopwords.
- `tfidf_vectorizer.pkl`: TF-IDF vectorizer.

### 4. ML-Based Approach
- **Testing**
  - `LGBM`
    - `LGBM_with_stopwords.ipynb`: Notebook for testing LGBM model with stopwords.
    - `LGBM_without_stopwords.ipynb`: Notebook for testing LGBM model without stopwords.
  - `naive_bayes`
    - `Naive_Bayes_with_stopwords.ipynb`: Notebook for testing Naive Bayes model with stopwords.
    - `Naive_Bayes_without_stopwords.ipynb`: Notebook for testing Naive Bayes model without stopwords.
  
- **Training**
  - `LGBM`
  - `naive_bayes`
    - `Naive_Bayes_with_stopwords_classifier.pkl`: Trained Naive Bayes model with stopwords.
    - `Naive_Bayes_with_stopwords.ipynb`: Notebook for training Naive Bayes model with stopwords.
    - `Naive_Bayes_without_stopwords_classifier.pkl`: Trained Naive Bayes model without stopwords.
    - `Naive_Bayes_without_stopwords.ipynb`: Notebook for training Naive Bayes model without stopwords.
  - `XGBoost`

- **Validation**
  - `naive_bayes_validation.ipynb`: Notebook for validating Naive Bayes model.

### 5. Rule-Based Approach
- **Testing**
  - `sentiwordnet_with_stopwords.ipynb`: Testing with SentiWordNet including stopwords.
  - `sentiwordnet_without_stopwords.ipynb`: Testing with SentiWordNet excluding stopwords.
  - `vader_with_stopwords.ipynb`: Testing with VADER including stopwords.
  - `vader_without_stopwords.ipynb`: Testing with VADER excluding stopwords.
  
- **Training**
  - `sentiwordnet_with_stopwords.ipynb`
  - `sentiwordnet_without_stopwords.ipynb`
  - `vader_with_stopwords.ipynb`
  - `vader_without_stopwords.ipynb`
  
- **Validation**
  - `sentiwordnet_without_stopwords.ipynb`

### Root Files
- `.gitignore`: Git ignore file.
- `README.md`: Project readme file.
- `requirements.txt`: List of required Python packages.



