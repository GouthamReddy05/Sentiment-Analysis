# IMDB Movie Review Sentiment Analysis 🎬

This project provides a comprehensive analysis of sentiment classification on the IMDB movie review dataset. It explores and compares various modeling techniques, from a baseline Logistic Regression to advanced deep learning architectures like LSTM and Bidirectional LSTM. A key focus is the comparative performance of custom-trained Word2Vec embeddings versus Google's pre-trained word2vec-google-news-300 model.

# 💾 Dataset

This project utilizes the well-known IMDB Movie Review dataset, which contains 50,000 labeled movie reviews. The dataset is split into an 80% training set and a 20% testing set to ensure robust evaluation of the models.

# ⚙️ Project Workflow

The project follows a structured and comparative machine learning pipeline:

### Data Loading & Cleaning:
The dataset is loaded, and a cleaning function is applied to each review to remove HTML tags and other non-alphanumeric characters.

### Text Preprocessing:
The cleaned reviews are tokenized into individual words, and common English stopwords are removed to reduce noise. Stemming was intentionally omitted from the preprocessing pipeline.

### Feature Extraction (Dual Approach):
Two distinct methods were used to convert the text data into numerical vectors:

### Custom Word Embeddings:
A Word2Vec model was trained from scratch on the project's training data corpus using gensim.

### Pre-trained Word Embeddings:
Google's powerful word2vec-google-news-300 model was loaded via gensim to generate high-quality, pre-trained vectors for the review text.

# Model Training & Tuning:

### Baseline Model:
A Logistic Regression classifier was trained to establish a performance baseline.

### Deep Learning Models:
Both standard LSTM and Bidirectional LSTM networks were built and trained on the features generated from both the custom and pre-trained embedding approaches.

### Hyperparameter Tuning:
Grid Search Cross-Validation (GridSearchCV) was employed to systematically find the optimal hyperparameters for the models, ensuring the best possible performance.

### Evaluation:
All models were rigorously evaluated on the 20% test set to compare their accuracy, precision, recall, and F1-score.

# 📊 Performance and Results 🏆

The final, optimized accuracy for the project is 87%.
This result was achieved by the LSTM model. An accuracy of 87% signifies that the model can correctly classify the sentiment of 87 out of every 100 unseen movie reviews, making it a robust and reliable classifier. This strong performance validates the effectiveness of using advanced deep learning architectures with pre-trained word embeddings for complex sentiment analysis tasks.


# 🛠️ Technologies Used

### Programming Language:
Python 3

### Libraries:

TensorFlow & Keras

Gensim (for Word2Vec)

Scikit-learn (for Logistic Regression, GridSearchCV, and metrics)

NLTK (Natural Language Toolkit)

Pandas & NumPy

# 🚀 How to Run this Project

To replicate this project on your local machine, follow these steps:

### Clone the repository:

git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
Set up a virtual environment (recommended):

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
Install the required libraries:

pip install tensorflow pandas scikit-learn nltk gensim
Download NLTK data:
Run the following commands in a Python shell to download the necessary NLTK models:

Python

import nltk
nltk.download('punkt')
nltk.download('stopwords')
Run the Jupyter Notebook:

jupyter notebook sentiment_analysis.ipynb
Execute the cells in order to load the data, run the preprocessing pipeline, and train the various models.
