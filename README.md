# Sentiment Analysis Project with NLTK

This project provides a complete end-to-end pipeline for sentiment analysis on movie reviews using Python, NLTK, and scikit-learn. The notebook walks through data preprocessing, visualization, feature extraction, machine learning model training, evaluation, and prediction.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Project Workflow](#project-workflow)
- [Results](#results)
- [Acknowledgements](#acknowledgements)

---

## Project Overview

Sentiment analysis aims to determine a writer’s attitude or emotional state towards a topic. In this project, we use a dataset of IMDB movie reviews to classify each review as either positive or negative using various NLP techniques and a Logistic Regression model.

## Dataset

- **Source:** IMDB Movie Reviews
- **Columns:**
  - `text`: The movie review content.
  - `label`: Sentiment label (1 = positive, 0 = negative).
- **Size:** 40,000 reviews (subset of 10,000 used for demonstration).

## Installation

Clone the repository and install the required Python libraries:

```bash
git clone https://github.com/shanmukaa/Sentiment-Analysis.git
cd Sentiment-Analysis
pip install -r requirements.txt
```

**Required Libraries:**
- pandas
- numpy
- matplotlib
- seaborn
- nltk
- scikit-learn

You may also need to download NLTK resources:

```python
import nltk
nltk.download('stopwords')
```

## Usage

1. Open the Jupyter notebook:
    ```bash
    jupyter notebook "Sentiment Analysis Project with NLTK .ipynb"
    ```
2. Run the cells step by step to:
    - Load and explore the data
    - Clean and preprocess the text
    - Visualize word distributions
    - Extract features using TF-IDF
    - Train and evaluate a machine learning model
    - Make predictions on new reviews

## Project Workflow

1. **Read Data:** Load the IMDB dataset using pandas.
2. **Explore Data:** Understand label distribution and text samples.
3. **Data Preprocessing:**
    - Remove HTML tags and emojis
    - Remove special characters and punctuation
    - Convert to lowercase
    - Remove stopwords
    - Tokenize and stem words
4. **Visualization:** 
    - Visualize label distribution (bar and pie charts)
    - Visualize most common words for positive and negative reviews
5. **Feature Extraction:** 
    - Use TF-IDF vectorizer to convert text to numeric features
6. **Model Training:** 
    - Split data into training and test sets
    - Train a Logistic Regression model with cross-validation
7. **Evaluation:** 
    - Compute accuracy on the test set
8. **Model Saving:** 
    - Save the trained model and vectorizer using pickle
9. **Prediction:** 
    - Use the trained model to classify new reviews as positive or negative

## Results

- **Model Accuracy:** ~87.56% on the test set
- The notebook includes code to print and interpret predictions for new reviews.
