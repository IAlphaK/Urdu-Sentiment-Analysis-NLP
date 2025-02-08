# Urdu Sentiment Analysis Tool NLP

## Overview
This project focuses on developing an **automatic sentiment analysis tool for Urdu text** on social media platforms. The pipeline includes data preprocessing, feature extraction, N-gram analysis, and a sentiment classification model. The system was trained using a Logistic Regression classifier with **TF-IDF** features, achieving an **accuracy of approximately 75%**.

## Features
- **Text Preprocessing:** Cleans text by removing special characters, URLs, hashtags, mentions, and emojis.
- **Stopword Removal:** Uses a custom Urdu stopword list while preserving sentiment words.
- **Stemming & Lemmatization:** Applies advanced Urdu-specific normalization techniques.
- **Feature Extraction:** Uses **TF-IDF** and **Word2Vec embeddings** for better text representation.
- **N-gram Analysis:** Extracts and analyzes **unigrams, bigrams, and trigrams** for linguistic insights.
- **Sentiment Classification:** Implements **Logistic Regression** to classify Urdu text as sarcastic or non-sarcastic.

## Installation
### Prerequisites
Ensure you have **Python 3.7+** installed along with the required dependencies.

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/sentiment-analysis-urdu.git
   ```
2. Import your dataset and stopwords list.
3. Run the main script:
   ```bash
   python main.py
   ```

## Methodology
1. **Data Preprocessing:**
   - Converts text to a standard format.
   - Removes unwanted characters and structures.
   - Filters out **short posts** (< 3 words).
2. **Feature Engineering:**
   - **TF-IDF** for frequency-based representations.
   - **Word2Vec** for contextual relationships.
3. **Classification:**
   - Splits data into **80% training** and **20% testing**.
   - Trains a **Logistic Regression model**.
   - Evaluates the model using **precision, recall, F1-score, and confusion matrix**.

## Project Results
### Model Performance
- **Accuracy:** ~75%
- **Precision & Recall:** ~75-76%

### Confusion Matrix
| Actual \ Predicted | Sarcastic | Non-Sarcastic |
|------------------|-----------|--------------|
| **Sarcastic**   | 320       | 80           |
| **Non-Sarcastic** | 85        | 315          |

### Proposed Improvements
- **Expanding the lemmatization dictionary** for better text normalization.
- **Exploring alternative classification models** like **SVM, Random Forests, and Gradient Boosting**.
- **Using BERT-based embeddings** to capture semantic relationships better.
- **Handling class imbalance** with **SMOTE or weighted loss functions**.
- **Applying k-fold cross-validation** for robust performance evaluation.

## Contributing
Feel free to contribute by submitting pull requests or reporting issues.

## License
This project is licensed under the **MIT License**.

## Author
**Abdullah Basit | 20I-0623**
