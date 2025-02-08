import pandas as pd
import re
import string
import nltk
nltk.download('punkt')
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from collections import Counter
from nltk.util import ngrams

# Import necessary libraries for classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Function to load the dataset
def load_dataset(excel_file_path):
    """
    Loads the dataset from an Excel file into a pandas DataFrame.

    Parameters:
    excel_file_path (str): The file path to the Excel file containing the dataset.

    Returns:
    DataFrame: A pandas DataFrame containing the dataset.
    """
    # Read the Excel file using pandas with openpyxl engine
    df = pd.read_excel(excel_file_path, engine='openpyxl')

    # Display the first few rows to verify
    print("Dataset loaded successfully. Here's a preview:")
    print(df.head())
    return df

# Function to load Urdu stopwords from a text file
def load_stopwords(stopwords_file_path):
    """
    Loads Urdu stopwords from a text file into a list.

    Parameters:
    stopwords_file_path (str): The file path to the text file containing Urdu stopwords.

    Returns:
    list: A list of Urdu stopwords.
    """
    with open(stopwords_file_path, 'r', encoding='utf-8') as f:
        stopwords = f.read().splitlines()

    # List of sentiment words to exclude from stopwords
    sentiment_words = [
        'نہیں',  # No
        'برا',  # Bad
        'خوش',  # Happy
        'غم',  # Sadness
        'محبت',  # Love
        'نفرت',  # Hate
        'پیار',  # Affection
        'خوبصورت',  # Beautiful
        'بدصورت',  # Ugly
        'خوشی',  # Joy
        'مایوسی',  # Disappointment
        'پریشان',  # Worried
        'افسردہ',  # Depressed
        'تنقید',  # Criticism
        'حیرت',  # Amazement
        'گھبراہٹ',  # Anxiety
        'غصہ',  # Anger
        'نرمی',  # Softness
        'حسد',  # Envy
        'خوشبو',  # Fragrance
        'تلخی',  # Bitterness
        'تکلیف',  # Pain
        'اطمینان',  # Satisfaction
        'ہنسی',  # Laughter
        'خودغرضی',  # Selfishness
        'سکون',  # Peace
        'حوصلہ',  # Courage
        'غمگین',  # Sad
        'مسرت',  # Delight
        'نفرت انگیز',  # Hateful
        'مظلوم',  # Oppressed
        'اچھا',  # Good
        'بہتر',  # Better
        'بری',  # Bad
        'دلکش',  # Charming
        'رحم',  # Compassion
        'بےوفا',  # Disloyal
        'وفادار',  # Loyal
        'شرمندگی',  # Embarrassment
        'فخر',  # Pride
        'خوف',  # Fear
        'امید',  # Hope
        'یقین',  # Trust
        'گندا',  # Dirty
        'صاف',  # Clean
        'مہربان',  # Kind
        'بدتمیز',  # Rude
        'طاقتور',  # Strong
        'کمزور',  # Weak
        'شکریہ'  # Thanks
    ]
    stopwords = [word for word in stopwords if word not in sentiment_words]

    print("\nStopwords loaded successfully. Number of stopwords:", len(stopwords))
    return stopwords

# Function to remove stopwords from a piece of text
def remove_stopwords(text, stopwords):
    """
    Removes stopwords from the given Urdu text.

    Parameters:
    text (str): The Urdu text from which to remove stopwords.
    stopwords (list): A list of Urdu stopwords.

    Returns:
    str: The text after stopword removal.
    """
    # Tokenize the text by splitting on spaces
    words = text.split()

    # Remove stopwords
    words_filtered = [word for word in words if word not in stopwords]

    # Rejoin the words into a single string
    cleaned_text = ' '.join(words_filtered)

    return cleaned_text

# Function to clean text by removing URLs, hashtags, mentions, punctuations, and emojis
def clean_text(text):
    """
    Cleans the text by removing URLs, hashtags, mentions, punctuations, and emojis.

    Parameters:
    text (str): The text to clean.

    Returns:
    str: The cleaned text.
    """
    # Ensure the input is a string
    text = str(text)

    # Remove URLs
    text = re.sub(r'http\S+|www.\S+', '', text)

    # Remove mentions (@username)
    text = re.sub(r'@\w+', '', text)

    # Remove hashtags (#hashtag)
    text = re.sub(r'#\w+', '', text)

    # Remove punctuations
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove emojis
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"  # Emoticons
        u"\U0001F300-\U0001F5FF"  # Symbols & Pictographs
        u"\U0001F680-\U0001F6FF"  # Transport & Map Symbols
        u"\U0001F1E0-\U0001F1FF"  # Flags (iOS)
        "]+",
        flags=re.UNICODE,
    )
    text = emoji_pattern.sub(r'', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# Function to filter out short posts with fewer than three words
def filter_short_posts(text):
    """
    Filters out text posts that have fewer than three words.

    Parameters:
    text (str): The text to evaluate.

    Returns:
    str: The text if it contains three or more words; otherwise, returns an empty string.
    """
    words = text.split()
    if len(words) >= 3:
        return text
    else:
        return ''

# Function to perform stemming on a word
def stem_word(word):
    """
    Stems the given Urdu word by removing common suffixes.

    Parameters:
    word (str): The Urdu word to stem.

    Returns:
    str: The stemmed word.
    """
    # List of common Urdu suffixes
    suffixes = [
        'یں', 'وں', 'ہ', 'ے', 'ی', 'ا', 'ات', 'وات', 'یات', 'اتی', 'اتیوں', 'اگا', 'اگی', 'ائیں',
        'ون', 'ہو', 'دار', 'اتی', 'اتیوں', 'اؤں', 'ئوں', 'ین', 'یگے', 'ینگے', 'اؤ', 'ئی',
        'یاں', 'یوں', 'وگی', 'وگا', 'گے', 'گی', 'یے', 'ہوئے', 'ہوگی', 'ہوگا', 'ہو', 'دے', 'دیں', 'تے',
        'ان', 'اوی', 'ائے', 'اتے', 'اتا', 'اتی', 'اتیوں', 'انے', 'اوں', 'پنا', 'پنے', 'وں', 'یپ', 'یگے',
        'ائی', 'او', 'نی', 'ندہ', 'ین', 'دی', 'اگی', 'ایا', 'پہ', 'ہائ', 'یت', 'ہات', 'ست', 'ائ', 'یات',
        'یلت', 'یاس', 'تی', 'شے', 'تہ', 'بھی', 'ارے', 'اروں', 'اڑ', 'اڑے', 'انہ', 'اوئے', 'ائیے', 'پے'
    ]

    # Remove suffixes
    for suffix in suffixes:
        if word.endswith(suffix):
            word = word[:-len(suffix)]
            break  # Remove only one suffix

    return word

# Function to stem all words in a piece of text
def stem_text(text):
    """
    Applies stemming to all words in the given text.

    Parameters:
    text (str): The text to stem.

    Returns:
    str: The text after stemming.
    """
    words = text.split()
    stemmed_words = [stem_word(word) for word in words]
    stemmed_text = ' '.join(stemmed_words)
    return stemmed_text

# Function to perform lemmatization on a word
def lemmatize_word(word, lemma_dict):
    """
    Lemmatizes the given Urdu word using the provided lemma dictionary.

    Parameters:
    word (str): The word to lemmatize.
    lemma_dict (dict): A dictionary mapping inflected forms to lemmas.

    Returns:
    str: The lemmatized word.
    """
    return lemma_dict.get(word, word)

# Function to lemmatize all words in a piece of text
def lemmatize_text(text, lemma_dict):
    """
    Applies lemmatization to all words in the given text.

    Parameters:
    text (str): The text to lemmatize.
    lemma_dict (dict): A dictionary mapping inflected forms to lemmas.

    Returns:
    str: The text after lemmatization.
    """
    words = text.split()
    lemmatized_words = [lemmatize_word(word, lemma_dict) for word in words]
    lemmatized_text = ' '.join(lemmatized_words)
    return lemmatized_text

# Function to tokenize text
def tokenize_text(text):
    """
    Tokenizes the given Urdu text into words.

    Parameters:
    text (str): The text to tokenize.

    Returns:
    list: A list of tokens.
    """
    # Define a tokenizer that matches Urdu words
    tokenizer = RegexpTokenizer(r'[\u0600-\u06FF]+')
    tokens = tokenizer.tokenize(text)
    return tokens

# Function to perform tokenization on the dataset
def tokenize_dataset(df):
    """
    Tokenizes the 'cleaned_text' column of the DataFrame.

    Parameters:
    df (DataFrame): The preprocessed DataFrame.

    Returns:
    DataFrame: The DataFrame with an added 'tokens' column.
    """
    df['tokens'] = df['cleaned_text'].apply(tokenize_text)
    print("\nTokenization completed.")
    return df

# Function to compute TF-IDF scores
def compute_tfidf(df):
    """
    Computes TF-IDF scores for the dataset.

    Parameters:
    df (DataFrame): The DataFrame containing the 'cleaned_text'.

    Returns:
    tuple: (tfidf_matrix, feature_names)
    """
    # Define the TfidfVectorizer with custom tokenizer
    vectorizer = TfidfVectorizer(
        tokenizer=tokenize_text,
        token_pattern=None,  # Tokenizer overrides the token_pattern
        max_features=1000  # Adjust as needed
    )

    # Fit and transform the cleaned text
    tfidf_matrix = vectorizer.fit_transform(df['cleaned_text'])
    feature_names = vectorizer.get_feature_names()

    print("\nTF-IDF computation completed.")
    return tfidf_matrix, feature_names

# Function to display top TF-IDF terms
def display_top_tfidf_terms(tfidf_matrix, feature_names, top_n=10):
    """
    Displays the top N terms with the highest TF-IDF scores.

    Parameters:
    tfidf_matrix (sparse matrix): The TF-IDF matrix.
    feature_names (list): The list of feature names (terms).
    top_n (int): The number of top terms to display.
    """
    # Compute the average TF-IDF score for each term across all documents
    avg_tfidf_scores = tfidf_matrix.mean(axis=0).A1
    term_scores = list(zip(feature_names, avg_tfidf_scores))

    # Sort the terms by TF-IDF score in descending order
    sorted_terms = sorted(term_scores, key=lambda x: x[1], reverse=True)

    print(f"\nTop {top_n} words with the highest TF-IDF scores:")
    for term, score in sorted_terms[:top_n]:
        print(f"{term}: {score:.4f}")

# Function to train Word2Vec model
def train_word2vec(df, vector_size=100, window=5, min_count=2, workers=4):
    """
    Trains a Word2Vec model on the tokenized text.

    Parameters:
    df (DataFrame): The DataFrame containing the 'tokens' column.
    vector_size (int): Dimensionality of the word vectors.
    window (int): Maximum distance between the current and predicted word.
    min_count (int): Ignores words with total frequency lower than this.
    workers (int): Number of worker threads to train the model.

    Returns:
    Word2Vec: The trained Word2Vec model.
    """
    # Prepare the sentences (list of tokens)
    sentences = df['tokens'].tolist()

    # Train the Word2Vec model
    model = Word2Vec(
        sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers
    )

    print("\nWord2Vec model training completed.")
    return model

# Function to find similar words using the Word2Vec model
def find_similar_words(model, word, top_n=5):
    """
    Finds the top N words most similar to the given word using the Word2Vec model.

    Parameters:
    model (Word2Vec): The trained Word2Vec model.
    word (str): The word for which to find similar words.
    top_n (int): The number of similar words to return.

    Returns:
    list: A list of tuples containing similar words and their similarity scores.
    """
    try:
        similar_words = model.wv.most_similar(word, topn=top_n)
        print(f"\nTop {top_n} words similar to '{word}':")
        for sim_word, score in similar_words:
            print(f"{sim_word}: {score:.4f}")
    except KeyError:
        print(f"\nWord '{word}' not found in the vocabulary.")
        similar_words = []
    return similar_words

# Function to perform N-gram analysis
def ngram_analysis(df, n):
    """
    Performs N-gram analysis on the tokenized text.

    Parameters:
    df (DataFrame): The DataFrame containing the 'tokens' column.
    n (int): The number of grams (e.g., 1 for unigrams, 2 for bigrams).

    Returns:
    Counter: A Counter object with N-gram frequencies.
    """
    ngram_counts = Counter()
    for tokens in df['tokens']:
        ngrams_generated = ngrams(tokens, n)
        ngram_counts.update(ngrams_generated)
    return ngram_counts

# Main function to process the dataset
# Main function to process the dataset
def preprocess_dataset(df, stopwords, lemma_dict):
    """
    Preprocesses the dataset by cleaning text, removing stopwords, filtering short posts,
    and applying stemming and lemmatization.

    Parameters:
    df (DataFrame): The pandas DataFrame containing the dataset.
    stopwords (list): A list of Urdu stopwords.
    lemma_dict (dict): A dictionary mapping inflected forms to lemmas.

    Returns:
    DataFrame: The preprocessed DataFrame.
    """
    # Convert all entries in 'urdu_text' to strings
    df['urdu_text'] = df['urdu_text'].astype(str)

    # Text cleaning
    df['cleaned_text'] = df['urdu_text'].apply(clean_text)
    print("\nText cleaning completed.")

    # Remove stopwords
    df['cleaned_text'] = df['cleaned_text'].apply(lambda x: remove_stopwords(x, stopwords))
    print("Stopword removal completed.")

    # Filter out short posts
    df['cleaned_text'] = df['cleaned_text'].apply(filter_short_posts)
    print("Short posts filtered.")

    # Apply stemming
    df['cleaned_text'] = df['cleaned_text'].apply(stem_text)
    print("Stemming completed.")

    # Apply lemmatization
    df['cleaned_text'] = df['cleaned_text'].apply(lambda x: lemmatize_text(x, lemma_dict))
    print("Lemmatization completed.")

    # Drop rows with empty 'cleaned_text'
    df = df[df['cleaned_text'] != '']
    df.reset_index(drop=True, inplace=True)

    return df
if __name__ == "__main__":
    # Replace with your actual file paths
    excel_file_path = 'dataset.xlsx'  # Update with your file path
    stopwords_file_path = 'stopwords-ur.txt'  # Update with your file path

    # Load the dataset
    df = load_dataset(excel_file_path)

    # Load the Urdu stopwords
    urdu_stopwords = load_stopwords(stopwords_file_path)

    # Create a lemmatization dictionary
    lemma_dict = {
        'کرتا': 'کر',
        'کرتی': 'کر',
        'کرتے': 'کر',
        'کیا': 'کر',
        'کی': 'کر',
        'ہیں': 'ہے',
        'تھا': 'ہے',
        'تھی': 'ہے',
        'تھے': 'ہے',
        'جاتا': 'جا',
        'جاتی': 'جا',
        'جاتے': 'جا',
        'گیا': 'جا',
        'گئی': 'جا',
        'گئے': 'جا',
        'آیا': 'آ',
        'آئی': 'آ',
        'آئے': 'آ',
        'لکھا': 'لکھ',
        'لکھتی': 'لکھ',
        'کھایا': 'کھا',
        'کھاتی': 'کھا',
        'چلتا': 'چل',
        'چلتی': 'چل',
        'دیتا': 'دے',
        'دیتی': 'دے',
        # Add more mappings as needed
    }

    # Preprocess the dataset
    df_preprocessed = preprocess_dataset(df, urdu_stopwords, lemma_dict)

    # Display the preprocessed DataFrame
    print("\nPreprocessing completed. Here's a preview of the preprocessed data:")
    print(df_preprocessed.head())

    # Save the preprocessed data to a new Excel file
    df_preprocessed.to_excel('preprocessed_dataset.xlsx', index=False, encoding='utf-8-sig')

    # Phase 3: Tokenization
    df_tokenized = tokenize_dataset(df_preprocessed)

    # Display tokenized text for sample sentences
    print("\nSample tokenized text:")
    print(df_tokenized[['cleaned_text', 'tokens']].head())

    # Phase 3: TF-IDF Calculation
    tfidf_matrix, feature_names = compute_tfidf(df_tokenized)

    # Display top TF-IDF terms
    display_top_tfidf_terms(tfidf_matrix, feature_names, top_n=10)

    # Phase 3: Word2Vec Training
    word2vec_model = train_word2vec(df_tokenized)

    # Find words similar to 'اچھا' (good)
    similar_words = find_similar_words(word2vec_model, 'اچھا', top_n=5)

    # Phase 4: N-gram Analysis
    # Unigrams (already tokenized words)
    unigram_counts = ngram_analysis(df_tokenized, 1)
    print("\nTop 10 unigrams:")
    for unigram, count in unigram_counts.most_common(10):
        print(f"{unigram[0]}: {count}")

    # Bigrams
    bigram_counts = ngram_analysis(df_tokenized, 2)
    print("\nTop 10 bigrams:")
    for bigram, count in bigram_counts.most_common(10):
        print(f"{bigram[0]} {bigram[1]}: {count}")

    # Trigrams
    trigram_counts = ngram_analysis(df_tokenized, 3)
    print("\nTop 10 trigrams:")
    for trigram, count in trigram_counts.most_common(10):
        print(f"{trigram[0]} {trigram[1]} {trigram[2]}: {count}")

    # ------------------------------
    # Phase 5: Sentiment Classification Model
    # ------------------------------

    # Ensure 'is_sarcastic' column is present
    if 'is_sarcastic' not in df_tokenized.columns:
        raise ValueError("The 'is_sarcastic' column is missing from the DataFrame.")

    # Prepare Features and Labels
    # Features: TF-IDF matrix
    # Labels: 'is_sarcastic' column
    X = tfidf_matrix  # TF-IDF features
    y = df_tokenized['is_sarcastic']  # Labels

    # Split the Dataset
    # Use train_test_split from scikit-learn to divide data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Build and Train the Model
    # Initialize the Logistic Regression classifier
    classifier = LogisticRegression(max_iter=1000)

    # Train the classifier on the training data
    classifier.fit(X_train, y_train)

    # Evaluate the Model
    # Make predictions on the test set
    y_pred = classifier.predict(X_test)

    # Compute evaluation metrics
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))