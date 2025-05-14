import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
import re
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import spacy
from collections import Counter

# Download all required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('vader_lexicon')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# Load spaCy model
try:
    nlp = spacy.load('en_core_web_sm')
except:
    import subprocess
    subprocess.run(['python', '-m', 'spacy', 'download', 'en_core_web_sm'])
    nlp = spacy.load('en_core_web_sm')

def calculate_readability(text):
    """Calculate Flesch Reading Ease score"""
    sentences = sent_tokenize(text)
    words = word_tokenize(text.lower())
    syllables = sum(count_syllables(word) for word in words)
    
    if len(sentences) == 0 or len(words) == 0:
        return 0
    
    avg_sentence_length = len(words) / len(sentences)
    avg_syllables_per_word = syllables / len(words)
    
    # Flesch Reading Ease formula
    score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
    return max(0, min(100, score))

def count_syllables(word):
    """Count the number of syllables in a word"""
    word = word.lower()
    count = 0
    vowels = "aeiouy"
    previous_is_vowel = False
    
    for char in word:
        is_vowel = char in vowels
        if is_vowel and not previous_is_vowel:
            count += 1
        previous_is_vowel = is_vowel
    
    if word.endswith("e"):
        count -= 1
    return max(1, count)

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    
    return ' '.join(tokens)

def extract_features(df):
    # Basic text features
    df['text_length'] = df['statement'].str.len().fillna(0)
    df['word_count'] = df['statement'].str.split().str.len().fillna(0)
    df['avg_word_length'] = (df['text_length'] / df['word_count']).fillna(0)
    df['sentence_count'] = df['statement'].str.count(r'[.!?]+').fillna(0)
    df['capital_count'] = df['statement'].str.count(r'[A-Z]').fillna(0)
    df['exclamation_count'] = df['statement'].str.count(r'\!').fillna(0)
    df['question_count'] = df['statement'].str.count(r'\?').fillna(0)
    
    # Sentiment analysis
    sia = SentimentIntensityAnalyzer()
    df['sentiment_scores'] = df['statement'].apply(lambda x: sia.polarity_scores(x))
    df['sentiment_compound'] = df['sentiment_scores'].apply(lambda x: x['compound'])
    df['sentiment_neg'] = df['sentiment_scores'].apply(lambda x: x['neg'])
    df['sentiment_neu'] = df['sentiment_scores'].apply(lambda x: x['neu'])
    df['sentiment_pos'] = df['sentiment_scores'].apply(lambda x: x['pos'])
    
    # TextBlob features
    df['textblob_polarity'] = df['statement'].apply(lambda x: TextBlob(x).sentiment.polarity)
    df['textblob_subjectivity'] = df['statement'].apply(lambda x: TextBlob(x).sentiment.subjectivity)
    
    # Readability score
    df['readability_score'] = df['statement'].apply(calculate_readability)
    
    # Named Entity Recognition features
    def count_entities(text):
        doc = nlp(text)
        entity_counts = Counter([ent.label_ for ent in doc.ents])
        return {
            'PERSON': entity_counts.get('PERSON', 0),
            'ORG': entity_counts.get('ORG', 0),
            'GPE': entity_counts.get('GPE', 0),
            'DATE': entity_counts.get('DATE', 0),
            'NUM': entity_counts.get('NUM', 0)
        }
    
    entity_features = df['statement'].apply(count_entities)
    df['person_count'] = entity_features.apply(lambda x: x['PERSON'])
    df['org_count'] = entity_features.apply(lambda x: x['ORG'])
    df['location_count'] = entity_features.apply(lambda x: x['GPE'])
    df['date_count'] = entity_features.apply(lambda x: x['DATE'])
    df['number_count'] = entity_features.apply(lambda x: x['NUM'])
    
    # Additional features
    df['has_numbers'] = df['statement'].str.contains(r'\d').astype(int)
    df['has_quotes'] = df['statement'].str.contains(r'["\']').astype(int)
    df['has_parentheses'] = df['statement'].str.contains(r'[()]').astype(int)
    df['has_emails'] = df['statement'].str.contains(r'[\w\.-]+@[\w\.-]+').astype(int)
    df['has_urls'] = df['statement'].str.contains(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+').astype(int)
    
    # POS tag ratios
    def get_pos_ratios(text):
        try:
            doc = nlp(text)
            if not doc:
                return {'noun_ratio': 0, 'verb_ratio': 0, 'adj_ratio': 0}
            
            total = len(doc)
            if total == 0:
                return {'noun_ratio': 0, 'verb_ratio': 0, 'adj_ratio': 0}
            
            noun_count = sum(1 for token in doc if token.pos_ in ['NOUN', 'PROPN'])
            verb_count = sum(1 for token in doc if token.pos_ == 'VERB')
            adj_count = sum(1 for token in doc if token.pos_ == 'ADJ')
            
            return {
                'noun_ratio': noun_count / total,
                'verb_ratio': verb_count / total,
                'adj_ratio': adj_count / total
            }
        except Exception as e:
            print(f"Warning: Error in POS tagging: {str(e)}")
            return {'noun_ratio': 0, 'verb_ratio': 0, 'adj_ratio': 0}
    
    pos_features = df['statement'].apply(get_pos_ratios)
    df['noun_ratio'] = pos_features.apply(lambda x: x['noun_ratio'])
    df['verb_ratio'] = pos_features.apply(lambda x: x['verb_ratio'])
    df['adj_ratio'] = pos_features.apply(lambda x: x['adj_ratio'])
    
    # Convert all numeric columns to float
    numeric_columns = [
        'text_length', 'word_count', 'avg_word_length', 'sentence_count',
        'capital_count', 'exclamation_count', 'question_count',
        'sentiment_compound', 'sentiment_neg', 'sentiment_neu', 'sentiment_pos',
        'textblob_polarity', 'textblob_subjectivity', 'readability_score',
        'person_count', 'org_count', 'location_count', 'date_count', 'number_count',
        'has_numbers', 'has_quotes', 'has_parentheses', 'has_emails', 'has_urls',
        'noun_ratio', 'verb_ratio', 'adj_ratio'
    ]
    
    for col in numeric_columns:
        df[col] = df[col].astype(float)
    
    return df

def train_model():
    # Load the dataset
    print("Loading LIAR dataset...")
    train_data = pd.read_csv('Datasets/train.tsv', sep='\t', header=None)
    test_data = pd.read_csv('Datasets/test.tsv', sep='\t', header=None)
    valid_data = pd.read_csv('Datasets/valid.tsv', sep='\t', header=None)
    
    # Define column names based on LIAR dataset format
    columns = [
        'id', 'label', 'statement', 'subject', 'speaker', 'job_title',
        'state', 'party', 'barely_true', 'false', 'half_true',
        'mostly_true', 'pants_on_fire', 'context'
    ]
    
    train_data.columns = columns
    test_data.columns = columns
    valid_data.columns = columns
    
    # Combine train and validation data for better training
    train_data = pd.concat([train_data, valid_data], ignore_index=True)
    
    # Convert labels to binary (true/false)
    # LIAR dataset has 6 labels: pants-fire, false, barely-true, half-true, mostly-true, true
    label_mapping = {
        'true': 1,
        'mostly-true': 1,
        'half-true': 1,
        'barely-true': 0,
        'false': 0,
        'pants-fire': 0
    }
    
    train_data['binary_label'] = train_data['label'].map(label_mapping)
    test_data['binary_label'] = test_data['label'].map(label_mapping)
    
    # Add speaker credibility features
    def calculate_speaker_credibility(row):
        total = row['barely_true'] + row['false'] + row['half_true'] + row['mostly_true'] + row['pants_on_fire']
        if total == 0:
            return 0.5  # neutral if no history
        true_count = row['mostly_true'] + row['half_true']
        return true_count / total
    
    train_data['speaker_credibility'] = train_data.apply(calculate_speaker_credibility, axis=1)
    test_data['speaker_credibility'] = test_data.apply(calculate_speaker_credibility, axis=1)
    
    # Extract additional features
    print("Extracting features from training data...")
    train_data = extract_features(train_data)
    print("Extracting features from test data...")
    test_data = extract_features(test_data)
    
    # Preprocess the text data
    print("Preprocessing text data...")
    train_data['processed_statement'] = train_data['statement'].apply(preprocess_text)
    test_data['processed_statement'] = test_data['statement'].apply(preprocess_text)
    
    # Define feature columns
    text_features = ['processed_statement']
    numeric_features = [
        'text_length', 'word_count', 'avg_word_length', 'sentence_count',
        'capital_count', 'exclamation_count', 'question_count',
        'sentiment_compound', 'sentiment_neg', 'sentiment_neu', 'sentiment_pos',
        'textblob_polarity', 'textblob_subjectivity', 'readability_score',
        'person_count', 'org_count', 'location_count', 'date_count', 'number_count',
        'has_numbers', 'has_quotes', 'has_parentheses', 'has_emails', 'has_urls',
        'noun_ratio', 'verb_ratio', 'adj_ratio', 'speaker_credibility'
    ]
    
    # Create TF-IDF vectorizer with improved parameters
    vectorizer = TfidfVectorizer(
        max_features=15000,
        ngram_range=(1, 3),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )
    
    # Transform text features
    print("Transforming text features...")
    X_train_text = vectorizer.fit_transform(train_data['processed_statement'])
    X_test_text = vectorizer.transform(test_data['processed_statement'])
    
    # Scale numeric features
    print("Scaling numeric features...")
    scaler = StandardScaler()
    X_train_numeric = scaler.fit_transform(train_data[numeric_features])
    X_test_numeric = scaler.transform(test_data[numeric_features])
    
    # Combine features
    X_train = np.hstack([X_train_text.toarray(), X_train_numeric])
    X_test = np.hstack([X_test_text.toarray(), X_test_numeric])
    
    y_train = train_data['binary_label']
    y_test = test_data['binary_label']
    
    # Create DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    # XGBoost parameters with improved settings
    params = {
        'objective': 'binary:logistic',
        'eval_metric': ['logloss', 'auc'],
        'tree_method': 'hist',  # Use CPU for better compatibility
        'max_depth': 8,
        'learning_rate': 0.03,
        'n_estimators': 500,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1
    }
    
    # Train the model
    print("Training the model...")
    print(f"Training data shape: {X_train.shape}")
    
    # Train with early stopping
    evallist = [(dtrain, 'train'), (dtest, 'eval')]
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=evallist,
        early_stopping_rounds=30,
        verbose_eval=10
    )
    
    # Make predictions
    y_pred = (model.predict(dtest) > 0.5).astype(int)
    
    # Evaluate the model
    print("\nEvaluating the model...")
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save the model and transformers
    print("\nSaving the model and transformers...")
    model.save_model('model/fake_news_model.json')
    joblib.dump(vectorizer, 'model/vectorizer.joblib')
    joblib.dump(scaler, 'model/scaler.joblib')
    print("Model and transformers saved successfully!")
    
    return model, vectorizer, scaler

if __name__ == "__main__":
    train_model() 