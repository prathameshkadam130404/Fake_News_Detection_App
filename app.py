import streamlit as st

# Set page config as the first Streamlit command
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide"
)

# Initialize session state for active tab if it doesn't exist
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "Top News"

# Import other libraries after st.set_page_config()
import joblib
import pandas as pd
import numpy as np
from newsapi import NewsApiClient
import os
from dotenv import load_dotenv
from train_model import preprocess_text, extract_features
import xgboost as xgb
import spacy
import requests
from bs4 import BeautifulSoup
import time
import re

# Load spaCy model
try:
    nlp = spacy.load('en_core_web_sm')
except:
    import subprocess
    subprocess.run(['python', '-m', 'spacy', 'download', 'en_core_web_sm'])
    nlp = spacy.load('en_core_web_sm')

# Load environment variables and secrets
try:
    # Load secrets from Streamlit
    api_key = st.secrets["NEWS_API_KEY"]
    model_path = st.secrets["MODEL_PATH"]
    vectorizer_path = st.secrets["VECTORIZER_PATH"]
    scaler_path = st.secrets["SCALER_PATH"]
    
    # Load app settings
    debug_mode = st.secrets["app"]["debug"]
    max_article_length = st.secrets["app"]["max_article_length"]
    max_search_results = st.secrets["app"]["max_search_results"]
except Exception as e:
    st.error(f"Error loading secrets: {str(e)}")
    st.info("Please make sure your secrets.toml file is properly configured.")
    st.stop()

# Initialize News API client
if not api_key or api_key == "your_actual_api_key":
    st.warning("⚠️ NEWS_API_KEY not found or not configured. Top News feature will not work.")
    st.info("To use this feature:")
    st.code("1. Get a free API key from https://newsapi.org/\n2. Configure your secrets.toml file with the NEWS_API_KEY")
    api_key = "dummy_key"  # Dummy key to prevent errors
newsapi = NewsApiClient(api_key=api_key)

# Load the model and transformers
@st.cache_resource
def load_model():
    try:
        model = xgb.Booster()
        model.load_model(model_path)
        vectorizer = joblib.load(vectorizer_path)
        scaler = joblib.load(scaler_path)
        return model, vectorizer, scaler
    except Exception as e:
        st.error(f"Error loading model files: {str(e)}")
        st.info("Please make sure all model files exist in the correct paths.")
        st.stop()

def fetch_article_content(url):
    """Fetch and parse article content from URL using BeautifulSoup with encoding detection"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        
        # Check if the response was successful
        response.raise_for_status()
        
        # Detect encoding - try from Content-Type header first, then from response.encoding
        encoding = None
        content_type = response.headers.get('Content-Type', '')
        if 'charset=' in content_type:
            encoding = content_type.split('charset=')[-1].split(';')[0].strip()
            
        if not encoding:
            # Use what requests detected
            encoding = response.encoding
        
        # Try to decode with the detected encoding
        try:
            html_content = response.content.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            # If that fails, try common encodings
            for enc in ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']:
                try:
                    html_content = response.content.decode(enc)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                # If all decodings fail, use latin-1 as a fallback (it never fails)
                html_content = response.content.decode('latin-1')
        
        # Parse the content
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Check if it's a valid HTML page
        if not soup.find('body'):
            return "Could not parse the article content. The URL might not be a valid HTML page."
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'iframe', 'noscript']):
            element.decompose()
        
        # Get article text from common elements that might contain the main content
        paragraphs = []
        for tag in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'article']):
            text = tag.get_text().strip()
            if text and len(text) > 20:  # Only include non-empty paragraphs of reasonable length
                paragraphs.append(text)
        
        if not paragraphs:
            # Fallback to div elements if no paragraphs found
            for tag in soup.find_all('div'):
                text = tag.get_text().strip()
                if text and len(text) > 50:  # Divs often contain more content, so use a higher threshold
                    paragraphs.append(text)
        
        # Join all paragraphs
        article_text = '\n\n'.join(paragraphs)
        
        # Clean up the text
        article_text = re.sub(r'\s+', ' ', article_text).strip()
        article_text = re.sub(r'[^\w\s.,!?;:()\-\'\"]+', '', article_text)
        
        if not article_text or len(article_text) < 100:
            return "Could not extract meaningful content from the article."
            
        return article_text
    
    except requests.exceptions.MissingSchema:
        return "Invalid URL. Make sure it starts with http:// or https://"
    except requests.exceptions.ConnectionError:
        return "Could not connect to the website. The URL might be invalid or the site might be down."
    except requests.exceptions.Timeout:
        return "The request timed out. The website might be slow or unresponsive."
    except requests.exceptions.HTTPError as e:
        return f"HTTP Error: {str(e)}"
    except Exception as e:
        return f"Error fetching article: {str(e)}"

def search_news_by_headline(headline):
    """Search for news articles by headline using News API"""
    try:
        # Search for articles with the headline
        response = newsapi.get_everything(
            q=headline,
            language='en',
            sort_by='relevancy',
            page_size=5
        )
        
        if response['status'] == 'ok':
            if response['articles']:
                st.success(f"Found {len(response['articles'])} articles matching your search.")
                return response['articles']
            else:
                # Instead of immediately marking as fake, provide more nuanced feedback
                st.warning("⚠️ No exact matches found for this headline.")
                st.info("This could mean:")
                st.write("1. The headline might be too recent and hasn't been indexed yet")
                st.write("2. The headline might be paraphrased or worded differently in news sources")
                st.write("3. The story might be from a source not covered by News API")
                st.write("4. The headline might contain false or unverified claims")
                
                # Try a broader search with key terms
                key_terms = ' '.join([word for word in headline.split() if len(word) > 3])
                if key_terms:
                    st.info("Searching for related articles with key terms...")
                    related_response = newsapi.get_everything(
                        q=key_terms,
                        language='en',
                        sort_by='relevancy',
                        page_size=3
                    )
                    if related_response['status'] == 'ok' and related_response['articles']:
                        st.success("Found some related articles. This suggests the topic exists but the exact headline may be paraphrased.")
                        return related_response['articles']
                
                return []
        else:
            st.error(f"API Error: {response.get('message', 'Unknown error')}")
            return []
    except Exception as e:
        error_msg = str(e)
        st.error(f"Error searching news: {error_msg}")
        
        if "401" in error_msg:
            st.error("API key error: Please make sure your NEWS_API_KEY is valid in the .env file")
        elif "426" in error_msg:
            st.error("You are using a developer API key which has limitations. Please upgrade your API key or reduce requests.")
        elif "429" in error_msg:
            st.error("Too many requests: You've reached the API request limit. Please try again later.")
        
        return []

def predict_news(text, model, vectorizer, scaler):
    # Create a DataFrame with the input text
    df = pd.DataFrame({
        'statement': [text],
        'processed_statement': [preprocess_text(text)]
    })
    
    # Add speaker credibility (neutral for single predictions)
    df['barely_true'] = 0
    df['false'] = 0
    df['half_true'] = 0
    df['mostly_true'] = 0
    df['pants_on_fire'] = 0
    df['speaker_credibility'] = 0.5
    
    # Extract features
    df = extract_features(df)
    
    # Transform text features
    text_features = vectorizer.transform(df['processed_statement'])
    
    # Transform numeric features
    numeric_features = [
        'text_length', 'word_count', 'avg_word_length', 'sentence_count',
        'capital_count', 'exclamation_count', 'question_count',
        'sentiment_compound', 'sentiment_neg', 'sentiment_neu', 'sentiment_pos',
        'textblob_polarity', 'textblob_subjectivity', 'readability_score',
        'person_count', 'org_count', 'location_count', 'date_count', 'number_count',
        'has_numbers', 'has_quotes', 'has_parentheses', 'has_emails', 'has_urls',
        'noun_ratio', 'verb_ratio', 'adj_ratio', 'speaker_credibility'
    ]
    numeric_data = scaler.transform(df[numeric_features])
    
    # Combine features
    X = np.hstack([text_features.toarray(), numeric_data])
    
    # Create DMatrix for prediction
    dtest = xgb.DMatrix(X)
    
    # Make prediction
    probability = model.predict(dtest)[0]
    prediction = int(probability > 0.5)
    return prediction, [1 - probability, probability]

def get_top_news(query, search_type='topic'):
    try:
        if search_type == 'topic':
            # Search by topic/keyword
            news = newsapi.get_everything(
                q=query,
                language='en',
                sort_by='publishedAt',
                page_size=10
            )
        else:
            # Search by domain
            if not query.startswith(('http://', 'https://')):
                query = query.replace('www.', '')
                if not re.match(r'^[a-zA-Z0-9][a-zA-Z0-9-]{1,61}[a-zA-Z0-9]\.[a-zA-Z]{2,}', query):
                    st.warning(f"Invalid domain format: {query}. Please use format like 'bbc.com'")
                    return []
            
            news = newsapi.get_everything(
                domains=query,
                language='en',
                sort_by='publishedAt',
                page_size=10
            )
        
        if not news.get('articles'):
            st.info(f"No articles found for {search_type}: {query}")
            return []
            
        st.success(f"Found {len(news['articles'])} articles for {search_type}: {query}")
        return news['articles']
    except Exception as e:
        error_msg = str(e)
        st.error(f"Error fetching news: {error_msg}")
        
        if "401" in error_msg:
            st.error("API key error: Please make sure your NEWS_API_KEY is valid in the .env file")
        elif "426" in error_msg:
            st.error("You are using a developer API key which has limitations. Please upgrade your API key or reduce requests.")
        elif "429" in error_msg:
            st.error("Too many requests: You've reached the API request limit. Please try again later.")
            
        return []

# Title and description
st.title("📰 Fake News Detection System")
st.markdown("""
This application helps you detect whether a news article is likely to be real or fake.
You can analyze articles by URL, headline, or view top news from specific domains.
""")

# Load the model
try:
    model, vectorizer, scaler = load_model()
except Exception as e:
    st.error("Error loading the model. Please make sure the model files exist in the 'model' directory.")
    st.stop()

# Create tabs for different features
tab1, tab2, tab3 = st.tabs(["Analyze Article", "Search by Headline", "Top News"])

with tab1:
    st.subheader("Analyze Article by URL")
    url = st.text_input("Enter the article URL:")
    
    if st.button("Analyze Article"):
        if url:
            with st.spinner("Fetching and analyzing article..."):
                article_text = fetch_article_content(url)
                
                if article_text and article_text.startswith(("Error", "Could not", "Invalid", "HTTP Error")):
                    st.error(article_text)
                    st.info("Please try a different URL or check if the website is accessible.")
                elif article_text:
                    try:
                        prediction, probability = predict_news(article_text, model, vectorizer, scaler)
                        st.markdown("---")
                        st.subheader("Analysis Result")
                        if prediction == 1:
                            st.success("This news appears to be REAL")
                        else:
                            st.error("This news appears to be FAKE")
                        st.write(f"Confidence: {max(probability)*100:.2f}%")
                        
                        # Show feature analysis
                        st.subheader("Feature Analysis")
                        df = pd.DataFrame({
                            'statement': [article_text],
                            'processed_statement': [preprocess_text(article_text)]
                        })
                        df['barely_true'] = 0
                        df['false'] = 0
                        df['half_true'] = 0
                        df['mostly_true'] = 0
                        df['pants_on_fire'] = 0
                        df['speaker_credibility'] = 0.5
                        df = extract_features(df)
                        
                        # Display key features
                        st.write("Text Statistics:")
                        st.write(f"- Word Count: {df['word_count'].iloc[0]:.0f}")
                        st.write(f"- Sentence Count: {df['sentence_count'].iloc[0]:.0f}")
                        st.write(f"- Readability Score: {df['readability_score'].iloc[0]:.1f}")
                        
                        st.write("\nSentiment Analysis:")
                        st.write(f"- Sentiment Score: {df['sentiment_compound'].iloc[0]:.2f}")
                        st.write(f"- Subjectivity: {df['textblob_subjectivity'].iloc[0]:.2f}")
                        
                        st.write("\nEntity Analysis:")
                        st.write(f"- Persons Mentioned: {df['person_count'].iloc[0]:.0f}")
                        st.write(f"- Organizations: {df['org_count'].iloc[0]:.0f}")
                        st.write(f"- Locations: {df['location_count'].iloc[0]:.0f}")
                        
                        # Show article preview
                        st.subheader("Article Preview")
                        st.write(article_text[:1000] + "..." if len(article_text) > 1000 else article_text)
                    except Exception as e:
                        st.error(f"Error analyzing article: {str(e)}")
                        st.info("This could be due to unexpected content format or an issue with the model.")
                else:
                    st.error("Unable to fetch article content.")
                    st.info("The URL might be invalid or the content might be behind a paywall.")
        else:
            st.warning("Please enter an article URL.")

with tab2:
    st.subheader("Search and Analyze by Headline")
    st.info("Enter a headline to check if it's real or fake news. If no articles are found, it's likely fake news.")
    headline = st.text_input("Enter the news headline:")
    
    if st.button("Search and Analyze"):
        if headline:
            with st.spinner("Searching for articles..."):
                articles = search_news_by_headline(headline)
                if articles:
                    st.markdown("---")
                    st.subheader("Found Articles")
                    
                    for article in articles:
                        with st.expander(f"{article['title']}"):
                            st.write(f"Source: {article['source']['name']}")
                            st.write(f"Published: {article['publishedAt']}")
                            
                            if article['url']:
                                with st.spinner("Fetching full article content..."):
                                    article_text = fetch_article_content(article['url'])
                                    
                                    if article_text and article_text.startswith(("Error", "Could not", "Invalid", "HTTP Error")):
                                        st.error(article_text)
                                        st.info("Please try a different article or check if the website is accessible.")
                                    elif article_text:
                                        try:
                                            prediction, probability = predict_news(article_text, model, vectorizer, scaler)
                                            if prediction == 1:
                                                st.success("This news appears to be REAL")
                                            else:
                                                st.error("This news appears to be FAKE")
                                            st.write(f"Confidence: {max(probability)*100:.2f}%")
                                            
                                            # Show feature analysis
                                            st.subheader("Feature Analysis")
                                            df = pd.DataFrame({
                                                'statement': [article_text],
                                                'processed_statement': [preprocess_text(article_text)]
                                            })
                                            df['barely_true'] = 0
                                            df['false'] = 0
                                            df['half_true'] = 0
                                            df['mostly_true'] = 0
                                            df['pants_on_fire'] = 0
                                            df['speaker_credibility'] = 0.5
                                            df = extract_features(df)
                                            
                                            # Display key features
                                            st.write("Text Statistics:")
                                            st.write(f"- Word Count: {df['word_count'].iloc[0]:.0f}")
                                            st.write(f"- Sentence Count: {df['sentence_count'].iloc[0]:.0f}")
                                            st.write(f"- Readability Score: {df['readability_score'].iloc[0]:.1f}")
                                            
                                            st.write("\nSentiment Analysis:")
                                            st.write(f"- Sentiment Score: {df['sentiment_compound'].iloc[0]:.2f}")
                                            st.write(f"- Subjectivity: {df['textblob_subjectivity'].iloc[0]:.2f}")
                                            
                                            st.write("\nEntity Analysis:")
                                            st.write(f"- Persons Mentioned: {df['person_count'].iloc[0]:.0f}")
                                            st.write(f"- Organizations: {df['org_count'].iloc[0]:.0f}")
                                            st.write(f"- Locations: {df['location_count'].iloc[0]:.0f}")
                                            
                                            # Show article preview
                                            st.subheader("Article Preview")
                                            st.write(article_text[:1000] + "..." if len(article_text) > 1000 else article_text)
                                        except Exception as e:
                                            st.error(f"Error analyzing article: {str(e)}")
                                            st.info("This could be due to unexpected content format or an issue with the model.")
                                    else:
                                        st.error("Unable to fetch article content.")
                                        st.info("The content might be behind a paywall or the site might block scrapers.")
                                
                                st.write(f"[Read original article]({article['url']})")
        else:
            st.warning("Please enter a headline to search.")

with tab3:
    st.subheader("Analyze Top News")
    
    # Display API key status
    if api_key == "dummy_key":
        st.error("⚠️ NEWS_API_KEY is not configured. This feature will not work without a valid API key.")
        st.info("To use this feature:")
        st.code("1. Get a free API key from https://newsapi.org/\n2. Configure your secrets.toml file with the NEWS_API_KEY")
    
    # Add search type selection
    search_type = st.radio(
        "Search by:",
        ["Topic", "Domain"],
        horizontal=True
    )
    
    if search_type == "Topic":
        query = st.text_input("Enter a topic (e.g., 'artificial intelligence', 'climate change', 'sports'):")
    else:
        query = st.text_input("Enter news domain (e.g., cnn.com, bbc.co.uk):")
    
    if st.button("Fetch and Analyze"):
        if not query:
            st.warning(f"Please enter a {search_type.lower()} to analyze.")
        elif api_key == "dummy_key":
            st.error("Cannot fetch news without a valid API key. Please configure your API key first.")
        else:
            with st.spinner(f"Fetching latest articles for {search_type.lower()}: {query}..."):
                articles = get_top_news(query, search_type.lower())
                if articles:
                    st.markdown("---")
                    st.subheader("Top News Analysis")
                    
                    # Display articles in a more organized way
                    for article in articles:
                        with st.expander(f"{article['title']}", expanded=True):
                            # Create two columns for article info and analysis
                            info_col, analysis_col = st.columns([1, 2])
                            
                            with info_col:
                                st.write(f"**Source:** {article['source']['name']}")
                                st.write(f"**Published:** {article['publishedAt']}")
                                if article['url']:
                                    st.write(f"[Read original article]({article['url']})")
                            
                            with analysis_col:
                                if article['url']:
                                    with st.spinner("Analyzing article..."):
                                        article_text = fetch_article_content(article['url'])
                                        
                                        if article_text and article_text.startswith(("Error", "Could not", "Invalid", "HTTP Error")):
                                            st.error(article_text)
                                            st.info("Please try a different article or check if the website is accessible.")
                                        elif article_text:
                                            try:
                                                prediction, probability = predict_news(article_text, model, vectorizer, scaler)
                                                if prediction == 1:
                                                    st.success("This news appears to be REAL")
                                                else:
                                                    st.error("This news appears to be FAKE")
                                                st.write(f"Confidence: {max(probability)*100:.2f}%")
                                                
                                                # Show feature analysis
                                                st.subheader("Feature Analysis")
                                                df = pd.DataFrame({
                                                    'statement': [article_text],
                                                    'processed_statement': [preprocess_text(article_text)]
                                                })
                                                df['barely_true'] = 0
                                                df['false'] = 0
                                                df['half_true'] = 0
                                                df['mostly_true'] = 0
                                                df['pants_on_fire'] = 0
                                                df['speaker_credibility'] = 0.5
                                                df = extract_features(df)
                                                
                                                # Display key features
                                                st.write("Text Statistics:")
                                                st.write(f"- Word Count: {df['word_count'].iloc[0]:.0f}")
                                                st.write(f"- Sentence Count: {df['sentence_count'].iloc[0]:.0f}")
                                                st.write(f"- Readability Score: {df['readability_score'].iloc[0]:.1f}")
                                                
                                                st.write("\nSentiment Analysis:")
                                                st.write(f"- Sentiment Score: {df['sentiment_compound'].iloc[0]:.2f}")
                                                st.write(f"- Subjectivity: {df['textblob_subjectivity'].iloc[0]:.2f}")
                                                
                                                st.write("\nEntity Analysis:")
                                                st.write(f"- Persons Mentioned: {df['person_count'].iloc[0]:.0f}")
                                                st.write(f"- Organizations: {df['org_count'].iloc[0]:.0f}")
                                                st.write(f"- Locations: {df['location_count'].iloc[0]:.0f}")
                                                
                                                # Show article preview
                                                st.subheader("Article Preview")
                                                st.write(article_text[:1000] + "..." if len(article_text) > 1000 else article_text)
                                            except Exception as e:
                                                st.error(f"Error analyzing article: {str(e)}")
                                                st.info("This could be due to unexpected content format or an issue with the model.")
                                        else:
                                            st.error("Unable to fetch article content.")
                                            st.info("The content might be behind a paywall or the site might block scrapers.")
                else:
                    st.warning("No articles found. Try a different topic/domain or check your NewsAPI key.")

# Add footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with ❤️ using Streamlit and Machine Learning</p>
    <p>Note: This is an AI-powered tool and should be used as a guide only.</p>
</div>
""", unsafe_allow_html=True) 