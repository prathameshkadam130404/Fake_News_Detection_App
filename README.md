# Fake News Detection System

This is a machine learning-powered application that helps detect whether news articles are real or fake. The system uses various natural language processing techniques and machine learning models to analyze news content.

## Features

- Analyze articles by URL
- Search and verify news headlines
- View and analyze top news from specific domains
- Detailed feature analysis including sentiment, readability, and entity recognition

## Deployment Instructions

### Local Development

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download required NLTK data and spaCy model:
   ```bash
   python -m nltk.downloader punkt stopwords averaged_perceptron_tagger wordnet vader_lexicon maxent_ne_chunker words
   python -m spacy download en_core_web_sm
   ```
4. Create a `.env` file with your News API key:
   ```
   NEWS_API_KEY=your_news_api_key_here
   ```
5. Run the application:
   ```bash
   streamlit run app.py
   ```

### Deploying to Streamlit Cloud

1. Push your code to a GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io/)
3. Sign in with your GitHub account
4. Click "New app"
5. Select your repository, branch, and main file (app.py)
6. Add your News API key in the secrets section:
   ```
   NEWS_API_KEY=your_news_api_key_here
   ```
7. Click "Deploy"

## Project Structure

- `app.py`: Main Streamlit application
- `train_model.py`: Script for training the machine learning model
- `requirements.txt`: Project dependencies
- `model/`: Directory containing trained model files
- `.streamlit/secrets.toml`: Configuration for Streamlit deployment

## Requirements

- Python 3.8+
- News API key (get one from [newsapi.org](https://newsapi.org/))
- All dependencies listed in requirements.txt

## Note

This is an AI-powered tool and should be used as a guide only. Always verify information from multiple reliable sources.
