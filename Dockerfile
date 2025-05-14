# Use Python 3.9 as base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data - added punkt_tab
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger'); nltk.download('wordnet'); nltk.download('vader_lexicon'); nltk.download('maxent_ne_chunker'); nltk.download('words'); nltk.download('punkt_tab')"

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Copy the application code
COPY . .

# Create model directory
RUN mkdir -p model

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Command to run the application
CMD ["python", "train_model.py"] 