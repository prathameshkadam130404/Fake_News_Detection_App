# Fake News Detection System

A machine learning-based fake news detection system with a Streamlit web interface.

## Features

- Train a fake news detection model using XGBoost
- Analyze articles by URL
- Search and analyze news by headline
- View and analyze top news from specific sources or topics
- Docker support for both training and serving

## Setup

### Prerequisites

- Docker and Docker Compose installed
- NewsAPI key (get one from [NewsAPI.org](https://newsapi.org/))

### Installation

1. Clone the repository
2. Create a `.env` file from the template:
   ```
   cp .env.template .env
   ```
3. Add your NewsAPI key to the `.env` file:
   ```
   NEWS_API_KEY=your_api_key_here
   ```

## Running with Docker

### First-time setup (train the model and run the app)

```bash
docker-compose up
```

This will:
1. Build both the training and app containers
2. Train the model (saving to the ./model directory)
3. Start the Streamlit app on port 8501

### Running just the app (after model is trained)

```bash
docker-compose up streamlit-app
```

### Retraining the model

```bash
docker-compose up model-training
```

## Accessing the Application

Once running, access the web interface at:
http://localhost:8501

## Dataset

This application uses the LIAR dataset for training. Make sure to place the dataset files in the `Datasets` directory:
- `train.tsv`: Training data
- `test.tsv`: Test data
- `valid.tsv`: Validation data

## License

[MIT License](LICENSE)
