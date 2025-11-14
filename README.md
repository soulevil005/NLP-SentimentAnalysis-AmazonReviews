# SentimentAnalysis-AmazonReviews

## Project Description

This project implements a **Sentiment Analysis model for Amazon Product Reviews** using a **Convolutional Neural Network (CNN)** built with PyTorch. The goal is to classify customer reviews as **positive** or **negative** based on their textual content.

The model is trained on the **Amazon Reviews Dataset** by *Julian McAuley & Bo Pang*, available publicly on Kaggle. The dataset contains millions of real Amazon product reviews in a lightweight FastText-style format:


### ✨ Key Features

- **Custom CNN-based Text Classifier** (word-level embedding + stacked conv layers)
- **Efficient Data Loading** for large text datasets (FastText format)
- **Tokenization using NLTK**
- **Vocabulary building** with configurable vocabulary size
- **TorchScript export** → enables fast loading and deployment without code dependencies
- **Clean training pipeline** with metrics (Accuracy, F1-score, AUC)
- **Inference script** for running predictions on new reviews
- **GPU acceleration support**

### 📦 Outputs Produced

- `models/amazon_reviews_sentiment.pt` → TorchScript-format trained model  
- `amazon_vocab.json` → vocabulary mapping used during training  
- Training logs, evaluation metrics, and test predictions  

This project is designed to be simple, easy to extend, and production-friendly. You can replace the CNN with an LSTM, Transformer, or BERT with minimal changes.
