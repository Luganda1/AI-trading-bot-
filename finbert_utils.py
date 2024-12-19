from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import Tuple 
device = "cuda:0" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(device)
labels = ["positive", "negative", "neutral"]

def estimate_sentiment(news):
    """Performs sentiment analysis on financial news text using a pre-trained machine learning model. Determines the sentiment probability and classification for the given input.

Processes the input news text through a sentiment analysis model to extract its emotional tone and confidence level. Returns a neutral sentiment with zero probability if no text is provided.

Args:
    news (str): The financial news text to be analyzed.

Returns:
    tuple: A tuple containing (sentiment_probability, sentiment_label).
        - sentiment_probability (float): Confidence score of the sentiment prediction.
        - sentiment_label (str): Predicted sentiment category (e.g., positive, negative, neutral).
""""
    
    if news:
        tokens = tokenizer(news, return_tensors="pt", padding=True).to(device)

        result = model(tokens["input_ids"], attention_mask=tokens["attention_mask"])[
            "logits"
        ]
        result = torch.nn.functional.softmax(torch.sum(result, 0), dim=-1)
        probability = result[torch.argmax(result)]
        sentiment = labels[torch.argmax(result)]
        return probability, sentiment
    else:
        return 0, labels[-1]


if __name__ == "__main__":
    tensor, sentiment = estimate_sentiment(['markets responded negatively to the news!','traders were displeased!'])
    print(tensor, sentiment)
    print(torch.cuda.is_available())