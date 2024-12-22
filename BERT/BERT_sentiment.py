import torch
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification


## Model loading
tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

def sentiment_calculator(review): 
    tokens = tokenizer.encode(review,return_tensors="pt")
    result = model(tokens)
    print(int(torch.argmax(result.logits))+1)

## Prediction
reviews = ["The movie is great. I would recommend it to anyone.",\
           "The movie is just there. Could have been better.",\
           "I wasted my time watching that movie. It was so absolutely terrible"]


for review in reviews:
    print(review)
    print("the sentiment score is : ")
    sentiment_calculator(review)
    print("..."*15)