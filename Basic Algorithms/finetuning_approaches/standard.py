"""
in the standard approach, Trainer API is used. 
All parameters of the pre-trained models are trained this includs:  
    Embedding layer (token, position and segment embedding)
    attention weights in transformer layers
    feedforward neural netwrok wieghts in each layer
    final classification/regression head 


"""
from huggingface_hub import login
login()
from datasets import load_dataset
from transformers import AutoTokenizer

dataset = load_dataset("yelp_review_full")
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-case")