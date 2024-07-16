import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from transformers import TFBartForConditionalGeneration, BartTokenizer

class TextSummarization:
    def __init__(self, model_name='facebook/bart-large-cnn'):
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.model = TFBartForConditionalGeneration.from_pretrained(model_name)

    def preprocess_text(self, text):
        # Tokenize text using pandas
        sentences = pd.Series(text.split('. ')).str.strip()
        cleaned_text = ". ".join(sentences)
        return cleaned_text

    def generate_summary(self, text, max_length=150, min_length=30, length_penalty=2.0, num_beams=4):
        inputs = self.tokenizer([text], max_length=1024, return_tensors='tf', truncation=True)
        summary_ids = self.model.generate(
            inputs['input_ids'],
            max_length=max_length,
            min_length=min_length,
            length_penalty=length_penalty,
            num_beams=num_beams,
            early_stopping=True
        )
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

def plot_summary_length_distribution(summaries):
    lengths = [len(summary.split()) for summary in summaries]
    plt.hist(lengths, bins=20, color='blue', edgecolor='black')
    plt.title('Summary Length Distribution')
    plt.xlabel('Length of Summaries')
    plt.ylabel('Frequency')
    plt.show()

def main():
    text = """Your long text document goes here. This is a sample text for summarization."""
    
    summarizer = TextSummarization()
    cleaned_text = summarizer.preprocess_text(text)
    summary = summarizer.generate_summary(cleaned_text)
    
    print("Original Text:\n", text)
    print("\nSummarized Text:\n", summary)

    # Example summaries for plotting
    example_summaries = [summary] * 10  # Replace with actual summaries if available
    plot_summary_length_distribution(example_summaries)

if __name__ == "__main__":
    main()
