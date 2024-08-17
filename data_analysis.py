import pandas as pd
import pickle
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
import numpy as np


class Analyse:
    def __init__(self):  
        # Load datasets
        with open("/nas-ctm01/datasets/public/ROCO/files/test.pkl", 'rb') as f:
            dataset_test = pickle.load(f)
        with open("/nas-ctm01/datasets/public/ROCO/files/train.pkl", 'rb') as f:
            dataset_train = pickle.load(f)
        with open("/nas-ctm01/datasets/public/ROCO/files/validation.pkl", 'rb') as f:
            dataset_val = pickle.load(f)
        
        # Initialize DataFrames
        self.df_test = pd.DataFrame(dataset_test)
        self.df_train = pd.DataFrame(dataset_train)
        self.df_val = pd.DataFrame(dataset_val)
        
        # Concatenate DataFrames
        self.df = pd.concat([self.df_test, self.df_train, self.df_val], ignore_index=True)
        self.captions = self.df['captions']

        # Initialize the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('Sharathhebbar24/chat_gpt2_dpo')

    def tokenize(self, caption):
        token = self.tokenizer(caption, return_tensors='pt')
        output=token['input_ids'].squeeze().tolist()
        print(output)
        if len(output)>175:
            print(caption)
        return output  # Convert to list

        
    def plot(self, filename='/nas-ctm01/homes/mtamorim/CapMed/images/token_distribution.png'):
        lengths = [len(self.tokenize(caption)) for caption in self.captions]
        
        # Calculate statistics
        mean_length = np.mean(lengths)
        q1_length = np.percentile(lengths, 25)
        median_length = np.median(lengths)
        q3_length = np.percentile(lengths, 75)
        min_length = np.min(lengths)
        max_length = np.max(lengths)
        
        # Plot histogram
        plt.hist(lengths, bins=50)
        plt.xlabel('Token Length')
        plt.ylabel('Frequency')
        plt.title('Token Length Distribution')
        
        # Add statistics to the plot
        plt.axvline(mean_length, color='r', linestyle='dashed', linewidth=1, label=f'Mean: {mean_length:.2f}')
        plt.axvline(q1_length, color='g', linestyle='dashed', linewidth=1, label=f'Q1: {q1_length:.2f}')
        plt.axvline(median_length, color='b', linestyle='dashed', linewidth=1, label=f'Median: {median_length:.2f}')
        plt.axvline(q3_length, color='y', linestyle='dashed', linewidth=1, label=f'Q3: {q3_length:.2f}')
        plt.axvline(min_length, color='k', linestyle='dashed', linewidth=1, label=f'Min: {min_length}')
        plt.axvline(max_length, color='m', linestyle='dashed', linewidth=1, label=f'Max: {max_length}')
        
        plt.legend()
        plt.xlim(0,200)
        
        # Save the plot as an image file
        plt.savefig(filename)
        plt.close()  # Close the plot to free up memory


if __name__ == '__main__':
    # Usage
    analyzer = Analyse()
    analyzer.plot()
