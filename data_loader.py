"""
Data loading and preprocessing for IMDB Movie Review Dataset
Handles downloading, tokenization, and batch creation
"""

import os
import re
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from collections import Counter
import requests
import zipfile
from tqdm import tqdm
import config

class IMDBDataset(Dataset):
    """Custom Dataset class for IMDB movie reviews"""
    
    def __init__(self, texts, labels, vocab, sequence_length=500):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Convert text to indices
        indices = self.text_to_indices(text)
        
        # Pad or truncate to fixed length
        if len(indices) > self.sequence_length:
            indices = indices[:self.sequence_length]
        else:
            indices = indices + [0] * (self.sequence_length - len(indices))
            
        return torch.tensor(indices, dtype=torch.long), torch.tensor(label, dtype=torch.float)
    
    def text_to_indices(self, text):
        """Convert text to sequence of vocabulary indices"""
        # Simple tokenization
        words = text.lower().split()
        # Clean words (remove punctuation, keep only alphanumeric)
        words = [re.sub(r'[^a-zA-Z0-9]', '', word) for word in words]
        words = [word for word in words if word]  # Remove empty strings
        
        indices = []
        for word in words:
            if word in self.vocab:
                indices.append(self.vocab[word])
            else:
                indices.append(self.vocab['<UNK>'])  # Unknown token
        
        return indices

class IMDBDataLoader:
    """Data loader for IMDB dataset with preprocessing"""
    
    def __init__(self):
        self.vocab = None
        self.vocab_size = 0
        
    def download_data(self):
        """Download IMDB dataset if not already present"""
        if not os.path.exists(config.DATA_PATH):
            os.makedirs(config.DATA_PATH)
            
        # Check if data already exists
        if os.path.exists(os.path.join(config.DATA_PATH, 'IMDB_Dataset.csv')):
            print("IMDB dataset already exists!")
            return
            
        print("Please download the IMDB dataset manually from:")
        print("https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")
        print("Or use the Hugging Face datasets library:")
        print("from datasets import load_dataset")
        print("dataset = load_dataset('imdb')")
        
        # Alternative: Load from Hugging Face datasets
        try:
            from datasets import load_dataset
            dataset = load_dataset('imdb')
            
            # Convert to pandas DataFrame
            train_df = pd.DataFrame({
                'review': dataset['train']['text'],
                'sentiment': ['positive' if label == 1 else 'negative' for label in dataset['train']['label']]
            })
            
            test_df = pd.DataFrame({
                'review': dataset['test']['text'],
                'sentiment': ['positive' if label == 1 else 'negative' for label in dataset['test']['label']]
            })
            
            # Combine and save
            full_df = pd.concat([train_df, test_df], ignore_index=True)
            full_df.to_csv(os.path.join(config.DATA_PATH, 'IMDB_Dataset.csv'), index=False)
            print("Dataset downloaded and saved successfully!")
            
        except ImportError:
            print("Please install the datasets library: pip install datasets")
            raise
    
    def build_vocabulary(self, texts, max_vocab_size=10000):
        """Build vocabulary from text data"""
        word_counts = Counter()
        
        for text in tqdm(texts, desc="Building vocabulary"):
            words = text.lower().split()
            words = [re.sub(r'[^a-zA-Z0-9]', '', word) for word in words]
            words = [word for word in words if word]
            word_counts.update(words)
        
        # Get most common words
        most_common = word_counts.most_common(max_vocab_size - 2)  # Reserve 2 for special tokens
        
        # Create vocabulary mapping
        self.vocab = {'<PAD>': 0, '<UNK>': 1}
        for word, _ in most_common:
            self.vocab[word] = len(self.vocab)
        
        self.vocab_size = len(self.vocab)
        print(f"Vocabulary size: {self.vocab_size}")
        
    def load_data(self):
        """Load and preprocess IMDB dataset"""
        # Download data if needed
        self.download_data()
        
        # Load data
        df = pd.read_csv(os.path.join(config.DATA_PATH, 'IMDB_Dataset.csv'))
        
        # Convert sentiment to binary labels
        df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})
        
        # Build vocabulary
        self.build_vocabulary(df['review'].values)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            df['review'].values,
            df['label'].values,
            test_size=config.TEST_RATIO,
            random_state=config.RANDOM_SEED,
            stratify=df['label'].values
        )
        
        print(f"Training samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def create_data_loaders(self, X_train, X_test, y_train, y_test):
        """Create PyTorch data loaders"""
        
        train_dataset = IMDBDataset(X_train, y_train, self.vocab, config.SEQUENCE_LENGTH)
        test_dataset = IMDBDataset(X_test, y_test, self.vocab, config.SEQUENCE_LENGTH)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=4,
            pin_memory=True if config.DEVICE == 'cuda' else False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True if config.DEVICE == 'cuda' else False
        )
        
        return train_loader, test_loader, test_dataset