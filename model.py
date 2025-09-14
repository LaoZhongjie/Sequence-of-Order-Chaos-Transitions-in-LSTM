"""
LSTM model definition for sentiment analysis
Implements the model architecture described in the paper
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import config

class SentimentLSTM(nn.Module):
    """
    LSTM model for sentiment analysis as described in the paper
    Architecture: Embedding -> LSTM -> Fully Connected -> Sigmoid
    """
    
    def __init__(self, vocab_size, embedding_dim=32, hidden_size=60, num_classes=1):
        super(SentimentLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        
        # Embedding layer - projects tokens into continuous vector space
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # LSTM layer for time-series analysis
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_size,
            batch_first=True,
            bidirectional=False
        )
        
        # Fully connected output layer for classification
        self.fc = nn.Linear(hidden_size, num_classes)
        
        # Sigmoid activation for binary classification
        self.sigmoid = nn.Sigmoid()
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights"""
        # Initialize embedding weights
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        self.embedding.weight.data[0].fill_(0)  # Padding token
        
        # Initialize LSTM weights
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
        
        # Initialize fully connected layer
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
    
    def forward(self, x, hidden=None):
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length)
            hidden: Hidden state tuple (h_0, c_0) for LSTM
            
        Returns:
            output: Sigmoid output for binary classification
            hidden: Final hidden state tuple
        """
        batch_size = x.size(0)
        
        # Embedding layer
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        
        # LSTM layer
        if hidden is None:
            hidden = self.init_hidden(batch_size)
        
        lstm_out, hidden = self.lstm(embedded, hidden)
        
        # Use the last output for classification
        # lstm_out shape: (batch_size, seq_len, hidden_size)
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_size)
        
        # Fully connected layer
        output = self.fc(last_output)  # (batch_size, num_classes)
        output = self.sigmoid(output)
        
        return output, hidden
    
    def init_hidden(self, batch_size):
        """
        Initialize hidden states for LSTM
        
        Args:
            batch_size: Batch size for initialization
            
        Returns:
            tuple: (h_0, c_0) hidden state tensors
        """
        device = next(self.parameters()).device
        h_0 = torch.zeros(1, batch_size, self.hidden_size, device=device)
        c_0 = torch.zeros(1, batch_size, self.hidden_size, device=device)
        return (h_0, c_0)
    
    def get_lstm_hidden_output(self, x, hidden=None):
        """
        Get LSTM hidden output for asymptotic analysis
        This method is used for chaos analysis in the paper
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length)
            hidden: Initial hidden state
            
        Returns:
            h_t: Hidden state at each timestep (batch_size, seq_len, hidden_size)
            final_hidden: Final hidden state tuple
        """
        batch_size = x.size(0)
        
        # Embedding
        embedded = self.embedding(x)
        
        # Initialize hidden state if not provided
        if hidden is None:
            hidden = self.init_hidden(batch_size)
        
        # LSTM forward pass
        h_t, final_hidden = self.lstm(embedded, hidden)
        
        return h_t, final_hidden
    
    def continue_lstm_iteration(self, initial_hidden, timesteps, input_dim=32):
        """
        Continue LSTM iteration with zero inputs for asymptotic analysis
        This implements the methodology described in Figure 1 of the paper
        
        Args:
            initial_hidden: Initial hidden state tuple (h_0, c_0)
            timesteps: Number of timesteps to iterate
            input_dim: Dimension of input (embedding_dim)
            
        Returns:
            final_hidden: Final hidden state after iteration
        """
        device = next(self.parameters()).device
        batch_size = initial_hidden[0].size(1)
        
        # Create zero input for continued iteration
        zero_input = torch.zeros(batch_size, timesteps, input_dim, device=device)
        
        # Continue LSTM iteration
        _, final_hidden = self.lstm(zero_input, initial_hidden)
        
        return final_hidden