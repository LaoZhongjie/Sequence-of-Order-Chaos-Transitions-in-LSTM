"""
Training script for LSTM model with checkpointing for chaos analysis
Implements the training procedure described in the paper
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime

import config
from model import SentimentLSTM
from data_loader import IMDBDataLoader
from chaos_analysis import ChaosAnalyzer

class LSTMTrainer:
    """Trainer class for LSTM model with comprehensive logging and checkpointing"""
    
    def __init__(self):
        self.device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Create directories
        self.create_directories()
        
        # Initialize data loader
        self.data_loader = IMDBDataLoader()
        
        # Initialize model, optimizer, and loss function
        self.model = None
        self.optimizer = None
        self.criterion = nn.BCELoss()
        
        # Training history
        self.training_history = {
            'train_loss': [],
            'test_loss': [],
            'train_accuracy': [],
            'test_accuracy': [],
            'epochs': []
        }
        
    def create_directories(self):
        """Create necessary directories for results and checkpoints"""
        os.makedirs(config.RESULTS_PATH, exist_ok=True)
        os.makedirs(config.CHECKPOINT_PATH, exist_ok=True)
        
    def load_data(self):
        """Load and prepare data"""
        print("Loading IMDB dataset...")
        X_train, X_test, y_train, y_test = self.data_loader.load_data()
        
        self.train_loader, self.test_loader, self.test_dataset = \
            self.data_loader.create_data_loaders(X_train, X_test, y_train, y_test)
        
        print(f"Vocabulary size: {self.data_loader.vocab_size}")
        return self.data_loader.vocab_size
        
    def initialize_model(self, vocab_size):
        """Initialize model and optimizer"""
        self.model = SentimentLSTM(
            vocab_size=vocab_size,
            embedding_dim=config.EMBEDDING_DIM,
            hidden_size=config.LSTM_HIDDEN_SIZE,
            num_classes=config.NUM_CLASSES
        ).to(self.device)
        
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.LEARNING_RATE
        )
        
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters())}")
        
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output, _ = self.model(data)
            output = output.squeeze()  # Remove extra dimension
            
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            predicted = (output > 0.5).float()
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def evaluate(self, epoch, save_wrong_predictions=False):
        """Evaluate model on test set"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        wrong_predictions_info = []
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.test_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                output, _ = self.model(data)
                output = output.squeeze()
                
                loss = self.criterion(output, target)
                total_loss += loss.item()
                
                predicted = (output > 0.5).float()
                total += target.size(0)
                correct += (predicted == target).sum().item()
                
                # Collect wrong predictions for analysis (Section IV of paper)
                if save_wrong_predictions:
                    wrong_mask = (predicted != target)
                    if wrong_mask.sum() > 0:
                        wrong_indices = torch.where(wrong_mask)[0]
                        for idx in wrong_indices:
                            wrong_predictions_info.append({
                                'batch_idx': batch_idx,
                                'sample_idx': idx.item(),
                                'predicted': predicted[idx].item(),
                                'actual': target[idx].item(),
                                'output': output[idx].item(),
                                'loss': self.criterion(output[idx:idx+1], target[idx:idx+1]).item()
                            })
        
        avg_loss = total_loss / len(self.test_loader)
        accuracy = 100. * correct / total
        
        if save_wrong_predictions:
            return avg_loss, accuracy, wrong_predictions_info
        
        return avg_loss, accuracy
    
    def save_checkpoint(self, epoch, train_loss, test_loss, train_acc, test_acc):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': train_loss,
            'test_loss': test_loss,
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'vocab_size': self.data_loader.vocab_size,
        }
        
        checkpoint_path = os.path.join(config.CHECKPOINT_PATH, f'model_epoch_{epoch}.pt')
        torch.save(checkpoint, checkpoint_path)
        
        # Also save the best model (lowest test loss)
        if not hasattr(self, 'best_test_loss') or test_loss < self.best_test_loss:
            self.best_test_loss = test_loss
            self.best_epoch = epoch
            best_path = os.path.join(config.CHECKPOINT_PATH, 'best_model.pt')
            torch.save(checkpoint, best_path)
    
    def save_training_history(self):
        """Save training history to JSON file"""
        history_path = os.path.join(config.RESULTS_PATH, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
    
    def train(self, max_epochs=None):
        """Main training loop with comprehensive logging"""
        if max_epochs is None:
            max_epochs = config.MAX_EPOCHS
            
        print(f"Starting training for {max_epochs} epochs...")
        print("=" * 60)
        
        for epoch in range(1, max_epochs + 1):
            # Training
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Evaluation
            test_loss, test_acc = self.evaluate(epoch)
            
            # Save to history
            self.training_history['epochs'].append(epoch)
            self.training_history['train_loss'].append(train_loss)
            self.training_history['test_loss'].append(test_loss)
            self.training_history['train_accuracy'].append(train_acc)
            self.training_history['test_accuracy'].append(test_acc)
            
            # Save checkpoint every epoch (needed for chaos analysis)
            self.save_checkpoint(epoch, train_loss, test_loss, train_acc, test_acc)
            
            # Print epoch summary
            print(f'Epoch {epoch:4d}: '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                  f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
            
            # Save training history periodically
            if epoch % 10 == 0:
                self.save_training_history()
        
        print("=" * 60)
        print(f"Training completed! Best epoch: {self.best_epoch}, Best test loss: {self.best_test_loss:.4f}")
        
        # Final save
        self.save_training_history()
        
        return self.training_history

def main():
    """Main training function"""
    # Set random seeds for reproducibility
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    
    # Initialize trainer
    trainer = LSTMTrainer()
    
    # Load data and initialize model
    vocab_size = trainer.load_data()
    trainer.initialize_model(vocab_size)
    
    # Train model
    history = trainer.train()
    
    print(f"Training results saved to {config.RESULTS_PATH}")
    print(f"Model checkpoints saved to {config.CHECKPOINT_PATH}")

if __name__ == "__main__":
    main()