"""
Fixed version of chaos analysis script with better JSON serialization
"""

import os
import torch
import numpy as np
import json
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import find_peaks

import config
from model import SentimentLSTM
from data_loader import IMDBDataLoader
from chaos_analysis import ChaosAnalyzer

class MultipleDescentAnalyzer:
    """Main analyzer class for reproducing paper results - Fixed version"""
    
    def __init__(self):
        self.device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize components
        self.data_loader = IMDBDataLoader()
        self.model = None
        self.chaos_analyzer = None
        
        # Results storage
        self.results = {
            'epochs': [],
            'train_loss': [],
            'test_loss': [],
            'train_accuracy': [],
            'test_accuracy': [],
            'asymptotic_distances': [],
            'bifurcation_data': [],
            'transitions': [],
            'phases': []
        }
        
    def load_data_and_model(self):
        """Load data and initialize model"""
        print("Loading data and model...")
        
        # Load data
        X_train, X_test, y_train, y_test = self.data_loader.load_data()
        _, _, self.test_dataset = self.data_loader.create_data_loaders(X_train, X_test, y_train, y_test)
        
        # Initialize model
        self.model = SentimentLSTM(
            vocab_size=self.data_loader.vocab_size,
            embedding_dim=config.EMBEDDING_DIM,
            hidden_size=config.LSTM_HIDDEN_SIZE,
            num_classes=config.NUM_CLASSES
        ).to(self.device)
        
        # Initialize chaos analyzer
        self.chaos_analyzer = ChaosAnalyzer(self.model, self.device)
        
        print(f"Data loaded. Test dataset size: {len(self.test_dataset)}")
        
    def load_training_history(self):
        """Load training history from file"""
        history_path = os.path.join(config.RESULTS_PATH, 'training_history.json')
        
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                history = json.load(f)
            
            self.results['epochs'] = history['epochs']
            self.results['train_loss'] = history['train_loss']
            self.results['test_loss'] = history['test_loss']
            self.results['train_accuracy'] = history['train_accuracy']
            self.results['test_accuracy'] = history['test_accuracy']
            
            print(f"Loaded training history for {len(self.results['epochs'])} epochs")
        else:
            print("No training history found. Please run training first.")
            return False
        
        return True
    
    def analyze_chaos_dynamics(self, start_epoch=config.START_EPOCH, end_epoch=config.END_EPOCH, interval=1):
        """Analyze chaos dynamics throughout training"""
        
        if end_epoch is None:
            end_epoch = len(self.results['epochs'])
            
        print(f"Analyzing chaos dynamics from epoch {start_epoch} to {end_epoch}...")
        
        epochs_to_analyze = list(range(start_epoch, end_epoch + 1, interval))
        asymptotic_distances = []
        bifurcation_data = []
        
        for epoch in tqdm(epochs_to_analyze, desc="Chaos analysis"):
            try:
                # Load model checkpoint
                checkpoint_path = os.path.join(config.CHECKPOINT_PATH, f'model_epoch_{epoch}.pt')
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                
                # Calculate asymptotic distance
                asym_dist, reduced_sums = self.chaos_analyzer.calculate_asymptotic_distance(
                    self.test_dataset, num_samples=config.NUM_TEST_SAMPLES
                )
                
                asymptotic_distances.append(float(asym_dist))  # Convert to Python float
                bifurcation_data.append([float(x) for x in reduced_sums])  # Convert array to list of floats
                
            except FileNotFoundError:
                print(f"Checkpoint not found for epoch {epoch}")
                asymptotic_distances.append(None)
                bifurcation_data.append([])
        
        # Store results
        self.results['analyzed_epochs'] = epochs_to_analyze
        self.results['asymptotic_distances'] = asymptotic_distances
        self.results['bifurcation_data'] = bifurcation_data
        
        # Detect transitions
        valid_distances = [d for d in asymptotic_distances if d is not None]
        transitions, phases = self.chaos_analyzer.detect_order_chaos_transitions(valid_distances)
        
        # Convert transitions to serializable format
        serializable_transitions = []
        for trans in transitions:
            serializable_transitions.append({
                'epoch': int(trans['epoch']),
                'transition': str(trans['transition']),
                'distance_before': float(trans['distance_before']),
                'distance_after': float(trans['distance_after'])
            })
        
        self.results['transitions'] = serializable_transitions
        self.results['phases'] = [str(p) for p in phases]
        
        print(f"Found {len(transitions)} phase transitions")
        
        return epochs_to_analyze, asymptotic_distances, bifurcation_data
    
    def detect_multiple_descents(self, test_loss=None, min_prominence=0.05):
        """Detect multiple descent cycles in test loss"""
        
        if test_loss is None:
            test_loss = np.array(self.results['test_loss'])
        
        # Find peaks in test loss (local maxima before descents)
        peaks, properties = find_peaks(test_loss, prominence=min_prominence, distance=10)
        
        # Find valleys (local minima after descents)
        valleys, _ = find_peaks(-test_loss, prominence=min_prominence, distance=10)
        
        descent_cycles = []
        
        # Match peaks with subsequent valleys
        for peak_idx in peaks:
            # Find the next valley after this peak
            subsequent_valleys = valleys[valleys > peak_idx]
            if len(subsequent_valleys) > 0:
                valley_idx = subsequent_valleys[0]
                
                cycle_info = {
                    'peak_epoch': int(peak_idx + 1),
                    'valley_epoch': int(valley_idx + 1),
                    'peak_loss': float(test_loss[peak_idx]),
                    'valley_loss': float(test_loss[valley_idx]),
                    'descent_magnitude': float(test_loss[peak_idx] - test_loss[valley_idx]),
                    'cycle_length': int(valley_idx - peak_idx)
                }
                descent_cycles.append(cycle_info)
        
        print(f"Detected {len(descent_cycles)} descent cycles")
        
        return descent_cycles
    
    def find_optimal_epochs(self):
        """Find optimal epochs based on test loss and transitions"""
        
        test_loss = np.array(self.results['test_loss'])
        
        # Global optimum (lowest test loss overall)
        global_opt_idx = np.argmin(test_loss)
        global_opt_epoch = int(global_opt_idx + 1)
        
        # First order-to-chaos transition
        first_transition_epoch = None
        if self.results['transitions']:
            for transition in self.results['transitions']:
                if transition['transition'] == 'order -> chaos':
                    first_transition_epoch = int(transition['epoch'])
                    break
        
        optimal_epochs = {
            'global_optimum': {
                'epoch': global_opt_epoch,
                'test_loss': float(test_loss[global_opt_idx]),
                'test_accuracy': float(self.results['test_accuracy'][global_opt_idx])
            }
        }
        
        if first_transition_epoch is not None:
            optimal_epochs['first_order_chaos_transition'] = {
                'epoch': first_transition_epoch,
                'test_loss': float(test_loss[first_transition_epoch - 1]),
                'test_accuracy': float(self.results['test_accuracy'][first_transition_epoch - 1])
            }
        
        return optimal_epochs
    
    def save_results(self):
        """Save all analysis results with robust error handling"""
        
        # Save as pickle first (always works)
        pickle_path = os.path.join(config.RESULTS_PATH, 'chaos_analysis_results.pkl')
        with open(pickle_path, 'wb') as f:
            pickle.dump(self.results, f)
        print(f"Complete results saved as pickle to {pickle_path}")
        
        # Try to save as JSON
        try:
            results_path = os.path.join(config.RESULTS_PATH, 'chaos_analysis_results.json')
            
            # Create a safe copy for JSON serialization
            json_safe_results = {}
            for key, value in self.results.items():
                if key == 'bifurcation_data':
                    # Skip this for JSON as it's very large
                    json_safe_results[key] = f"Large array of size {len(value)} - see pickle file"
                else:
                    json_safe_results[key] = value
            
            with open(results_path, 'w') as f:
                json.dump(json_safe_results, f, indent=2)
            print(f"JSON results saved to {results_path}")
            
        except Exception as e:
            print(f"Could not save JSON (not critical): {e}")
            
            # Save simplified version
            simplified_results = {
                'summary': {
                    'total_epochs': len(self.results['epochs']),
                    'analyzed_epochs': len(self.results.get('analyzed_epochs', [])),
                    'num_transitions': len(self.results['transitions']),
                    'num_descent_cycles': len(self.results.get('descent_cycles', [])),
                    'global_optimum_epoch': int(np.argmin(self.results['test_loss']) + 1),
                    'min_test_loss': float(min(self.results['test_loss']))
                },
                'training_curves': {
                    'epochs': self.results['epochs'],
                    'test_loss': self.results['test_loss'],
                    'test_accuracy': self.results['test_accuracy']
                }
            }
            
            simple_path = os.path.join(config.RESULTS_PATH, 'analysis_summary.json')
            with open(simple_path, 'w') as f:
                json.dump(simplified_results, f, indent=2)
            print(f"Simplified summary saved to {simple_path}")

def main():
    """Main analysis function"""
    
    # Initialize analyzer
    analyzer = MultipleDescentAnalyzer()
    
    # Load data and model
    analyzer.load_data_and_model()
    
    # Load training history
    if not analyzer.load_training_history():
        print("Please run training first using train.py")
        return
    
    # Analyze chaos dynamics
    print("Starting chaos analysis...")
    analyzer.analyze_chaos_dynamics(start_epoch=1, end_epoch=20, interval=1)
    
    # Detect multiple descents
    descent_cycles = analyzer.detect_multiple_descents()
    analyzer.results['descent_cycles'] = descent_cycles
    
    # Find optimal epochs
    optimal_epochs = analyzer.find_optimal_epochs()
    analyzer.results['optimal_epochs'] = optimal_epochs
    
    print("\n" + "="*60)
    print("ANALYSIS RESULTS")
    print("="*60)
    
    print(f"Detected {len(descent_cycles)} descent cycles")
    if optimal_epochs.get('first_order_chaos_transition'):
        print(f"First order-chaos transition at epoch {optimal_epochs['first_order_chaos_transition']['epoch']}")
        print(f"Global optimum at epoch {optimal_epochs['global_optimum']['epoch']}")
        
        # Check if they match
        transition_epoch = optimal_epochs['first_order_chaos_transition']['epoch']
        optimal_epoch = optimal_epochs['global_optimum']['epoch']
        if abs(transition_epoch - optimal_epoch) <= 5:
            print("✓ CONFIRMED: Global optimum occurs near first order-chaos transition!")
        else:
            print(f"Global optimum ({optimal_epoch}) differs from first transition ({transition_epoch})")
            print("Note: This might be due to limited epochs analyzed. Try more epochs for full pattern.")
    
    # Save results
    analyzer.save_results()
    
    print(f"\nAll results saved to {config.RESULTS_PATH}")

if __name__ == "__main__":
    main()