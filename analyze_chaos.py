import os
import torch
import numpy as np
import json
import pickle
from tqdm import tqdm
from scipy.signal import find_peaks

import config
from model import SentimentLSTM
from data_loader import IMDBDataLoader
from chaos_analysis import ChaosAnalyzer

class MultipleDescentAnalyzer:
    """Fixed analyzer following paper methodology exactly"""
    
    def __init__(self):
        self.device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.data_loader = IMDBDataLoader()
        self.model = None
        self.chaos_analyzer = None
        
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
        
        X_train, X_test, y_train, y_test = self.data_loader.load_data()
        _, _, self.test_dataset = self.data_loader.create_data_loaders(X_train, X_test, y_train, y_test)
        
        self.model = SentimentLSTM(
            vocab_size=self.data_loader.vocab_size,
            embedding_dim=config.EMBEDDING_DIM,
            hidden_size=config.LSTM_HIDDEN_SIZE,
            num_classes=config.NUM_CLASSES
        ).to(self.device)
        
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
            print("No training history found.")
            return False
        
        return True
    
    def analyze_chaos_dynamics(self, start_epoch=1, end_epoch=None, interval=1):
        """Analyze chaos dynamics"""
        
        if end_epoch is None:
            end_epoch = len(self.results['epochs'])
            
        print(f"Analyzing chaos dynamics from epoch {start_epoch} to {end_epoch}...")
        
        epochs_to_analyze = list(range(start_epoch, min(end_epoch + 1, len(self.results['epochs']) + 1), interval))
        asymptotic_distances = []
        bifurcation_data = []
        
        for epoch in tqdm(epochs_to_analyze, desc="Chaos analysis"):
            try:
                checkpoint_path = os.path.join(config.CHECKPOINT_PATH, f'model_epoch_{epoch}.pt')
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                
                # Calculate asymptotic distance with fixed method
                asym_dist, reduced_sums = self.chaos_analyzer.calculate_asymptotic_distance(
                    self.test_dataset, num_samples=config.NUM_TEST_SAMPLES
                )
                
                asymptotic_distances.append(float(asym_dist))
                bifurcation_data.append([float(x) for x in reduced_sums])
                
            except FileNotFoundError:
                print(f"Checkpoint not found for epoch {epoch}")
                asymptotic_distances.append(None)
                bifurcation_data.append([])
        
        self.results['analyzed_epochs'] = epochs_to_analyze
        self.results['asymptotic_distances'] = asymptotic_distances
        self.results['bifurcation_data'] = bifurcation_data
        
        # Detect transitions with corrected method
        valid_distances = [d for d in asymptotic_distances if d is not None]
        transitions, phases = self.chaos_analyzer.detect_order_chaos_transitions(valid_distances, threshold=-14.5)
        
        # Convert to serializable format
        serializable_transitions = []
        for trans in transitions:
            serializable_transitions.append({
                'epoch': int(trans['epoch']),
                'transition': str(trans['transition']),
                'distance_before': float(trans['distance_before']) if trans['distance_before'] is not None else None,
                'distance_after': float(trans['distance_after'])
            })
        
        self.results['transitions'] = serializable_transitions
        self.results['phases'] = [str(p) for p in phases]
        
        print(f"Found {len(transitions)} phase transitions")
        
        return epochs_to_analyze, asymptotic_distances, bifurcation_data
    
    def detect_multiple_descents(self, test_loss=None, min_prominence=0.5, min_distance=10):
        """
        Detect multiple descent cycles with stricter criteria
        
        From paper: these are significant drops in test loss during overfitting phase
        Need to be more selective to match paper's ~8 cycles in 1000 epochs
        """
        
        if test_loss is None:
            test_loss = np.array(self.results['test_loss'])
        
        # Focus on overfitting phase (after epoch 100 or when loss starts increasing)
        # Find when overfitting starts
        min_loss_idx = np.argmin(test_loss)
        overfitting_start = max(min_loss_idx + 10, len(test_loss) // 4)  # Start after minimum + buffer
        
        if overfitting_start >= len(test_loss) - 10:
            print("Not enough epochs in overfitting phase for descent detection")
            return []
        
        # Focus on overfitting region
        overfitting_loss = test_loss[overfitting_start:]
        overfitting_epochs = list(range(overfitting_start + 1, len(test_loss) + 1))
        
        # Find peaks with stricter criteria
        peaks, properties = find_peaks(
            overfitting_loss, 
            prominence=min_prominence,  # Increased prominence
            distance=min_distance,      # Minimum distance between peaks
            height=np.mean(overfitting_loss)  # Only consider peaks above average
        )
        
        # Find valleys
        valleys, _ = find_peaks(
            -overfitting_loss, 
            prominence=min_prominence * 0.7,  # Slightly lower prominence for valleys
            distance=min_distance // 2
        )
        
        descent_cycles = []
        
        # Match peaks with subsequent valleys
        for peak_idx in peaks:
            subsequent_valleys = valleys[valleys > peak_idx]
            if len(subsequent_valleys) > 0:
                valley_idx = subsequent_valleys[0]
                
                # Convert back to original epoch numbering
                peak_epoch = overfitting_epochs[peak_idx]
                valley_epoch = overfitting_epochs[valley_idx]
                peak_loss = overfitting_loss[peak_idx]
                valley_loss = overfitting_loss[valley_idx]
                
                # Additional criteria: significant descent
                descent_magnitude = peak_loss - valley_loss
                if descent_magnitude > min_prominence:  # Must be significant drop
                    cycle_info = {
                        'peak_epoch': int(peak_epoch),
                        'valley_epoch': int(valley_epoch),
                        'peak_loss': float(peak_loss),
                        'valley_loss': float(valley_loss),
                        'descent_magnitude': float(descent_magnitude),
                        'cycle_length': int(valley_idx - peak_idx)
                    }
                    descent_cycles.append(cycle_info)
        
        print(f"Detected {len(descent_cycles)} significant descent cycles in overfitting phase")
        
        return descent_cycles
    
    def find_optimal_epochs(self):
        """Find optimal epochs based on paper methodology"""
        
        test_loss = np.array(self.results['test_loss'])
        
        # Global optimum
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
        
        if first_transition_epoch is not None and first_transition_epoch <= len(test_loss):
            optimal_epochs['first_order_chaos_transition'] = {
                'epoch': first_transition_epoch,
                'test_loss': float(test_loss[first_transition_epoch - 1]),
                'test_accuracy': float(self.results['test_accuracy'][first_transition_epoch - 1])
            }
        
        return optimal_epochs
    
    def save_results(self):
        """Save results with proper error handling"""
        
        # Save as pickle (complete data)
        pickle_path = os.path.join(config.RESULTS_PATH, 'chaos_analysis_results.pkl')
        with open(pickle_path, 'wb') as f:
            pickle.dump(self.results, f)
        print(f"Complete results saved to {pickle_path}")
        
        # Save summary as JSON
        summary = {
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
            },
            'transitions': self.results['transitions'],
            'descent_cycles': self.results.get('descent_cycles', [])
        }
        
        summary_path = os.path.join(config.RESULTS_PATH, 'analysis_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved to {summary_path}")

def main():
    """Main analysis function"""
    
    analyzer = MultipleDescentAnalyzer()
    
    analyzer.load_data_and_model()
    
    if not analyzer.load_training_history():
        print("Please run training first.")
        return
    
    # Analyze chaos dynamics
    print("Starting chaos analysis...")
    analyzer.analyze_chaos_dynamics(start_epoch=1, end_epoch=1000, interval=1)
    
    # Detect descent cycles with corrected method
    descent_cycles = analyzer.detect_multiple_descents(min_prominence=0.3, min_distance=5)
    analyzer.results['descent_cycles'] = descent_cycles
    
    # Find optimal epochs
    optimal_epochs = analyzer.find_optimal_epochs()
    analyzer.results['optimal_epochs'] = optimal_epochs
    
    print("\n" + "="*60)
    print("ANALYSIS RESULTS")
    print("="*60)
    
    print(f"Detected {len(descent_cycles)} significant descent cycles")
    
    if optimal_epochs.get('first_order_chaos_transition'):
        trans_epoch = optimal_epochs['first_order_chaos_transition']['epoch']
        opt_epoch = optimal_epochs['global_optimum']['epoch']
        print(f"First order-chaos transition: Epoch {trans_epoch}")
        print(f"Global optimum: Epoch {opt_epoch}")
        
        if abs(trans_epoch - opt_epoch) <= 3:
            print("✓ CONFIRMED: Global optimum occurs near first order-chaos transition!")
        else:
            print(f"Global optimum ({opt_epoch}) differs from first transition ({trans_epoch})")
    else:
        print("No order-chaos transitions detected in analyzed epochs")
        print("Note: May need more epochs or different threshold")
    
    analyzer.save_results()
    print(f"\nResults saved to {config.RESULTS_PATH}")

if __name__ == "__main__":
    main()