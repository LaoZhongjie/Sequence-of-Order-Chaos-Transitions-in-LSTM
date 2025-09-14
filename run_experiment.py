"""
Main experiment runner - orchestrates the entire reproduction pipeline
Run this script to reproduce the paper's results from start to finish
"""

import os
import sys
import argparse
import time
from datetime import datetime
import torch

import config
from train import LSTMTrainer
from analyze_chaos import MultipleDescentAnalyzer
from visualize_results import ResultsVisualizer

class ExperimentRunner:
    """Main class to orchestrate the entire experiment pipeline"""
    
    def __init__(self):
        self.start_time = time.time()
        self.setup_experiment()
        
    def setup_experiment(self):
        """Setup experiment environment"""
        print("="*80)
        print("REPRODUCING: Multiple Descents in Deep Learning as a Sequence of Order-Chaos Transitions")
        print("="*80)
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Device: {torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')}")
        print(f"Random seed: {config.RANDOM_SEED}")
        print()
        
        # Create all necessary directories
        for path in [config.DATA_PATH, config.RESULTS_PATH, config.CHECKPOINT_PATH]:
            os.makedirs(path, exist_ok=True)
            
    def run_training(self, max_epochs=None):
        """Run LSTM training with checkpointing"""
        
        print("STEP 1: TRAINING LSTM MODEL")
        print("-" * 40)
        
        if max_epochs is None:
            max_epochs = config.MAX_EPOCHS
            
        # Check if training already completed
        history_path = os.path.join(config.RESULTS_PATH, 'training_history.json')
        if os.path.exists(history_path):
            response = input(f"Training history found. Skip training? (y/n): ").lower()
            if response == 'y':
                print("Skipping training step.")
                return True
        
        try:
            # Initialize trainer
            trainer = LSTMTrainer()
            
            # Load data and train
            vocab_size = trainer.load_data()
            trainer.initialize_model(vocab_size)
            
            print(f"Starting training for {max_epochs} epochs...")
            print("This will take several hours depending on your hardware.")
            print()
            
            # Train model
            history = trainer.train(max_epochs)
            
            print(f"✓ Training completed successfully!")
            print(f"✓ Best epoch: {trainer.best_epoch}")
            print(f"✓ Best test loss: {trainer.best_test_loss:.4f}")
            print()
            
            return True
            
        except Exception as e:
            print(f"✗ Training failed: {str(e)}")
            return False
    
    def run_chaos_analysis(self, max_analysis_epochs=config.END_EPOCH):
        """Run chaos analysis on trained model"""
        
        print("STEP 2: CHAOS ANALYSIS")
        print("-" * 40)
        
        # Check if analysis already completed
        analysis_path = os.path.join(config.RESULTS_PATH, 'chaos_analysis_results.json')
        if os.path.exists(analysis_path):
            response = input(f"Chaos analysis results found. Skip analysis? (y/n): ").lower()
            if response == 'y':
                print("Skipping chaos analysis step.")
                return True
        
        try:
            # Initialize analyzer
            analyzer = MultipleDescentAnalyzer()
            
            # Load data and model
            analyzer.load_data_and_model()
            
            # Load training history
            if not analyzer.load_training_history():
                print("✗ No training history found. Please run training first.")
                return False
            
            print(f"Starting chaos analysis for first {max_analysis_epochs} epochs...")
            print("This is computationally intensive and may take several hours.")
            print("Consider running on a smaller epoch range first for testing.")
            print()
            
            # Run analysis
            epochs, distances, bifurcation_data = analyzer.analyze_chaos_dynamics(
                start_epoch=config.START_EPOCH, 
                end_epoch=max_analysis_epochs, 
                interval=1
            )
            
            # Detect multiple descents
            descent_cycles = analyzer.detect_multiple_descents()
            analyzer.results['descent_cycles'] = descent_cycles
            
            # Find optimal epochs
            optimal_epochs = analyzer.find_optimal_epochs()
            analyzer.results['optimal_epochs'] = optimal_epochs
            
            # Save results
            analyzer.save_results()
            
            # Print summary
            print("✓ Chaos analysis completed successfully!")
            print(f"✓ Analyzed {len(epochs)} epochs")
            print(f"✓ Found {len(descent_cycles)} descent cycles")
            
            if optimal_epochs.get('first_order_chaos_transition'):
                trans_epoch = optimal_epochs['first_order_chaos_transition']['epoch']
                opt_epoch = optimal_epochs['global_optimum']['epoch']
                print(f"✓ First order-chaos transition: Epoch {trans_epoch}")
                print(f"✓ Global optimum: Epoch {opt_epoch}")
                
                if abs(trans_epoch - opt_epoch) <= 5:
                    print("🎉 KEY FINDING CONFIRMED: Global optimum occurs near first order-chaos transition!")
                else:
                    print("⚠️  Global optimum differs from first transition - may need longer analysis")
            
            print()
            return True
            
        except Exception as e:
            print(f"✗ Chaos analysis failed: {str(e)}")
            return False
    
    def run_visualization(self):
        """Generate all visualization plots"""
        
        print("STEP 3: GENERATING VISUALIZATIONS")
        print("-" * 40)
        
        try:
            visualizer = ResultsVisualizer()
            visualizer.generate_all_plots()
            
            print("✓ All visualizations generated successfully!")
            print(f"✓ Plots saved to {os.path.join(config.RESULTS_PATH, 'figures')}")
            print()
            
            return True
            
        except Exception as e:
            print(f"✗ Visualization failed: {str(e)}")
            return False
    
    def print_summary(self):
        """Print experiment summary"""
        
        total_time = time.time() - self.start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        
        print("="*80)
        print("EXPERIMENT SUMMARY")
        print("="*80)
        print(f"Total runtime: {hours}h {minutes}m")
        print(f"Results directory: {config.RESULTS_PATH}")
        print(f"Checkpoints directory: {config.CHECKPOINT_PATH}")
        print()
        
        # Check what was completed
        files_to_check = [
            ('training_history.json', 'Training completed'),
            ('chaos_analysis_results.json', 'Chaos analysis completed'),
            ('figures/multiple_descents_overview.png', 'Visualizations generated')
        ]
        
        for filename, description in files_to_check:
            filepath = os.path.join(config.RESULTS_PATH, filename)
            status = "✓" if os.path.exists(filepath) else "✗"
            print(f"{status} {description}")
        
        print()
        print("To reproduce specific components:")
        print("  python train.py              # Training only")
        print("  python analyze_chaos.py      # Analysis only")  
        print("  python visualize_results.py  # Visualization only")
        print()
        
        print("Paper findings to verify:")
        print("1. Multiple descent cycles in overfitting regime")
        print("2. Global optimum occurs at first order-chaos transition")
        print("3. Each descent corresponds to order-chaos transition")
        print("4. LSTM dynamics resemble tanh map bifurcation")
        print("="*80)

def main():
    """Main experiment function with command line arguments"""
    
    parser = argparse.ArgumentParser(description='Reproduce Multiple Descents Paper')
    parser.add_argument('--train-epochs', type=int, default=config.MAX_EPOCHS,
                       help='Number of training epochs')
    parser.add_argument('--analysis-epochs', type=int, default=config.END_EPOCH,
                       help='Number of epochs to analyze for chaos (computationally intensive)')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip training if results exist')
    parser.add_argument('--skip-analysis', action='store_true', 
                       help='Skip chaos analysis if results exist')
    parser.add_argument('--skip-visualization', action='store_true',
                       help='Skip visualization generation')
    parser.add_argument('--quick-test', action='store_true',
                       help='Quick test run (50 train epochs, 20 analysis epochs)')
    
    args = parser.parse_args()
    
    # Quick test mode
    if args.quick_test:
        args.train_epochs = 50
        args.analysis_epochs = 20
        print("QUICK TEST MODE: Reduced epochs for fast testing")
        print()
    
    # Initialize runner
    runner = ExperimentRunner()
    
    success = True
    
    # Step 1: Training
    if not args.skip_training:
        success = success and runner.run_training(args.train_epochs)
        if not success:
            print("Training failed. Stopping experiment.")
            return
    
    # Step 2: Chaos Analysis
    if not args.skip_analysis:
        success = success and runner.run_chaos_analysis(args.analysis_epochs)
        if not success:
            print("Analysis failed. Stopping experiment.")
            return
    
    # Step 3: Visualization
    if not args.skip_visualization:
        success = success and runner.run_visualization()
    
    # Print summary
    runner.print_summary()
    
    if success:
        print("🎉 Experiment completed successfully!")
    else:
        print("⚠️  Experiment completed with some issues.")

if __name__ == "__main__":
    main()