"""
Fixed visualization script to reproduce Figure 2 and other plots from the paper
Handles all edge cases and data format issues
"""

import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')  # Suppress overflow warnings

import config

class ResultsVisualizer:
    """Visualizer for multiple descents and chaos analysis results - Fixed version"""
    
    def __init__(self):
        self.results = None
        self.setup_plotting_style()
        
    def setup_plotting_style(self):
        """Setup publication-quality plotting style"""
        plt.style.use('default')  # Use default style to avoid seaborn issues
        
        # Set default parameters
        plt.rcParams.update({
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'xtick.labelsize': 11,
            'ytick.labelsize': 11,
            'legend.fontsize': 12,
            'figure.titlesize': 18,
            'lines.linewidth': 2,
            'axes.linewidth': 1.5,
            'figure.facecolor': 'white',
            'axes.facecolor': 'white'
        })
    
    def load_results(self):
        """Load analysis results from available files"""
        results_path = os.path.join(config.RESULTS_PATH, 'chaos_analysis_results.json')
        pickle_path = os.path.join(config.RESULTS_PATH, 'chaos_analysis_results.pkl')
        summary_path = os.path.join(config.RESULTS_PATH, 'analysis_summary.json')
        history_path = os.path.join(config.RESULTS_PATH, 'training_history.json')
        
        # Try to load from different sources in order of preference
        if os.path.exists(pickle_path):
            try:
                with open(pickle_path, 'rb') as f:
                    self.results = pickle.load(f)
                print(f"Loaded results from pickle file for {len(self.results['epochs'])} epochs")
                return True
            except Exception as e:
                print(f"Could not load pickle file: {e}")
        
        if os.path.exists(results_path):
            try:
                with open(results_path, 'r') as f:
                    self.results = json.load(f)
                print(f"Loaded results from JSON file for {len(self.results['epochs'])} epochs")
                return True
            except Exception as e:
                print(f"Could not load JSON file: {e}")
        
        if os.path.exists(summary_path):
            try:
                with open(summary_path, 'r') as f:
                    summary = json.load(f)
                # Create minimal results structure
                self.results = summary.get('training_curves', {})
                if 'summary' in summary:
                    self.results['optimal_epochs'] = {
                        'global_optimum': {
                            'epoch': summary['summary']['global_optimum_epoch'],
                            'test_loss': summary['summary']['min_test_loss']
                        }
                    }
                print(f"Loaded results from summary file")
                return True
            except Exception as e:
                print(f"Could not load summary file: {e}")
        
        if os.path.exists(history_path):
            try:
                with open(history_path, 'r') as f:
                    self.results = json.load(f)
                print(f"Loaded training history as fallback for {len(self.results['epochs'])} epochs")
                return True
            except Exception as e:
                print(f"Could not load training history: {e}")
        
        print(f"No results files found in {config.RESULTS_PATH}")
        return False
    
    def plot_multiple_descents_overview(self, save_path=None):
        """
        Reproduce Figure 2(a) from the paper: Multiple descents with order-chaos transitions
        """
        
        if not self.results:
            print("No results loaded")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        epochs = self.results.get('epochs', [])
        test_loss = self.results.get('test_loss', [])
        test_accuracy = self.results.get('test_accuracy', [])
        
        if not epochs or not test_loss:
            print("No training data available")
            return
        
        # Plot 1: Test loss and asymptotic distances
        ax1_twin = ax1.twinx()
        
        # Test loss (brown line in paper)
        line1 = ax1.plot(epochs, test_loss, color='brown', linewidth=2, label='Test Loss')
        ax1.set_ylabel('Test Loss', color='brown', fontsize=14)
        ax1.tick_params(axis='y', labelcolor='brown')
        
        # Asymptotic distances (green line in paper) if available
        if 'analyzed_epochs' in self.results and 'asymptotic_distances' in self.results:
            analyzed_epochs = self.results['analyzed_epochs']
            asym_distances = self.results['asymptotic_distances']
            
            # Filter out None/NaN values
            valid_data = [(e, d) for e, d in zip(analyzed_epochs, asym_distances) 
                         if d is not None and not (isinstance(d, float) and np.isnan(d))]
            
            if valid_data:
                valid_epochs, valid_distances = zip(*valid_data)
                line2 = ax1_twin.plot(valid_epochs, valid_distances, color='green', linewidth=2, label='Asymptotic Distance')
                ax1_twin.set_ylabel('Log Asymptotic Distance', color='green', fontsize=14)
                ax1_twin.tick_params(axis='y', labelcolor='green')
                
                # Highlight order-chaos transitions if available
                if 'transitions' in self.results:
                    for transition in self.results['transitions']:
                        epoch = transition['epoch']
                        if transition['transition'] == 'order -> chaos':
                            ax1.axvline(x=epoch, color='red', linestyle='--', alpha=0.7, linewidth=1)
                            ax1.text(epoch, max(test_loss) * 0.9, f'Epoch {epoch}', rotation=90, 
                                    ha='right', va='top', fontsize=10, color='red')
        else:
            # Add note about missing chaos analysis
            ax1.text(0.7, 0.8, 'Chaos analysis data\nnot available', transform=ax1.transAxes,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                    fontsize=10)
        
        ax1.set_xlabel('Training Epochs', fontsize=14)
        ax1.set_title('Multiple Descents and Order-Chaos Transitions', fontsize=16)
        ax1.grid(True, alpha=0.3)
        
        # Add legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        if 'analyzed_epochs' in self.results:
            try:
                lines2, labels2 = ax1_twin.get_legend_handles_labels()
                ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
            except:
                ax1.legend()
        else:
            ax1.legend()
        
        # Plot 2: Training accuracy for context
        if test_accuracy:
            ax2.plot(epochs, test_accuracy, color='blue', linewidth=2, label='Test Accuracy')
            ax2.set_xlabel('Training Epochs', fontsize=14)
            ax2.set_ylabel('Accuracy (%)', fontsize=14)
            ax2.set_title('Test Accuracy Over Training', fontsize=14)
            ax2.grid(True, alpha=0.3)
            ax2.legend()
        else:
            ax2.text(0.5, 0.5, 'Test accuracy data not available', transform=ax2.transAxes,
                    ha='center', va='center', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        
        plt.show()
    
    def plot_bifurcation_diagram(self, save_path=None):
        """
        Reproduce Figure 2(b) from the paper: Bifurcation diagram
        """
        
        if not self.results:
            print("No results loaded")
            return
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Check if we have bifurcation data
        has_bifurcation_data = ('analyzed_epochs' in self.results and 
                              'bifurcation_data' in self.results and
                              self.results['bifurcation_data'])
        
        if has_bifurcation_data:
            analyzed_epochs = self.results['analyzed_epochs']
            bifurcation_data = self.results['bifurcation_data']
            
            # Create bifurcation plot (blue points in paper)
            all_epochs = []
            all_values = []
            
            for epoch, reduced_sums in zip(analyzed_epochs, bifurcation_data):
                if isinstance(reduced_sums, list) and len(reduced_sums) > 0:
                    # Convert to numpy array and filter valid data
                    try:
                        reduced_sums_array = np.array(reduced_sums, dtype=float)
                        valid_mask = np.isfinite(reduced_sums_array)
                        valid_sums = reduced_sums_array[valid_mask]
                        
                        if len(valid_sums) > 0:
                            epoch_array = np.full_like(valid_sums, epoch)
                            all_epochs.extend(epoch_array)
                            all_values.extend(valid_sums)
                    except:
                        continue
            
            # Plot bifurcation points if we have valid data
            if len(all_epochs) > 0:
                ax.scatter(all_epochs, all_values, s=0.5, color='blue', alpha=0.6, label='Bifurcation Map')
                
                # Overlay test loss (scaled)
                if 'test_loss' in self.results:
                    test_loss = self.results['test_loss']
                    loss_epochs = self.results['epochs']
                    
                    # Scale test loss to fit
                    values_min, values_max = min(all_values), max(all_values)
                    loss_min, loss_max = min(test_loss), max(test_loss)
                    
                    if values_max != values_min and loss_max != loss_min:
                        scaled_loss = [values_min + (l - loss_min) * (values_max - values_min) / (loss_max - loss_min) 
                                     for l in test_loss]
                        ax.plot(loss_epochs, scaled_loss, color='brown', linewidth=2, alpha=0.8, label='Test Loss (scaled)')
            else:
                ax.text(0.5, 0.5, 'No valid bifurcation data\n(Data processing issue)', 
                       transform=ax.transAxes, ha='center', va='center',
                       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        else:
            # No bifurcation data - just show test loss
            if 'test_loss' in self.results:
                ax.plot(self.results['epochs'], self.results['test_loss'], color='brown', linewidth=2, label='Test Loss')
            
            ax.text(0.5, 0.7, 'Bifurcation analysis not available\n(Limited chaos analysis)', 
                   transform=ax.transAxes, ha='center', va='center',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        ax.set_xlabel('Training Epochs', fontsize=14)
        ax.set_ylabel('Reduced Sum h_T · 1 (normalized)', fontsize=14)
        ax.set_title('Bifurcation Diagram and Test Loss', fontsize=16)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
            
        plt.show()
    
    def plot_descent_cycles_analysis(self, save_path=None):
        """Plot detailed analysis of individual descent cycles"""
        
        if not self.results:
            print("No results loaded")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        epochs = self.results.get('epochs', [])
        test_loss = self.results.get('test_loss', [])
        
        if not epochs or not test_loss:
            ax1.text(0.5, 0.5, 'No training data available', transform=ax1.transAxes,
                    ha='center', va='center', fontsize=14)
            ax2.text(0.5, 0.5, 'No training data available', transform=ax2.transAxes,
                    ha='center', va='center', fontsize=14)
            return
        
        # Plot test loss
        ax1.plot(epochs, test_loss, color='navy', linewidth=2, label='Test Loss')
        
        # Check if we have descent cycle data
        descent_cycles = self.results.get('descent_cycles', [])
        
        if descent_cycles and len(descent_cycles) > 0:
            # Highlight descent cycles
            colors = plt.cm.Set3(np.linspace(0, 1, len(descent_cycles)))
            
            for i, (cycle, color) in enumerate(zip(descent_cycles, colors)):
                peak_epoch = cycle.get('peak_epoch', 0)
                valley_epoch = cycle.get('valley_epoch', 0)
                peak_loss = cycle.get('peak_loss', 0)
                valley_loss = cycle.get('valley_loss', 0)
                
                if peak_epoch > 0 and valley_epoch > 0:
                    # Highlight the descent region
                    ax1.axvspan(peak_epoch, valley_epoch, alpha=0.3, color=color, 
                               label=f'Cycle {i+1}')
                    
                    # Mark peak and valley
                    ax1.plot(peak_epoch, peak_loss, 'ro', markersize=8)
                    ax1.plot(valley_epoch, valley_loss, 'go', markersize=8)
            
            # Plot cycle statistics
            cycle_lengths = [cycle.get('cycle_length', 0) for cycle in descent_cycles]
            descent_magnitudes = [cycle.get('descent_magnitude', 0) for cycle in descent_cycles]
            cycle_numbers = list(range(1, len(descent_cycles) + 1))
            
            if any(cycle_lengths) and any(descent_magnitudes):
                ax2_twin = ax2.twinx()
                
                bars1 = ax2.bar([x - 0.2 for x in cycle_numbers], cycle_lengths, width=0.4, 
                               color='skyblue', alpha=0.7, label='Cycle Length (epochs)')
                bars2 = ax2_twin.bar([x + 0.2 for x in cycle_numbers], descent_magnitudes, width=0.4,
                                   color='lightcoral', alpha=0.7, label='Descent Magnitude')
                
                ax2.set_xlabel('Descent Cycle Number', fontsize=14)
                ax2.set_ylabel('Cycle Length (epochs)', color='blue', fontsize=14)
                ax2_twin.set_ylabel('Descent Magnitude', color='red', fontsize=14)
                ax2.set_title('Descent Cycle Characteristics', fontsize=14)
                
                # Combine legends
                lines1, labels1 = ax2.get_legend_handles_labels()
                lines2, labels2 = ax2_twin.get_legend_handles_labels()
                ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
            else:
                ax2.text(0.5, 0.5, 'Descent cycle statistics not available', 
                        transform=ax2.transAxes, ha='center', va='center', fontsize=12)
        else:
            ax1.text(0.7, 0.8, f'Only {len(descent_cycles)} descent cycles detected\n(Need more training epochs)', 
                    transform=ax1.transAxes, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
            ax2.text(0.5, 0.5, 'No descent cycles to analyze', 
                    transform=ax2.transAxes, ha='center', va='center', fontsize=12)
        
        ax1.set_xlabel('Training Epochs', fontsize=14)
        ax1.set_ylabel('Test Loss', fontsize=14)
        ax1.set_title('Individual Descent Cycles Analysis', fontsize=16)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
            
        plt.show()
    
    def plot_optimal_epoch_analysis(self, save_path=None):
        """Analyze and visualize optimal epochs vs order-chaos transitions"""
        
        if not self.results:
            print("No results loaded")
            return
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        epochs = self.results.get('epochs', [])
        test_loss = self.results.get('test_loss', [])
        
        if not epochs or not test_loss:
            ax.text(0.5, 0.5, 'No training data available', transform=ax.transAxes,
                    ha='center', va='center', fontsize=14)
            return
        
        # Plot test loss
        ax.plot(epochs, test_loss, color='navy', linewidth=2, label='Test Loss')
        
        # Find and mark global optimum
        min_loss_idx = np.argmin(test_loss)
        global_opt_epoch = epochs[min_loss_idx]
        global_opt_loss = test_loss[min_loss_idx]
        
        ax.plot(global_opt_epoch, global_opt_loss, 'ro', markersize=12, 
               label=f"Global Optimum (Epoch {global_opt_epoch})")
        
        # Mark transitions if available
        transition_found = False
        if 'transitions' in self.results and self.results['transitions']:
            for transition in self.results['transitions']:
                if transition['transition'] == 'order -> chaos':
                    trans_epoch = transition['epoch']
                    if trans_epoch <= len(test_loss):
                        trans_loss = test_loss[trans_epoch - 1]  # Convert to 0-based index
                        ax.plot(trans_epoch, trans_loss, 'gs', markersize=12,
                               label=f"First Order-Chaos Transition (Epoch {trans_epoch})")
                        transition_found = True
                        break
        
        # Add phase background if available
        if 'phases' in self.results and 'analyzed_epochs' in self.results:
            analyzed_epochs = self.results['analyzed_epochs']
            phases = self.results['phases']
            
            for i, (epoch, phase) in enumerate(zip(analyzed_epochs, phases)):
                if i < len(analyzed_epochs) - 1:
                    next_epoch = analyzed_epochs[i + 1]
                    
                    if phase == 'order':
                        ax.axvspan(epoch, next_epoch, alpha=0.2, color='lightblue', 
                                  label='Order Phase' if i == 0 else "")
                    elif phase == 'chaos':
                        ax.axvspan(epoch, next_epoch, alpha=0.2, color='lightcoral',
                                  label='Chaos Phase' if i == 0 else "")
        
        ax.set_xlabel('Training Epochs', fontsize=14)
        ax.set_ylabel('Test Loss', fontsize=14)
        ax.set_title('Optimal Epochs and Order-Chaos Transitions', fontsize=16)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add text box with findings
        textstr = [f"Global Optimum: Epoch {global_opt_epoch} (Loss: {global_opt_loss:.4f})"]
        
        if transition_found:
            trans_epoch = None
            for transition in self.results['transitions']:
                if transition['transition'] == 'order -> chaos':
                    trans_epoch = transition['epoch']
                    break
            
            if trans_epoch:
                textstr.append(f"First Transition: Epoch {trans_epoch}")
                if abs(global_opt_epoch - trans_epoch) <= 5:
                    textstr.append("✓ Global optimum occurs at first order-chaos transition!")
                else:
                    textstr.append("✗ Global optimum differs from first transition")
        else:
            textstr.append("Note: Limited chaos analysis - need more epochs")
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.02, 0.98, '\n'.join(textstr), transform=ax.transAxes, fontsize=11,
               verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
            
        plt.show()
    
    def plot_tanh_map_comparison(self, save_path=None):
        """Plot tanh map for comparison with LSTM dynamics (Figure 3 from paper)"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        def safe_tanh_map(k, r):
            """Numerically stable tanh map"""
            k = np.clip(k, -5, 5)  # Prevent extreme values
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = r * k * (1 - np.tanh(k))
                if np.isnan(result) or np.isinf(result):
                    return 0.0
                return np.clip(result, -10, 10)
        
        def calculate_distance_safe(r_val, iterations=200, samples=50):
            """Calculate asymptotic distance safely"""
            distances = []
            
            for _ in range(samples):
                k0 = np.random.uniform(-0.2, 0.2)  # Small initial range
                
                # Original trajectory
                k = k0
                for _ in range(iterations):
                    k = safe_tanh_map(k, r_val)
                k_final = k
                
                # Perturbed trajectory
                k_pert = k0 + 1e-6
                for _ in range(iterations):
                    k_pert = safe_tanh_map(k_pert, r_val)
                k_final_pert = k_pert
                
                distance = abs(k_final_pert - k_final) + np.exp(-15)
                if np.isfinite(distance):
                    distances.append(distance)
            
            if distances:
                return np.mean(np.log(distances))
            return -15  # Default to order
        
        # Generate tanh map data
        r_values = np.linspace(1, 6, 50)  # Reduced range and resolution
        asymptotic_distances = []
        bifurcation_points = []
        
        print("Calculating tanh map data (this may take a moment)...")
        
        for r in r_values:
            # Calculate asymptotic distance
            asym_dist = calculate_distance_safe(r)
            asymptotic_distances.append(asym_dist)
            
            # Calculate bifurcation points
            final_values = []
            for _ in range(30):  # Fewer samples
                k = np.random.uniform(-0.2, 0.2)
                for _ in range(200):  # Fewer iterations
                    k = safe_tanh_map(k, r)
                if np.isfinite(k):
                    final_values.append(k)
            
            bifurcation_points.append(final_values)
        
        # Plot 1: Asymptotic distances
        ax1.plot(r_values, asymptotic_distances, color='green', linewidth=2)
        ax1.set_xlabel('Parameter r', fontsize=14)
        ax1.set_ylabel('Log Asymptotic Distance', fontsize=14)
        ax1.set_title('Tanh Map: Asymptotic Distance', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=-10, color='red', linestyle='--', alpha=0.7, label='Order-Chaos Threshold')
        ax1.legend()
        
        # Plot 2: Bifurcation diagram
        all_r = []
        all_vals = []
        
        for r, vals in zip(r_values, bifurcation_points):
            if vals:
                all_r.extend([r] * len(vals))
                all_vals.extend(vals)
        
        if all_r and all_vals:
            ax2.scatter(all_r, all_vals, s=1.0, color='blue', alpha=0.6)
        
        ax2.set_xlabel('Parameter r', fontsize=14)
        ax2.set_ylabel('Final Values k_T', fontsize=14)
        ax2.set_title('Tanh Map: Bifurcation Diagram', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
            
        plt.show()
    
    def generate_all_plots(self):
        """Generate all plots and save them"""
        
        if not self.load_results():
            print("Cannot load results. Please run training and analysis first.")
            return
        
        print("Generating all visualization plots...")
        
        # Create figures directory
        figures_dir = os.path.join(config.RESULTS_PATH, 'figures')
        os.makedirs(figures_dir, exist_ok=True)
        
        # Generate each plot
        plots_to_generate = [
            ('multiple_descents_overview', self.plot_multiple_descents_overview),
            ('bifurcation_diagram', self.plot_bifurcation_diagram),
            ('descent_cycles_analysis', self.plot_descent_cycles_analysis),
            ('optimal_epoch_analysis', self.plot_optimal_epoch_analysis),
            #('tanh_map_comparison', self.plot_tanh_map_comparison)
        ]
        
        for plot_name, plot_function in plots_to_generate:
            try:
                save_path = os.path.join(figures_dir, f'{plot_name}.png') if config.SAVE_FIGURES else None
                plot_function(save_path=save_path)
                print(f"✓ Generated {plot_name}")
            except Exception as e:
                print(f"✗ Error generating {plot_name}: {str(e)}")
                import traceback
                traceback.print_exc()
        
        print(f"\nAll plots saved to {figures_dir}")

def main():
    """Main visualization function"""
    
    visualizer = ResultsVisualizer()
    
    # Generate all plots
    visualizer.generate_all_plots()

if __name__ == "__main__":
    main()