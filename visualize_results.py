import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import config

class ResultsVisualizer:
    """Final fixed visualizer"""
    
    def __init__(self):
        self.results = None
        self.setup_plotting_style()
        
    def setup_plotting_style(self):
        """Setup plotting style"""
        plt.style.use('default')
        plt.rcParams.update({
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'xtick.labelsize': 11,
            'ytick.labelsize': 11,
            'legend.fontsize': 10,
            'figure.titlesize': 18,
            'lines.linewidth': 2,
            'axes.linewidth': 1.5,
            'figure.facecolor': 'white',
            'axes.facecolor': 'white'
        })
    
    def load_results(self):
        """Load analysis results"""
        pickle_path = os.path.join(config.RESULTS_PATH, 'chaos_analysis_results.pkl')
        summary_path = os.path.join(config.RESULTS_PATH, 'analysis_summary.json')
        history_path = os.path.join(config.RESULTS_PATH, 'training_history.json')
        
        # Try pickle first
        if os.path.exists(pickle_path):
            try:
                with open(pickle_path, 'rb') as f:
                    self.results = pickle.load(f)
                print(f"Loaded results from pickle file for {len(self.results['epochs'])} epochs")
                return True
            except Exception as e:
                print(f"Could not load pickle: {e}")
        
        # Try summary
        if os.path.exists(summary_path):
            try:
                with open(summary_path, 'r') as f:
                    data = json.load(f)
                self.results = data.get('training_curves', {})
                if 'transitions' in data:
                    self.results['transitions'] = data['transitions']
                if 'descent_cycles' in data:
                    self.results['descent_cycles'] = data['descent_cycles']
                print(f"Loaded results from summary file")
                return True
            except Exception as e:
                print(f"Could not load summary: {e}")
        
        # Fallback to training history
        if os.path.exists(history_path):
            try:
                with open(history_path, 'r') as f:
                    self.results = json.load(f)
                print(f"Loaded training history as fallback")
                return True
            except Exception as e:
                print(f"Could not load history: {e}")
        
        return False
    
    def plot_multiple_descents_overview(self, save_path=None):
        """Plot Figure 2(a) reproduction"""
        
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
        
        # Test loss (brown line)
        ax1.plot(epochs, test_loss, color='brown', linewidth=2, label='Test Loss')
        ax1.set_ylabel('Test Loss', color='brown', fontsize=14)
        ax1.tick_params(axis='y', labelcolor='brown')
        
        min_loss_idx = np.argmin(test_loss)
        global_opt_epoch = epochs[min_loss_idx]
        
        ax1.axvline(x=global_opt_epoch, color='r', linestyle='--', 
                    label=f"Global Optimum (Epoch {global_opt_epoch})")
        
        # Asymptotic distances (green line) if available
        plot_chaos_data = False
        if ('analyzed_epochs' in self.results and 'asymptotic_distances' in self.results):
            analyzed_epochs = self.results.get('analyzed_epochs', [])
            asym_distances = self.results.get('asymptotic_distances', [])
            
            # Filter valid data
            valid_data = [(e, d) for e, d in zip(analyzed_epochs, asym_distances) 
                         if d is not None and not np.isnan(d)]
            
            if len(valid_data) > 0:
                valid_epochs, valid_distances = zip(*valid_data)
                ax1_twin.plot(valid_epochs, valid_distances, color='green', linewidth=2, 
                            label='Asymptotic Distance')
                ax1_twin.set_ylabel('Log Asymptotic Distance', color='green', fontsize=14)
                ax1_twin.tick_params(axis='y', labelcolor='green')
                plot_chaos_data = True
                
                # # Mark transitions
                # if 'transitions' in self.results:
                #     for i, transition in enumerate(self.results['transitions']):
                #         if transition['transition'] == 'order -> chaos':
                #             epoch = transition['epoch']
                #             if epoch <= max(epochs):
                #                 ax1.axvline(x=epoch, color='red', linestyle='--', alpha=0.7)
                #                 if i == 0:  # Only label first one
                #                     ax1.text(epoch + 2, max(test_loss) * 0.9, 
                #                            f'1st O→C\nEpoch {epoch}', 
                #                            fontsize=10, color='red')
        
        if not plot_chaos_data:
            ax1.text(0.7, 0.8, 'Chaos analysis data\nnot available\n(Run longer analysis)', 
                    transform=ax1.transAxes,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                    fontsize=10)
        
        ax1.set_xlabel('Training Epochs', fontsize=14)
        ax1.set_title('Multiple Descents and Order-Chaos Transitions', fontsize=16)
        ax1.grid(True, alpha=0.3)
        
        # Legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        if plot_chaos_data:
            lines2, labels2 = ax1_twin.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        else:
            ax1.legend()
        
        # Plot 2: Test accuracy
        if test_accuracy:
            ax2.plot(epochs, test_accuracy, color='blue', linewidth=2, label='Test Accuracy')
            ax2.set_xlabel('Training Epochs', fontsize=14)
            ax2.set_ylabel('Accuracy (%)', fontsize=14)
            ax2.set_title('Test Accuracy Over Training', fontsize=14)
            ax2.grid(True, alpha=0.3)
            ax2.legend()
        else:
            ax2.text(0.5, 0.5, 'Test accuracy data not available', 
                    transform=ax2.transAxes, ha='center', va='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        
        plt.show()
    
    def plot_descent_cycles_analysis(self, save_path=None):
        """Plot descent cycles with fixed legend"""
        
        if not self.results:
            print("No results loaded")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        epochs = self.results.get('epochs', [])
        test_loss = self.results.get('test_loss', [])
        
        if not epochs or not test_loss:
            ax1.text(0.5, 0.5, 'No training data available', transform=ax1.transAxes,
                    ha='center', va='center', fontsize=14)
            return
        
        # Plot test loss
        ax1.plot(epochs, test_loss, color='navy', linewidth=2, label='Test Loss')
        
        # Get descent cycles
        descent_cycles = self.results.get('descent_cycles', [])
        
        if len(descent_cycles) > 0:
            print(f"Plotting {len(descent_cycles)} descent cycles")
            
            # Limit colors to avoid legend overflow
            max_cycles_to_show = 10
            cycles_to_plot = descent_cycles[:max_cycles_to_show]
            
            if len(descent_cycles) > max_cycles_to_show:
                print(f"Showing first {max_cycles_to_show} of {len(descent_cycles)} cycles")
            
            # Use distinct colors
            colors = plt.cm.tab10(np.linspace(0, 1, min(len(cycles_to_plot), 10)))
            
            # Track if we've added legend labels
            added_peak_label = False
            added_valley_label = False
            
            for i, (cycle, color) in enumerate(zip(cycles_to_plot, colors)):
                peak_epoch = cycle.get('peak_epoch', 0)
                valley_epoch = cycle.get('valley_epoch', 0)
                peak_loss = cycle.get('peak_loss', 0)
                valley_loss = cycle.get('valley_loss', 0)
                
                if peak_epoch > 0 and valley_epoch > 0:
                    # Highlight descent region (only show first few to avoid clutter)
                    if i < 5:
                        ax1.axvspan(peak_epoch, valley_epoch, alpha=0.2, color=color, 
                                   label=f'Cycle {i+1}' if i < 3 else "")
                    
                    # Mark peaks and valleys (only first few)
                    if i < 8:
                        ax1.plot(peak_epoch, peak_loss, 'ro', markersize=6,
                                label='Peak' if not added_peak_label else "")
                        ax1.plot(valley_epoch, valley_loss, 'go', markersize=6,
                                label='Valley' if not added_valley_label else "")
                        added_peak_label = True
                        added_valley_label = True
            
            # Add summary text
            if len(descent_cycles) > max_cycles_to_show:
                ax1.text(0.02, 0.95, f'Showing {max_cycles_to_show}/{len(descent_cycles)} cycles\n(Overfitting phase)', 
                        transform=ax1.transAxes, 
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                        fontsize=10, verticalalignment='top')
            
            # Plot cycle statistics (limit to reasonable number)
            cycle_lengths = [cycle.get('cycle_length', 0) for cycle in cycles_to_plot]
            descent_magnitudes = [cycle.get('descent_magnitude', 0) for cycle in cycles_to_plot]
            cycle_numbers = list(range(1, len(cycles_to_plot) + 1))
            
            if cycle_lengths and descent_magnitudes:
                ax2_twin = ax2.twinx()
                
                ax2.bar([x - 0.2 for x in cycle_numbers], cycle_lengths, width=0.4, 
                       color='skyblue', alpha=0.7, label='Cycle Length (epochs)')
                ax2_twin.bar([x + 0.2 for x in cycle_numbers], descent_magnitudes, width=0.4,
                           color='lightcoral', alpha=0.7, label='Descent Magnitude')
                
                ax2.set_xlabel('Descent Cycle Number', fontsize=14)
                ax2.set_ylabel('Cycle Length (epochs)', color='blue', fontsize=14)
                ax2_twin.set_ylabel('Descent Magnitude', color='red', fontsize=14)
                ax2.set_title(f'Descent Cycle Characteristics (First {len(cycles_to_plot)} cycles)', fontsize=14)
                ax2.set_xticks(cycle_numbers)
                
                # Combined legend
                lines1, labels1 = ax2.get_legend_handles_labels()
                lines2, labels2 = ax2_twin.get_legend_handles_labels()
                ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
            else:
                ax2.text(0.5, 0.5, 'No cycle statistics available', 
                        transform=ax2.transAxes, ha='center', va='center')
        else:
            ax1.text(0.7, 0.8, f'No significant descent cycles detected\n(Need overfitting phase)', 
                    transform=ax1.transAxes, 
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
            ax2.text(0.5, 0.5, 'No descent cycles to analyze', 
                    transform=ax2.transAxes, ha='center', va='center')
        
        ax1.set_xlabel('Training Epochs', fontsize=14)
        ax1.set_ylabel('Test Loss', fontsize=14)
        ax1.set_title('Individual Descent Cycles Analysis', fontsize=16)
        ax1.grid(True, alpha=0.3)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Move legend outside
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
            
        plt.show()
    
    def plot_optimal_epoch_analysis(self, save_path=None):
        """Plot optimal epochs vs transitions with corrected phase detection"""
        
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
        
        # Mark first order-chaos transition if available
        transition_found = False
        first_transition_epoch = None
        
        if 'transitions' in self.results and self.results['transitions']:
            for transition in self.results['transitions']:
                if transition['transition'] == 'order -> chaos':
                    trans_epoch = transition['epoch']
                    if trans_epoch <= len(test_loss):
                        trans_loss = test_loss[trans_epoch - 1]  # Convert to 0-based
                        ax.plot(trans_epoch, trans_loss, 'gs', markersize=12,
                               label=f"First Order→Chaos Transition (Epoch {trans_epoch})")
                        first_transition_epoch = trans_epoch
                        transition_found = True
                        break
        
        # Add phase background if available
        phase_background = False
        if ('phases' in self.results and 'analyzed_epochs' in self.results and 
            len(self.results['phases']) > 0):
            
            analyzed_epochs = self.results.get('analyzed_epochs', [])
            phases = self.results.get('phases', [])
            
            order_shown = False
            chaos_shown = False
            
            for i, (epoch, phase) in enumerate(zip(analyzed_epochs, phases)):
                if i < len(analyzed_epochs) - 1:
                    next_epoch = analyzed_epochs[i + 1]
                    
                    if phase == 'order' and not order_shown:
                        ax.axvspan(epoch, next_epoch, alpha=0.2, color='lightblue', 
                                  label='Order Phase')
                        order_shown = True
                        phase_background = True
                    elif phase == 'order':
                        ax.axvspan(epoch, next_epoch, alpha=0.2, color='lightblue')
                    elif phase == 'chaos' and not chaos_shown:
                        ax.axvspan(epoch, next_epoch, alpha=0.2, color='lightcoral',
                                  label='Chaos Phase')
                        chaos_shown = True  
                        phase_background = True
                    elif phase == 'chaos':
                        ax.axvspan(epoch, next_epoch, alpha=0.2, color='lightcoral')
        
        ax.set_xlabel('Training Epochs', fontsize=14)
        ax.set_ylabel('Test Loss', fontsize=14)
        ax.set_title('Optimal Epochs and Order-Chaos Transitions', fontsize=16)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add findings text box
        textstr = [f"Global Optimum: Epoch {global_opt_epoch} (Loss: {global_opt_loss:.4f})"]
        
        if transition_found and first_transition_epoch:
            textstr.append(f"First O→C Transition: Epoch {first_transition_epoch}")
            if abs(global_opt_epoch - first_transition_epoch) <= 3:
                textstr.append("✓ Global optimum near first transition!")
                textstr.append("(Paper's key finding confirmed)")
            else:
                textstr.append("✗ Global optimum differs from first transition")
                textstr.append("(May need more epochs for full pattern)")
        else:
            textstr.append("No order-chaos transitions detected")
            textstr.append("Note: Need longer chaos analysis")
        
        if not phase_background:
            textstr.append("Phase analysis unavailable")
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.02, 0.98, '\n'.join(textstr), transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
            
        plt.show()
    
    def plot_bifurcation_diagram(self, save_path=None):
        """Plot bifurcation diagram with proper data handling"""
        
        if not self.results:
            print("No results loaded")
            return
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Check for bifurcation data
        has_data = ('analyzed_epochs' in self.results and 
                   'bifurcation_data' in self.results and 
                   self.results['bifurcation_data'])
        
        if has_data:
            analyzed_epochs = self.results['analyzed_epochs']
            bifurcation_data = self.results['bifurcation_data']
            
            all_epochs = []
            all_values = []
            
            for epoch, reduced_sums in zip(analyzed_epochs, bifurcation_data):
                if isinstance(reduced_sums, list) and len(reduced_sums) > 0:
                    try:
                        valid_sums = [x for x in reduced_sums if np.isfinite(x)]
                        if len(valid_sums) > 10:  # Need reasonable amount of data
                            all_epochs.extend([epoch] * len(valid_sums))
                            all_values.extend(valid_sums)
                    except:
                        continue
            
            if len(all_epochs) > 100:  # Need sufficient data for meaningful plot
                ax.scatter(all_epochs, all_values, s=0.3, color='blue', alpha=0.4, 
                          label='Bifurcation Map')
                
                # Overlay test loss if available
                if 'test_loss' in self.results:
                    test_loss = self.results['test_loss']
                    loss_epochs = self.results['epochs']
                    
                    # Scale to fit
                    if len(all_values) > 0:
                        val_range = max(all_values) - min(all_values)
                        loss_range = max(test_loss) - min(test_loss)
                        if val_range > 0 and loss_range > 0:
                            val_min = min(all_values)
                            loss_min = min(test_loss)
                            scaled_loss = [val_min + (l - loss_min) * val_range / loss_range 
                                         for l in test_loss]
                            ax.plot(loss_epochs, scaled_loss, color='brown', linewidth=2, 
                                   alpha=0.8, label='Test Loss (scaled)')
                
                ax.set_ylabel('Reduced Sum h_T · 1 (normalized)', fontsize=14)
            else:
                ax.text(0.5, 0.5, 'Insufficient bifurcation data\n(Need more samples)', 
                       transform=ax.transAxes, ha='center', va='center',
                       bbox=dict(boxstyle='round', facecolor='lightyellow'))
        else:
            # Show just test loss
            if 'test_loss' in self.results:
                ax.plot(self.results['epochs'], self.results['test_loss'], 
                       color='brown', linewidth=2, label='Test Loss')
            
            ax.text(0.5, 0.7, 'Bifurcation analysis not available\n(Limited chaos analysis)', 
                   transform=ax.transAxes, ha='center', va='center',
                   bbox=dict(boxstyle='round', facecolor='lightyellow'))
            ax.set_ylabel('Test Loss', fontsize=14)
        
        ax.set_xlabel('Training Epochs', fontsize=14)
        ax.set_title('Bifurcation Diagram and Test Loss', fontsize=16)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
            
        plt.show()
    
    def generate_all_plots(self):
        """Generate all plots"""
        
        if not self.load_results():
            print("Cannot load results. Please run analysis first.")
            return
        
        print("Generating visualization plots...")
        
        figures_dir = os.path.join(config.RESULTS_PATH, 'figures')
        os.makedirs(figures_dir, exist_ok=True)
        
        plots_to_generate = [
            ('multiple_descents_overview', self.plot_multiple_descents_overview),
            ('descent_cycles_analysis', self.plot_descent_cycles_analysis),
            ('optimal_epoch_analysis', self.plot_optimal_epoch_analysis),
            ('bifurcation_diagram', self.plot_bifurcation_diagram)
        ]
        
        for plot_name, plot_function in plots_to_generate:
            try:
                save_path = os.path.join(figures_dir, f'{plot_name}.png') if config.SAVE_FIGURES else None
                plot_function(save_path=save_path)
                print(f"✓ Generated {plot_name}")
            except Exception as e:
                print(f"✗ Error generating {plot_name}: {str(e)}")
        
        print(f"\nPlots saved to {figures_dir}")

def main():
    """Main visualization function"""
    visualizer = ResultsVisualizer()
    visualizer.generate_all_plots()

if __name__ == "__main__":
    main()