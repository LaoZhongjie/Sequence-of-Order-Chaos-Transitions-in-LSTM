"""
Asymptotic stability analysis for detecting order-chaos transitions
Implements the methodology described in Section II of the paper
"""

import torch
import numpy as np
from tqdm import tqdm
import config

class ChaosAnalyzer:
    """
    Analyzer for detecting order-chaos transitions in LSTM models
    Based on asymptotic trajectory separation under perturbation
    """
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()  # Set to evaluation mode
    
    def calculate_asymptotic_distance(self, test_dataset, num_samples=500):
        """
        Calculate asymptotic distance for order-chaos analysis
        
        This implements the methodology from Figure 1 and Section II.B of the paper:
        1. Pass review through embedding layer (first 500 timesteps)
        2. Continue LSTM iteration with zero inputs (timesteps 500-1599)
        3. Add perturbation to initial hidden state
        4. Calculate distance between perturbed and original trajectories
        
        Args:
            test_dataset: Test dataset containing reviews
            num_samples: Number of samples to analyze (500 in paper)
            
        Returns:
            asymptotic_distance: Log of geometric mean of distances
            reduced_sums: Reduced sums for bifurcation visualization
        """
        
        # Randomly sample reviews from test dataset
        indices = torch.randperm(len(test_dataset))[:num_samples]
        
        distances = []
        reduced_sums = []
        
        with torch.no_grad():
            for idx in tqdm(indices, desc="Calculating asymptotic distances"):
                # Get review and convert to tensor
                review, _ = test_dataset[idx]
                review = review.unsqueeze(0).to(self.device)  # Add batch dimension
                
                # Step 1: Process real review (first 500 timesteps)
                h_t, hidden = self.model.get_lstm_hidden_output(review)
                
                # Step 2: Create perturbed initial hidden state
                h_0, c_0 = hidden
                
                # Add Gaussian noise perturbation to hidden state
                noise = torch.randn_like(h_0) * config.NOISE_SCALE
                h_0_perturbed = h_0 + noise
                hidden_perturbed = (h_0_perturbed, c_0)
                
                # Step 3: Continue iteration with zero inputs
                zero_timesteps = config.ASYMPTOTIC_TIMESTEPS - config.REVIEW_TIMESTEPS
                
                # Original trajectory
                final_hidden = self.model.continue_lstm_iteration(
                    hidden, zero_timesteps, config.EMBEDDING_DIM
                )
                h_final = final_hidden[0].squeeze(0)  # Remove layer dimension
                
                # Perturbed trajectory
                final_hidden_perturbed = self.model.continue_lstm_iteration(
                    hidden_perturbed, zero_timesteps, config.EMBEDDING_DIM
                )
                h_final_perturbed = final_hidden_perturbed[0].squeeze(0)
                
                # Step 4: Calculate asymptotic distance
                distance = torch.norm(h_final_perturbed - h_final, p=2).item()
                
                # Add machine precision threshold (exp(-15) in paper)
                distance_adjusted = distance + np.exp(config.MACHINE_PRECISION_THRESHOLD)
                distances.append(distance_adjusted)
                
                # Calculate reduced sum for bifurcation visualization (h_T · 1)
                reduced_sum = torch.sum(h_final).item()
                reduced_sums.append(reduced_sum)
        
        # Calculate geometric mean of distances (step 6 in paper)
        distances = np.array(distances)
        geometric_mean = np.exp(np.mean(np.log(distances)))
        asymptotic_distance = np.log(geometric_mean)
        
        # Normalize reduced sums by subtracting mean (for visualization)
        reduced_sums = np.array(reduced_sums)
        reduced_sums_normalized = reduced_sums - np.mean(reduced_sums)
        
        return asymptotic_distance, reduced_sums_normalized
    
    def analyze_training_dynamics(self, test_dataset, checkpoint_dir, 
                                 start_epoch=0, end_epoch=1000, interval=1):
        """
        Analyze order-chaos transitions throughout training
        
        Args:
            test_dataset: Test dataset for analysis
            checkpoint_dir: Directory containing model checkpoints
            start_epoch: Starting epoch for analysis
            end_epoch: Ending epoch for analysis
            interval: Interval between analyzed epochs
            
        Returns:
            epochs: List of analyzed epochs
            asymptotic_distances: Asymptotic distances for each epoch
            bifurcation_data: Reduced sums for bifurcation visualization
        """
        
        epochs = list(range(start_epoch, end_epoch + 1, interval))
        asymptotic_distances = []
        bifurcation_data = []
        
        for epoch in tqdm(epochs, desc="Analyzing training dynamics"):
            try:
                # Load model checkpoint for this epoch
                checkpoint_path = f"{checkpoint_dir}/model_epoch_{epoch}.pt"
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                
                # Calculate asymptotic distance
                asym_dist, reduced_sums = self.calculate_asymptotic_distance(test_dataset)
                
                asymptotic_distances.append(asym_dist)
                bifurcation_data.append(reduced_sums)
                
            except FileNotFoundError:
                print(f"Checkpoint not found for epoch {epoch}, skipping...")
                asymptotic_distances.append(np.nan)
                bifurcation_data.append(np.array([np.nan] * config.NUM_TEST_SAMPLES))
        
        return epochs, asymptotic_distances, bifurcation_data
    
    def detect_order_chaos_transitions(self, asymptotic_distances, threshold=-10):
        """
        Detect transitions between order and chaos phases
        
        Args:
            asymptotic_distances: List of asymptotic distances over epochs
            threshold: Threshold for detecting order phase (default: -10)
            
        Returns:
            transitions: List of transition points
            phases: List of phase labels ('order' or 'chaos')
        """
        
        phases = []
        transitions = []
        
        for i, distance in enumerate(asymptotic_distances):
            if np.isnan(distance):
                phases.append('unknown')
                continue
                
            # Order phase: distance close to machine precision threshold
            if distance <= threshold:
                current_phase = 'order'
            else:
                current_phase = 'chaos'
            
            phases.append(current_phase)
            
            # Detect transition
            if i > 0 and phases[i-1] != 'unknown' and phases[i-1] != current_phase:
                transitions.append({
                    'epoch': i,
                    'transition': f"{phases[i-1]} -> {current_phase}",
                    'distance_before': asymptotic_distances[i-1],
                    'distance_after': distance
                })
        
        return transitions, phases
    
    def calculate_loss_on_wrong_predictions(self, test_loader, criterion):
        """
        Calculate loss specifically on incorrectly predicted samples
        This implements the analysis from Section IV of the paper
        
        Args:
            test_loader: DataLoader for test data
            criterion: Loss function (BCELoss)
            
        Returns:
            wrong_prediction_loss: Loss calculated only on wrong predictions
            num_wrong: Number of wrong predictions
        """
        
        self.model.eval()
        wrong_losses = []
        total_wrong = 0
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                output, _ = self.model(data)
                output = output.squeeze()
                
                # Find wrong predictions
                predictions = (output > 0.5).float()
                wrong_mask = (predictions != target)
                
                if wrong_mask.sum() > 0:
                    wrong_outputs = output[wrong_mask]
                    wrong_targets = target[wrong_mask]
                    
                    # Calculate loss for wrong predictions
                    wrong_loss = criterion(wrong_outputs, wrong_targets)
                    wrong_losses.append(wrong_loss.item())
                    total_wrong += wrong_mask.sum().item()
        
        if wrong_losses:
            avg_wrong_loss = np.mean(wrong_losses)
        else:
            avg_wrong_loss = 0.0
        
        return avg_wrong_loss, total_wrong