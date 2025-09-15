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
        Calculate asymptotic distance exactly as described in paper Section II.B
        
        Key points from paper:
        1. Use first 500 timesteps with real review data
        2. Continue with zero input for timesteps 500-1599 (1100 additional steps)
        3. Add small Gaussian noise perturbation to h_{-1}
        4. Calculate |h'_T - h_T| at T=1599
        5. Add exp(-15) for machine precision
        6. Take log of geometric mean across 500 samples
        """
        
        # Randomly sample reviews from test dataset
        indices = torch.randperm(len(test_dataset))[:num_samples]
        
        distances = []
        reduced_sums = []
        
        with torch.no_grad():
            for idx in tqdm(indices, desc="Calculating asymptotic distances", leave=False):
                # Get review
                review, _ = test_dataset[idx]
                review = review.unsqueeze(0).to(self.device)  # (1, 500)
                
                # Step 1: Process review through embedding
                embedded = self.model.embedding(review)  # (1, 500, 32)
                
                # Step 2: Initialize hidden states
                batch_size = 1
                h_0 = torch.zeros(1, batch_size, self.model.hidden_size, device=self.device)
                c_0 = torch.zeros(1, batch_size, self.model.hidden_size, device=self.device)
                hidden = (h_0, c_0)
                
                # Step 3: Run LSTM for first 500 timesteps (with real input)
                lstm_out, hidden_after_review = self.model.lstm(embedded, hidden)
                
                # Step 4: Create perturbed initial state
                h_after, c_after = hidden_after_review
                # Add Gaussian noise to hidden state (perturbation ε)
                epsilon = torch.randn_like(h_after) * config.NOISE_SCALE
                h_perturbed = h_after + epsilon
                hidden_perturbed = (h_perturbed, c_after)  # Only perturb h, not c
                
                # Step 5: Continue iteration with zero inputs for remaining timesteps
                remaining_steps = config.ASYMPTOTIC_TIMESTEPS - config.REVIEW_TIMESTEPS  # 1100
                zero_input = torch.zeros(1, remaining_steps, config.EMBEDDING_DIM, device=self.device)
                
                # Original trajectory
                _, final_hidden = self.model.lstm(zero_input, hidden_after_review)
                h_final = final_hidden[0]  # (1, 1, 60)
                
                # Perturbed trajectory  
                _, final_hidden_perturbed = self.model.lstm(zero_input, hidden_perturbed)
                h_final_perturbed = final_hidden_perturbed[0]  # (1, 1, 60)
                
                # Step 6: Calculate asymptotic distance |h'_T - h_T|
                distance = torch.norm(h_final_perturbed - h_final, p=2).item()
                
                # Add machine precision threshold as in paper
                distance_adjusted = distance + np.exp(config.MACHINE_PRECISION_THRESHOLD)  # exp(-15)
                distances.append(distance_adjusted)
                
                # Calculate reduced sum h_T · 1 for bifurcation analysis
                reduced_sum = torch.sum(h_final).item()
                reduced_sums.append(reduced_sum)
        
        # Step 7: Calculate geometric mean and take log (paper equation)
        # D̃ = log(∏(D'_i)^(1/500)) = (1/500) * Σ log(D'_i)
        distances = np.array(distances)
        log_distances = np.log(distances)
        asymptotic_distance = np.mean(log_distances)  # This is the geometric mean in log space
        
        # Normalize reduced sums by subtracting mean for visualization
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
    
    def detect_order_chaos_transitions(self, asymptotic_distances, threshold=-15):
        """
        Detect transitions based on paper methodology
        
        From paper: "if the asymptotic distance D̃ is very negative (minimal value is −15 
        due to machine precision adjustment), it indicates the model is in the ordered phase"
        
        Adjusted threshold to -12 to be more sensitive than -10 used before
        """
        
        phases = []
        transitions = []
        
        for i, distance in enumerate(asymptotic_distances):
            if distance is None or np.isnan(distance):
                phases.append('unknown')
                continue
                
            # Order phase: distance close to machine precision threshold (-15)
            # Use -12 as threshold (closer to -15 than the previous -10)
            if distance <= threshold:
                current_phase = 'order'
            else:
                current_phase = 'chaos'
            
            phases.append(current_phase)
            
            # Detect transition
            if i > 0 and phases[i-1] != 'unknown' and phases[i-1] != current_phase:
                transitions.append({
                    'epoch': i + 1,  # Convert to 1-based epoch numbering
                    'transition': f"{phases[i-1]} -> {current_phase}",
                    'distance_before': asymptotic_distances[i-1] if i > 0 else None,
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