import torch
import numpy as np

class StructuralPlasticityOptimizer:
    def __init__(self, model, pruning_rate=0.1, regrowth_rate=0.1):
        """
        Implements Structural Plasticity (Pruning + Regrowth) influenced by spatial topology.
        
        We assign virtual 2D coordinates to neurons.
        - Input neurons: Grid 28x28
        - Hidden neurons: Random or Grid
        
        Connectivity cost is proportional to distance.
        """
        self.model = model
        self.pruning_rate = pruning_rate
        self.regrowth_rate = regrowth_rate
        self.masks = {}
        
        # Initialize Masks (all 1s initially)
        for name, param in model.named_parameters():
            if 'weight' in name:
                self.masks[name] = torch.ones_like(param, dtype=torch.bool)
                
        # Assign Positions (Virtual Embedding)
        # Input: 28x28 grid
        self.pos_input = self._generate_grid_positions(28) # (784, 2)
        
        # Hidden: Random in same unit square
        hidden_dim = model.fc1.out_features
        self.pos_hidden = torch.rand(hidden_dim, 2)
        
        # Output: Random
        out_dim = model.fc2.out_features
        self.pos_output = torch.rand(out_dim, 2)
        
        # Precompute Distances for FC1 (784 -> 128)
        # weight shape (128, 784) -> (Hidden, Input)
        # We want Dist[h, i] = |pos_h - pos_i|
        p_h = self.pos_hidden.unsqueeze(1) # (128, 1, 2)
        p_i = self.pos_input.unsqueeze(0)  # (1, 784, 2)
        self.dist_matrix_fc1 = torch.norm(p_h - p_i, dim=2) # (128, 784)
        
        # Max distance for normalization
        self.max_dist = np.sqrt(2)

    def _generate_grid_positions(self, side):
        x = torch.linspace(0, 1, side)
        y = torch.linspace(0, 1, side)
        grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
        return torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)

    def step(self):
        """
        Apply pruning and regrowth.
        1. Prune weakest weights.
        2. Regrow weights, preferring short connections (Low Distance).
        """
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name not in self.masks: continue
                
                # --- PRUNING ---
                # Prune bottom k% of *active* weights magnitude
                weight_abs = param.abs()
                current_mask = self.masks[name]
                active_weights = weight_abs[current_mask]
                
                if len(active_weights) == 0: continue
                
                # Dynamic threshold
                k = int(len(active_weights) * self.pruning_rate)
                if k > 0:
                    threshold = torch.topk(active_weights, k, largest=False).values.max()
                    # Update mask: Kill weights below threshold
                    # But don't kill previously dead weights (already 0)
                    new_prune = (weight_abs < threshold) & current_mask
                    self.masks[name][new_prune] = False
                    
                    # Zero out pruned weights
                    param.data[new_prune] = 0.0

                # --- REGROWTH ---
                # Regrow k connections.
                # Prioritize: Low Distance?
                # Regrowth Probability P ~ exp(-Dist^2 / sigma) (Like our sim!)
                
                if 'fc1.weight' in name:
                    dist_mat = self.dist_matrix_fc1.to(param.device)
                    # Prob of regrowth
                    # Only calculate for inactive weights? Too expensive to check all.
                    # Just sample random indices and accept based on probability.
                    
                    # Number of spots to regrow = k
                    n_regrow = k
                    
                    # Random candidates (indices of inactive)
                    # This is slow if we do it exactly. 
                    # Heuristic: Pick random indices, if inactive, activate with prob(dist).
                    
                    zeros_idx = torch.nonzero(~self.masks[name]) # (N_zeros, 2)
                    if len(zeros_idx) > 0:
                        perm = torch.randperm(len(zeros_idx))
                        candidates = zeros_idx[perm[:n_regrow*2]] # Overselect
                        
                        # Calculate acceptance prob
                        dists = dist_mat[candidates[:, 0], candidates[:, 1]]
                        probs = torch.exp(-(dists**2) / (2 * 0.3**2)) # Sigma=0.3
                        
                        accept = torch.rand(len(probs)).to(param.device) < probs
                        final_regrow = candidates[accept][:n_regrow]
                        
                        # Activate
                        row, col = final_regrow[:, 0], final_regrow[:, 1]
                        self.masks[name][row, col] = True
                        # Initialize with small random value
                        param.data[row, col] = torch.randn(len(row)).to(param.device) * 0.01

    def apply_mask(self):
        """Ensure zero weights stay zero."""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.masks:
                    param.data *= self.masks[name].float()
