import numpy as np

class NeuralNetwork:
    def __init__(self, n_neurons, neuron_types, connectivity_sigma=0.2, weight_scale=10.0):
        """
        Manage synaptic connections, spatial topology, and plasticity.
        
        Args:
            n_neurons (int): Total neurons.
            neuron_types (np.array): 0=Excitatory, 1=Inhibitory.
            connectivity_sigma (float): Gaussian width for spatial connectivity (0.0-1.0).
            weight_scale (float): Scaling factor for weights.
        """
        self.n = n_neurons
        self.neuron_types = neuron_types # 0=Exc, 1=Inh
        
        # 1. Spatial Embedding (Random positions in 2D unit square)
        self.positions = np.random.rand(n_neurons, 2)
        
        # 2. Initialize Connectivity based on distance
        # Calculate pairwise distances
        # diff[i, j, :] = pos[i] - pos[j]
        # dist[i, j] = norm(diff)
        pos_i = self.positions[:, np.newaxis, :]
        pos_j = self.positions[np.newaxis, :, :]
        distances = np.linalg.norm(pos_i - pos_j, axis=2)
        
        # Probability P(i,j) = exp(-dist^2 / sigma^2)
        # We perform a random draw against this probability
        prob_matrix = np.exp(-(distances**2) / (2 * connectivity_sigma**2))
        
        # Remove self-connections probability
        np.fill_diagonal(prob_matrix, 0.0)
        
        # To keep total connections reasonable, we might scale prob
        # Let's say we want ~10% connectivity overall on average if sigma is large
        # If sigma is small, it's local. We'll just trust the sigma.
        
        self.connectivity = np.random.rand(n_neurons, n_neurons) < prob_matrix
        
        # 3. Initialize Weights obeying Dale's Principle
        # Weights are stored as POSITIVE magnitudes for easier STDP math, 
        # but we apply signs during current calculation.
        # Actually, let's store them with signs to be explicit, but STDP needs care.
        # Decision: Store raw signed weights.
        
        self.weights = np.random.rand(n_neurons, n_neurons) * weight_scale
        self.weights *= self.connectivity
        
        # Apply signs based on Pre-synaptic neuron type (Columns)
        # If neuron j is Inhibitory, column j should be negative.
        inh_indices = np.where(neuron_types == 1)[0]
        self.weights[:, inh_indices] *= -1.0
        
        # STDP parameters
        self.A_plus = 0.05    # Reduced learning rate
        self.A_minus = 0.06 
        self.tau_stdp = 20.0 
        self.w_max = weight_scale * 2.0 
        
        self.spike_trace = np.zeros(n_neurons)

    def compute_synaptic_currents(self, spiked_indices):
        """
        Compute I = W @ S. Parallelized.
        """
        input_current = np.zeros(self.n)
        if np.any(spiked_indices):
            spike_vec = np.zeros(self.n)
            spike_vec[spiked_indices] = 1.0
            input_current = self.weights @ spike_vec
            
        return input_current

    def update_stdp(self, spiked_indices, dt=1.0):
        """
        Apply STDP. Enforce Dale's Principle:
        - Excitatory weights (positive) can grow/shrink but stay >= 0.
        - Inhibitory weights (negative) can grow/shrink (magnitude) but stay <= 0.
        
        Actually, standard STDP is often only applied to Excitatory synapses.
        Plasticity on Inhibition is more complex (iSTDP). 
        For simplicity Phase 2.5: ONLY Excitatory connections are plastic.
        Inhibitory connections are fixed.
        """
        # Decay traces
        self.spike_trace *= np.exp(-dt / self.tau_stdp)
        
        # Identify Excitatory Pre-synaptic neurons (Columns that are plastic)
        exc_indices = np.where(self.neuron_types == 0)[0]
        
        if np.any(spiked_indices):
            
            # LTP: Pre(Exc) -> Post(Any)
            # If Pre trace is high and Post spikes NOW -> Increase weight
            # We only modify columns j where j is Excitatory
            
            # spiked_indices are the POST neurons i that spiked NOW
            # self.spike_trace has the PRE neurons j activity
            
            # We want to add A_plus * trace[j] to weights[i, j]
            # But only for j in exc_indices
            
            # Create a mask for plastic weights: [i in spiked, j in exc]
            # This is complex to vectorize efficiently without touching everything.
            # Using outer product:
            
            # Post spikes (i) vector:
            post_spikes_vec = np.zeros(self.n)
            post_spikes_vec[spiked_indices] = 1.0
            
            # Pre trace (j) masked by Excitatory types:
            pre_trace_masked = self.spike_trace.copy()
            pre_trace_masked[self.neuron_types == 1] = 0.0 # Zero out traces from I neurons
            
            delta_w_ltp = self.A_plus * np.outer(post_spikes_vec, pre_trace_masked)
            
            # LTD: Post (old trace) -> Pre (Exc) spikes NOW
            # If Post trace is high and Pre (Exc) spikes NOW -> Decrease weight
            
            # Post trace (i):
            post_trace_vec = self.spike_trace.copy() # i
            
            # Pre spikes (j) vector (only Exc):
            pre_spikes_vec = np.zeros(self.n)
            # Intersection of spiked_indices and exc_indices
            spiked_exc = np.intersect1d(spiked_indices, exc_indices)
            pre_spikes_vec[spiked_exc] = 1.0
            
            delta_w_ltd = -self.A_minus * np.outer(post_trace_vec, pre_spikes_vec)
            
            # Apply changes masked by existing connectivity
            total_delta = (delta_w_ltp + delta_w_ltd) * self.connectivity
            
            # Update weights
            self.weights += total_delta
            
            # Clip weights for Excitatory to [0, w_max]
            # We only modified Excitatory columns, so we can clamp them.
            # But simpler: Loop over Exc columns or just clip all positive weights
            # Since I columns are negative and we didn't touch them (traces masked), 
            # they shouldn't change. But let's be safe.
            
            # Enforce 0 lower bound for Exc weights
            # And -w_max lower bound for Inh weights? No, they are fixed.
            
            # Logic check: We only added delta to Exc columns (pre_trace_masked had 0 for I, pre_spikes_vec had 0 for I)
            # So only Exc columns changed.
            # Just clamp Exc columns to [0, w_max]
            
            self.weights[:, exc_indices] = np.clip(self.weights[:, exc_indices], 0, self.w_max)

            # Update trace
            self.spike_trace[spiked_indices] += 1.0

    def synaptic_scaling(self, target_sum=15.0):
        """
        Homeostatic scaling: Normalize incoming Excitatory weights for each neuron 
        so the sum of weights equals target_sum.
        Prevents runaway excitation.
        """
        # Only scale Excitatory inputs (columns where type == 0)
        exc_indices = np.where(self.neuron_types == 0)[0]
        
        # Sum of Exc weights per row (per post-synaptic neuron)
        # axis=1 means sum over columns
        current_sums = np.sum(self.weights[:, exc_indices], axis=1)
        
        # Avoid division by zero
        current_sums[current_sums < 0.001] = 1.0
        
        # Compute scaling factors
        scale_factors = target_sum / current_sums
        
        # Apply scaling row-wise to Excitatory columns only
        # self.weights[i, exc] *= scale_factors[i]
        # Reshape scale_factors for broadcasting: (N, 1) * (N, n_exc) ? No
        # (N, n_exc) = (N, n_exc) * (N, 1)
        
        self.weights[:, exc_indices] *= scale_factors[:, np.newaxis]
