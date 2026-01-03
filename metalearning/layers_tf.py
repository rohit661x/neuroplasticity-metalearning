import tensorflow as tf
import numpy as np

class PlasticDense(tf.keras.layers.Layer):
    def __init__(self, units, activation=None, 
                 connectivity_sigma=0.3, 
                 pruning_rate=0.05, 
                 regrowth_rate=0.05,
                 **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.connectivity_sigma = connectivity_sigma
        self.pruning_rate = pruning_rate
        self.regrowth_rate = regrowth_rate
        
    def build(self, input_shape):
        dim_input = input_shape[-1]
        self.dim_input = dim_input
        self.dim_output = self.units
        
        # Weights
        self.kernel = self.add_weight(
            shape=(dim_input, self.units),
            initializer="glorot_uniform",
            trainable=True,
            name="kernel"
        )
        self.bias = self.add_weight(
            shape=(self.units,),
            initializer="zeros",
            trainable=True,
            name="bias"
        )
        
        # Mask (Non-trainable state)
        self.mask = self.add_weight(
            shape=(dim_input, self.units),
            initializer="ones",
            trainable=False,
            dtype=tf.bool,
            name="mask"
        )
        
        # Topology Precomputation (Virtual 2D)
        # Input: Grid
        side_in = int(np.ceil(np.sqrt(dim_input)))
        grid_x, grid_y = np.meshgrid(
            np.linspace(0, 1, side_in), 
            np.linspace(0, 1, side_in)
        )
        self.pos_input = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)[:dim_input]
        
        # Hidden: Random (Numpy)
        self.pos_hidden = np.random.uniform(0, 1, (self.units, 2)).astype(np.float32)
        
        # Distances: (Input, Hidden) in Numpy
        p_i = np.expand_dims(self.pos_input, 1)  # (In, 1, 2)
        p_h = np.expand_dims(self.pos_hidden, 0) # (1, Out, 2)
        
        # Compute proper distance matrix (Euclidean)
        dist_init = np.linalg.norm(p_i - p_h, axis=2).astype(np.float32)
        
        self.dist_matrix = self.add_weight(
            name="dist_matrix",
            shape=(dim_input, self.units),
            initializer=tf.constant_initializer(dist_init), 
            trainable=False
        )
        
        super().build(input_shape)

    def call(self, inputs):
        # Enforce mask in forward pass
        masked_kernel = self.kernel * tf.cast(self.mask, self.kernel.dtype)
        output = tf.matmul(inputs, masked_kernel) + self.bias
        if self.activation:
            output = self.activation(output)
        return output
        
    def plasticity_step(self):
        """Perform pruning and regrowth on this layer."""
        # --- PRUNING ---
        # Look at masked_kernel magnitude
        eff_kernel = self.kernel * tf.cast(self.mask, self.kernel.dtype)
        w_abs = tf.abs(eff_kernel)
        
        # Get active weights
        active_weights = tf.boolean_mask(w_abs, self.mask)
        n_active = tf.size(active_weights)
        if n_active == 0: return

        # Calculate threshold
        k = tf.cast(tf.cast(n_active, tf.float32) * self.pruning_rate, tf.int32)
        if k > 0:
            threshold = tf.sort(active_weights)[k]
            # Prune weights below threshold inside the current mask
            msg_prune = (w_abs < threshold) & self.mask
            
            # Update Mask
            self.mask.assign(self.mask & tf.logical_not(msg_prune))
            
            # Zero out weights explicitly (optional since call() masks, but good for optimizer)
            self.kernel.assign(self.kernel * tf.cast(self.mask, self.kernel.dtype))
            
        # --- REGROWTH ---
        total_params = self.dim_input * self.dim_output
        n_regrow = int(total_params * self.regrowth_rate)
        
        zeros_mask = tf.logical_not(self.mask)
        zeros_indices = tf.where(zeros_mask)
        n_zeros = tf.shape(zeros_indices)[0]
        
        if n_zeros > 0 and n_regrow > 0:
            # Shuffle candidates
            shuffled_indices = tf.random.shuffle(zeros_indices)
            n_check = min(n_zeros, n_regrow * 3)
            candidates = shuffled_indices[:n_check]
            
            # Probabilities based on distance
            dists = tf.gather_nd(self.dist_matrix, candidates)
            probs = tf.exp(-(dists**2) / (2 * self.connectivity_sigma**2))
            
            # Accept
            accept = tf.random.uniform(tf.shape(probs)) < probs
            final_indices = tf.boolean_mask(candidates, accept)[:n_regrow]
            
            if tf.shape(final_indices)[0] > 0:
                # Enable in mask
                # Need to do scatter update on mask variable
                # tf.Variable doesn't support direct scatter boolean?
                # Cast to int, scatter, cast back
                mask_int = tf.cast(self.mask, tf.int32)
                mask_int = tf.tensor_scatter_nd_update(
                    mask_int, final_indices, tf.ones(tf.shape(final_indices)[0], dtype=tf.int32)
                )
                self.mask.assign(tf.cast(mask_int, tf.bool))
                
                # Initialize new weights
                new_vals = tf.random.normal([tf.shape(final_indices)[0]]) * 0.01
                # Use functional update + assign for safety
                kernel_updated = tf.tensor_scatter_nd_update(self.kernel, final_indices, new_vals)
                self.kernel.assign(kernel_updated)

    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "activation": tf.keras.activations.serialize(self.activation),
            "connectivity_sigma": self.connectivity_sigma,
            "pruning_rate": self.pruning_rate,
            "regrowth_rate": self.regrowth_rate
        })
        return config
