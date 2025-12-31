import numpy as np
from .neuron import IzhikevichNeuron
from .network import NeuralNetwork

class Simulation:
    def __init__(self, n_neurons=100, duration_ms=1000, dt=0.5):
        self.n = n_neurons
        self.duration = duration_ms
        self.dt = dt
        self.steps = int(duration_ms / dt)
        
        # 1. Generate Neuron Types (80% Exc, 20% Inh)
        # 0 = Excitatory, 1 = Inhibitory
        n_inh = int(0.2 * n_neurons)
        n_exc = n_neurons - n_inh
        self.neuron_types = np.array([0]*n_exc + [1]*n_inh)
        np.random.shuffle(self.neuron_types)
        
        # 2. Initialize Components
        self.neurons = IzhikevichNeuron(n_neurons, neuron_types=self.neuron_types)
        self.network = NeuralNetwork(n_neurons, neuron_types=self.neuron_types, connectivity_sigma=0.25, weight_scale=5.0)
        
        # Data recording
        self.spike_times = []
        self.spike_indices = []
        self.weight_history = [] 
        
    def run(self, stimulus_func=None):
        """
        Run the simulation.
        
        Args:
            stimulus_func (callable, optional): function(step, n_neurons) -> np.array of currents.
        """
        print(f"Starting simulation for {self.duration}ms (dt={self.dt}ms)...")
        
        curr_time = 0
        
        # Save initial weights
        self.weight_history.append(self.network.weights.copy())
        
        for step in range(self.steps):
            # Thalamic noise (Reduced)
            noise = np.random.randn(self.n) * 2.0 
            if step < 200: # Kickstart longer but gentler
                noise += 5.0
            
            # External Stimulus
            ext_stim = np.zeros(self.n)
            if stimulus_func:
                ext_stim = stimulus_func(step, self.n)
                
            # Synaptic inputs
            last_spikes = self.neurons.spiked
            synaptic_input = self.network.compute_synaptic_currents(last_spikes)
            
            total_input = noise + synaptic_input + ext_stim
            
            # Step Neurons
            spiked_bool = self.neurons.step(total_input, self.dt)
            
            # Plasticity
            self.network.update_stdp(spiked_bool, self.dt)
            
            # Homeostasis (Synaptic Scaling) more frequent
            if step % 50 == 0 and step > 0:
                self.network.synaptic_scaling()
            
            # Record spikes
            if np.any(spiked_bool):
                spike_idxs = np.where(spiked_bool)[0]
                self.spike_indices.extend(spike_idxs)
                self.spike_times.extend([curr_time] * len(spike_idxs))
            
            curr_time += self.dt
            
            # Record weights every 1 sec
            if step % 1000 == 0 and step > 0:
                self.weight_history.append(self.network.weights.copy())
        
        self.weight_history.append(self.network.weights.copy())
        print("Simulation complete.")

    def get_results(self):
        return {
            "spike_times": np.array(self.spike_times),
            "spike_indices": np.array(self.spike_indices),
            "weight_history": self.weight_history,
            "positions": self.network.positions,
            "neuron_types": self.neuron_types
        }
