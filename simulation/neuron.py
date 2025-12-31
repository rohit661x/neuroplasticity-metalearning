import numpy as np

class IzhikevichNeuron:
    def __init__(self, n_neurons, neuron_types=None):
        """
        Initialize a population of Izhikevich neurons.
        
        Args:
            n_neurons (int): Number of neurons in the population.
            neuron_types (np.array): Array of ints (0=Excitatory, 1=Inhibitory). 
                                     If None, all are assummed Excitatory.
        """
        self.n = n_neurons
        
        # Default all to Excitatory if types not provided
        if neuron_types is None:
            neuron_types = np.zeros(n_neurons, dtype=int)
        
        self.neuron_types = neuron_types
        
        # Initialize parameters vectors
        self.a = np.zeros(n_neurons)
        self.b = np.zeros(n_neurons)
        self.c = np.zeros(n_neurons)
        self.d = np.zeros(n_neurons)
        
        # Randomness factor for heterogeneity
        re = np.random.rand(n_neurons)
        
        # Set parameters based on type
        # 0 = Excitatory (Regular Spiking - RS)
        # a=0.02, b=0.2, c=-65, d=8
        exc_mask = (neuron_types == 0)
        self.a[exc_mask] = 0.02
        self.b[exc_mask] = 0.2
        self.c[exc_mask] = -65.0 + 15.0 * re[exc_mask]**2
        self.d[exc_mask] = 8.0 - 6.0 * re[exc_mask]**2
        
        # 1 = Inhibitory (Fast Spiking - FS)
        # a=0.1, b=0.2, c=-65, d=2
        inh_mask = (neuron_types == 1)
        self.a[inh_mask] = 0.1
        self.b[inh_mask] = 0.2
        self.c[inh_mask] = -65.0
        self.d[inh_mask] = 2.0
        
        # State variables
        self.v = -65.0 * np.ones(n_neurons)
        self.u = self.b * self.v             # Recovery variable
        
        # Spike history for the current step
        self.spiked = np.zeros(n_neurons, dtype=bool)

    def step(self, i_accumulated, dt=1.0):
        """
        Advance the simulation by one time step dt (ms).
        
        Args:
            i_accumulated (np.array): Input current for each neuron.
            dt (float): Time step size in ms (default 1.0).
        """
        # Izhikevich dynamics (Euler integration)
        # v dot = 0.04*v^2 + 5*v + 140 - u + I
        # u dot = a * (b*v - u)
        
        v = self.v
        u = self.u
        
        self.v += dt * (0.04 * v**2 + 5 * v + 140 - u + i_accumulated)
        self.u += dt * (self.a * (self.b * v - u))
        
        # Check for spikes
        self.spiked = self.v >= 30.0
        
        # Reset after spike
        self.v[self.spiked] = self.c[self.spiked]
        self.u[self.spiked] += self.d[self.spiked]

        return self.spiked
