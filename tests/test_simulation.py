import pytest
import numpy as np
from simulation.neuron import IzhikevichNeuron
from simulation.network import NeuralNetwork

def test_neuron_spiking():
    """Test that neurons spike when injected with high current."""
    n = 10
    # All Excitatory
    types = np.zeros(n, dtype=int)
    neurons = IzhikevichNeuron(n, neuron_types=types)
    
    # Run for 100 ms with high input
    spikes_detected = False
    for _ in range(100):
        # Inject huge current to force spiking
        spikes = neurons.step(np.full(n, 100.0))
        if np.any(spikes):
            spikes_detected = True
            break
            
    assert spikes_detected, "Neurons should spike with high input current."

def test_stdp_learning():
    """Test that STDP changes Excitatory weights."""
    n = 2
    types = np.array([0, 0]) # Both Excitatory
    # Use huge sigma to ensure connection
    net = NeuralNetwork(n, neuron_types=types, connectivity_sigma=100.0)
    
    # Ensure connection 0->1 exists
    # Force connection if random draw failed (though with sigma=100 it shouldn't)
    net.connectivity[1, 0] = True
    net.weights[1, 0] = 5.0
    
    initial_weight = net.weights[1, 0]
    
    # Trigger LTP: 0 spikes then 1 spikes
    
    # Step 1: 0 spikes
    spikes = np.array([True, False])
    net.update_stdp(spikes, dt=1.0)
    
    # Step 2: 1 spikes
    spikes = np.array([False, True])
    net.update_stdp(spikes, dt=1.0)
    
    # Check if weight increased (LTP)
    assert net.weights[1, 0] > initial_weight, "Excitatory weight should increase after LTP event."

def test_dales_principle():
    """Test that Inhibitory neurons have negative weights."""
    n = 10
    # Half Exc, Half Inh
    types = np.array([0]*5 + [1]*5)
    net = NeuralNetwork(n, neuron_types=types, connectivity_sigma=10.0)
    
    # Check weights from Inhibitory neurons (columns 5-9)
    inh_weights = net.weights[:, 5:]
    # Should be all <= 0
    assert np.all(inh_weights <= 0), "Inhibitory weights must be non-positive."
    
    # Check weights from Excitatory neurons (columns 0-4)
    exc_weights = net.weights[:, :5]
    # Should be all >= 0
    assert np.all(exc_weights >= 0), "Excitatory weights must be non-negative."
