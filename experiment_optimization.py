import numpy as np
import matplotlib.pyplot as plt
from simulation.simulation import Simulation
from analysis.topology import TopologyAnalyzer
import time

def engram_stimulus(step, n_neurons):
    """
    Protocol:
    0-3000ms: Training (Stim A + Stim B)
    3000-6000ms: Rest
    6000-8000ms: Recall (Stim A only)
    """
    current_input = np.zeros(n_neurons)
    
    # 0-20: Group A
    # 20-40: Group B
    group_A = np.arange(0, 20)
    group_B = np.arange(20, 40)
    
    # Training: A + B
    if 0 <= step < 3000:
        # Strong Poisson-like kick
        if np.random.rand() < 0.1:
            current_input[group_A] = 20.0
            current_input[group_B] = 20.0
            
    # Recall: A only
    elif 6000 <= step < 8000:
        if np.random.rand() < 0.2: # Higher frequency
            current_input[group_A] = 30.0 # Stronger drive
            
    return current_input

def run_trial(sigma, a_plus, duration=8000):
    sim = Simulation(n_neurons=100, duration_ms=duration, dt=1.0)
    
    # Override parameters
    sim.network.connectivity_sigma = sigma
    sim.network.A_plus = a_plus
    sim.network.w_max = 10.0
    
    # Re-initialize weights with new sigma
    # (Note: Simulation.__init__ already called NeuralNetwork, so we need to rebuild it or re-init weights)
    # Re-building is safer
    from simulation.network import NeuralNetwork
    sim.network = NeuralNetwork(sim.n, neuron_types=sim.neuron_types, connectivity_sigma=sigma, weight_scale=5.0)
    # Apply learning rate
    sim.network.A_plus = a_plus
    
    # Run
    sim.run(stimulus_func=engram_stimulus)
    
    # Score
    res = sim.get_results()
    spike_times = res['spike_times']
    spike_indices = res['spike_indices']
    
    recall_mask = (spike_times >= 6000) & (spike_times < 8000)
    recall_idx = spike_indices[recall_mask]
    
    count_B = np.sum((recall_idx >= 20) & (recall_idx < 40))
    count_Control = np.sum((recall_idx >= 40) & (recall_idx < 60))
    
    score = count_B / (count_Control + 1.0) # Avoid div bu zero
    
    # Topology extraction
    analyzer = TopologyAnalyzer(res['positions'])
    topo_metrics = analyzer.compute_graph_metrics(res['weight_history'][-1], threshold=0.5)
    
    return score, topo_metrics

def main():
    print("Starting Optimization Sweep...")
    
    sigmas = [0.1, 0.25, 0.4, 0.6]
    a_pluses = [0.01, 0.05, 0.08, 0.15]
    
    best_score = -1
    best_params = None
    best_metrics = None
    
    results = []
    
    for sigma in sigmas:
        for a_plus in a_pluses:
            print(f"Testing Sigma={sigma}, A_plus={a_plus}...")
            try:
                score, metrics = run_trial(sigma, a_plus)
                print(f"  -> Score: {score:.2f} (Modularity: {metrics.get('modularity', 0):.2f})")
                
                results.append((sigma, a_plus, score))
                
                if score > best_score:
                    best_score = score
                    best_params = (sigma, a_plus)
                    best_metrics = metrics
            except Exception as e:
                print(f"  -> Failed: {e}")

    print("\n--- Optimization Complete ---")
    print(f"Best Score: {best_score:.2f}")
    print(f"Best Params: Sigma={best_params[0]}, A_plus={best_params[1]}")
    print(f"Associated Metrics: {best_metrics}")
    
    # Save optimized constants to a file
    with open('neuroplasticity_constant.txt', 'w') as f:
        f.write(f"OPTIMAL_SIGMA={best_params[0]}\n")
        f.write(f"OPTIMAL_A_PLUS={best_params[1]}\n")
        f.write(f"MODULARITY={best_metrics.get('modularity', 0)}\n")
        f.write(f"CLUSTERING={best_metrics.get('clustering', 0)}\n")

if __name__ == "__main__":
    main()
