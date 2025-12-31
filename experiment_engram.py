import numpy as np
import matplotlib.pyplot as plt
from simulation.simulation import Simulation
from analysis.topology import TopologyAnalyzer

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

def main():
    print("Running Engram Experiment...")
    
    # Setup
    sim = Simulation(n_neurons=200, duration_ms=8000, dt=1.0)
    
    # HACK: Boost plasticity moderately
    sim.network.A_plus = 0.08 
    sim.network.w_max = 10.0
    
    # HACK: Ensure Group A and B are connected?
    # Actually, with sigma=0.25 and random positions, they might be far apart.
    # Group A is 0-19, B is 20-39.
    # We can rely on indirect paths or force proximity, but boosting A_plus is first step.
    
    # Run
    sim.run(stimulus_func=engram_stimulus)
    
    res = sim.get_results()
    spike_times = res['spike_times']
    spike_indices = res['spike_indices']
    weights = res['weight_history'][-1]
    
    # Analysis: Check firing rates logic
    # We want to see if Group B fires during Recall (6000-8000)
    # Compare with Control Group (e.g., 40-60)
    
    recall_mask = (spike_times >= 6000) & (spike_times < 8000)
    recall_idx = spike_indices[recall_mask]
    
    count_A = np.sum((recall_idx >= 0) & (recall_idx < 20))
    count_B = np.sum((recall_idx >= 20) & (recall_idx < 40))
    count_Control = np.sum((recall_idx >= 40) & (recall_idx < 60))
    
    print(f"Recall Phase Firing Counts:")
    print(f"Group A (Stimulated): {count_A}")
    print(f"Group B (Memory Target): {count_B}")
    print(f"Group C (Control): {count_Control}")
    
    if count_B > count_Control * 1.5:
        print("SUCCESS: Associative Memory (Engram) detected! B fired significantly more than Control.")
    else:
        print("FAILURE: No significant memory association.")

    # Small-World Analysis
    analyzer = TopologyAnalyzer(res['positions'])
    sigma_start = analyzer.compute_small_worldness(res['weight_history'][0], threshold=0.1)
    sigma_end = analyzer.compute_small_worldness(res['weight_history'][-1], threshold=0.1)
    
    print(f"Small-World Sigma: {sigma_start:.2f} (Start) -> {sigma_end:.2f} (End)")
    
    # Plot
    plt.figure(figsize=(12, 6))
    
    exc_mask = np.isin(spike_indices, np.where(sim.neuron_types==0)[0])
    inh_mask = ~exc_mask
    plt.plot(spike_times[exc_mask], spike_indices[exc_mask], '.b', markersize=2, alpha=0.3)
    plt.plot(spike_times[inh_mask], spike_indices[inh_mask], '.r', markersize=2, alpha=0.3)
    
    # Highlight Groups
    plt.axhspan(0, 20, color='green', alpha=0.1, label='Group A')
    plt.axhspan(20, 40, color='orange', alpha=0.1, label='Group B')
    
    # Highlight Phases
    plt.axvspan(0, 3000, color='gray', alpha=0.1, label='Training (A+B)')
    plt.axvspan(3000, 6000, color='white', alpha=0.0)
    plt.axvspan(6000, 8000, color='yellow', alpha=0.1, label='Recall (A only)')
    
    plt.title(f"Engram Experiment (B spikes={count_B} vs Control={count_Control})")
    plt.legend(loc='upper right')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron Index')
    plt.tight_layout()
    plt.savefig('viz/engram_result.png')
    print("Saved viz/engram_result.png")

if __name__ == "__main__":
    main()
