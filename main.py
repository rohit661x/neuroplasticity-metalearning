import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from simulation.simulation import Simulation

def main():
    # Parameters
    N = 200 # Increased neuron count for better spatial viz
    DURATION = 10000 # ms - Increased to allow plasticity to stabilize
    
    print(f"Initializing simulation with {N} neurons (Spatial + Inhibition)...")
    sim = Simulation(n_neurons=N, duration_ms=DURATION, dt=1.0)
    
    sim.run()
    
    results = sim.get_results()
    spike_times = results['spike_times']
    spike_indices = results['spike_indices']
    weight_history = results['weight_history']
    positions = results['positions']
    neuron_types = results['neuron_types']
    
    # ... (After simulation run)
    
    from analysis.topology import TopologyAnalyzer
    from analysis.plasticity_metrics import derive_constant
    
    print("Performing Topological Analysis...")
    analyzer = TopologyAnalyzer(positions)
    
    # Analyze evolution of metrics
    metrics_history = {
        'time': [],
        'modularity': [],
        'global_efficiency': [],
        'betti_1': []
    }
    
    # We recorded weights every 1 sec (1000 ms = 1000 steps if dt=1)
    # Actually wait, weight_history has snapshots.
    # Snapshots were taken every 1000 steps.
    
    snapshot_dt_ms = 1000 
    
    for i, weights in enumerate(weight_history):
        t = i * snapshot_dt_ms
        metrics_history['time'].append(t)
        
        # Graph Metrics
        g_metrics = analyzer.compute_graph_metrics(weights, threshold=0.2)
        metrics_history['modularity'].append(g_metrics['modularity'])
        metrics_history['global_efficiency'].append(g_metrics['global_efficiency'])
        
        # TDA (Betti numbers)
        # This can be slow, so maybe only do it if the matrix isn't huge
        tda_res = analyzer.compute_persistence(weights)
        metrics_history['betti_1'].append(tda_res['betti_1'])

    # Derive Constants
    print("Deriving Plasticity Constants...")
    time_arr = np.array(metrics_history['time'])
    
    mod_fit = derive_constant(time_arr, np.array(metrics_history['modularity']))
    eff_fit = derive_constant(time_arr, np.array(metrics_history['global_efficiency']))
    
    print(f"Modularity Rate Constant (k): {mod_fit['k']:.5f} (R2={mod_fit['r2']:.2f})")
    print(f"Efficiency Rate Constant (k): {eff_fit['k']:.5f} (R2={eff_fit['r2']:.2f})")
    
    # Visualization
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3)
    
    ax0 = fig.add_subplot(gs[0, 0]) # Raster
    ax1 = fig.add_subplot(gs[0, 1]) # Spatial Topology
    ax2 = fig.add_subplot(gs[0, 2]) # Weight Dist
    ax3 = fig.add_subplot(gs[1, :]) # Metrics Evolution
    
    # 1. Raster plot
    if len(spike_times) > 0:
        exc_mask = np.isin(spike_indices, np.where(neuron_types==0)[0])
        inh_mask = ~exc_mask
        ax0.plot(spike_times[exc_mask], spike_indices[exc_mask], '.b', markersize=2, label='Exc')
        ax0.plot(spike_times[inh_mask], spike_indices[inh_mask], '.r', markersize=2, label='Inh')
        ax0.set_title('Raster Plot')
        ax0.legend(loc='upper right')
        ax0.set_xlim([0, DURATION])
    else:
        ax0.text(0.5, 0.5, "No spikes recorded", ha='center')

    # 2. Spatial Network Topology
    final_weights = weight_history[-1]
    exc_weights = np.where(final_weights > 0, final_weights, 0)
    if exc_weights.max() > 0:
        threshold = np.percentile(exc_weights[exc_weights > 0], 90)
    else:
        threshold = 0.1
        
    exc_pos = positions[neuron_types == 0]
    inh_pos = positions[neuron_types == 1]
    ax1.scatter(exc_pos[:, 0], exc_pos[:, 1], c='b', s=20, alpha=0.6)
    ax1.scatter(inh_pos[:, 0], inh_pos[:, 1], c='r', s=20, alpha=0.6)
    
    rows, cols = np.where(exc_weights > threshold)
    count = 0
    for r, c in zip(rows, cols):
        if count > 500: break
        p1 = positions[r]
        p2 = positions[c]
        ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', alpha=0.1, linewidth=0.5)
        count += 1
    ax1.set_title('Spatial Connectivity')

    # 3. Weight Distribution
    w_flat = final_weights.flatten()
    w_flat = w_flat[w_flat != 0]
    ax2.hist(w_flat, bins=50, color='purple', alpha=0.7)
    ax2.set_title('Weight Distribution')
    
    # 4. Metrics Evolution
    t = metrics_history['time']
    ax3.plot(t, metrics_history['modularity'], 'g-o', label='Modularity')
    ax3.plot(t, metrics_history['global_efficiency'], 'b-o', label='Global Efficiency')
    # Scale Betti to fit?
    b1_norm = np.array(metrics_history['betti_1'])
    if b1_norm.max() > 0: b1_norm = b1_norm / b1_norm.max()
    ax3.plot(t, b1_norm, 'k--', label='Betti-1 (Normalized)')
    
    ax3.set_title(f"Topological Evolution (k_mod={mod_fit['k']:.4f})")
    ax3.set_xlabel('Time (ms)')
    ax3.legend()

    plt.tight_layout()
    plt.savefig('viz/simulation_result_phase3.png')
    print("Results saved to viz/simulation_result_phase3.png")

if __name__ == "__main__":
    main()
