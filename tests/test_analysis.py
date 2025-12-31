import pytest
import numpy as np
import networkx as nx
from analysis.topology import TopologyAnalyzer
from analysis.plasticity_metrics import derive_constant, exponential_decay

def test_graph_metrics():
    """Test that graph metrics are computed correctly."""
    # Simple Ring graph: 0-1-2-0
    # Weights are distances? No, weights are adjacency.
    adj = np.array([
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0]
    ])
    
    positions = np.zeros((3, 2))
    analyzer = TopologyAnalyzer(positions)
    
    metrics = analyzer.compute_graph_metrics(adj, threshold=0.5)
    
    # Fully connected triangle has clustering coeff 1.0
    assert metrics['clustering'] == 1.0
    assert metrics['global_efficiency'] > 0

def test_betti_numbers():
    """Test that TDA finds a loop."""
    # A discrete loop of 4 points
    # 0-1, 1-2, 2-3, 3-0
    # If weights are strong, distance is small.
    # Distances:
    # 0  1  2  3
    # 0 [. . .]
    
    # We pass weights.
    # Strong weights 0-1, 1-2, 2-3, 3-0
    weights = np.zeros((4, 4))
    weights[0, 1] = 10; weights[1, 0] = 10
    weights[1, 2] = 10; weights[2, 1] = 10
    weights[2, 3] = 10; weights[3, 2] = 10
    weights[3, 0] = 10; weights[0, 3] = 10
    
    positions = np.zeros((4, 2))
    analyzer = TopologyAnalyzer(positions)
    
    res = analyzer.compute_persistence(weights)
    
    # Should have 1 connected component (H0=1) and 1 loop (H1=1)
    # Note: Ripser might return H0 as N points, but infinite survival is 1.
    # Our simple counter counts features > threshold.
    # For a connected ring, we expect 1 loop.
    assert res['betti_1'] >= 1

def test_constant_derivation():
    """Test that we can recover an exponential rate constant."""
    t = np.linspace(0, 10, 20)
    k_true = 0.5
    y = exponential_decay(t, A=10, K=k_true, C=5)
    
    res = derive_constant(t, y)
    
    assert np.isclose(res['k'], k_true, rtol=0.1)
    assert res['r2'] > 0.95
