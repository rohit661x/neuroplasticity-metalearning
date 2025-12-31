import numpy as np
import networkx as nx
from ripser import ripser
from persim import plot_diagrams
import matplotlib.pyplot as plt

class TopologyAnalyzer:
    def __init__(self, positions):
        """
        Args:
            positions (np.array): (N, 2) array of neuron coordinates.
        """
        self.positions = positions
        
    def compute_graph_metrics(self, weights, threshold=0.1):
        """
        Compute standard graph theory metrics.
        Args:
            weights (np.array): Weighted adjacency matrix.
            threshold (float): Cutoff to treat connection as binary edge.
        """
        # Create graph from adjacency matrix
        # Use absolute weights for graph structure
        adj = np.abs(weights)
        adj[adj < threshold] = 0
        
        G = nx.from_numpy_array(adj)
        
        metrics = {}
        if G.number_of_edges() > 0:
            metrics['global_efficiency'] = nx.global_efficiency(G)
            try:
                metrics['clustering'] = nx.average_clustering(G)
            except:
                metrics['clustering'] = 0
                
            # Community structure (Modularity)
            try:
                communities = nx.community.greedy_modularity_communities(G)
                metrics['modularity'] = nx.community.modularity(G, communities)
            except:
                metrics['modularity'] = 0
        else:
            metrics['global_efficiency'] = 0
            metrics['clustering'] = 0
            metrics['modularity'] = 0
            
        return metrics

    def compute_small_worldness(self, weights, threshold=0.1):
        """
        Compute Small-World Sigma = (C/C_rand) / (L/L_rand).
        """
        adj = np.abs(weights)
        adj[adj < threshold] = 0
        G = nx.from_numpy_array(adj)
        
        # Must be connected for Path Length
        if not nx.is_connected(G):
            # Take largest component
            G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
            
        n = G.number_of_nodes()
        m = G.number_of_edges()
        
        if n < 4 or m < 4: return 0.0
        
        # 1. Real Metrics
        try:
            C = nx.average_clustering(G)
            L = nx.average_shortest_path_length(G)
        except:
            return 0.0
            
        # 2. Random Equivalents (Erdos-Renyi with same N, M)
        # Approximate C_rand ~ <k>/N and L_rand ~ ln(N)/ln(<k>)
        k_avg = 2 * m / n
        
        if k_avg <= 1: return 0.0
        
        C_rand_est = k_avg / n
        L_rand_est = np.log(n) / np.log(k_avg)
        
        # Avoid division by zero
        if C_rand_est == 0 or L_rand_est == 0: return 0.0
        
        sigma = (C / C_rand_est) / (L / L_rand_est)
        return sigma

    def compute_persistence(self, weights):
        """
        Compute persistent homology (Betti numbers) using Ripser.
        We interpret 'weights' as 'similarity'. Ripser expects 'distance'.
        So we transform weights to distance: dist = 1 - (weight / max_weight)
        or dist = 1/weight
        
        Actually, for checking "loops" in the functional network, 
        strong connection = short distance.
        """
        # Filter for Excitatory connections mostly? 
        # Or just take absolute magnitude.
        
        w_abs = np.abs(weights)
        max_w = np.max(w_abs)
        if max_w == 0:
            return {'betti_0': 0, 'betti_1': 0}
            
        # Transform connection strength to "distance"
        # Strong weight -> Small distance
        # 1.0 - (w / max)  (Simple inversion)
        # Add epsilon to avoid 0 distance?
        
        dist_matrix = 1.0 - (w_abs / max_w)
        np.fill_diagonal(dist_matrix, 0)
        
        # Ripser
        # dimension 1 = loops (Betti-1)
        # dimension 0 = connected components (Betti-0)
        result = ripser(dist_matrix, distance_matrix=True, maxdim=1)
        diagrams = result['dgms']
        
        # Summarize diagrams
        # Betti numbers are essentially the number of features that "persist" 
        # for a significant range.
        # Simple proxy: Count features with persistence > threshold
        
        # dgms[0] is H0 (components), dgms[1] is H1 (loops)
        
        # Lifetime threshold (e.g., 0.1)
        # Persistence = death - birth
        
        def count_features(dgm, threshold=0.1):
            if len(dgm) == 0: return 0
            # Infinity is typically max distance
            diffs = dgm[:, 1] - dgm[:, 0]
            # remove infinity (death = inf)
            finite_diffs = diffs[np.isfinite(diffs)]
            return np.sum(finite_diffs > threshold)
            
        b0 = count_features(diagrams[0])
        b1 = count_features(diagrams[1]) if len(diagrams) > 1 else 0
        
        return {
            'betti_0': b0,
            'betti_1': b1,
            'diagrams': diagrams
        }
