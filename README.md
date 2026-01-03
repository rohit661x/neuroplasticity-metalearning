# Learning Compositional Generalization from Biological Plasticity Dynamics

> **Central Thesis: Structure is Learning.**
> Artificial Neural Networks typically start dense and learn weights. Biological brains start sparse and learn *structure* (topology). This project demonstrates that applying biological structural priors (derived from neuroplasticity simulations) to AI models leads to efficient, self-organizing sparsity without sacrificing accuracy.

## Overview

This project bridges Computational Neuroscience and Deep Learning. It consists of three components:
1.  **Biological Simulation**: A ground-up simulation of Izhikevich neurons, Spike-Timing-Dependent Plasticity (STDP), and Homeostatic Scaling to model how "engrams" (memories) form in biological tissue.
2.  **Meta-Learning Optimization**: A framework to scientifically derive the "Optimal Neuroplasticity Constant" (connectivity sigma, learning rates) from the biological simulation.
3.  **Dual-Stack AI Implementation**: A validation suite that enforces these derived biological rules in standard Deep Learning models using both **PyTorch** and **TensorFlow**.

## Key Results

-   **Derivation**: Successfully derived optimal `sigma` and `A_plus` parameters that maximize modularity and small-worldness in biological networks.
-   **Validation**: Implemented these rules in a custom `PlasticDense` layer for TensorFlow and a `StructuralPlasticityOptimizer` for PyTorch.
-   **Performance**: The bio-regularized AI models achieved **~52% sparsity** while maintaining **~98% accuracy** on MNIST, proving that the network can self-organize into an efficient topology.

## Repository Structure

-   `simulation/`: Pure Python implementation of the biological brain model (Neurons, STDP, Network).
-   `analysis/`: Tools for Topological Data Analysis (Betti numbers, Persistence) to measure network structure.
-   `metalearning/`:
    -   `layers_tf.py`: **TensorFlow** Custom Layer (`PlasticDense`) implementing self-contained structural plasticity.
    -   `model.py` / `regularizer.py`: **PyTorch** implementation.
-   `train_benchmark_tf.py`: TensorFlow benchmark script.
-   `train_benchmark.py`: PyTorch benchmark script.
-   `experiment_optimization.py`: The meta-learning loop to find optimal biological constants.

## Installation

```bash
pip install -r requirements.txt
```

*Requires Python 3.8+. Dependencies include `tensorflow`, `torch`, `numpy`, `matplotlib`, `brian2` (optional), `ripser` (for topology).*

## Usage

### 1. Run the Biological Simulation
Explore how plasticity parameters affect network topology.
```bash
python experiment_optimization.py
```

### 2. Run the AI Benchmark (TensorFlow)
Train a standard MLP vs. a Bio-Regularized MLP to see structural learning in action.
```bash
python train_benchmark_tf.py
```
*Check `viz/benchmark_result_tf.png` for results.*

### 3. Run the AI Benchmark (PyTorch)
```bash
python train_benchmark.py
```

## Citation
If you use this code for research in Neuro-AI or Structural Plasticity, please cite:
> Suryadevara, R. (2025). *Learning Compositional Generalization from Biological Plasticity Dynamics*.
