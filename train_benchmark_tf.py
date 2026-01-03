import tensorflow as tf
from metalearning.model_tf import SimpleMLP
from metalearning.layers_tf import PlasticDense
import matplotlib.pyplot as plt
import numpy as np
import os

def main():
    # Settings
    batch_size = 64
    epochs = 5
    
    # Load Data
    print("Loading MNIST...")
    try:
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
    except Exception as e:
        print(f"Failed to load MNIST: {e}. Using Fake Data.")
        x_train = np.random.randn(1000, 28, 28).astype('float32')
        y_train = np.random.randint(0, 10, size=(1000,))
        x_test = np.random.randn(200, 28, 28).astype('float32')
        y_test = np.random.randint(0, 10, size=(200,))

    # Prepare datasets
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(batch_size)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(1000)

    # --- Baseline Training ---
    print("\n--- Baseline Training ---")
    model_base = SimpleMLP()
    # Note: Baseline is same class but we won't trigger plasticity, so it's effectively a dense net
    # (Actually PlasticDense starts dense, so if we don't call plasticity_step, it stays dense)
    model_base.compile(optimizer='adam',
                       loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                       metrics=['accuracy'])
    
    history_base = model_base.fit(train_ds, epochs=epochs, validation_data=test_ds)
    acc_base = history_base.history['val_accuracy']
    
    # --- Bio-Regularized Training ---
    print("\n--- Bio-Regularized Training ---")
    model_bio = SimpleMLP()
    model_bio.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
    
    # Orchestrator Callback
    class PlasticityOrchestrator(tf.keras.callbacks.Callback):
        def __init__(self, step_interval=100):
            self.step_interval = step_interval
            self.batch_count = 0
            
        def on_train_batch_end(self, batch, logs=None):
            self.batch_count += 1
            if self.batch_count % self.step_interval == 0:
                # Trigger plasticity on all PlasticDense layers
                self._trigger_plasticity()
        
        def _trigger_plasticity(self):
            for layer in self.model.layers:
                # Check directly or via recursion (SimpleMLP is flat-ish)
                if isinstance(layer, PlasticDense):
                    layer.plasticity_step()

    # Sparsity Logger
    final_sparsity = 0.0
    class SparsityLogger(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            nonlocal final_sparsity
            for layer in self.model.layers:
                if isinstance(layer, PlasticDense):
                    # Sparsity = 1.0 - (sum(mask) / size)
                    flattened_mask = tf.reshape(layer.mask, [-1])
                    n_active = tf.reduce_sum(tf.cast(flattened_mask, tf.float32))
                    total = tf.cast(tf.size(flattened_mask), tf.float32)
                    sparsity = 1.0 - (n_active / total)
                    print(f" - Sparsity {layer.name}: {sparsity:.4f}")
                    final_sparsity = sparsity.numpy()

    # Pruning strong: 0.1, Regrowth weak: 0.05 => Net sparsification
    # We must configure layers? SimpleMLP constructor sets defaults.
    # To override, we would need to access layers after build or pass args.
    # SimpleMLP uses Hardcoded rates in __init__? 
    # Ah, I added kwargs to PlasticDense but SimpleMLP hardcodes them?
    # Let's rebuild SimpleMLP to allow overrides or just accept defaults. 
    # PlasticDense defaults: Prune 0.05, Regrow 0.05 (Equilibrium).
    # To demonstrate sparsity, we want Prune > Regrow.
    # We can modify the layer instances directly before training!
    
    # Build first to init layers
    model_bio.build((None, 28, 28))
    model_bio.fc1.pruning_rate = 0.1
    model_bio.fc1.regrowth_rate = 0.05
    
    history_bio = model_bio.fit(train_ds, epochs=epochs, validation_data=test_ds, 
                                callbacks=[PlasticityOrchestrator(step_interval=100), SparsityLogger()])
    acc_bio = history_bio.history['val_accuracy']

    # Validation
    if final_sparsity < 0.05:
        print(f"[WARNING] Sparsity is low ({final_sparsity:.4f}). Plasticity might not be working!")
    else:
        print(f"[SUCCESS] Final Sparsity achieved: {final_sparsity:.4f}")

    # --- Plotting ---
    if not os.path.exists('viz'):
        os.makedirs('viz')
        
    plt.figure()
    plt.plot(range(1, epochs+1), acc_base, label='Baseline (TF)')
    plt.plot(range(1, epochs+1), acc_bio, label='Bio-Regularized (TF)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Meta-Learning Benchmark (TensorFlow)')
    plt.savefig('viz/benchmark_result_tf.png')
    print("Saved viz/benchmark_result_tf.png")

if __name__ == "__main__":
    main()
