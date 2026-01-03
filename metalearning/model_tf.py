import tensorflow as tf

from metalearning.layers_tf import PlasticDense

class SimpleMLP(tf.keras.Model):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.flatten = tf.keras.layers.Flatten()
        # Use PlasticDense for the hidden layer
        self.fc1 = PlasticDense(128, activation='relu', connectivity_sigma=0.3)
        self.fc2 = tf.keras.layers.Dense(10)
        
    def call(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        return self.fc2(x)
