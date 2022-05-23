import tensorflow as tf 

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Conv2DTranspose, Reshape
from tensorflow.keras.models import Sequential

from tensorflow.keras.losses import categorical_crossentropy as loss_fn
from tensorflow.keras.metrics import categorical_accuracy as metric_fn


def create_model(n_classes):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(units=n_classes))
    return model

def create_autoencoder():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Dense(14*14))
    model.add(Reshape((14, 14, 1)))
    model.add(Conv2DTranspose(1, (3,3), strides=2, activation='relu', padding='same'))
    return model

# @tf.function
def train_step(model, x, y, optimizer):
    with tf.GradientTape() as tape:
        y_pred = model(x, training=True)
        loss_value = loss_fn(y, y_pred, from_logits=True)
        acc_value = metric_fn(y, y_pred)
        loss_value = tf.reduce_mean(loss_value)
        acc_value = tf.reduce_mean(acc_value)
        
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    return loss_value, acc_value

# @tf.function
def test_step(model, x, y):
    y_pred = model(x, training=False)
    loss_value = loss_fn(y, y_pred, from_logits=True)
    acc_value = metric_fn(y, y_pred)
    
    loss_value = tf.reduce_mean(loss_value)
    acc_value = tf.reduce_mean(acc_value)
    return loss_value, acc_value

class MyDenseLayer(tf.keras.layers.Layer):
    def __init__(self, num_outputs, dropout=0.5, name=''):
        super(MyDenseLayer, self).__init__(name=name)
        self.num_outputs = num_outputs
        self.bn_0 = tf.keras.layers.LayerNormalization(name='BN')
        self.bn_1 = tf.keras.layers.LayerNormalization(name='BN')
        self.drop = tf.keras.layers.Dropout(dropout, name='Dropout')
        self.layer_0 = tf.keras.layers.Dense(self.num_outputs, name='layer_0')
        self.layer_1 = tf.keras.layers.Dense(self.num_outputs, name='layer_1')
        
    def call(self, inputs):
        with tf.name_scope("First_Part") as scope0:
            x = self.layer_0(inputs)
            x = self.bn_0(x)
            x = self.drop(x)
        
        with tf.name_scope("Second_Part") as scope1:
            y = self.layer_1(inputs)
            y = self.bn_1(y)
        
        return x, y