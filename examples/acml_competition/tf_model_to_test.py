import tensorflow as tf

##### load your own model trained on cifar10
# here we just take random weights 
model = tf.keras.applications.resnet50.ResNet50(weights=None, input_shape=(32, 32, 3), classes=10)

def predict(X):
    """run inference on a batch"""
    return model(X)
