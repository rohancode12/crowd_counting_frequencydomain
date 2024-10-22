import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

def make_layers(cfg, use_batch_norm=True):
    '''
    Constructs VGG-like feature extraction layers.
    :param cfg: List specifying the layer configurations (conv layers, max pooling).
    :param use_batch_norm: Whether to use batch normalization.
    :return: TensorFlow Sequential model of layers.
    '''
    layers_list = []
    in_channels = 3  # Starting with 3 input channels (RGB)
    
    for v in cfg:
        if v == 'M':
            layers_list.append(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        else:
            layers_list.append(layers.Conv2D(v, (3, 3), padding='same'))
            if use_batch_norm:
                layers_list.append(layers.BatchNormalization())
            layers_list.append(layers.ReLU())
    
    return models.Sequential(layers_list)


# Configurations for VGG-like architecture
cfg = {
    'Baysian_Ma': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
    'Regular': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}


class VGG_Baysian_Ma(tf.keras.Model):
    '''
    VGG model used for Bayesian crowd counting loss (Ma).
    '''
    def __init__(self, features):
        super(VGG_Baysian_Ma, self).__init__()
        self.features = features
        self.reg_layer = models.Sequential([
            layers.Conv2D(256, (3, 3), padding='same'),
            layers.ReLU(),
            layers.Conv2D(128, (3, 3), padding='same'),
            layers.ReLU(),
            layers.Conv2D(1, (1, 1))
        ])

    def call(self, x):
        x = self.features(x)
        # Bilinear upsampling (like F.upsample_bilinear in PyTorch)
        x = tf.image.resize(x, size=(x.shape[1] * 2, x.shape[2] * 2), method='bilinear')
        x = self.reg_layer(x)
        return tf.abs(x)


def vgg19(use_batch_norm=True, layers_cfg='Baysian_Ma', weights_path=None):
    '''
    VGG 19-layer model (configuration "Baysian_Ma")
    Model pre-trained on ImageNet if weights_path is not specified.
    :param use_batch_norm: Whether to use batch normalization.
    :param layers_cfg: The configuration of layers (Baysian_Ma or Regular).
    :param weights_path: Optional path to weights. If not provided, model weights are not preloaded.
    :return: A VGG_Baysian_Ma model.
    '''
    features = make_layers(cfg[layers_cfg], use_batch_norm)
    model = VGG_Baysian_Ma(features)
    
    if weights_path:
        model.load_weights(weights_path)
    
    return model


# Example usage for the ShanghaiTech dataset
if __name__ == "__main__":
    # Load model
    model = vgg19(use_batch_norm=True, layers_cfg='Baysian_Ma', weights_path=None)
    
    # Example input: a batch of RGB images (height=384, width=384, channels=3)
    input_tensor = tf.random.normal([1, 384, 384, 3])
    
    # Run the model
    output = model(input_tensor)
    
    print("Output shape:", output.shape)
