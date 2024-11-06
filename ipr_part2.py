
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np
from PIL import Image
import os

# Define the custom Fourier Transform layer
class FourierTransformLayer(layers.Layer):
    def call(self, inputs):
        # Apply 2D FFT
        fft = tf.signal.fft2d(tf.cast(inputs, tf.complex64))
        magnitude = tf.abs(fft)
        phase = tf.math.angle(fft)
        # Concatenate magnitude and phase along the channel dimension
        return tf.concat([magnitude, phase], axis=-1)

# Define the generator function to yield image and density map
def data_generator(image_dir, gt_dir):
    images = os.listdir(image_dir)
    for img_name in images:
        img_path = os.path.join(image_dir, img_name)
        gt_name = img_name.replace('IMG', 'GT_IMG').replace('.jpg', '.npy')  # Adjust for GT format
        gt_path = os.path.join(gt_dir, gt_name)

        # Load and resize image to 512x512
        image = Image.open(img_path).convert('RGB').resize((512, 512))
        image = np.array(image) / 255.0  # Normalize image to [0, 1]

        # Load and resize density map to 128x128 to match model output
        density_map = np.load(gt_path)
        density_map = np.array(Image.fromarray(density_map).resize((128, 128), Image.BILINEAR))
        
        yield image.astype(np.float32), density_map[..., np.newaxis].astype(np.float32)

# Define a function to wrap the generator with tf.data.Dataset
def create_dataset(image_dir, gt_dir, batch_size=8):
    dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(image_dir, gt_dir),
        output_signature=(
            tf.TensorSpec(shape=(512, 512, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(128, 128, 1), dtype=tf.float32),
        )
    )
    dataset = dataset.batch(batch_size)
    return dataset

# Create Spatial Path of Dual-Path Network
def create_spatial_path():
    inputs = layers.Input(shape=(512, 512, 3))
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = Dropout(0.3)(x)

    x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = Dropout(0.3)(x)

    x = layers.Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = layers.Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # ASPP Module with multiple dilation rates
    aspp1 = layers.Conv2D(256, (3, 3), padding='same', dilation_rate=1, activation='relu')(x)
    aspp2 = layers.Conv2D(256, (3, 3), padding='same', dilation_rate=6, activation='relu')(x)
    aspp3 = layers.Conv2D(256, (3, 3), padding='same', dilation_rate=12, activation='relu')(x)
    aspp4 = layers.Conv2D(256, (3, 3), padding='same', dilation_rate=18, activation='relu')(x)

    aspp_concat = layers.Concatenate()([aspp1, aspp2, aspp3, aspp4])
    aspp_output = layers.Conv2D(512, (1, 1), activation='relu')(aspp_concat)

    # Decoder
    x = layers.UpSampling2D((2, 2))(aspp_output)
    x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    
    return models.Model(inputs, x)

# Create Frequency Path of Dual-Path Network
def create_frequency_path():
    inputs = layers.Input(shape=(512, 512, 3))

    # Apply the custom Fourier transform layer
    freq_input = FourierTransformLayer()(inputs)
    
    x = layers.Conv2D(64, (3, 3), strides=2, padding='same', activation='relu')(freq_input)
    x = layers.Conv2D(128, (3, 3), strides=2, padding='same', activation='relu')(x)
    x = layers.Conv2D(256, (3, 3), strides=2, padding='same', activation='relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Reshape((1, 1, 256))(x)
    
    return models.Model(inputs, x)

# Fusion and final prediction layers
def create_dual_path_network():
    spatial_path = create_spatial_path()
    frequency_path = create_frequency_path()

    inputs = layers.Input(shape=(512, 512, 3))
    spatial_output = spatial_path(inputs)
    frequency_output = frequency_path(inputs)
    
    # Upsample frequency output to match spatial dimensions
    frequency_upsampled = layers.UpSampling2D((128, 128))(frequency_output)

    # Fusion layer
    x = layers.Concatenate()([spatial_output, frequency_upsampled])
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.Conv2D(1, (1, 1), activation='relu')(x)

    return models.Model(inputs, x)

# Create the model
model = create_dual_path_network()
model.summary()

# Compile the model with a lower learning rate
model.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_absolute_error')

# Paths for training and testing data
train_image_dir = 'C:\\Users\\hp\\Downloads\\ShanghaiTech_Crowd_Counting_Dataset\\part_A_final\\train_data\\images'
train_gt_dir = 'C:\\Users\\hp\\Downloads\\ShanghaiTech_Crowd_Counting_Dataset\\part_A_final\\train_data\\ground_truth_npy'
test_image_dir = 'C:\\Users\\hp\\Downloads\\ShanghaiTech_Crowd_Counting_Dataset\\part_A_final\\test_data\\images'
test_gt_dir = 'C:\\Users\\hp\\Downloads\\ShanghaiTech_Crowd_Counting_Dataset\\part_A_final\\test_data\\ground_truth_npy'

# Create train and test datasets
train_dataset = create_dataset(train_image_dir, train_gt_dir, batch_size=8)
test_dataset = create_dataset(test_image_dir, test_gt_dir, batch_size=8)

# Train the model for more epochs
model.fit(train_dataset, epochs=50, validation_data=test_dataset)

# Evaluate on the test set to compute MSE and MAE
all_preds, all_gts = [], []
for image_batch, gt_batch in test_dataset:
    pred_batch = model.predict(image_batch)
    all_preds.append(pred_batch)
    all_gts.append(gt_batch)

# Concatenate all batches to form arrays
all_preds = np.concatenate(all_preds, axis=0)
all_gts = np.concatenate(all_gts, axis=0)

# Calculate Mean Squared Error (MSE) and Mean Absolute Error (MAE)
mse = np.mean((all_preds - all_gts) ** 2)
mae = np.mean(np.abs(all_preds - all_gts))

print(f"Test MSE: {mse:.4f}")
print(f"Test MAE: {mae:.4f}")
