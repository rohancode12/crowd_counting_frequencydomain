import tensorflow as tf
import numpy as np
import logging

from data_preprocessing import load_and_preprocess_dataset  # Your custom data preprocessing
from model_architecture import vgg19  # Your custom VGG19 model
from loss_function import Chf_Loss  # Your custom CHF loss function
from utils import save_model  # Custom model saving utility


def optimizer_parser(params):
    if params['optimizer'].lower() == 'sgd':
        optimizer = tf.keras.optimizers.SGD(
            learning_rate=params.get('lr', 0.01),
            momentum=params.get('momentum', 0),
            nesterov=params.get('nesterov', False)
        )
    elif params['optimizer'].lower() == 'adam':
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=params.get('lr', 0.001),
            beta_1=params.get('betas', (0.9, 0.999))[0],
            beta_2=params.get('betas', (0.9, 0.999))[1],
            epsilon=params.get('eps', 1e-08)
        )
    return optimizer

def dataloader_parser(params, dataset_dir, category=('train', 'val')):
    batch_size = params.get('batch_size', 1)
    image_size = params.get('img_size', 512)

    # Load and preprocess the dataset
    datasets = {}
    for split in category:
        dataset = load_and_preprocess_dataset(f"{dataset_dir}/{split}", image_size=image_size)
        datasets[split] = tf.data.Dataset.from_tensor_slices(dataset).shuffle(100).batch(batch_size)

    return datasets

class CrowdCountingTrainer:
    def __init__(self, logger, model, optimizer, dataset_dir, dataloader_params, chf_loss_params, epochs=1000, sample_interval=8):
        self.logger = logger
        self.model = model
        self.optimizer = optimizer
        self.epochs = epochs
        self.sample_interval = sample_interval

        # Load dataset
        self.dataloaders = dataloader_parser(dataloader_params, dataset_dir)

        # Define CHF loss function
        self.chf_loss = Chf_Loss(**chf_loss_params)

    def train(self, val_interval=2, val_start=0):
        for epoch in range(1, self.epochs + 1):
            print(f"Epoch {epoch}/{self.epochs}")
            self.train_one_epoch()

            if epoch % val_interval == 0 and epoch >= val_start:
                self.validate('val')
                # Save model after validation
                save_model(self.model, f"best_model_epoch_{epoch}.h5")
    
    def train_one_epoch(self):
        for images, chfs in self.dataloaders['train']:
            with tf.GradientTape() as tape:
                predictions = self.model(images, training=True)
                loss = self.chf_loss(predictions, chfs)

            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            print(f"Training loss: {loss.numpy()}")

    def validate(self, split='val'):
        mse_list = []
        mae_list = []
        for images, chfs in self.dataloaders[split]:
            predictions = self.model(images, training=False)
            mse = tf.reduce_mean(tf.square(predictions - chfs))
            mae = tf.reduce_mean(tf.abs(predictions - chfs))
            mse_list.append(mse.numpy())
            mae_list.append(mae.numpy())
        print(f"Validation MSE: {np.mean(mse_list)}, MAE: {np.mean(mae_list)}")

class Chf_Loss(tf.Module):
    def __init__(self, chf_step, chf_tik, bandwidth):
        self.chf_step = chf_step
        self.chf_tik = chf_tik
        self.bandwidth = bandwidth

    def __call__(self, predictions, ground_truth):
        pred_chf = self.compute_chf(predictions)
        gt_chf = self.compute_chf(ground_truth)
        loss = tf.reduce_mean(tf.square(pred_chf - gt_chf))
        return loss

    def compute_chf(self, density_map):
        # Compute characteristic function using Fourier transform (or custom implementation)
        chf = tf.signal.fft2d(tf.cast(density_map, tf.complex64))
        return chf

if __name__ == '__main__':
    dataset_dir = "C:\Users\hp\Downloads\ShanghaiTech_Crowd_Counting_Dataset\part_A_final\train_data"  # Path to ShanghaiTech dataset
    model = vgg19(input_shape=(512, 512, 3))

    optimizer_params = {
        'optimizer': 'adam',
        'lr': 1e-5,
    }
    dataloader_params = {
        'datahandler': 'chf_rcrop',
        'batch_size': 16,
        'img_size': 512,
    }
    chf_loss_params = {
        'chf_step': 30,
        'chf_tik': 0.01,
        'bandwidth': 8,
    }

    trainer = CrowdCountingTrainer(
        logger=logging,
        model=model,
        optimizer=optimizer_parser(optimizer_params),
        dataset_dir=dataset_dir,
        dataloader_params=dataloader_params,
        chf_loss_params=chf_loss_params,
        epochs=100,
        sample_interval=8
    )
    trainer.train()
