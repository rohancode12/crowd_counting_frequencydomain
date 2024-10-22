from Utils import model_test_tf as mt  # Assuming model_test_tf is the TensorFlow version of model_test
from Model import vgg_19_tf as vgg_19  # TensorFlow version of vgg_19
import sys

if __name__ == '__main__':
    # Initialize the model for testing
    model = vgg_19.vgg19()  # Load the TensorFlow version of the VGG-19 model
    
    # Create an instance of the test class
    d = mt.Chf_Model_Test(model, sys.argv[2])

    # Run evaluation based on the dataset
    
        # For the ShanghaiTech dataset
 d.evaluate_on_test_set(sys.argv[1], sys.argv[1] + '_result.txt')


class Chf_Model_Test:
    def __init__(self, model, weights_path):
        # Load model weights if provided
        self.model = model
        if weights_path:
            self.model.load_weights(weights_path)

    def evaluate_on_test_set(self, dataset_path, result_file, max_size=None):
        # Load the dataset (assuming you use tf.data.Dataset or similar)
        test_dataset = self.load_dataset(dataset_path, max_size)
        
        # Prepare result file
        with open(result_file, 'w') as f:
            for images, labels in test_dataset:
                predictions = self.model(images, training=False)  # Run inference

                # Calculate MAE and MSE
                mae = self.mean_absolute_error(predictions, labels)
                mse = self.mean_squared_error(predictions, labels)

                # Write results to the file
                f.write(f"MAE: {mae.numpy()}, MSE: {mse.numpy()}\n")


    def mean_absolute_error(self, predictions, labels):
        """
        Calculate Mean Absolute Error (MAE)
        :param predictions: TensorFlow model predictions
        :param labels: Ground truth labels
        :return: MAE value as a TensorFlow scalar
        """
        return tf.reduce_mean(tf.abs(predictions - labels))

    def mean_squared_error(self, predictions, labels):
        """
        Calculate Mean Squared Error (MSE)
        :param predictions: TensorFlow model predictions
        :param labels: Ground truth labels
        :return: MSE value as a TensorFlow scalar
        """
        return tf.reduce_mean(tf.square(predictions - labels))