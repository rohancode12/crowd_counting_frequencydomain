import tensorflow as tf
import numpy as np

class ChfLikelihood(tf.Module):
    def likelihood(self, input, target, scale):
        raise NotImplementedError

class CentralGaussian(ChfLikelihood):
    '''
    General i.i.d. noise-robust window for noisy crowd counting.
    '''
    def __init__(self, chf_step, chf_tik, var: str, coeff):
        '''
        :param chf_step:
        :param chf_tik:
        :param var: path to the empirical error variance map (.npy file)
        :param coeff: set it as 1 or 0.5 depending on the noise level
        '''
        self.chf_step = chf_step
        self.chf_tik = chf_tik
        # Load the variance map from a .npy file
        self.h_r = tf.convert_to_tensor(np.load(var), dtype=tf.float32)
        self.coeff = coeff

    def likelihood(self, input, target, scale):
        people_count = target[:, int(target.shape[1] / 2), int(target.shape[2] / 2), 0]
        if not isinstance(scale, int):
            scale = tf.reshape(scale, [self.h_r.shape[0], self.h_r.shape[1]])
        var = (scale ** 2 * self.h_r[None, ...]) * people_count[:, None, None]
        loss = tf.norm(input - target, axis=-1) / tf.sqrt(self.coeff * var + 1)
        loss = tf.reduce_sum(loss) / target.shape[0]

        return loss


class CentralGaussianWithGaussianNoise(ChfLikelihood):
    '''
    i.i.d. Gaussian noise-robust window for noisy crowd counting.
    '''
    def __init__(self, chf_step, chf_tik, noise_bandwidth, ground_truth_bandwidth):
        '''
        :param chf_step:
        :param chf_tik:
        :param noise_bandwidth: the bandwidth of the Gaussian distribution of the annotation noise
        :param ground_truth_bandwidth: the bandwidth of the ground truth density map
        '''
        self.chf_step = chf_step
        self.chf_tik = chf_tik
        
        plane = tf.stack([
            tf.repeat(tf.range(-self.chf_step, self.chf_step)[None, :], 2 * self.chf_step, axis=0) * self.chf_tik,
            tf.tile(tf.range(-self.chf_step, self.chf_step)[:, None], [1, 2 * self.chf_step]) * self.chf_tik
        ], axis=-1)

        original_chf_factor = tf.exp(-0.5 * tf.reduce_sum(plane ** 2, axis=-1) * ground_truth_bandwidth ** 2)
        noise_chf_factor = tf.exp(-0.5 * tf.reduce_sum(plane ** 2, axis=-1) * noise_bandwidth ** 2)
        self.h_r = (1 - noise_chf_factor) * original_chf_factor

    def likelihood(self, input, target, scale):
        people_count = target[:, int(target.shape[1] / 2), int(target.shape[2] / 2), 0]
        if not isinstance(scale, int):
            scale = tf.reshape(scale, [self.h_r.shape[0], self.h_r.shape[1]])
        var = (scale ** 2 * self.h_r[None, ...]) * people_count[:, None, None]
        var = tf.cast(var, input.dtype)
        loss = tf.norm(input - target, axis=-1) / tf.sqrt(var + 1)
        loss = tf.reduce_sum(loss) / target.shape[0]

        return loss


class ChfLoss(tf.Module):
    def __init__(self, chf_step: int, chf_tik: float, sample_step: float, is_dense: bool):
        '''
        :param chf_step: Number of steps in the characteristic function plane.
        :param chf_tik: Sampling interval in the characteristic function domain.
        :param sample_step: Sampling interval in the image plane.
        :param is_dense: Boolean flag for whether to use dense datasets.
        '''
        super(ChfLoss, self).__init__()
        self.chf_step = chf_step
        self.chf_tik = chf_tik
        self.sample_step = sample_step
        self.is_dense = is_dense

        self.plane_shape = None
        self.real_template = None
        self.img_template = None

    def make_template(self, dnn_output: tf.Tensor):
        # Construct the spatial domain
        x_axis = tf.linspace(self.sample_step / 2, dnn_output.shape[-1] * self.sample_step - self.sample_step / 2, dnn_output.shape[-1])
        y_axis = tf.linspace(self.sample_step / 2, dnn_output.shape[-2] * self.sample_step - self.sample_step / 2, dnn_output.shape[-2])
        sample_coordinates = tf.stack([tf.repeat(x_axis, len(y_axis)), tf.tile(y_axis, [len(x_axis)])], axis=0)

        # Construct the ch.f. domain
        plane = tf.stack([
            tf.repeat(tf.range(-self.chf_step, self.chf_step)[None, :], 2 * self.chf_step, axis=0) * self.chf_tik,
            tf.tile(tf.range(-self.chf_step, self.chf_step)[:, None], [1, 2 * self.chf_step]) * self.chf_tik
        ], axis=-1)

        # Calculate the characteristic function template
        angle = tf.matmul(plane, sample_coordinates)
        self.real_template = tf.cos(angle)
        self.img_template = tf.sin(angle)

    def __call__(self, dnn_output: tf.Tensor, chf: tf.Tensor):
        if dnn_output.shape[-2:] != self.plane_shape:
            self.make_template(dnn_output)
            self.plane_shape = dnn_output.shape[-2:]

        flatten_output = tf.reshape(tf.transpose(dnn_output, perm=[0, 2, 1]), [dnn_output.shape[0], -1])
        chf_real = tf.reduce_sum(self.real_template * flatten_output[:, None, None, :], axis=3, keepdims=True)
        chf_img = tf.reduce_sum(self.img_template * flatten_output[:, None, None, :], axis=3, keepdims=True)
        derived_chf = tf.concat([chf_real, chf_img], axis=3)

        if not self.is_dense:
            loss = tf.reduce_sum(tf.norm(derived_chf - chf, axis=2) * self.chf_tik)
        else:
            loss = tf.reduce_sum(tf.norm(derived_chf - chf, axis=2)) * self.chf_tik ** 2

        return loss / chf.shape[0]


class ChfLikelihoodLoss(tf.Module):
    '''
    Loss function for noisy crowd counting.
    '''
    def __init__(self, chf_step: int, chf_tik: float, sample_step: float, likelihood):
        '''
        :param chf_step: Number of steps in the characteristic function plane.
        :param chf_tik: Sampling interval in the characteristic function domain.
        :param sample_step: Sampling interval in the image plane.
        :param likelihood: Likelihood function used for crowd counting.
        '''
        super(ChfLikelihoodLoss, self).__init__()
        self.chf_step = chf_step
        self.chf_tik = chf_tik
        self.sample_step = sample_step

        self.likelihood = likelihood
        self.scale = 1

        self.plane_shape = None
        self.real_template = None
        self.img_template = None

    def make_template(self, dnn_output: tf.Tensor):
        # Construct the spatial domain
        x_axis = tf.linspace(self.sample_step / 2, dnn_output.shape[-1] * self.sample_step - self.sample_step / 2, dnn_output.shape[-1])
        y_axis = tf.linspace(self.sample_step / 2, dnn_output.shape[-2] * self.sample_step - self.sample_step / 2, dnn_output.shape[-2])
        sample_coordinates = tf.stack([tf.repeat(x_axis, len(y_axis)), tf.tile(y_axis, [len(x_axis)])], axis=0)

        # Construct the ch.f. domain
        plane = tf.stack([
            tf.repeat(tf.range(-self.chf_step, self.chf_step)[None, :], 2 * self.chf_step, axis=0) * self.chf_tik,
            tf.tile(tf.range(-self.chf_step, self.chf_step)[:, None], [1, 2 * self.chf_step]) * self.chf_tik
        ], axis=-1)

        # Calculate the characteristic function template
        angle = tf.matmul(plane, sample_coordinates)
        self.real_template = tf.cos(angle)
        self.img_template = tf.sin(angle)

    def __call__(self, dnn_output: tf.Tensor, chf: tf.Tensor):
        if dnn_output.shape != self.plane_shape:
            self.make_template(dnn_output)
            self.plane_shape = dnn_output.shape

        flatten_output = tf.reshape(tf.transpose(dnn_output, perm=[0, 2, 1]), [dnn_output.shape[0], -1])
        chf_real = tf.reduce_sum(self.real_template * flatten_output[:, None, None, :], axis=3, keepdims=True)
        chf_img = tf.reduce_sum(self.img_template * flatten_output[:, None, None, :], axis=3, keepdims=True)

        derived_chf = tf.concat([chf_real, chf_img], axis=3)
        loss = self.likelihood.likelihood(derived_chf, chf, self.scale)

        return loss
