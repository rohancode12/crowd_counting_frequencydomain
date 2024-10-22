import numpy as np
from PIL import Image
from glob import glob
import os
import math
import tensorflow as tf

def cal_new_size(im_h, im_w, min_size, max_size):
    if im_h < im_w:
        if im_h < min_size:
            ratio = 1.0 * min_size / im_h
            im_h = min_size
            im_w = round(im_w * ratio)
        elif im_h > max_size:
            ratio = 1.0 * max_size / im_h
            im_h = max_size
            im_w = round(im_w * ratio)
        else:
            ratio = 1.0
    else:
        if im_w < min_size:
            ratio = 1.0 * min_size / im_w
            im_w = min_size
            im_h = round(im_h * ratio)
        elif im_w > max_size:
            ratio = 1.0 * max_size / im_w
            im_w = max_size
            im_h = round(im_h * ratio)
        else:
            ratio = 1.0
    return im_h, im_w, ratio

class Collate_F():
    @staticmethod
    def train_collate(batch):
        batch = list(zip(*batch))
        images = np.stack(batch[0], 0)
        chfs = np.stack(batch[1], 0)
        return images, chfs

class CrowdData(tf.keras.utils.Sequence):
    def __init__(self, img_path: str, dot_ann_path: str, mode: str, is_gray: bool = False, 
                 min_size: int = 0, max_size: int = np.inf):
        '''
        Args:
            img_path (str): Path to images
            dot_ann_path (str): Path to annotation files (.npy)
            mode (str): 'train', 'val', or 'test'
            is_gray (bool): If True, convert images to grayscale
            min_size (int): Minimum size for image resizing
            max_size (int): Maximum size for image resizing
        '''
        self.im_list = sorted(glob(os.path.join(img_path, '*.jpg')))
        self.dot_ann_list = sorted(glob(os.path.join(dot_ann_path, '*.npy')))
        self.mode = mode
        self.is_gray = is_gray
        self.shorter_length_min = min_size
        self.shorter_length_max = max_size

        if self.is_gray:
            self.transform = lambda x: tf.image.rgb_to_grayscale(tf.image.convert_image_dtype(x, tf.float32))
        else:
            self.transform = lambda x: tf.image.convert_image_dtype(x, tf.float32)

        self.people_counts = []
        self.dealt_imgs = []
        self.dealt_dotmap = []

        for item in range(0, len(self.im_list)):
            self.single_img_path = self.im_list[item]
            self.single_dot_ann_path = self.dot_ann_list[item]

            if self.is_gray:
                img = Image.open(self.single_img_path).convert('L')
            else:
                img = Image.open(self.single_img_path).convert('RGB')

            gt_data = np.load(self.single_dot_ann_path)
            if gt_data.shape[0] > 0:
                if gt_data.ndim == 1:
                    gt_data = np.expand_dims(gt_data[:2], axis=0)
                dot_ann = gt_data[:, :2]
            else:
                dot_ann = gt_data

            w, h = img.size
            if min([w, h]) < self.shorter_length_min:
                r = self.shorter_length_min / min([w, h])
                img, dot_ann = Image_dotmap_processing.resize(img, dot_ann,
                                                             np.ceil(np.array([w * r, h * r])).astype(int))
            if min([w, h]) > self.shorter_length_max:
                r = self.shorter_length_max / min([w, h])
                img, dot_ann = Image_dotmap_processing.resize(img, dot_ann,
                                                             np.ceil(np.array([w * r, h * r])).astype(int))

            self.process(img, dot_ann)

    def process(self, img, dot_ann):
        '''
        Args:
            img (PIL Image):  image
            gt_data (ndarray):  dotted_annotation with or without other information

        Returns: Processed image and annotations for training or testing
        '''
        raise NotImplemented

    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, item):
        if self.mode.lower().startswith('train'):
            return self.dealt_imgs[item], self.dealt_dotmap[item]
        else:
            return self.dealt_imgs[item], self.dealt_dotmap[item], self.people_counts[item], self.im_list[item]

class ImgTensor_dotTensor_processing():
    @staticmethod
    def crop(img_tensor, dot_tensor, crop_position):
        '''
        Args:
            img_tensor (tensor): image tensor
            dot_tensor (ndarray): dot annotation positions
            crop_position (tuple): Crop coordinates (left, upper, right, lower)

        Returns: Cropped image tensor and cropped dot_tensor
        '''
        img_tensor = tf.image.crop_to_bounding_box(img_tensor, crop_position[1], crop_position[0], 
                                                   crop_position[3] - crop_position[1], crop_position[2] - crop_position[0])

        if dot_tensor.shape[0] > 0:
            mask = (dot_tensor[:, 0] > crop_position[0]) & (dot_tensor[:, 0] < crop_position[2]) & (
                    dot_tensor[:, 1] > crop_position[1]) & (dot_tensor[:, 1] < crop_position[3])
            dot_tensor = dot_tensor[mask] - np.array(crop_position[0:2], dtype=dot_tensor.dtype)

        return img_tensor, dot_tensor

    @staticmethod
    def random_crop(img_tensor, dot_tensor, size):
        '''
        Randomly crop the image
        Args:
            img_tensor (tensor): Image tensor
            dot_tensor (ndarray): Dot annotation
            size (tuple): Crop size (height, width)
        Returns: Cropped image and corresponding dot annotations
        '''
        return tf.image.random_crop(img_tensor, size), dot_tensor

    @staticmethod
    def random_mirror(img_tensor, dot_tensor):
        '''
        Randomly flip the image horizontally
        Args:
            img_tensor (tensor): Image tensor
            dot_tensor (ndarray): Dot annotation positions

        Returns: Flipped image tensor and dot_tensor
        '''
        if np.random.rand() > 0.5:
            img_tensor = tf.image.flip_left_right(img_tensor)
            if dot_tensor.shape[0] > 0:
                dot_tensor[:, 0] = img_tensor.shape[1] - dot_tensor[:, 0]
        return img_tensor, dot_tensor

class Image_dotmap_processing():
    @staticmethod
    def crop(img, dotted_map, crop_position=(0, 0, 512, 512)):
        '''
        General crop for image and annotation
        Args:
            img (Image): PIL Image
            dotted_map (n × 2 ndarray): Dot annotation positions
            crop_position (tuple): Crop position

        Returns: Cropped image and dot annotations
        '''
        img = img.crop(crop_position)
        if dotted_map.shape[0] > 0:
            mask = (dotted_map[:, 0] > crop_position[0]) & (dotted_map[:, 0] < crop_position[2]) & (
                    dotted_map[:, 1] > crop_position[1]) & (dotted_map[:, 1] < crop_position[3])
            dotted_map = dotted_map[mask]
            dotted_map[:, 0:2] = dotted_map[:, 0:2] - crop_position[0:2]

        return img, dotted_map

    @staticmethod
    def resize(img, dotted_map, size=512):
        '''
        Resize image and annotations
        Args:
            img (Image): PIL Image
            dotted_map ((n×2 ndarray): Dot annotation positions
            size (tuple): Target size

        Returns: Resized image and annotations
        '''
        size = np.array(size)
        ratio = size / np.array(img.size)
        image = img.resize(size, Image.ANTIALIAS)

        if dotted_map.shape[0] > 0:
            dotted_map = dotted_map * ratio

        return image, dotted_map

    @staticmethod
    def random_mirror(img, dotted_map):
        '''
        Randomly flip the image horizontally
        Args:
            img (Image): PIL Image
            dotted_map (ndarray): Dot annotation positions

        Returns: Flipped image and annotations
        '''
        w, h = img.size
        if np.random.rand() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if dotted_map.shape[0] > 0:
                dotted_map[:, 0] = w - dotted_map[:, 0]
        return img, dotted_map

class Generating_data_from_dotted_annotation:
    @staticmethod
    def construct_characteristic_function(head_position, bandwidth, origin=0,
                                          step=30, step_length=0.01):
        '''
        Generate characteristic function from head positions

        Args:
            head_position (n×2 ndarray): Head position annotations
            bandwidth (int): Bandwidth parameter for Gaussian function
            origin (int): Origin for characteristic function
            step (int): Step count
            step_length (float): Step length

        Returns: Characteristic function tensor
        '''
        if head_position.shape[0] > 0:
            gauss_mean = head_position - origin
            plane = tf.stack([
                tf.range(-step, step, dtype=tf.float32) * step_length,
                tf.range(-step, step, dtype=tf.float32) * step_length
            ], axis=-1)
            angle = tf.linalg.matvec(plane, gauss_mean, transpose_a=True)
            length = tf.exp(-0.5 * tf.reduce_sum(tf.square(plane), axis=-1, keepdims=True) * bandwidth ** 2)
            cf_real = tf.reduce_sum(tf.cos(angle) * length, axis=-1)
            cf_img = tf.reduce_sum(tf.sin(angle) * length, axis=-1)
            return tf.stack([cf_real, cf_img], axis=-1)
        else:    
            return tf.zeros([step * 2, step * 2, 2], dtype=tf.float32)

import os
import glob
import numpy as np
from PIL import Image
from scipy.io import loadmat
from Dataset.dataprocessor import Image_dotmap_processing as idp
from Dataset.dataloader import cal_new_size


class Trans_gt_to_ndarray:
    @staticmethod
    def trans_ann_to_npy_SHTC(target_path: str, save_path: str):
        """
        Convert the MATLAB ground truth annotations to NumPy .npy files.

        Args:
            target_path (str): Directory where the original annotation (.mat) files are.
            save_path (str): Directory where the converted .npy annotation maps will be saved.
        """
        assert target_path != save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        for file in glob.glob(target_path + '/*.mat'):
            dot = loadmat(file)
            # Extract location information from the .mat file
            x = dot[list(dot.keys())[-1]][0, 0]['location'][0, 0].astype(np.float)
            # Save the .npy file
            np.save(os.path.join(save_path, os.path.splitext(os.path.split(file)[-1])[0] + '.npy'), x)


class Directory_path:
    @staticmethod
    def prefix_suffix(dataset_name: str):
        """
        Get the directory structure for ShanghaiTech Part A and B datasets.

        Args:
            dataset_name (str): Name of the dataset.

        Returns:
            Tuple of prefixes and suffixes for the dataset directories.
        """
        if 'a' in dataset_name.lower():
            return 'Dataset/ShanghaiTech/part_A_final/', '_data/images', '_data/ground_truth_npy'
        elif 'b' in dataset_name.lower():
            return 'Dataset/ShanghaiTech/part_B_final/', '_data/images', '_data/ground_truth_npy'

    @staticmethod
    def get_name_from_no(dataset_name: str, set: str, prefix: str, img_suffix: str, dotmap_suffix: str, img_no):
        """
        Get the image and annotation file paths given an image number.

        Args:
            dataset_name (str): Name of the dataset.
            set (str): Set type ('train', 'val', 'test').
            prefix (str): Prefix directory path.
            img_suffix (str): Image directory suffix.
            dotmap_suffix (str): Annotation directory suffix.
            img_no (int or str): Image number or identifier.

        Returns:
            Paths to the image and dotmap file.
        """
        if isinstance(img_no, int):
            img_path = os.path.join(prefix, set, img_suffix, f'IMG_{img_no}.jpg')
            dotmap_path = os.path.join(prefix, set, dotmap_suffix, f'GT_IMG_{img_no}.npy')
        elif isinstance(img_no, str):
            img_path = os.path.join(prefix, set, img_suffix, f'{img_no}.jpg')
            dotmap_path = os.path.join(prefix, set, dotmap_suffix, f'GT_{img_no}.npy')
        else:
            raise ValueError("img_no should be int or str")
        
        return img_path, dotmap_path

    @staticmethod
    def get_data(dataset_name: str, set: str, img_no, min_side_length=0, max_side_length=np.inf, is_gray=False):
        """
        Retrieve the image and dot annotation, and resize if needed.

        Args:
            dataset_name (str): Name of the dataset.
            set (str): Set type ('train', 'test').
            img_no (int or str): Image number.
            min_side_length (int): Minimum size for the image's shorter edge.
            max_side_length (int): Maximum size for the image's shorter edge.
            is_gray (bool): If True, convert the image to grayscale.

        Returns:
            Resized image and corresponding dot annotation.
        """
        prefix, img_suffix, dotmap_suffix = Directory_path.prefix_suffix(dataset_name)
        img_path, dotmap_path = Directory_path.get_name_from_no(dataset_name, set, prefix, img_suffix, dotmap_suffix, img_no)

        # Load the image and annotation
        img = Image.open(img_path).convert('L' if is_gray else 'RGB')
        dot_ann = np.load(dotmap_path)
        
        if dot_ann.shape[0] > 0:
            if dot_ann.ndim == 1:
                dot_ann = np.expand_dims(dot_ann[:2], axis=0)
            dot_ann = dot_ann[:, :2]

        # Resize image and annotation if necessary
        w, h = img.size
        h, w, ratio = cal_new_size(h, w, min_side_length, max_side_length)
        if ratio != 1:
            img, dot_ann = idp.resize(img, dot_ann, np.array([w, h]))

        return img, dot_ann


class Batch_image_dotmap_processing:
    @staticmethod
    def resize(dataset_name: str, set: str, min_side_length: int, max_side_length: int, dotmap_together=True, is_gray=False):
        """
        Resize images and their corresponding dotmaps in the dataset.

        Args:
            dataset_name (str): Name of the dataset.
            set (str): Set type ('train', 'val', 'test').
            min_side_length (int): Minimum size for the shorter edge of the image.
            max_side_length (int): Maximum size for the shorter edge of the image.
            dotmap_together (bool): Whether to resize the dotmap along with the image.
            is_gray (bool): If True, convert the image to grayscale.
        """
        prefix, img_suffix, dotmap_suffix = Directory_path.prefix_suffix(dataset_name)

        for img_path in glob.glob(os.path.join(prefix + set + img_suffix, '*.jpg')):
            num = Directory_path.get_no_from_name(dataset_name, img_path)
            img_path, dotmap_path = Directory_path.get_name_from_no(dataset_name, set, prefix, img_suffix, dotmap_suffix, int(num))
            
            if is_gray:
                img = Image.open(img_path).convert('L')
            else:
                img = Image.open(img_path).convert('RGB')

            if dotmap_together:
                dot_ann = np.load(dotmap_path)
                if dot_ann.shape[0] > 0:
                    if dot_ann.ndim == 1:
                        dot_ann = np.expand_dims(dot_ann, axis=0)
                    dot_ann = dot_ann[:, :2]

            w, h = img.size
            h, w, ratio = cal_new_size(h, w, min_side_length, max_side_length)
            if ratio != 1:
                img = img.resize(np.array([w, h]))
                img.save(img_path, quality=95)
                if dotmap_together and dot_ann.shape[0] > 0:
                    dot_ann = dot_ann * ratio
                    np.save(dotmap_path, dot_ann)


class Dataset_preparation:
    @staticmethod
    def SHTCA():
        """
        Prepare ShanghaiTech Part A dataset by converting annotations to .npy format.
        """
        Trans_gt_to_ndarray.trans_ann_to_npy_SHTC(
            'C:\Users\hp\Downloads\ShanghaiTech_Crowd_Counting_Dataset\part_A_final\train_data\ground_truth',
            'C:\Users\hp\Downloads\ShanghaiTech_Crowd_Counting_Dataset\part_A_final\train_data\ground_truth\ground_truth_npy')
        Trans_gt_to_ndarray.trans_ann_to_npy_SHTC(
            'C:\Users\hp\Downloads\ShanghaiTech_Crowd_Counting_Dataset\part_A_final\test_data\ground_truth',
            'C:\Users\hp\Downloads\ShanghaiTech_Crowd_Counting_Dataset\part_A_final\test_data\ground_truth\ground_truth_npy')

    @staticmethod
    def SHTCB():
        """
        Prepare ShanghaiTech Part B dataset by converting annotations to .npy format.
        """
        Trans_gt_to_ndarray.trans_ann_to_npy_SHTC(
            'C:\Users\hp\Downloads\ShanghaiTech_Crowd_Counting_Dataset\part_B_final\train_data\ground_truth'',
            'C:\Users\hp\Downloads\ShanghaiTech_Crowd_Counting_Dataset\part_B_final\train_data\ground_truth\ground_truth_npy')
        Trans_gt_to_ndarray.trans_ann_to_npy_SHTC(
            'C:\Users\hp\Downloads\ShanghaiTech_Crowd_Counting_Dataset\part_B_final\test_data\ground_truth',
            'C:\Users\hp\Downloads\ShanghaiTech_Crowd_Counting_Dataset\part_A_final\test_data\ground_truth\ground_truth_npy')
