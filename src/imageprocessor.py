import cv2
import numpy as np
import config


class ImageProcessor:

    def __init__(self, num_images_to_transform):
        self.num_images_to_transform = num_images_to_transform
        self.random_noise_mode = config.RandomNoiseConfig.mode
        self.noisy_image_indices = np.random.choice(50000, size=25, replace=False)
        self.transformed_image_indices = np.random.choice(50000, size=25, replace=False)
        self.mean = config.RandomNoiseConfig.mean
        self.var = config.RandomNoiseConfig.var
        self.clip = config.RandomNoiseConfig.clip
        self.src_points = np.float32([[0, 0], [32 - 1, 0], [0, 32 - 1], [32 - 1, 32 - 1]])
        self.dst_points = np.float32([[0, 0], [32 - 1, 0], [int(0.33 * 32), 32 - 1], [int(0.66 * 32), 32 - 1]])

    def apply_random_noise_to_image(self, image):
        """
        This function adds a random noise to the given input image and returns the noisy image
        :param image:
        :return: noisy_image
        """
        self.random_noise_mode = self.random_noise_mode.lower()
        if image.min() < 0:
            low_clip = -1
        else:
            low_clip = 0

        if self.random_noise_mode == 'gaussian':
            noise = np.random.normal(self.mean, self.var ** 0.5, image.shape)
            noisy_image = image + noise

        if self.clip:
            noisy_image = np.clip(noisy_image, low_clip, 1.0)

        return noisy_image

    def apply_projective_transform_to_image(self, image):
        """
        This function applies perspective transformation to the given input image and returns the transformed image
        :param image:
        :return: transformed_image
        """
        projective_matrix = cv2.getPerspectiveTransform(self.src_points, self.dst_points)
        transformed_image = cv2.warpPerspective(image, projective_matrix, (32, 32))

        return transformed_image

    def apply_random_noise_to_images(self, images):
        """
        This function receives as input a numpy array of images. To each image it adds random noise and returns the numpy
        array of noisy images
        :param images:
        :return: noisy_images:
        """
        noisy_images = np.empty((self.num_images_to_transform, 32, 32, 3))
        for image in images:
            noisy_images = np.append(noisy_images, self.apply_random_noise_to_image(image).reshape(1, 32, 32, 3),
                                     axis=0)
        return noisy_images

    def apply_projective_transform_to_images(self, images):
        """
        This function receives as input a numpy array of images. To each image it applies perspective transformation using
        openCV library methods and returns the numpy array of transformed images
        :param images:
        :return: transformed_images
        """
        transformed_images = np.empty((self.num_images_to_transform, 32, 32, 3))
        for image in images:
            transformed_images = np.append(transformed_images, self.apply_projective_transform__to_image(image).reshape(1, 32, 32, 3),
                                           axis=0)
        return transformed_images


