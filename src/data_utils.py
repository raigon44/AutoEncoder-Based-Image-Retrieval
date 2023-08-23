import tensorflow as tf
import numpy as np
from imageprocessor import ImageProcessor


def prepare_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    return x_train, y_train, x_test, y_test


def check_data_properties():
    x_train, y_train, x_test, y_test = prepare_data()
    print("Checking the results........")

    print("Verifying if value of image array is [0,1]")
    if (x_train >= 0).all() and (x_train <= 1).all() and (x_test >= 0).all() and (x_test <= 1).all():
        print("All values in x_train are in the range [0, 1]")
    else:
        print("There are values in the image array which are not inside the range [0, 1]")
        exit(0)

    print("Checking the data type of the image arrays")
    if str(x_train.dtype) == "float32" and str(x_test.dtype) == "float32":
        print("x_train data type: " + str(x_train.dtype))
        print("x_test data type: " + str(x_test.dtype))
    else:
        print("Data types of the image array are not Float32")
        exit(0)

    print("Checking the shape of all the numpy arrays")
    if x_train.shape == (50000, 32, 32, 3) and x_test.shape == (10000, 32, 32, 3) and y_train.shape == (
    50000, 1) and y_test.shape == (10000, 1):
        print("x_train shape: " + str(x_train.shape))
        print("y_train shape: " + str(y_train.shape))
        print("x_test shape: " + str(x_test.shape))
        print("y_test shape: " + str(y_test.shape))
        print("Shape of all the arrays looks fine!!!")
    else:
        print("Array share do not match the expected value.")
        exit(0)

    return


# def create_transformed_dataset(x_train, num_images) :
#
#     image_processor_obj = ImageProcessor(num_images)
#
#     noisy_images = image_processor_obj.apply_random_noise_to_images()
#     transformed_images = image_processor_obj.apply_projective_transform__to_images()
#
#     x_train_adapted = x_train[:300]
#     for index in image_processor_obj.noisy_image_indices:
#         x_train_adapted = np.append(x_train_adapted, x_train[index].reshape(1, 32, 32, 3), axis=0)
#
#     for index in image_processor_obj.transformed_image_indices:
#         x_train_adapted = np.append(x_train_adapted, x_train[index].reshape(1, 32, 32, 3), axis=0)
#
#     # indices_noisy_images = np.random.choice(50000, size=25, replace=False)
#     # indices_transformed_images = np.random.choice(50000, size=25, replace=False)
#     # x_train_new_1, noisy_images = random_noise(x_train[indices_noisy_images], x_train[:300], 'gaussian', mean=0.1,
#     #                                            var=0.01)
#     # x_train_new_1, transformed_images = projective_transformation(x_train[indices_transformed_images], x_train_new_1)
#
#     x_train_adapted_1 = x_train[:300]
#
#     for index in indices_noisy_images:
#         x_train_adapted_1 = np.append(x_train_adapted_1, x_train[index].reshape(1, 32, 32, 3), axis=0)
#
#     for index in indices_transformed_images:
#         x_train_adapted_1 = np.append(x_train_adapted_1, x_train[index].reshape(1, 32, 32, 3), axis=0)
#
#     transformed_images_test_1 = x_train_new_1[200:]
#
#     transformed_images_test_1 = transformed_images_test_1[:10]
#
#     return x_train_new_1, x_train_adapted_1, transformed_images_test_1
#
#     pass


if __name__ == "__main__":
    check_data_properties()
