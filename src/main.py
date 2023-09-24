from model import Model
import data_utils
import config
from argparse import ArgumentParser
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_distances
import matplotlib.pyplot as plt
import numpy as np
import logging

parser = ArgumentParser()
parser.add_argument('--operation', required=True)
args = parser.parse_args()


def train():
    data_utils.check_data_properties(x_train, y_train, x_test, y_test)

    model_obj = Model(config.ModelConfig)

    model_obj.train_model(x_train, x_test)

    return


def load_encoder_model(model_path):
    """Loads the encoder model from the model saved during training."""
    auto_encoder = tf.keras.models.load_model(model_path)
    return tf.keras.models.Model(auto_encoder.input, auto_encoder.layers[13].output)


def plot_similar_images(similar_images, query_images):
    for i in range(5):
        index_list = similar_images[i]
        count = 1
        plt.figure(figsize=(20, 4))
        for index in index_list:
            ax = plt.subplot(2, 10, count)
            plt.imshow(query_images[i])
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax = plt.subplot(2, 10, count + 10)
            plt.imshow(x_dataset[index])
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            count = count + 1
        plt.show()


def data_query(encoder_model):
    """Randomly selects 5 image from the dataset.
       Encode the 5 images and the whole dataset using the encoder model.
       Compute the pairwise distance between each query encoding all the encodings in the whole dataset. """
    query_images = x_test[np.random.choice(10000, size=5, replace=False)]
    query_encodings = encoder_model.predict(query_images)

    dataset_encodings = encoder_model.predict(x_dataset)

    similar_images_cosine = []

    for i in range(0, 5):

        dist_cosine = cosine_distances(query_encodings[i].reshape(1, 10), dataset_encodings)
        dist_cosine = dist_cosine[0]
        indices_cosine = sorted(range(len(dist_cosine)), key=lambda sub: dist_cosine[sub])[:10]
        similar_images_cosine.append(indices_cosine)

    logger.info("Plotting the similar images using the Cosine distance")
    plot_similar_images(similar_images_cosine, query_images)


def main():

    if args.operation == 'train':
        train()
    elif args.operation == 'query':
        data_query(load_encoder_model('models/AutoEncoder_Final.hdf5'))
    else:
        logger.error('The entered operation is unknown! Only the operations train and inference are allowed.')
        exit(1)
    return


def get_logger():
    """
    This function configures and returns the logger which will be used by the rest of the python modules in this project.
    :return:
    """
    logger = logging.getLogger('AutoEncoder_Log')
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    logging_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(logging_format)

    logger.addHandler(console_handler)

    return logger


if __name__ == '__main__':
    logger = get_logger()
    x_train, y_train, x_test, y_test = data_utils.prepare_data()
    x_dataset = np.concatenate((x_train, x_test), axis=0)
    main()





