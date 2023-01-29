import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from datetime import datetime
from sklearn.metrics.pairwise import cosine_distances


def random_noise(images, x_noisy, mode='gaussian', seed=None, clip=True, **kwargs):

    for image in images:
        mode = mode.lower()
        if image.min() < 0:
            low_clip = -1
        else:
            low_clip = 0
        if seed is not None:
            np.random.seed(seed=seed)

        if mode == 'gaussian':
            noise = np.random.normal(kwargs['mean'], kwargs['var'] ** 0.5,
                                     image.shape)
            out = image + noise
        if clip:
            out = np.clip(out, low_clip, 1.0)

        x_noisy = np.append(x_noisy, out.reshape(1, 32, 32, 3), axis=0)

    return x_noisy, x_noisy[49999:]


def projective_transformation(images, x_transformed):
    src_points = np.float32([[0, 0], [32 - 1, 0], [0, 32 - 1], [32 - 1, 32 - 1]])
    dst_points = np.float32([[0, 0], [32 - 1, 0], [int(0.33 * 32), 32 - 1], [int(0.66 * 32), 32 - 1]])

    for image in images:

        projective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)

        img_output = cv2.warpPerspective(image, projective_matrix, (32, 32))
        x_transformed = np.append(x_transformed, img_output.reshape(1, 32, 32, 3), axis=0)
    return x_transformed, x_transformed[49999:]


def create_transformed_dataset():

    indices_noisy_images = np.random.choice(50000, size=25, replace=False)
    indices_transformed_images = np.random.choice(50000, size=25, replace=False)
    x_train_new_1, noisy_images = random_noise(x_train[indices_noisy_images], x_train[:300], 'gaussian', mean=0.1,
                                             var=0.01)
    x_train_new_1, transformed_images = projective_transformation(x_train[indices_transformed_images], x_train_new_1)

    x_train_adapted_1 = x_train[:300]

    for index in indices_noisy_images:
        x_train_adapted_1 = np.append(x_train_adapted_1, x_train[index].reshape(1, 32, 32, 3), axis=0)

    for index in indices_transformed_images:
        x_train_adapted_1 = np.append(x_train_adapted_1, x_train[index].reshape(1, 32, 32, 3), axis=0)

    transformed_images_test_1 = x_train_new_1[200:]

    transformed_images_test_1 = transformed_images_test_1[:10]

    return x_train_new_1, x_train_adapted_1, transformed_images_test_1


def plot_similar_images(similar_images, query_images, n):
    for i in range(n):
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

def retrain_model():

    auto_encoder = tf.keras.models.load_model('tmp/retrain9')

    logdir = "logs/scalars/retrainwithtranform" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath='tmp/retraintrnf',
        save_weights_only=False,
        monitor='val_loss',
        mode='min',
        save_best_only=True
    )

    auto_encoder.fit(x_train_new, x_train_adapted,
                     epochs=100,
                     batch_size=256,
                     shuffle=True,
                     callbacks=[tensorboard_callback, model_checkpoint_callback],
                     validation_data=(x_test, x_test))

    auto_encoder.save('AutoEncoder_Transformed.hdf5')


def evaluate(test_images):

    auto_encoder_old = tf.keras.models.load_model('AutoEncoder_Final.hdf5')
    auto_encoder_new = tf.keras.models.load_model('AutoEncoder_Transformed.hdf5')

    decoded_img_old = auto_encoder_old.predict(test_images)
    decoded_img_new = auto_encoder_new.predict(test_images)

    print("Reconstructed images for old model")
    n = 10
    plt.figure(figsize=(20, 4))
    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(test_images[i])
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_img_old[i].reshape(32, 32, 3))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

    print("Reconstructed images for new model")
    n = 10
    plt.figure(figsize=(20, 4))
    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(test_images[i])
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_img_new[i].reshape(32, 32, 3))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


def image_querying():
    auto_encoder_old = tf.keras.models.load_model('AutoEncoder_Final.hdf5')
    auto_encoder_new = tf.keras.models.load_model('AutoEncoder_Transformed.hdf5')

    #Reconstruct the encoder model for similar image retrieval

    encoder_old = tf.keras.models.Model(auto_encoder_old.input, auto_encoder_old.layers[13].output)
    encoder_new = tf.keras.models.Model(auto_encoder_new.input, auto_encoder_new.layers[13].output)

    query_images = x_test[np.random.choice(10000, size=2, replace=False)]

    query_encodings_old = encoder_old.predict(query_images)
    dataset_encodings_old = encoder_old.predict(x_dataset)

    similar_images_cosine_old = []

    for i in range(2):
        dist_cosine = cosine_distances(query_encodings_old[i].reshape(1, 10), dataset_encodings_old)
        dist_cosine = dist_cosine[0]
        indices_cosine = sorted(range(len(dist_cosine)), key=lambda sub: dist_cosine[sub])[:10]
        similar_images_cosine_old.append(indices_cosine)

    query_encodings_new = encoder_new.predict(query_images)
    dataset_encodings_new = encoder_new.predict(x_dataset)

    similar_images_cosine_new = []

    for i in range(2):
        dist_cosine = cosine_distances(query_encodings_new[i].reshape(1, 10), dataset_encodings_new)
        dist_cosine = dist_cosine[0]
        indices_cosine = sorted(range(len(dist_cosine)), key=lambda sub: dist_cosine[sub])[:10]
        similar_images_cosine_new.append(indices_cosine)

    print("Plotting the similar images for 2 random images using the old encoder model (Cosine distance)")
    plot_similar_images(similar_images_cosine_old, query_images, 2)

    print("Plotting the similar images for 2 random images using the old encoder model (Cosine distance)")
    plot_similar_images(similar_images_cosine_new, query_images, 2)


if __name__ == "__main__":

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    x_dataset = np.concatenate((x_train, x_test), axis=0)

    x_train_new, x_train_adapted, transformed_images_test = create_transformed_dataset()  #Commented out for final submission

    #print("Retraining with the new dataset containing transformed images!!!")

    #retrain_model()    #Commented out for final submission

    #print("Evaluating the reconstruction of image for the old and new encoder models!!!")

    #evaluate(transformed_images_test)

    print("Image querying:")

    image_querying()