import tensorflow as tf
from datetime import datetime
import numpy as np


def create_model():

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    latent_space_dim = 10

    logdir = "logs/scalars/retrainwitheight" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath='tmp/retrain10',
        save_weights_only=False,
        monitor='val_loss',
        mode='min',
        save_best_only=True
    )

    input_tensor = tf.keras.Input(shape=(32, 32, 3))

    encoder_layer = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(input_tensor)
    encoder_layer = tf.keras.layers.MaxPooling2D((2, 2))(encoder_layer)
    encoder_layer = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(encoder_layer)
    encoder_layer = tf.keras.layers.MaxPooling2D((2, 2))(encoder_layer)
    encoder_layer = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(encoder_layer)
    encoder_layer = tf.keras.layers.MaxPooling2D((2, 2))(encoder_layer)
    encoder_layer = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(encoder_layer)
    encoder_layer = tf.keras.layers.MaxPooling2D((2, 2))(encoder_layer)
    encoder_layer = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(encoder_layer)
    encoder_layer = tf.keras.layers.MaxPooling2D((2, 2))(encoder_layer)
    encoder_layer = tf.keras.layers.Flatten()(encoder_layer)
    encoder_layer = tf.keras.layers.Dense(32, activation='relu')(encoder_layer)
    encoder_output = tf.keras.layers.Dense(latent_space_dim)(encoder_layer)

    encoder_model = tf.keras.Model(input_tensor, encoder_output)
    print("Encoder model summary:")
    encoder_model.summary()

    decoder_layer = tf.keras.layers.Dense(27, activation='relu')(encoder_output)
    decoder_layer = tf.keras.layers.Reshape((3, 3, 3))(decoder_layer)
    decoder_layer = tf.keras.layers.Conv2D(8, (2, 2), activation='relu', padding='same')(decoder_layer)
    decoder_layer = tf.keras.layers.UpSampling2D((2, 2))(decoder_layer)
    decoder_layer = tf.keras.layers.Conv2D(16, (2, 2), activation='relu', padding='same')(decoder_layer)
    decoder_layer = tf.keras.layers.UpSampling2D((2, 2))(decoder_layer)
    decoder_layer = tf.keras.layers.Conv2D(32, (2, 2), activation='relu', padding='same')(decoder_layer)
    decoder_layer = tf.keras.layers.UpSampling2D((2, 2))(decoder_layer)
    decoder_layer = tf.keras.layers.Conv2D(64, (2, 2), activation='relu', padding='same')(decoder_layer)
    decoder_layer = tf.keras.layers.UpSampling2D((2, 2))(decoder_layer)
    decoder_layer = tf.keras.layers.Flatten()(decoder_layer)
    decoder_layer = tf.keras.layers.Dense(64, activation='relu')(decoder_layer)
    decoder_layer = tf.keras.layers.Dense(3072)(decoder_layer)
    decoder_output = tf.keras.layers.Reshape((32, 32, 3))(decoder_layer)

    auto_encoder = tf.keras.Model(input_tensor, decoder_output)

    print("Auto encoder model summary:")
    auto_encoder.summary()

    auto_encoder.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

    tf.keras.utils.plot_model(
        auto_encoder,
        to_file="autoencoder.png",
        show_shapes=False,
        show_layer_names=True,
        rankdir="TB",
        expand_nested=False,
        dpi=200,
    )

    training_history = auto_encoder.fit(x_train, x_train,
                                        epochs=100,
                                        batch_size=256,
                                        shuffle=True,
                                        callbacks=[tensorboard_callback, model_checkpoint_callback],
                                        validation_data=(x_test, x_test))

    encoder_model.save('Encoder_Final.hdf5')
    auto_encoder.save('AutoEncoder_Final.hdf5')

    print("Average test loss: ", np.average(training_history.history['loss']))


if __name__ == '__main__':
    create_model()
