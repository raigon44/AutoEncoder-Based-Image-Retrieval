import tensorflow as tf
from datetime import datetime
import numpy as np
import config
import os


class Model:

    def __init__(self, latent_space_dim):
        self.tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=config.FileLocation.log_dir + datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=config.FileLocation.save_dir + str(len(next(os.walk(config.FileLocation.save_dir)))),
            save_weights_only=False,
            monitor='val_loss',
            mode='min',
            save_best_only=True
        )
        self.latent_space_dim = latent_space_dim
        self.auto_encoder_model, self.encoder_model = self.create_model()

    def create_model(self):
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
        encoder_output = tf.keras.layers.Dense(self.latent_space_dim)(encoder_layer)

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

        return auto_encoder, encoder_model

    def train_model(self, x_train, x_test):
        training_history = self.auto_encoder.fit(x_train, x_train,
                                                 epochs=config.ModelConfig.epochs,
                                                 batch_size=config.ModelConfig.batch_size,
                                                 shuffle=True,
                                                 callbacks=[self.tensorboard_callback, self.model_checkpoint_callback],
                                                 validation_data=(x_test, x_test))

        print("Average test loss: ", np.average(training_history.history['loss']))

        self.encoder_model.save('Encoder_Final.hdf5')
        self.auto_encoder.save('AutoEncoder_Final.hdf5')
