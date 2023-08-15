class FileLocation:
    log_dir = "logs/scalars/retrainwitheight"
    save_dir = "savedModel/model"


class ModelConfig:
    latent_space_dimension = 10
    epochs = 65
    batch_size = 256


class RandomNoiseConfig:
    mode = 'gaussian'
    mean = 0.1
    var = 0.01
    clip = True