class FileLocation:
    log_dir = "logs/scalars/train_"
    save_dir = "savedModel/model"


class ModelConfig:
    latent_space_dim = 10
    epochs = 65
    batch_size = 256
    optimizer = 'adam'
    metrics = 'accuracy'
    loss = 'mse'


class RandomNoiseConfig:
    mode = 'gaussian'
    mean = 0.1
    var = 0.01
    clip = True