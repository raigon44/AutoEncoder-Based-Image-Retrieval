from model import Model
import data_utils
import config


def main():

    # Let's load and perform a sanity check to the dataset
    x_train, y_train, x_test, y_test = data_utils.prepare_data()
    data_utils.check_data_properties(x_train, y_train, x_test, y_test)

    model_obj = Model(config.ModelConfig)

    # Let's start the training
    model_obj.train_model(x_train, x_test)

    return


if __name__ == '__main__':
    main()





