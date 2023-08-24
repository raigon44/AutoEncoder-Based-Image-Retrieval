import tensorflow as tf


def prepare_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    return x_train, y_train, x_test, y_test


def check_data_properties(x_train, y_train, x_test, y_test):
    # x_train, y_train, x_test, y_test = prepare_data()
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


if __name__ == "__main__":
    check_data_properties()
