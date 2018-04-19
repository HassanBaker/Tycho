params = {
    "input_shape": [batch_size, image_size, image_size, channels],

    "conv1": {
        "num_channels": 3,
        "output_size": 32,
        "filter_size": 6,
        "pooling": 2,
        "name": "conv1"

    },

    "conv2": {
        "num_channels": 32,
        "output_size": 64,
        "filter_size": 5,
        "pooling": 2,
        "name": "conv2"

    },

    "conv3": {
        "num_channels": 64,
        "output_size": 128,
        "filter_size": 3,
        "pooling": None,
        "name": "conv3"

    },

    "conv4": {
        "num_channels": 128,
        "output_size": 128,
        "filter_size": 3,
        "pooling": 2,
        "name": "conv4"

    },

    "dense1": {
        "weight_shape": [512, 128],
        "bias_shape": [128],
        "weight_stddev": 0.01,
        "activation": "relu",
        "name": "dense1"
    },
    "dense2": {
        "weight_shape": [128, 64],
        "bias_shape": [64],
        "weight_stddev": 0.01,
        "activation": "relu",
        "name": "dense2"
    },

    "dense3": {
        "weight_shape": [64, num_labels],
        "bias_shape": [num_labels],
        "weight_stddev": 0.001,
        "activation": "sigmoid",
        "name": "dense3"
    },

    "output_shape": [16, num_labels],

    "loss": "OLS",

    "train_step": {
        "type": "Adam",
        "learning_rate": 0.004,
        "momentum": 0.9
    },

    "error_estimator": "RMSE"
}