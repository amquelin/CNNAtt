import tensorflow as tf
from tensorflow.keras import layers, models

def create_model(params, network_params):
    """
    Function to create and compile a model based on provided parameters.
    """
    input1 = layers.Input(shape=(params["nb_gen"], params["num_snps"], 1))  # Population 1
    input2 = layers.Input(shape=(params["nb_gen"], params["num_snps"], 1))  # Population 2
    input_pos = layers.Input(shape=(1, params["num_snps"], 1))  # Positions

    # Convolutional layers
    x1 = layers.Conv2D(network_params["nb_fil_inputs"], network_params["kernel_pop"], activation='relu')(input1)
    x2 = layers.Conv2D(network_params["nb_fil_inputs"], network_params["kernel_pop"], activation='relu')(input2)
    pos = layers.Conv2D(network_params["nb_fil_inputs"], network_params["kernel_pos"], activation='relu')(input_pos)

    # MaxPooling
    x1 = layers.MaxPooling2D(pool_size=network_params["pool_size_inputs"])(x1)
    x2 = layers.MaxPooling2D(pool_size=network_params["pool_size_inputs"])(x2)
    pos = layers.MaxPooling2D(pool_size=network_params["pool_size_inputs"])(pos)

    # Average over dimension 0 for each population
    x1_avg = layers.Lambda(
        lambda x: tf.reduce_mean(x, axis=1, keepdims=True),
        output_shape=lambda input_shape: (input_shape[0], 1, input_shape[2], input_shape[3])
    )(x1)

    x2_avg = layers.Lambda(
        lambda x: tf.reduce_mean(x, axis=1, keepdims=True),
        output_shape=lambda input_shape: (input_shape[0], 1, input_shape[2], input_shape[3])
    )(x2)

    # Concatenate the results of the populations and positions
    x_concat = layers.concatenate([x1_avg, x2_avg, pos], axis=1)

    # Post concatenation
    x = layers.Conv2D(network_params["nb_fil_pc"], network_params["kernel_pc"], activation='relu')(x_concat)
    x = layers.MaxPooling2D(pool_size=network_params["pool_size_pc"])(x)

    # Flatten to pass to the dense part
    x = layers.Flatten()(x)

    # Fully connected part
    x = layers.Dense(network_params["dense_units"][0], activation='relu')(x)
    x = layers.Dense(network_params["dense_units"][1], activation='relu')(x)
    x = layers.Dense(network_params["dense_units"][2], activation='relu')(x)

    # Output of the model
    output = layers.Dense(len(params["target"]))(x)

    # Create the model
    model = models.Model(inputs=[input1, input2, input_pos], outputs=output)

    # Compile the model
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=network_params["learning_rate"],
        decay_steps=network_params["decay_steps"],
        decay_rate=network_params["decay_rate"]
    )

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                  loss='mse', metrics=['mae'])

    return model

