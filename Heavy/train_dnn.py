import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
from collections import deque
import random
import time


# Model Hyper Parameters
BATCH_SIZE = 32
PHASES = 2
LANES = 12
CELLS = 12
MAX_SPEED = 10
LEARNING_RATE = 2e-4
BETA = 1e-3
GAMMA = 0.95
EPSILON = 0.1
N = 2000


def update_model_weights(TargetModel, TrainModel, beta=BETA):
    theta_array = TrainModel.get_weights()
    theta_prime_array = TargetModel.get_weights()
    output_weights = []

    for i in range(len(theta_array)):
        theta = theta_array[i]
        theta_prime = theta_prime_array[i]

        output_weights.append(beta*theta + (1-beta)*theta_prime)

    # Set these new weights to our target model
    TargetModel.set_weights(output_weights)


def init_model(num_lanes, num_cells, num_phases, model_name):
    """
    This function takes shape inputs and returns a neural network model
    that matches the structure found in the Gao research paper.
    :param num_lanes: Total number of lanes in the simulation (Int or Float)
    :param num_cells: Number of position cells per lane (Int or Float)
    :param num_phases: Number of light phases (Int or Float)
    :return: model: The tensorflow model for this dnn
    """

    position_input = tf.keras.Input(shape=(num_lanes, num_cells))
    velocity_input = tf.keras.Input(shape=(num_lanes, num_cells))
    light_phase_input = tf.keras.Input(shape=(1, num_phases))
    queue_len_input = tf.keras.Input(shape=(4,1))

    # First convolutional layer has 16 filters of size 4 by 4
    pos1 = tf.keras.layers.Conv1D(filters=num_lanes, strides=2,
                                  kernel_size=4,
                                  activation="relu")(position_input)

    vel1 = tf.keras.layers.Conv1D(filters=num_lanes, strides=2,
                                  kernel_size=4,
                                  activation="relu")(velocity_input)

    # Second Layer has 32 filters of size 2 by 2 with a stride of 1
    pos2 = tf.keras.layers.Conv1D(filters=num_lanes, strides=1, kernel_size=2,
                                  activation="relu")(pos1)
    vel2 = tf.keras.layers.Conv1D(filters=num_lanes, strides=1, kernel_size=2,
                                  activation="relu")(vel1)

    # Flatten our convolutional layers
    pos3 = tf.keras.layers.Flatten()(pos2)
    vel3 = tf.keras.layers.Flatten()(vel2)
    light3 = tf.keras.layers.Flatten()(light_phase_input)
    q_len3 = tf.keras.layers.Flatten()(queue_len_input)

    # Third layer concatenates position, velocity and light phase together
    combined = tf.keras.layers.concatenate([pos3, vel3, light3, q_len3], axis=1)
    layer3 = tf.keras.layers.Dense(128, activation="relu")(combined)

    # Layer 4: 64 nodes
    layer4 = tf.keras.layers.Dense(64, activation="relu")(layer3)

    # Output layer
    q_estimate = tf.keras.layers.Dense(num_phases, activation="linear")(layer4)

    # Create a model for the DNN
    model = tf.keras.Model(inputs=[position_input, velocity_input, light_phase_input, queue_len_input],
                               outputs=q_estimate, name=model_name)
    model.compile(optimizer="rmsprop", loss=None, metrics=None)
    #model.summary()

    return model


@tf.function
def train_step(x_train, x_test, reward, train_model, test_model, optimizer, num_phases):
    """
    This function takes input data for two different DNN models (same model structure)
    and updates the model weights of the training model according to the MSE loss and
    the optimizer specified for this function.

    """
    with tf.GradientTape() as tape:
        # Predict
        q = train_model(x_train)
        q_prime = test_model(x_test)

        # Reshape
        q = tf.reshape(q, [-1, num_phases])
        action = tf.reshape(x_test[2], [-1, num_phases])
        q_prime = tf.reshape(q_prime, [-1, num_phases])

        # Find max of q_prime
        q_prime_max = tf.math.reduce_max(q_prime, axis=1)
        #print("q_prime_max", q_prime_max)
        #print("reward", reward)

        # Find y values (Reward must be a float)
        y = tf.math.add(tf.cast(reward, tf.float32), tf.math.multiply(GAMMA, q_prime_max))

        # Find the q value of the chosen action
        q_action = tf.gather_nd(q, indices=tf.where(action == 1))

        # Find the MSE between y and q action
        loss = tf.keras.losses.MSE(y, q_action)

    # Find the gradients using tape
    grads = tape.gradient(loss, train_model.trainable_variables)
    optimizer.apply_gradients(zip(grads, train_model.trainable_variables))

    return loss


def train_batch(batch, train_model, test_model, optimizer, num_phases, verbose=False):
    """
    :param batch: A list of dictionaries. Each dictionary contains an experience from
                  a SUMO simulation
    :param train_model: This is the tensorflow model to be trained (DNN model)
    :param test_model: This is the tensorflow model used for evaluating the training model
                       Usually called the target model
    :param optimizer: This is a tensorflow optimizer to be used for training and updating model
                      weights for the train model
    :return: None
    """

    # First extract the test data and train data from the batch
    pos = np.zeros((BATCH_SIZE, LANES, CELLS))
    vel = np.zeros_like(pos)
    phase = np.zeros((BATCH_SIZE, 1, num_phases))
    q_len = np.zeros((BATCH_SIZE, 4, 1))
    rewards = np.zeros(BATCH_SIZE)
    pos2 = np.zeros_like(pos)
    vel2 = np.zeros_like(vel)
    phase2 = np.zeros_like(phase)
    q_len2 = np.zeros_like(q_len)

    for i in range(BATCH_SIZE):
        exp = batch[i]

        # Setting variables
        pos[i] = exp["pos"]
        vel[i] = exp["vel"]
        phase[i] = exp["phase"]
        q_len[i] = exp["qlen"]
        rewards[i] = exp["reward"]
        pos2[i] = exp["pos2"]
        vel2[i] = exp["vel2"]
        phase2[i] = exp["phase2"]
        q_len2[i] = exp["qlen2"]

    x_tr = [pos, vel, phase, q_len]
    x_te = [pos2, vel2, phase2, q_len2]

    # Set DNN model weights
    start = time.time()
    batch_mse = train_step(x_tr, x_te, rewards, train_model, test_model, optimizer, num_phases)

    # Set target model weights
    update_model_weights(test_model, train_model)
    stop = time.time()

    if verbose:
        print(f"Model weights updated on batch of {BATCH_SIZE} sample(s)")
        print(f"Training time {(stop-start)*1000:.1f} ms")
        tf.print("MSE:", batch_mse)


# ********************DEPRICATED***********************
def init_target_model(num_lanes, num_cells, num_phases):
    """
    This function takes three input shapes and returns a tensorflow
    model that follows the structure for the target model of the
    Gao paper
    :param num_lanes: Total number of incoming lanes in the simulation
    :param num_cells: The number of position cells per lane
    :param num_phases: The total number of light phases
    :return: target_model: The tensorflow model
    """

    # Input Layer
    pos_input_target = tf.keras.Input(shape=(num_lanes, num_cells))
    vel_input_target = tf.keras.Input(shape=(num_lanes, num_cells))
    phase_input_target = tf.keras.Input(shape=(1, num_phases))

    # Layer 1
    pos1_target = tf.keras.layers.Conv1D(filters=num_lanes, strides=2,
                                         kernel_size=4, activation="relu")(pos_input_target)
    vel1_target = tf.keras.layers.Conv1D(filters=num_lanes, strides=2,
                                         kernel_size=4, activation="relu")(vel_input_target)

    # Layer 2
    pos2_target = tf.keras.layers.Conv1D(filters=num_lanes * 2, strides=1, kernel_size=2,
                                         activation="relu")(pos1_target)
    vel2_target = tf.keras.layers.Conv1D(filters=num_lanes * 2, strides=1, kernel_size=2,
                                         activation="relu")(vel1_target)

    # Layer 3: Concatenation
    pos3_target = tf.keras.layers.Flatten()(pos2_target)
    vel3_target = tf.keras.layers.Flatten()(vel2_target)
    light3_target = tf.keras.layers.Flatten()(phase_input_target)
    combined_target = tf.keras.layers.concatenate([pos3_target, vel3_target, light3_target], axis=1)
    layer3_target = tf.keras.layers.Dense(128, activation="relu")(combined_target)

    # Layer 4
    layer4_target = tf.keras.layers.Dense(64, activation="relu")(layer3_target)

    # Target output
    q_prime = tf.keras.layers.Dense(num_phases)(layer4_target)

    # Create the target model
    target_model = tf.keras.Model(inputs=[pos_input_target, vel_input_target, phase_input_target],
                                  outputs=q_prime, name="target_model")
    target_model.compile(optimizer="rmsprop", loss=None, metrics=None)

    return target_model
# **********************************************************************

if __name__ == '__main__':

    position_data = np.random.randint(0, 2, (BATCH_SIZE, LANES, CELLS))
    velocity_data = MAX_SPEED * np.random.rand(BATCH_SIZE, LANES, CELLS)
    phase_data = np.zeros((BATCH_SIZE, 1, PHASES))
    reward = np.random.randint(-50, 50, BATCH_SIZE).astype("float32")
    position2 = np.random.randint(0, 2, (BATCH_SIZE, LANES, CELLS))
    velocity2 = MAX_SPEED * np.random.rand(BATCH_SIZE, LANES, CELLS)
    phase2 = np.zeros((BATCH_SIZE, 1, PHASES))

    for i in range(BATCH_SIZE):
        phase_data[i, 0, np.random.randint(0,2)] = 1
        phase2[i, 0, np.random.randint(0,2)] = 1

    print("Position shape:", position2.shape, "\tVelocity shape:", velocity2.shape,
          "\tPhase shape:", phase2.shape)

    exp = {"pos": position_data, "vel": velocity_data, "phase": phase_data,
           "reward": reward, "pos2": position2, "vel2": velocity2, "phase2": phase2}

    # Initialize our models
    target_model = init_model(LANES, CELLS, PHASES, model_name="target_model")
    dnn_model = init_model(LANES, CELLS, PHASES, model_name="dnn_model")

    # Initialize the optimizer
    opt = tf.keras.optimizers.RMSprop(learning_rate=LEARNING_RATE)

    # Testing tensorflow training method
    X_tr = [position_data, velocity_data, phase_data]
    X_te = [position2, velocity2, phase2]