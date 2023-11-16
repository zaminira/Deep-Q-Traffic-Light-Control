import os
import sys
import optparse
import random
import numpy as np
import traci
from traci import edge as dg
from collections import deque
import random
from train_dnn import update_model_weights, init_model, train_batch, train_step
from complex import states
import tensorflow as tf
import time

# Model Hyper Parameters
BATCH_SIZE = 32
LEARNING_RATE = 2e-4
BETA = 1e-3
GAMMA = 0.95
EPSILON = 0.1
N = 2000

# Simulation Hyper Parameters
PHASES = 2
LANES = 12
CELLS = 12


def eval_model(target_model):
    # Find the traffic light id and phase duration
    light_id = traci.trafficlight.getIDList()[0]
    current_phase = traci.trafficlight.getPhase(light_id)
    switch_time = traci.trafficlight.getNextSwitch(light_id)

    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()

        if traci.simulation.getTime() == switch_time:

            # Get state information
            [position, speed, light, qlen] = states()

            # Predict next action based on q values from the simulation
            q_val = target_model.predict([position.reshape(1, LANES, CELLS),
                                          speed.reshape(1, LANES, CELLS), light.reshape(1, 1, PHASES),
                                          qlen.reshape(1, 4, 1)])

            if np.random.rand() > EPSILON:
                action = np.argwhere(q_val[0] == np.max(q_val[0]))[0, 0]
                # print("Action", action, type(action))
            else:
                action = np.random.randint(0, 2, 1)[0]
                # print("EPSILON!!")

            # print("q val:", q_val)
            # print("action:", action)

            # Change the light
            if 4 * action == traci.trafficlight.getPhase(light_id):
                traci.trafficlight.setPhase(light_id, str(4 * action))
            else:
                # print("LIGHT CHANGE INITIATED")
                current_phase = traci.trafficlight.getPhase(light_id)
                traci.trafficlight.setPhase(light_id, str(1 + current_phase))

                # Take simulation step
                traci.simulationStep()

                current_phase = traci.trafficlight.getPhase(light_id)
                while current_phase not in [0, 4]:
                    traci.simulationStep()
                    current_phase = traci.trafficlight.getPhase(light_id)

            # Get the new switch time
            switch_time = traci.trafficlight.getNextSwitch(light_id)
            # print("New switch time:", switch_time)


if __name__ == '__main__':

    # Initialize models
    target_model = init_model(LANES, CELLS, PHASES, model_name="target_model")
    dnn_model = init_model(LANES, CELLS, PHASES, model_name="dnn_model")

    # Load the weights
    #train_path = "models/train_model/my_checkpoint"
    target_path = "models/target_model/new_best_model_checkpoint"
    target_model.load_weights(target_path)
    #dnn_model.load_weights(train_path)

    # Open sumo gui
    sumoBinary = "sumo-gui"

    traci.start([sumoBinary, "-c", "complex.sumocfg"])
    eval_model(target_model)
    traci.close()
