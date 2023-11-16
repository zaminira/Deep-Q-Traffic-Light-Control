# -*- coding: utf-8 -*-
"""
Spyder Editor
Created by Zahra on 5.13.2021 for RL project to run the SUMO files

This is a temporary script file.
"""
import os
import sys
import optparse
import random
import numpy as np
from traci import edge as dg
from collections import deque
import random
from train_dnn import update_model_weights, init_model, train_batch, train_step
import tensorflow as tf
import time
import pickle
import matplotlib.pyplot as plt

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

# we need to import python modules from the $SUMO_HOME/tools directory
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

from sumolib import checkBinary  # noqa
import traci  # noqa

# We wnat to generate routes 
def generate_routefile():
    random.seed(42)  # make tests reproducible
    N = 3600  # number of time steps
    # demand per second from different directions
    pEWWE = 1. / 5 # from paper WE and EW  
    pSNNS = 1. / 10
    pRT = 1. / 30  # I picked this 
    pLT = 1. / 20
    with open("complex.rou.xml", "w") as routes:
        print("""<routes>
        <vType id="car" accel="0.8" decel="4.5" sigma="0" length="5" minGap="2" maxSpeed="90"/>
        <route id="W_LT" edges="51i 1i 3o 53o" />
        <route id="W_E" edges="51i 1i 1o 51o" />
        <route id="W_RT" edges="51i 1i 4o 54o" />
        
        <route id="E_W" edges="52i 2i 2o 52o" />
        <route id="E_RT" edges="52i 2i 3o 53o" />
        <route id="E_LT" edges="52i 2i 4o 54o" />
        
        <route id="S_N" edges="53i 3i 3o 53o" />
        <route id="S_LT" edges="53i 3i 2o 52o" />
        <route id="S_RT" edges="53i 3i 1o 51o" />
        
        <route id="N_S" edges="54i 4i 4o 54o" />
        <route id="N_LT" edges="54i 4i 1o 51o" />
        <route id="N_RT" edges="54i 4i 2o 52o" />""", file=routes)
        
        vehNr = 0
        for i in range(N):
            if random.uniform(0, 1) < pEWWE:
                print('    <vehicle id="EW_%i" type="car" route="E_W" depart="%i" />' % (
                    vehNr, i), file=routes)
                vehNr += 1
                print('    <vehicle id="WE_%i" type="car" route="W_E" depart="%i" />' % (
                    vehNr, i), file=routes)        
                vehNr += 1
                
            if random.uniform(0, 1) < pSNNS:
                print('    <vehicle id="SN_%i" type="car" route="S_N" depart="%i" />' % (
                    vehNr, i), file=routes)
                vehNr += 1
                print('    <vehicle id="NS_%i" type="car" route="N_S" depart="%i" />' % (
                    vehNr, i), file=routes)
                vehNr += 1
                
            if random.uniform(0, 1) < pRT:
                print('    <vehicle id="WRT_%i" type="car" route="W_RT" depart="%i" />' % (
                    vehNr, i), file=routes)
                vehNr += 1
                print('    <vehicle id="SRT_%i" type="car" route="S_RT" depart="%i" />' % (
                    vehNr, i), file=routes)
                vehNr += 1
                print('    <vehicle id="ERT_%i" type="car" route="E_RT" depart="%i" />' % (
                    vehNr, i), file=routes)
                vehNr += 1
                print('    <vehicle id="NRT_%i" type="car" route="N_RT" depart="%i" />' % (
                    vehNr, i), file=routes)
                vehNr += 1
                
            if random.uniform(0, 1) < pLT: # generates cars to turn left 
                print('    <vehicle id="WLT_%i" type="car" route="W_LT" depart="%i" />' % (
                    vehNr, i), file=routes)
                vehNr += 1
                print('    <vehicle id="SLT_%i" type="car" route="S_LT" depart="%i" />' % (
                    vehNr, i), file=routes)
                vehNr += 1
                print('    <vehicle id="ELT_%i" type="car" route="E_LT" depart="%i" />' % (
                    vehNr, i), file=routes)
                vehNr += 1
                print('    <vehicle id="NLT_%i" type="car" route="N_LT" depart="%i" />' % (
                    vehNr, i), file=routes)
                vehNr += 1
             
        print("</routes>", file=routes)
    return [vehNr]
 
def get_options():
   optParser = optparse.OptionParser()
   optParser.add_option("--nogui", action="store_true",
                        default=False, help="run the commandline version of sumo")
   options, args = optParser.parse_args()
   return options   


def states():
    cell=7# length of each cell
    roadLen=12 # road length that is covered
    position=np.zeros((12,12))
    speed=np.zeros((12,12))
    speedLimit=30
    startP=14.6 # 
    
    
    # It seems like the cars behind the lights start from 0 up to 480 
    
    # P0 we get the ID of all the vehicles in this lane from the last
    # simulation step. Then use this information to extract the position of each car 
    # The queue length would be always the sum of the row
    # P[R L U D]
    R_cars = traci.edge.getLastStepVehicleIDs('1i') #P0
    L_cars = traci.edge.getLastStepVehicleIDs('2i') #P1
    U_cars= traci.edge.getLastStepVehicleIDs('3i') #P2
    D_cars= traci.edge.getLastStepVehicleIDs('4i') #P3
    JunctionPosition= traci.junction.getPosition('0') # Gets the position of the junction
    
    # The position matrix is a 3*12 matrix for each lane 
    # The index starts [junction ---- outter most  ]
    R_position=np.zeros((3,12))
    R_speed=np.zeros((3,12))
    for car in R_cars:
        loc=round((abs((traci.vehicle.getPosition(car)[0])-(-startP)))/cell)
        if loc<roadLen:
            R_position[traci.vehicle.getLaneIndex(car)][loc] = 1
            R_speed[traci.vehicle.getLaneIndex(car)][loc] = traci.vehicle.getSpeed(car) / speedLimit
    R_position=np.array(R_position)
    R_pos=R_position.reshape((3,12))
    R_speed=np.array(R_speed)
    R_v=R_speed.reshape((3,12))
        
    L_position=np.zeros((3,12))
    L_speed=np.zeros((3,12))      
    for car in L_cars:
        loc=round((abs((traci.vehicle.getPosition(car)[0])-startP))/cell)
        if loc<roadLen:
            L_position[traci.vehicle.getLaneIndex(car)][loc] = 1
            L_speed[traci.vehicle.getLaneIndex(car)][loc] = traci.vehicle.getSpeed(car) / speedLimit
    L_position=np.array(L_position)
    L_pos=L_position.reshape((3,12))
    L_speed=np.array(L_speed)
    L_v=L_speed.reshape((3,12))
    
    
    U_position=np.zeros((3,12))
    U_speed=np.zeros((3,12))       
    for car in U_cars:
        loc=round((abs((traci.vehicle.getPosition(car)[1])-(-startP)))/cell)
        if loc<roadLen:
            U_position[traci.vehicle.getLaneIndex(car)][loc]=1
            U_speed[traci.vehicle.getLaneIndex(car)][loc] = traci.vehicle.getSpeed(car) / speedLimit
    U_position=np.array(U_position)
    U_pos=U_position.reshape((3,12))
    U_speed=np.array(U_speed)
    U_v=U_speed.reshape((3,12))
                                                       
    D_position=np.zeros((3,12))
    D_speed=np.zeros((3,12))                
    for car in D_cars:
        loc=round((abs((traci.vehicle.getPosition(car)[1])-startP))/cell)
        if loc<roadLen:
            D_position[traci.vehicle.getLaneIndex(car)][loc]=1
            D_speed[traci.vehicle.getLaneIndex(car)][loc] = traci.vehicle.getSpeed(car) / speedLimit
    D_position=np.array(D_position)
    D_pos=D_position.reshape((3,12))
    D_speed=np.array(D_speed)
    D_v=D_speed.reshape((3,12))
    
    position=np.concatenate((R_pos,U_pos,L_pos,D_pos))
    speed=np.concatenate((R_v,U_v,L_v,D_v))
    
    
                                                       
    qlen=np.zeros((4,1))     
    qlen[0] = traci.edge.getLastStepHaltingNumber('1i')
    qlen[1] = traci.edge.getLastStepHaltingNumber('3i')
    qlen[2] = traci.edge.getLastStepHaltingNumber('2i')
    qlen[3] = traci.edge.getLastStepHaltingNumber('4i')

    if traci.trafficlight.getPhase('0') == 0:
        # Light phase is green EW
        light = np.array([[0, 1]])
    elif traci.trafficlight.getPhase("0") == 4:
        # Green light for NS
        light = np.array([[1, 0]])
    else:
        # We are at some in between state
        # Either a yellow light or some left turn
        light = np.array([[0, 0]])
            
    return[position,speed,light,qlen]       
    
# for the static average waiting time by the default of the NetEdit    
def default_static(memory_deque, target_model, dnn_model, optimizer, epsilon=EPSILON):
    """execute the TraCI control loop"""

    print("Epsilon:", epsilon)
    # Keep track of total wait time for metrics
    twt = 0
    total_vehicles = 0

    # Find the traffic light id and phase duration
    light_id = traci.trafficlight.getIDList()[0]
    current_phase = traci.trafficlight.getPhase(light_id)
    switch_time = traci.trafficlight.getNextSwitch(light_id)

    # Initialize variables that can be referenced before assignment
    prev_wait = 0
    prev_pos = np.zeros((LANES, CELLS))
    prev_vel = np.zeros_like(prev_pos)
    prev_phase = np.zeros((1, PHASES))
    prev_qlen = np.zeros((4, 1))

    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()

        # Increment total waiting time and total vehicles
        twt += (traci.edge.getWaitingTime('1i') + traci.edge.getWaitingTime('2i')
                + traci.edge.getWaitingTime('3i') + traci.edge.getWaitingTime('4i'))

        total_vehicles += traci.simulation.getDepartedNumber()

        if traci.simulation.getTime() == switch_time:
            # We check for switch time - 1 because all the information
            # we receive about the simulation is delayed by one time step
            wTime = (traci.edge.getWaitingTime('1i') + traci.edge.getWaitingTime('2i')
                     + traci.edge.getWaitingTime('3i') + traci.edge.getWaitingTime('4i'))

            # Get state information
            [position, speed, light, qlen] = states()

            # Predict next action based on q values from the simulation
            q_val = target_model.predict([position.reshape(1, LANES, CELLS),
                                          speed.reshape(1, LANES, CELLS), light.reshape(1, 1, PHASES),
                                          qlen.reshape(1, 4, 1)])

            if np.random.rand() > epsilon:
                action = np.argwhere(q_val[0] == np.max(q_val[0]))[0,0]
                #print("Action", action, type(action))
            else:
                action = np.random.randint(0, 2, 1)[0]
                #print("EPSILON!!")

            #print("q val:", q_val)
            #print("action:", action)

            # Change the light
            if 4*action == traci.trafficlight.getPhase(light_id):
                traci.trafficlight.setPhase(light_id, str(4*action))
            else:
                #print("LIGHT CHANGE INITIATED")
                current_phase = traci.trafficlight.getPhase(light_id)
                traci.trafficlight.setPhase(light_id, str(1 + current_phase))

                # Take simulation step
                traci.simulationStep()

                # Increment total waiting time
                twt += (traci.edge.getWaitingTime('1i') + traci.edge.getWaitingTime('2i')
                        + traci.edge.getWaitingTime('3i') + traci.edge.getWaitingTime('4i'))

                total_vehicles += traci.simulation.getDepartedNumber()

                current_phase = traci.trafficlight.getPhase(light_id)
                while current_phase not in [0, 4]:
                    traci.simulationStep()

                    # Increment total waiting time
                    twt += (traci.edge.getWaitingTime('1i') + traci.edge.getWaitingTime('2i')
                            + traci.edge.getWaitingTime('3i') + traci.edge.getWaitingTime('4i'))

                    total_vehicles += traci.simulation.getDepartedNumber()

                    current_phase = traci.trafficlight.getPhase(light_id)

            # Get the new switch time
            switch_time = traci.trafficlight.getNextSwitch(light_id)
            #print("New switch time:", switch_time)

            if traci.simulation.getTime() >= 40:
                # Define an experience
                reward = prev_wait - wTime
                #print("Reward: ", reward)
                exp = {"pos": prev_pos, "vel": prev_vel, "phase": prev_phase,
                       "qlen": prev_qlen, "reward": reward, "pos2": position,
                       "vel2": speed, "phase2": light, "qlen2": qlen}

                # Append our experience to our memory
                memory_deque.append(exp)

            # Train our models
            if len(memory_deque) >= BATCH_SIZE:
                batch = random.choices(memory_deque, k=BATCH_SIZE)
                train_batch(batch, dnn_model, target_model, optimizer, PHASES, verbose=False)

            # Save state information for next iteration
            prev_pos = position
            prev_vel = speed
            prev_phase = light
            prev_qlen = qlen
            prev_wait = wTime

    print(f"Total wait time: {twt/3600:.1f} hours")
    print(f"Total vehicles: {total_vehicles}")

    return twt/total_vehicles


def validate_phase_static(phase):
    assert phase in [0, 1, 2, 3, 4, 5, 6, 7]

    light_id = traci.trafficlight.getIDList()[0]

    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        print("Step:", traci.simulation.getTime())


def random_rate(episode):

    if episode in range(10):
        epsilon = 0.5
    elif episode in range(10, 50):
        epsilon = 0.25
    elif episode in range(50, 250):
        epsilon = 0.1
    elif episode in range(250, 400):
        epsilon = 0.05
    else:
        epsilon = 0.01

    return epsilon

    
# this is the main entry point of this script
if __name__ == "__main__":
    options = get_options()

    # this script has been called from the command line. It will start sumo as a
    # server, then connect and run
    if options.nogui:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')

    # Testing no gui mode
    sumoBinary = "sumo"

    # Average total wait time for all episodes
    avg_twt = []

    # Initialize a deque to keep track of some experiences
    total_memory = 10000
    memory = deque([], maxlen=total_memory)

    # Initialize target and dnn models
    targetModel = init_model(LANES, CELLS, PHASES, model_name="target_model")
    dnnModel = init_model(LANES, CELLS, PHASES, model_name="dnn_model")
    rmsprop = tf.keras.optimizers.RMSprop(learning_rate=LEARNING_RATE)

    # A checkpoint path for saving the models at end
    train_path = "models/train_model/final_checkpoint"
    target_path = "models/target_model/final_checkpoint"

    # Add path for "BEST" model
    old_best_target_path = "models/target_model/best_model_checkpoint"
    new_best_target_path = "models/target_model/new_best_model_checkpoint"
    newest_best_target_path = "models/target_model/newest_best_model_checkpoint"
    best_avg_wait = np.inf
    checkpoint = 6

    # Check if we already have a best checkpoint and load in from that point
    #if os.path.isfile(old_best_target_path+".index"):
        #dnnModel.load_weights(old_best_target_path)

    # Set model weights to be the same
    targetModel.set_weights(dnnModel.get_weights())

    # Set total number of episodes for this run
    N = 2000

    for episode in range(N):
        # Generate Route file
        vehNr = generate_routefile()

        print("*"*20, f"SIMULATION {episode}", "*"*20)
        print("\n")

        traci.start([sumoBinary, "-c", "complex.sumocfg"])
        avg_wait = default_static(memory, targetModel, dnnModel, rmsprop, epsilon=EPSILON)
        traci.close()

        # average wait time
        avg_twt.append(avg_wait)
        print(f"Average vehicle wait time: {avg_wait:.2f} seconds")

        if avg_wait < best_avg_wait and episode > 100:
            # We have a new lowest avg wait time
            targetModel.save_weights(newest_best_target_path)
            best_avg_wait = avg_wait
            print("New shortest avg wait!")
        else:
            pass

    # Save model weights
    targetModel.save_weights(target_path)
    dnnModel.save_weights(train_path)
    print("Checkpoints saved")

    # Plot our average wait time vs episode
    plt.figure()
    plt.plot(range(N), avg_twt)
    plt.xlabel("Episodes")
    plt.ylabel("Average wait time per vehicle (seconds)")
    plt.title("Average Wait Time vs Episodes")
    plt.show()

    # Pickle our twt so we can adjust the plot later if we like
    with open("total_wait_and_range.pickle", "wb") as f:
        pickle.dump([range(N), avg_twt], f)

sys.stdout.flush()
