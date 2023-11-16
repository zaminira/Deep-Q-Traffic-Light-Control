# -*- coding: utf-8 -*-
"""
Spyder Editor
Created by Zahra on 5.13.2021 for RL project to run the SUMO files

Thif file is created for our traffic light control RL project spring 2021.
Team members are Zahra , Dylan and Greesan. 
The program is out light traffic flow control 
"""


#nLibraries and packages required for running this program 
from __future__ import absolute_import
from __future__ import print_function

from IPython import get_ipython
get_ipython().magic('reset -sf')

import os
import sys
import optparse
import random
import numpy as np
from traci import edge as dg
#from traci.edge import getLastStepHaltingNumber as LHN

# we need to import python modules from the $SUMO_HOME/tools directory
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

from sumolib import checkBinary  # noqa
import traci  # noqa
import keras
from keras.layers import Dense, Flatten, Conv2D, Dropout,Input
from keras.optimizers import Adam
from keras.models import Model
from collections import deque
import tensorflow as tf
import matplotlib.pyplot as plt







############################################ CNN model

def CNN_model_build():
    
    # our CNN_model built of convolutional layer and dense layer and flattening
    init=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=43)
    input_position=Input(shape=(12,12,1))
    position=Conv2D(filters = 32, kernel_size = (2, 2), padding = 'valid', activation = 'relu')(input_position)
    position=Flatten()(position)
    
    input_speed=Input(shape=(12,12,1))
    speed=Conv2D(filters = 32, kernel_size = (2, 2), padding = 'valid', activation = 'relu')(input_speed)
    speed=Flatten()(speed)
    
    input_qlen=Input(shape=(4,1))
    qlen=Flatten()(input_qlen)

    input_light=Input(shape=(2,1))
    light=Flatten()(input_light)
    
    qValue=keras.layers.concatenate([position,speed,light])
    qValue=Dense(128,activation='relu')(qValue)
    qValue=Dense(64,activation='relu')(qValue)
    qValue=Dense(2,activation='relu')(qValue)
    
    CNN_model=Model(inputs=[input_position,input_speed,input_light],outputs=[qValue])
    #print(CNN_model.summary())
    
    # In this model we use Adam optimizer 
    init_lr = 1e-4 
    optimizer = Adam(lr = init_lr)
    CNN_model.compile(optimizer = optimizer, loss = 'mse') # So this takes the mean square error 
    return CNN_model

##################################################### Useful functions 
    

# this function saves the experience in memory which will be used to update the NN
# for 1 hour the done is False then turns True >> true when the episode is over 
def memory_push(buffer, state, action, reward, next_state, done):
    experience = (state, action, reward , next_state, done)   # make sure that reward is a numpy array 
    buffer.append(experience)

# our greedy policy to be greedy with probability epsilon    
def get_action(state):
    epsilon= 0.1
    Num_actions=2
    r=np.random.rand()
    if r<epsilon:
        action=random.randrange(Num_actions)
    else:
        predict_action=CNN_model.predict(state) # takes the states as the input 
        action=np.argmax(predict_action) 
    return action

    
#### ENVIRONMENT ###################################################################################

# We wnat to generate routes 
def generate_routefile():
    random.seed(42)  # make tests reproducible
    N =3600   #575 
    # demand per second from different directions
    pEWWE = 1. / 5 # from paper WE and EW  
    pSNNS = 1. / 10
    pRT = 1. / 30
    pLT = 1. / 20
    with open("complex2.rou.xml", "w") as routes:
        print("""<routes>
        <vType id="car" accel="0.8" decel="4.5" sigma="0.5" length="5" minGap="2" maxSpeed="70"/>
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
                print('    <vehicle id="SN_%i" type="car" route="S_N" depart="%i" color="0,1,0" />' % (
                    vehNr, i), file=routes)
                vehNr += 1
                print('    <vehicle id="NS_%i" type="car" route="N_S" depart="%i" color="0,1,0" />' % (
                    vehNr, i), file=routes)
                vehNr += 1
                
            if random.uniform(0, 1) < pRT:
                print('    <vehicle id="WRT_%i" type="car" route="W_RT" depart="%i" color="0,0,1" />' % (
                    vehNr, i), file=routes)
                vehNr += 1
                print('    <vehicle id="SRT_%i" type="car" route="S_RT" depart="%i" color="0,0,1"/>' % (
                    vehNr, i), file=routes)
                vehNr += 1
                print('    <vehicle id="ERT_%i" type="car" route="E_RT" depart="%i" />' % (
                    vehNr, i), file=routes)
                vehNr += 1
                print('    <vehicle id="NRT_%i" type="car" route="N_RT" depart="%i" color="1,0,1"/>' % (
                    vehNr, i), file=routes)
                vehNr += 1
                
            if random.uniform(0, 1) < pLT: # generates cars to turn left 
                print('    <vehicle id="WLT_%i" type="car" route="W_LT" depart="%i" color="1,0,1"/>' % (
                    vehNr, i), file=routes)
                vehNr += 1
                print('    <vehicle id="SLT_%i" type="car" route="S_LT" depart="%i" color="1,1,1" />' % (
                    vehNr, i), file=routes)
                vehNr += 1
                print('    <vehicle id="ELT_%i" type="car" route="E_LT" depart="%i" color="1,0,1" />' % (
                    vehNr, i), file=routes)
                vehNr += 1
                print('    <vehicle id="NLT_%i" type="car" route="N_LT" depart="%i" color="0,1,1" />' % (
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
#### ENVIRONMENT DONE ##################################################################
   

#### Getting STATES #################################################################
def states():
    cell=7# length of each cell
    roadLen=12 # road length that is covered
    position=np.zeros((12,12))
    speed=np.zeros((12,12))
    speedLimit=30
    startP=14.6 # 
    
    R_cars = traci.edge.getLastStepVehicleIDs('1i') #P0
    L_cars = traci.edge.getLastStepVehicleIDs('2i') #P1
    U_cars= traci.edge.getLastStepVehicleIDs('3i') #P2
    D_cars= traci.edge.getLastStepVehicleIDs('4i') #P3
    
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
    
    # the position matrix is a 12*12 matrix 
    position=np.concatenate((R_pos,U_pos,L_pos,D_pos))
    speed=np.concatenate((R_v,U_v,L_v,D_v))

    # qlen matrix of the length of each queue at the roads                                                   
    qlen=np.zeros((4,1))     
    qlen[0] = traci.edge.getLastStepHaltingNumber('1i') # right cars
    qlen[1] = traci.edge.getLastStepHaltingNumber('3i') # up cars
    qlen[2] = traci.edge.getLastStepHaltingNumber('2i') #left cars
    qlen[3] = traci.edge.getLastStepHaltingNumber('4i') # down cars
    qlen=np.array(qlen)
    
    light=[]
    if (traci.trafficlight.getPhase('0')==4): #NS is green
        light=[0,1] # from paper when green light is on for NS 
    else:
        light=[1,0] # Green light is on for EW 
    light=np.array(light)
    
    
    # Setting up the input dimesions of the NN 
    position = np.array(position)
    position = position.reshape(1, 12, 12, 1)
    
    speed = np.array(speed)
    speed = speed.reshape(1, 12, 12, 1)
    
    light = np.array(light)
    light= light.reshape(1, 2, 1)
    
    qlen=np.array(qlen)
    qlen=qlen.reshape(1,4,1)
       
    return[position,speed,light] 

def q_length():
    qlen=np.zeros((4,1))     
    qlen[0] = traci.edge.getLastStepHaltingNumber('1i')
    qlen[1] = traci.edge.getLastStepHaltingNumber('3i')
    qlen[2] = traci.edge.getLastStepHaltingNumber('2i')
    qlen[3] = traci.edge.getLastStepHaltingNumber('4i')
    qlen=np.array(qlen)

    return qlen
    
#### RETURN STATES ########################################################################### 


# for the static average waiting time of the NetEdit  has green time of 10 s , and yellow of 6s  
def default_static():
    """execute the TraCI control loop"""
    step = 0
    wTime=0 # The wainting time of all the cars in the simualtion
    while traci.simulation.getMinExpectedNumber() > 0:
        wTime+=(traci.edge.getLastStepHaltingNumber('1i') + traci.edge.getLastStepHaltingNumber(
            '2i') + traci.edge.getLastStepHaltingNumber('3i') + traci.edge.getLastStepHaltingNumber('4i'))
        traci.simulationStep() 
        step+=1
    return [wTime,step]


# Running the simulation with our trained agent 
def tarined_NN():
    
    CNN_model=CNN_model_build()
    CNN_model.load_weights("weight_file_new.h5")
    wTime=0
    gtau=10
    ytau=6
    s=0
    epsilon= 0.1
    Num_actions=2
 
    while traci.simulation.getMinExpectedNumber() > 0:    
 
        
        wTime+=(traci.edge.getLastStepHaltingNumber('1i') + traci.edge.getLastStepHaltingNumber(
            '2i') + traci.edge.getLastStepHaltingNumber('3i') + traci.edge.getLastStepHaltingNumber('4i'))
        traci.simulationStep() 
        state=states()
        light=state[2] 
        r=np.random.rand()
        if r<epsilon:
            action=random.randrange(Num_actions)
        else:
            predict_action=CNN_model.predict(state) # takes the states as the input 
            action=np.argmax(predict_action)

                
        if (action==0 and light[0][0][0]== 1) : # does nothing just keep the EW green on 
            for i in range(gtau):
                s+=1
                traci.trafficlight.setPhase("0", 0)
                wTime+=(traci.edge.getLastStepHaltingNumber('1i') + traci.edge.getLastStepHaltingNumber(
                    '2i') + traci.edge.getLastStepHaltingNumber('3i') + traci.edge.getLastStepHaltingNumber('4i'))
                traci.simulationStep()
            
        if (action==0 and light[0][0][0]==0):  # has to change the light phase to 0 transition 

            # the transition 
            for i in range(ytau):
                s+=1
                traci.trafficlight.setPhase("0", 5)
                wTime+=(traci.edge.getLastStepHaltingNumber('1i') + traci.edge.getLastStepHaltingNumber(
                    '2i') + traci.edge.getLastStepHaltingNumber('3i') + traci.edge.getLastStepHaltingNumber('4i'))
                traci.simulationStep()
                
            for i in range(gtau):
                s+=1
                traci.trafficlight.setPhase("0", 6)
                wTime+=(traci.edge.getLastStepHaltingNumber('1i') + traci.edge.getLastStepHaltingNumber(
                    '2i') + traci.edge.getLastStepHaltingNumber('3i') + traci.edge.getLastStepHaltingNumber('4i'))
                traci.simulationStep()
                
            for i in range(ytau):
                s+=1
                traci.trafficlight.setPhase("0", 7)
                wTime+=(traci.edge.getLastStepHaltingNumber('1i') + traci.edge.getLastStepHaltingNumber(
                    '2i') + traci.edge.getLastStepHaltingNumber('3i') + traci.edge.getLastStepHaltingNumber('4i'))
                traci.simulationStep()


            for i in range(gtau):
                s+=1
                traci.trafficlight.setPhase("0", 0)
                wTime+=(traci.edge.getLastStepHaltingNumber('1i') + traci.edge.getLastStepHaltingNumber(
                    '2i') + traci.edge.getLastStepHaltingNumber('3i') + traci.edge.getLastStepHaltingNumber('4i'))
                traci.simulationStep()
                
                
            # second condition when action is 1 >> NS green which is phase 4
        if (action==1 and light[0][1][0]== 1) : # does nothing just keep the NS green on 
            
            for i in range(gtau):
                s+=1
                traci.trafficlight.setPhase("0", 4)
                wTime+=(traci.edge.getLastStepHaltingNumber('1i') + traci.edge.getLastStepHaltingNumber(
                    '2i') + traci.edge.getLastStepHaltingNumber('3i') + traci.edge.getLastStepHaltingNumber('4i'))
                traci.simulationStep()   
                
                
        if (action==1 and light[0][1][0]==0):  # has to change the light phase to 4 transition 
            
            for i in range(ytau):
                s+=1
                traci.trafficlight.setPhase("0", 1)
                wTime+=(traci.edge.getLastStepHaltingNumber('1i') + traci.edge.getLastStepHaltingNumber(
                    '2i') + traci.edge.getLastStepHaltingNumber('3i') + traci.edge.getLastStepHaltingNumber('4i'))
                traci.simulationStep()
                
            for i in range(gtau):
                s+=1
                traci.trafficlight.setPhase("0", 2)
                wTime+=(traci.edge.getLastStepHaltingNumber('1i') + traci.edge.getLastStepHaltingNumber(
                    '2i') + traci.edge.getLastStepHaltingNumber('3i') + traci.edge.getLastStepHaltingNumber('4i'))
                traci.simulationStep()
                
            for i in range(ytau):
                s+=1
                traci.trafficlight.setPhase("0", 3)
                wTime+=(traci.edge.getLastStepHaltingNumber('1i') + traci.edge.getLastStepHaltingNumber(
                    '2i') + traci.edge.getLastStepHaltingNumber('3i') + traci.edge.getLastStepHaltingNumber('4i'))
                traci.simulationStep()

            for i in range(gtau):
                s+=1
                traci.trafficlight.setPhase("0", 4)
                wTime+=(traci.edge.getLastStepHaltingNumber('1i') + traci.edge.getLastStepHaltingNumber(
                    '2i') + traci.edge.getLastStepHaltingNumber('3i') + traci.edge.getLastStepHaltingNumber('4i'))
                traci.simulationStep()

    return [wTime,s]
    

  

# this is the main entry point of this script
if __name__ == "__main__":
    options = get_options()

    # this script has been called from the command line. It will start sumo as a
    # server, then connect and run. 
    if options.nogui:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')
    #sumoBinary = checkBinary('sumo')
    # first, generate the route file for this simulation
    vehNr=generate_routefile()
    #traci.start([sumoBinary, "-c", "complex.sumocfg", '--start'])  #"--tripinfo-output", "tripinfo.xml"
    # this is the normal way of using traci. sumo is started as a
    # subprocess and then the python script connects and runs
    
    # This section runs the cyclic policy and returns the avergae waiting time of
    # all the vehicles. 
    traci.start([sumoBinary, "-c", "complex2.sumocfg", "--tripinfo-output", "tripinfo.xml"])
    #[wTime,step]=tarined_NN()
    [wTime,step]=default_static()
    ave_w=wTime
    print("Average delay time of each vehicle in seconds:")
    print(wTime/1830)
    traci.close(wait=False)
sys.stdout.flush()






    
    