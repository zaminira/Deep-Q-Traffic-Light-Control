import pickle
import matplotlib.pyplot as plt
import traci


def default_static():
    wait_time = 0
    total_vehicles = 0
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        wait_time += (traci.edge.getWaitingTime('1i') + traci.edge.getWaitingTime('2i')
                      + traci.edge.getWaitingTime('3i') + traci.edge.getWaitingTime('4i'))

        total_vehicles += traci.simulation.getDepartedNumber()

    return wait_time/total_vehicles


# Find the average waiting time of cars from the static case
traci.start(["sumo", "-c", "complex.sumocfg"])
avg_wait_static = default_static()
traci.close()

with open("total_wait.pickle", "rb") as f:
    avg_twt = pickle.load(f)

plt.figure()
plt.plot(range(500), avg_twt, label="Deep Q Agent")
plt.plot(range(500), [avg_wait_static]*500, linestyle="--", color="r", label="Cyclic Phase")
plt.ylim([0, 1000])
plt.xlabel("Episodes")
plt.ylabel("Average wait time per vehicle (seconds)")
plt.title("Average Wait Time vs Episode")
plt.legend()
plt.show()

with open("best_graph.pickle", "wb") as f:
    pickle.dump([range(500), avg_twt], f)
