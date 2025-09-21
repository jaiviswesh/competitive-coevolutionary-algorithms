import random
import numpy as np
import multiprocessing
from functools import partial
import traceback
import sys
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from deap import base, creator, tools, algorithms
# This script requires the final version of hunted_sim.py
from hunted_sim import HuntedSim, example_config

# --- GA parameters ---
POP_SIZE = 20
NGEN = 500
CXPB = 0.9  # Crossover probability

# --- Genome & Simulation Constants ---
cfg = example_config()
DRONE_COUNT = sum([p['count'] for p in cfg['drones']])
PREY_COUNT = len(cfg['prey'])

# Bounds for the evolvable Boids parameters
SEP_MIN, SEP_MAX = 0.0, 3.0  # Separation weight
ALI_MIN, ALI_MAX = 0.0, 3.0  # Alignment weight
COH_MIN, COH_MAX = 0.0, 3.0  # Cohesion weight

# Prey genome constants
SPEED_MIN, SPEED_MAX = 1.0, 5.0
TIME_MIN, TIME_MAX = 0.0, cfg['sim_time']
AVOID_MIN, AVOID_MAX = 0.0, 3.0

HOF_SIZE = 5

# --- DEAP Creator Setup ---
if "FitnessMin" not in creator.__dict__:
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
if "Drone" not in creator.__dict__:
    creator.create("Drone", list, fitness=creator.FitnessMin)
if "FitnessMax" not in creator.__dict__:
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if "Prey" not in creator.__dict__:
    creator.create("Prey", list, fitness=creator.FitnessMax)

# --- Genome Initializers ---
def drone_init():
    """Genome is a flat list of [sep1, ali1, coh1, sep2, ali2, coh2, ...]"""
    genome = []
    for _ in range(DRONE_COUNT):
        genome.extend([
            random.uniform(SEP_MIN, SEP_MAX),
            random.uniform(ALI_MIN, ALI_MAX),
            random.uniform(COH_MIN, COH_MAX),
        ])
    return genome

def prey_init():
    """Genome defines evolved behaviors like speed, start time, and avoidance."""
    genome = []
    for _ in range(PREY_COUNT):
        genome.extend([
            0,  # Placeholder for unused border_idx gene
            0,  # Placeholder for unused coord gene
            random.uniform(SPEED_MIN, SPEED_MAX),
            random.uniform(TIME_MIN, TIME_MAX),
            random.uniform(AVOID_MIN, AVOID_MAX),
        ])
    return genome

# --- Genome Decoders ---
def apply_drone_genome(sim, drone_genome):
    """Assigns the evolved Boids weights to each drone instance in the simulator."""
    if drone_genome is None: return
    for i, drone in enumerate(sim.drones):
        idx = i * 3
        if idx + 2 < len(drone_genome):
            drone.sep_weight = drone_genome[idx]
            drone.ali_weight = drone_genome[idx + 1]
            drone.coh_weight = drone_genome[idx + 2]

def decode_prey(prey_genome):
    """Takes a prey genome and returns a list of config dictionaries."""
    if prey_genome is None: return example_config()['prey']
    prey_list = []
    for i in range(0, len(prey_genome), 5):
        prey_list.append({
            'speed': float(np.clip(prey_genome[i+2], SPEED_MIN, SPEED_MAX)),
            'escape_time': float(np.clip(prey_genome[i+3], TIME_MIN, TIME_MAX)),
            'avoidance': float(np.clip(prey_genome[i+4], AVOID_MIN, AVOID_MAX)),
        })
    return prey_list

# --- Evaluation Function ---
def evaluate(individual, opponent_genome, individual_type):
    """Runs a simulation and returns the fitness score."""
    try:
        cfg = example_config()
        sim = HuntedSim(cfg, seed=random.randint(0, 10000))

        if individual_type == 'drone':
            apply_drone_genome(sim, individual)
            sim.update_prey_list(decode_prey(opponent_genome))
        else: # individual is prey
            apply_drone_genome(sim, opponent_genome)
            sim.update_prey_list(decode_prey(individual))
        
        stats, _ = sim.run()
        print(".", end='', flush=True)
        return (stats['F'],) # Return the score directly for both types
    except Exception:
        traceback.print_exc()
        # Return a penalized (worst possible) fitness on crash
        return (0.0,) if individual_type == 'drone' else (0.0,)

# --- Custom Mutation for Prey ---
def mutate_prey(individual, indpb=0.1):
    for i in range(0, len(individual), 5):
        if random.random() < indpb: individual[i+2] = random.uniform(SPEED_MIN, SPEED_MAX)
        if random.random() < indpb: individual[i+3] = random.uniform(TIME_MIN, TIME_MAX)
        if random.random() < indpb: individual[i+4] = random.uniform(AVOID_MIN, AVOID_MAX)
    return (individual,)

# --- Bounds enforcing decorator for Drone mutation ---
def check_bounds(min_bounds, max_bounds):
    """A decorator to clamp mutated values within the defined bounds."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            offspring = func(*args, **kwargs)
            for i, (min_val, max_val) in enumerate(zip(min_bounds, max_bounds)):
                offspring[0][i] = np.clip(offspring[0][i], min_val, max_val)
            return offspring
        return wrapper
    return decorator

# --- Plotting Function ---
def plot_final_simulation(best_drone, best_prey):
    """Runs one simulation with the best evolved agents and plots the result."""
    print("\n--- Running final simulation to generate plot ---")
    cfg = example_config()
    sim = HuntedSim(cfg, seed=42)
    apply_drone_genome(sim, best_drone)
    sim.update_prey_list(decode_prey(best_prey))

    stats, traj = sim.run(record_trajectories=True)
    print("Stats from plotted simulation:", stats)

    plt.figure(figsize=(8, 8))
    for base_pos in sim.bases:
        plt.scatter(base_pos[0], base_pos[1], c='green', marker='s', s=100, zorder=5)
    
    drone_label_set, prey_label_set = False, False
    for tlist in traj['drone'].values():
        arr = np.array(tlist)
        if arr.shape[0] > 0:
            label = 'Drone' if not drone_label_set else '_nolegend_'
            plt.plot(arr[:,0], arr[:,1], color='blue', lw=0.7, alpha=0.8, label=label)
            drone_label_set = True
    for tlist in traj['prey'].values():
        arr = np.array(tlist)
        if arr.shape[0] > 0:
            label = 'Prey' if not prey_label_set else '_nolegend_'
            plt.plot(arr[:,0], arr[:,1], '--', color='red', lw=1.5, label=label)
            plt.scatter(arr[0,0], arr[0,1], color='red', marker='x', s=100)
            prey_label_set = True

    handles, labels = plt.gca().get_legend_handles_labels()
    base_handle = Line2D([0],[0], marker='s', color='w', label='Base', mfc='green', ms=10)
    if "Base" not in labels: handles.append(base_handle)
    
    plt.xlim(0, cfg['map_w']); plt.ylim(0, cfg['map_h'])
    plt.title('Base Defense Trajectories (Best Evolved Agents)'); plt.xlabel('X Coordinate'); plt.ylabel('Y Coordinate')
    plt.grid(True, linestyle=':', alpha=0.6); plt.legend(handles=handles)
    plt.gca().set_aspect('equal', adjustable='box'); plt.show()

# --- Main Execution ---
def main():
    # --- Drone Toolbox Setup ---
    toolbox_drone = base.Toolbox()
    toolbox_drone.register("individual", tools.initIterate, creator.Drone, drone_init)
    toolbox_drone.register("population", tools.initRepeat, list, toolbox_drone.individual)
    toolbox_drone.register("mate", tools.cxUniform, indpb=0.5) # Switched to stable uniform crossover
    # Switched to stable Gaussian mutation with a bounds decorator
    low_bounds = [SEP_MIN, ALI_MIN, COH_MIN] * DRONE_COUNT
    up_bounds = [SEP_MAX, ALI_MAX, COH_MAX] * DRONE_COUNT
    toolbox_drone.register("mutate", tools.mutGaussian, mu=0, sigma=0.5, indpb=0.1)
    toolbox_drone.decorate("mutate", check_bounds(low_bounds, up_bounds))
    toolbox_drone.register("select", tools.selTournament, tournsize=3)

    # --- Prey Toolbox Setup ---
    toolbox_prey = base.Toolbox()
    toolbox_prey.register("individual", tools.initIterate, creator.Prey, prey_init)
    toolbox_prey.register("population", tools.initRepeat, list, toolbox_prey.individual)
    toolbox_prey.register("mate", tools.cxBlend, alpha=0.5)
    toolbox_prey.register("mutate", mutate_prey, indpb=0.1)
    toolbox_prey.register("select", tools.selTournament, tournsize=3)
    
    # --- Evolution Start ---
    pool = multiprocessing.Pool()
    toolbox_drone.register("map", pool.map)
    toolbox_prey.register("map", pool.map)
    
    pop_drone = toolbox_drone.population(n=POP_SIZE)
    pop_prey = toolbox_prey.population(n=POP_SIZE)
    hof_drone = tools.HallOfFame(HOF_SIZE)
    hof_prey = tools.HallOfFame(HOF_SIZE)
    
    print("--- Starting Co-evolution (Boids Model) ---")
    for gen in range(NGEN):
        best_prey_genome = list(hof_prey)[0] if len(hof_prey) > 0 else None
        best_drone_genome = list(hof_drone)[0] if len(hof_drone) > 0 else None

        eval_drone_func = partial(evaluate, opponent_genome=best_prey_genome, individual_type='drone')
        fitnesses_drone = toolbox_drone.map(eval_drone_func, pop_drone)
        for ind, fit in zip(pop_drone, fitnesses_drone): ind.fitness.values = fit
        hof_drone.update(pop_drone)

        eval_prey_func = partial(evaluate, opponent_genome=best_drone_genome, individual_type='prey')
        fitnesses_prey = toolbox_prey.map(eval_prey_func, pop_prey)
        for ind, fit in zip(pop_prey, fitnesses_prey): ind.fitness.values = fit
        hof_prey.update(pop_prey)

        offspring_drone = algorithms.varAnd(pop_drone, toolbox_drone, CXPB, 1.0)
        pop_drone[:] = toolbox_drone.select(offspring_drone, k=len(offspring_drone))
        
        offspring_prey = algorithms.varAnd(pop_prey, toolbox_prey, CXPB, 1.0)
        pop_prey[:] = toolbox_prey.select(offspring_prey, k=len(offspring_prey))

        drone_best_F = hof_drone[0].fitness.values[0] if len(hof_drone) > 0 else float('nan')
        prey_best_F = hof_prey[0].fitness.values[0] if len(hof_prey) > 0 else float('nan')
        print(f"\nGen {gen:02d}: Drone Best F (min): {drone_best_F:.2f} | Prey Best F (max): {prey_best_F:.2f}")

    print("\n--- Co-evolution Finished ---")
    best_drone = list(hof_drone)[0] if len(hof_drone) > 0 else None
    best_prey = list(hof_prey)[0] if len(hof_prey) > 0 else None

    if best_drone: print("Best drone genome:", best_drone)
    if best_prey: print("Best prey genome:", best_prey)

    pool.close()
    pool.join()
    
    if best_drone and best_prey:
        plot_final_simulation(best_drone, best_prey)

if __name__ == "__main__":
    main()