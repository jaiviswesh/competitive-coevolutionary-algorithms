import random
import numpy as np
import multiprocessing
import json
from functools import partial
from deap import base, creator, tools, algorithms
# Import helper functions from your updated hunted_sim
from hunted_sim import HuntedSim, example_config, apply_drone_genome, decode_prey

POP_SIZE = 70
NGEN = 75
CXPB, MUTPB, HOF_SIZE = 0.6, 0.2, 5

cfg = example_config()
DRONE_COUNT = sum([p['count'] for p in cfg['drones']])
PREY_COUNT = len(cfg['prey'])

# Bounds
SEP_MIN, SEP_MAX = 0.0, 7.0
ALI_MIN, ALI_MAX = 0.0, 3.0
COH_MIN, COH_MAX = 0.0, 1.0
SEARCH_MIN, SEARCH_MAX = -1.0, 1.0

# Setup DEAP
# Drone minimizes Fitness (Wants low score = fast catches)
if "FitnessMin" not in creator.__dict__: creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
if "Drone" not in creator.__dict__: creator.create("Drone", list, fitness=creator.FitnessMin)

# Prey maximizes Fitness (Wants high score = fast escape or long survival)
if "FitnessMax" not in creator.__dict__: creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if "Prey" not in creator.__dict__: creator.create("Prey", list, fitness=creator.FitnessMax)

def drone_init():
    genome = []
    for _ in range(DRONE_COUNT):
        genome.extend([random.uniform(SEP_MIN, SEP_MAX), random.uniform(ALI_MIN, ALI_MAX),
                       random.uniform(COH_MIN, COH_MAX), random.uniform(SEARCH_MIN, SEARCH_MAX)])
    return genome

def prey_init():
    genome = []
    for _ in range(PREY_COUNT):
        genome.extend([random.uniform(1, 5), random.uniform(5, 600), random.uniform(0, 4)])
    return genome

def evaluate(individual, opponent_genome, individual_type):
    try:
        sim = HuntedSim(example_config(), seed=random.randint(0, 10000))
        if individual_type == 'drone':
            apply_drone_genome(sim, individual)
            sim.update_prey_list(decode_prey(opponent_genome))
        else:
            apply_drone_genome(sim, opponent_genome)
            sim.update_prey_list(decode_prey(individual))
        stats, _ = sim.run()
        return (stats['F'],)
    except Exception:
        return (100000.0 if individual_type == 'drone' else 0.0,)

def main():
    toolbox_drone = base.Toolbox()
    toolbox_drone.register("individual", tools.initIterate, creator.Drone, drone_init)
    toolbox_drone.register("population", tools.initRepeat, list, toolbox_drone.individual)
    toolbox_drone.register("mate", tools.cxUniform, indpb=0.5)
    toolbox_drone.register("mutate", tools.mutGaussian, mu=0, sigma=0.5, indpb=0.1)
    toolbox_drone.register("select", tools.selTournament, tournsize=3)

    toolbox_prey = base.Toolbox()
    toolbox_prey.register("individual", tools.initIterate, creator.Prey, prey_init)
    toolbox_prey.register("population", tools.initRepeat, list, toolbox_prey.individual)
    toolbox_prey.register("mate", tools.cxBlend, alpha=0.5)
    toolbox_prey.register("mutate", tools.mutGaussian, mu=0, sigma=0.5, indpb=0.1)
    toolbox_prey.register("select", tools.selTournament, tournsize=3)
    
    # Enable parallel evaluation
    pool = multiprocessing.Pool()
    toolbox_drone.register("map", pool.map)
    toolbox_prey.register("map", pool.map)
    
    # Run simple evolution
    pop_drone = toolbox_drone.population(n=POP_SIZE)
    pop_prey = toolbox_prey.population(n=POP_SIZE)
    hof_drone = tools.HallOfFame(HOF_SIZE)
    hof_prey = tools.HallOfFame(HOF_SIZE)

    print("--- Training Started ---")
    for gen in range(NGEN):
        # Assess Drones
        best_prey = list(hof_prey)[0] if len(hof_prey) > 0 else None
        eval_drone = partial(evaluate, opponent_genome=best_prey, individual_type='drone')
        fits_d = toolbox_drone.map(eval_drone, pop_drone)
        for ind, fit in zip(pop_drone, fits_d): ind.fitness.values = fit
        hof_drone.update(pop_drone)
        
        # Assess Prey
        best_drone = list(hof_drone)[0] if len(hof_drone) > 0 else None
        eval_prey = partial(evaluate, opponent_genome=best_drone, individual_type='prey')
        fits_p = toolbox_prey.map(eval_prey, pop_prey)
        for ind, fit in zip(pop_prey, fits_p): ind.fitness.values = fit
        hof_prey.update(pop_prey)

        # Breed
        off_d = algorithms.varAnd(pop_drone, toolbox_drone, CXPB, MUTPB)
        pop_drone[:] = toolbox_drone.select(pop_drone + off_d, POP_SIZE)
        off_p = algorithms.varAnd(pop_prey, toolbox_prey, CXPB, MUTPB)
        pop_prey[:] = toolbox_prey.select(pop_prey + off_p, POP_SIZE)
        
        print(f"Gen {gen+1}/{NGEN} complete.")

    # SAVE TO JSON
    print("\n--- Saving Model to best_model.json ---")
    best_drone = list(hof_drone)[0]
    best_prey = list(hof_prey)[0]
    
    data = {
        "drone_genome": list(best_drone),
        "prey_genome": list(best_prey)
    }
    with open("best_model.json", "w") as f:
        json.dump(data, f)
    print("Done.")
    
    pool.close()
    pool.join()

if __name__ == "__main__":
    main()