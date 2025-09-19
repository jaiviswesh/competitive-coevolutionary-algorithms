import random
import numpy as np
import multiprocessing

from deap import base, creator, tools, algorithms
from hunted_sim import HuntedSim, example_config

# --- GA parameters ---
POP_SIZE = 20
NGEN = 50  # keep small for testing
CXPB = 1.0

# --- Predator genome: prox radii for each predator ---
PREDATOR_COUNT = sum([p['count'] for p in example_config()['predators']])
PREDATOR_MIN = 5.0
PREDATOR_MAX = 100.0

# --- Escaper genome ---
ESCAPER_COUNT = len(example_config()['escapers'])
BORDERS = ['N', 'S', 'E', 'W']
COORD_MIN, COORD_MAX = 0.0, 400.0
SPEED_MIN, SPEED_MAX = 1.0, 5.0
TIME_MIN, TIME_MAX = 0.0, 600.0
AVOID_MIN, AVOID_MAX = 0.0, 2.0

# --- Hall of Fame size ---
HOF_SIZE = 10

# --- Predator individual: list of prox radii ---
def predator_init():
    return [random.uniform(PREDATOR_MIN, PREDATOR_MAX) for _ in range(PREDATOR_COUNT)]

# --- Escaper individual ---
def escaper_init():
    genome = []
    for _ in range(ESCAPER_COUNT):
        border = random.choice(BORDERS)
        coord = random.uniform(COORD_MIN, COORD_MAX)
        speed = random.uniform(SPEED_MIN, SPEED_MAX)
        time = random.uniform(TIME_MIN, TIME_MAX)
        avoidance = random.uniform(AVOID_MIN, AVOID_MAX)
        genome.extend([BORDERS.index(border), coord, speed, time, avoidance])
    return genome

# --- Decode escaper genome into list of dicts for simulator ---
def decode_escaper(genome):
    escapers = []
    for i in range(0, len(genome), 5):
        border = BORDERS[int(genome[i]) % 4]
        coord = genome[i+1]
        speed = genome[i+2]
        time = genome[i+3]
        avoidance = genome[i+4]
        escapers.append({
            'border': border,
            'coord': coord,
            'speed': speed,
            'escape_time': time,
            'avoidance': avoidance
        })
    return escapers

# --- Evaluation functions ---
def evaluate_predator(pred_ind, escapers=None):
    cfg = example_config()
    idx = 0
    for p in cfg['predators']:
        for _ in range(p['count']):
            p['prox_r'] = pred_ind[idx]
            idx += 1
    if escapers:
        cfg['escapers'] = escapers
    sim = HuntedSim(cfg)
    stats, _ = sim.run()
    return (stats['F'],)

def evaluate_escaper(esc_ind, predators=None):
    cfg = example_config()
    cfg['escapers'] = decode_escaper(esc_ind)
    if predators:
        idx = 0
        for p in cfg['predators']:
            for _ in range(p['count']):
                p['prox_r'] = predators[idx]
                idx += 1
    sim = HuntedSim(cfg)
    stats, _ = sim.run()
    return (stats['F'],)

# --- Escaper mutation (custom, handles mix of discrete + floats) ---
def mutate_escaper(individual, indpb_border=0.02, indpb_float=0.1):
    for i in range(0, len(individual), 5):
        # border index
        if random.random() < indpb_border:
            individual[i] = random.randrange(0, len(BORDERS))
        # coord
        if random.random() < indpb_float:
            individual[i+1] = random.uniform(COORD_MIN, COORD_MAX)
        # speed
        if random.random() < indpb_float:
            individual[i+2] = random.uniform(SPEED_MIN, SPEED_MAX)
        # time
        if random.random() < indpb_float:
            individual[i+3] = random.uniform(TIME_MIN, TIME_MAX)
        # avoidance
        if random.random() < indpb_float:
            individual[i+4] = random.uniform(AVOID_MIN, AVOID_MAX)
    return (individual,)

# --- DEAP creator setup ---
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Predator", list, fitness=creator.FitnessMin)
creator.create("Escaper", list, fitness=creator.FitnessMin)

# --- Toolbox setup ---
toolbox_pred = base.Toolbox()
toolbox_pred.register("individual", tools.initIterate, creator.Predator, predator_init)
toolbox_pred.register("population", tools.initRepeat, list, toolbox_pred.individual)
toolbox_pred.register("mate", tools.cxUniform, indpb=0.5)
toolbox_pred.register(
    "mutate",
    tools.mutPolynomialBounded,
    low=PREDATOR_MIN,
    up=PREDATOR_MAX,
    eta=20.0,
    indpb=1.0/PREDATOR_COUNT
)
toolbox_pred.register("select", tools.selTournament, tournsize=3)

toolbox_esc = base.Toolbox()
toolbox_esc.register("individual", tools.initIterate, creator.Escaper, escaper_init)
toolbox_esc.register("population", tools.initRepeat, list, toolbox_esc.individual)
toolbox_esc.register("mate", tools.cxUniform, indpb=0.5)
toolbox_esc.register("mutate", mutate_escaper)
toolbox_esc.register("select", tools.selTournament, tournsize=3)

# --- Parallelization ---
pool = multiprocessing.Pool()
toolbox_pred.register("map", pool.map)
toolbox_esc.register("map", pool.map)

# --- Main coevolution loop ---
def coevolution():
    pop_pred = toolbox_pred.population(n=POP_SIZE)
    pop_esc = toolbox_esc.population(n=POP_SIZE)
    hof_pred = tools.HallOfFame(HOF_SIZE)
    hof_esc = tools.HallOfFame(HOF_SIZE)

    for gen in range(NGEN):
        # evaluate predators
        fitnesses_pred = toolbox_pred.map(evaluate_predator, pop_pred)
        for ind, fit in zip(pop_pred, fitnesses_pred):
            ind.fitness.values = fit
        hof_pred.update(pop_pred)

        # evaluate escapers
        fitnesses_esc = toolbox_esc.map(evaluate_escaper, pop_esc)
        for ind, fit in zip(pop_esc, fitnesses_esc):
            ind.fitness.values = fit
        hof_esc.update(pop_esc)

        # GA steps
        pop_pred = toolbox_pred.select(
            algorithms.varAnd(pop_pred, toolbox_pred, CXPB, 1.0/PREDATOR_COUNT),
            POP_SIZE
        )
        pop_esc = toolbox_esc.select(
            algorithms.varAnd(pop_esc, toolbox_esc, CXPB, 1.0/(ESCAPER_COUNT*5)),
            POP_SIZE
        )

        print(f"Gen {gen}: Predator best {hof_pred[0].fitness.values[0]:.2f}, Escaper best {hof_esc[0].fitness.values[0]:.2f}")

    return hof_pred, hof_esc

if __name__ == "__main__":
    hof_pred, hof_esc = coevolution()
    print("Best predator config:", hof_pred[0])
    print("Best escaper config:", hof_esc[0])
    pool.close(); pool.join()
