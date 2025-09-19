# coevolve.py
import random
import numpy as np
import multiprocessing
from functools import partial

# DEAP
from deap import base, creator, tools, algorithms

# Import your simulator and example config
from hunted_sim import HuntedSim, example_config

# -------------------------
# User-tunable parameters
# -------------------------
POP_SIZE = 20
NGEN = 100            # small by default; paper used ~1000 gens with many evals
CXPB = 1.0
# number of parallel workers
NWORKERS = max(1, multiprocessing.cpu_count() - 1)

# How many opponent HOF members to sample per evaluation
HOF_SAMPLE_SIZE = 5

# How many simulation repeats per evaluation (to reduce stochastic noise)
SIM_REPEATS = 3

# ---------- Build problem-specific bounds from example_config ----------
cfg_template = example_config()
PREDATOR_COUNT = sum([p['count'] for p in cfg_template['predators']])
ESCAPER_COUNT = len(cfg_template['escapers'])
MAP_W = cfg_template['map_w']
MAP_H = cfg_template['map_h']

# Predator prox radius bounds (meters)
PREDATOR_MIN = 5.0
PREDATOR_MAX = 100.0

# Escaper parameter bounds
BORDERS = ['N', 'S', 'E', 'W']         # border encoded as index 0..3
COORD_MIN = 0.0
COORD_MAX = MAP_W                      # 0..map width/height
SPEED_MIN = 1.0
SPEED_MAX = 5.0
TIME_MIN = 0.0
TIME_MAX = cfg_template['sim_time']    # within sim duration
AVOID_MIN = 0.0
AVOID_MAX = 2.0

# Hall of fame size
HOF_SIZE = 30

# -------------------------
# DEAP setup (safe-guard re-creation)
# -------------------------
if "FitnessMin" not in creator.__dict__:
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

# predator and escaper individual classes
if "Predator" not in creator.__dict__:
    creator.create("Predator", list, fitness=creator.FitnessMin)
if "Escaper" not in creator.__dict__:
    creator.create("Escaper", list, fitness=creator.FitnessMin)

toolbox_pred = base.Toolbox()
toolbox_esc = base.Toolbox()

# -------------------------
# Genotype initializers
# -------------------------
def predator_init():
    # list of floats: prox radii for each predator agent
    return [random.uniform(PREDATOR_MIN, PREDATOR_MAX) for _ in range(PREDATOR_COUNT)]

def escaper_init():
    # genome layout: for each escaper -> [border_index, coord, speed, time, avoidance]
    genome = []
    for _ in range(ESCAPER_COUNT):
        border_idx = random.randrange(0, len(BORDERS))
        coord = random.uniform(COORD_MIN, COORD_MAX)
        speed = random.uniform(SPEED_MIN, SPEED_MAX)
        time = random.uniform(TIME_MIN, TIME_MAX)
        avoidance = random.uniform(AVOID_MIN, AVOID_MAX)
        genome.extend([border_idx, coord, speed, time, avoidance])
    return genome

# register
toolbox_pred.register("individual", tools.initIterate, creator.Predator, predator_init)
toolbox_pred.register("population", tools.initRepeat, list, toolbox_pred.individual)
toolbox_esc.register("individual", tools.initIterate, creator.Escaper, escaper_init)
toolbox_esc.register("population", tools.initRepeat, list, toolbox_esc.individual)

# -------------------------
# Genetic operators
# -------------------------
# Predators: floats -> use mutUniformFloat
toolbox_pred.register("mate", tools.cxUniform, indpb=0.5)
toolbox_pred.register(
    "mutate",
    tools.mutUniformFloat,
    low=PREDATOR_MIN,
    up=PREDATOR_MAX,
    indpb=1.0 / PREDATOR_COUNT
)
toolbox_pred.register("select", tools.selTournament, tournsize=3)

# Escapers: genome contains discrete (border idx) + floats.
# We'll apply two mutations: mutate border with small prob (random choice) and mutate floats with mutUniformFloat.
def mutate_escaper(individual, indpb_border=0.02, indpb_float=0.1):
    # each escaper has 5 entries
    L = len(individual)
    for i in range(0, L, 5):
        # border slot
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

toolbox_esc.register("mate", tools.cxUniform, indpb=0.5)
toolbox_esc.register("mutate", mutate_escaper, indpb_border=0.02, indpb_float=1.0/(ESCAPER_COUNT*5))
toolbox_esc.register("select", tools.selTournament, tournsize=3)

# -------------------------
# Helpers: decode genomes -> simulation config
# -------------------------
def decode_predator_genome(pred_genome, cfg):
    """Apply predator proximity radii from genome into a copy of cfg and return new cfg."""
    cfg_copy = dict(cfg)  # shallow; we'll deep-copy predators list
    cfg_copy['predators'] = [dict(p) for p in cfg['predators']]
    idx = 0
    for p in cfg_copy['predators']:
        # for each agent in this swarm set prox_r to genome[idx]
        p.setdefault('prox_r', PREDATOR_MIN)
        # update all agents in this swarm individually from genome
        # Note: the original example_config bundles swarms; we will assign genome values sequentially to each agent
        for _ in range(p['count']):
            p['prox_r'] = pred_genome[idx]
            idx += 1
    return cfg_copy

def decode_escaper_genome(esc_genome):
    """Return list of escaper dicts compatible with example_config()['escapers'] entries."""
    escapers = []
    L = len(esc_genome)
    assert L == ESCAPER_COUNT * 5
    for i in range(0, L, 5):
        border_idx = int(round(esc_genome[i])) % len(BORDERS)
        coord = float(esc_genome[i+1])
        speed = float(esc_genome[i+2])
        time = float(esc_genome[i+3])
        avoidance = float(esc_genome[i+4])
        escapers.append({
            'border': BORDERS[border_idx],
            'coord': float(np.clip(coord, COORD_MIN, COORD_MAX)),
            'speed': float(np.clip(speed, SPEED_MIN, SPEED_MAX)),
            'escape_time': float(np.clip(time, TIME_MIN, TIME_MAX)),
            'avoidance': float(np.clip(avoidance, AVOID_MIN, AVOID_MAX))
        })
    return escapers

# -------------------------
# Evaluation functions
# -------------------------
def evaluate_predator(pred_individual, esc_opponents, base_cfg, sim_repeats=SIM_REPEATS, seed_base=0):
    """
    Evaluate predator individual against a list of escaper individuals (genomes).
    Returns a 1-tuple (mean_F,) because we minimize F.
    """
    scores = []
    for esc_ind in esc_opponents:
        # decode esc genome to escaper cfg
        esc_list = decode_escaper_genome(esc_ind)
        # create cfg with escapers set
        cfg = dict(base_cfg)
        cfg['predators'] = [dict(p) for p in base_cfg['predators']]
        # apply predator prox radii from pred_individual
        cfg = decode_predator_genome(pred_individual, cfg)
        # place the escapers list into cfg (HINT: example_config uses dicts; our sim's init expects 'escapers' as list of dicts)
        cfg['escapers'] = esc_list
        # run a few repeats with different seeds to smooth stochasticity
        for r in range(sim_repeats):
            sim = HuntedSim(cfg, seed=seed_base + r)
            stats, _ = sim.run(record_trajectories=False)
            scores.append(stats['F'])
    # average across all matchups & repeats
    if len(scores) == 0:
        return (1e9,)   # large penalty if nothing evaluated
    return (float(np.mean(scores)),)

def evaluate_escaper(esc_individual, pred_opponents, base_cfg, sim_repeats=SIM_REPEATS, seed_base=10000):
    """Evaluate escaper individual against list of predator genomes. Return 1-tuple (mean_F,)."""
    scores = []
    for pred_ind in pred_opponents:
        # build cfg: use base_cfg predators overridden with pred_ind values
        cfg = dict(base_cfg)
        cfg['predators'] = [dict(p) for p in base_cfg['predators']]
        cfg = decode_predator_genome(pred_ind, cfg)
        # set escapers: the evaluated escaper plus (optionally) others -- here we just evaluate all escapers as clones of this
        # Simpler: create cfg['escapers'] from decode_escaper_genome(esc_ind)
        cfg['escapers'] = decode_escaper_genome(esc_ind)
        for r in range(sim_repeats):
            sim = HuntedSim(cfg, seed=seed_base + r)
            stats, _ = sim.run(record_trajectories=False)
            # For escaper, a higher F means predators did worse -> in our minimization formulation we keep same metric
            scores.append(stats['F'])
    if len(scores) == 0:
        return (1e9,)
    return (float(np.mean(scores)),)

# -------------------------
# Parallelization: register toolbox.map with a Pool
# -------------------------
pool = multiprocessing.Pool(NWORKERS)
toolbox_pred.register("map", pool.map)
toolbox_esc.register("map", pool.map)

# -------------------------
# Main coevolution loop
# -------------------------
def coevolution():
    base_cfg = example_config()

    pop_pred = toolbox_pred.population(n=POP_SIZE)
    pop_esc = toolbox_esc.population(n=POP_SIZE)
    hof_pred = tools.HallOfFame(HOF_SIZE)
    hof_esc = tools.HallOfFame(HOF_SIZE)

    # Pre-evaluate initial random populations (so fitness attributes exist)
    # We'll use a small random sample of opponents for initial evaluation
    for ind in pop_pred:
        ind.fitness.values = evaluate_predator(ind, [toolbox_esc.individual() for _ in range(HOF_SAMPLE_SIZE)], base_cfg)
    for ind in pop_esc:
        ind.fitness.values = evaluate_escaper(ind, [toolbox_pred.individual() for _ in range(HOF_SAMPLE_SIZE)], base_cfg)

    hof_pred.update(pop_pred)
    hof_esc.update(pop_esc)

    for gen in range(1, NGEN+1):
        # --- Evaluate predators against a sample from escaper HOF or population ---
        if len(hof_esc) >= 1:
            esc_pool = list(hof_esc)
        else:
            esc_pool = pop_esc
        esc_sample = random.sample(esc_pool, min(HOF_SAMPLE_SIZE, len(esc_pool)))

        # Evaluate predator population in parallel
        eval_fun_pred = partial(evaluate_predator, esc_opponents=esc_sample, base_cfg=base_cfg)
        fitnesses_pred = toolbox_pred.map(eval_fun_pred, pop_pred)
        for ind, fit in zip(pop_pred, fitnesses_pred):
            ind.fitness.values = fit
        hof_pred.update(pop_pred)

        # --- Evaluate escapers against a sample from predator HOF or population ---
        if len(hof_pred) >= 1:
            pred_pool = list(hof_pred)
        else:
            pred_pool = pop_pred
        pred_sample = random.sample(pred_pool, min(HOF_SAMPLE_SIZE, len(pred_pool)))

        eval_fun_esc = partial(evaluate_escaper, pred_opponents=pred_sample, base_cfg=base_cfg)
        fitnesses_esc = toolbox_esc.map(eval_fun_esc, pop_esc)
        for ind, fit in zip(pop_esc, fitnesses_esc):
            ind.fitness.values = fit
        hof_esc.update(pop_esc)

        # --- Variation (crossover + mutation) and selection ---
        offspring_pred = algorithms.varAnd(pop_pred, toolbox_pred, CXPB, 1.0 / PREDATOR_COUNT)
        offspring_esc = algorithms.varAnd(pop_esc, toolbox_esc, CXPB, 1.0 / (ESCAPER_COUNT*5))

        pop_pred = toolbox_pred.select(offspring_pred, POP_SIZE)
        pop_esc = toolbox_esc.select(offspring_esc, POP_SIZE)

        # Print progress (average fitness)
        avg_pred = np.mean([ind.fitness.values[0] for ind in pop_pred])
        avg_esc = np.mean([ind.fitness.values[0] for ind in pop_esc])
        print(f"Gen {gen:3d}: Predator avg F = {avg_pred:.2f}, Escaper avg F = {avg_esc:.2f}")

    return hof_pred, hof_esc

if __name__ == "__main__":
    hof_pred, hof_esc = coevolution()
    print("Best predator config (prox radii):", hof_pred[0])
    print("Best escaper genome:", hof_esc[0])
    pool.close()
    pool.join()
