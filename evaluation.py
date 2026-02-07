import numpy as np
from hunted_sim import HuntedSim, example_config, apply_drone_genome, decode_prey
import json

def load_model():
    try:
        with open("best_model.json", "r") as f:
            data = json.load(f)
        return data['drone_genome'], data['prey_genome']
    except:
        return None, None

def evaluate(runs=100):
    print(f"--- Running {runs} Simulations ---")
    drone_genome, prey_genome = load_model()
    
    # Storage for metrics
    intercept_distances = [] # How far from base were they caught?
    catch_times = []         # How long did it take?
    swarm_dispersions = []   # How spread out were the drones?
    total_caught = 0
    total_infiltrated = 0

    for i in range(runs):
        sim = HuntedSim(example_config(), seed=i)
        
        if drone_genome:
            apply_drone_genome(sim, drone_genome)
            sim.update_prey_list(decode_prey(prey_genome))

        # Run Simulation
        stats, trajectories = sim.run(record_trajectories=True) # Enable recording for dispersion
        
        # 1. Capture & Infiltration Counts
        total_caught += stats['n_detected']
        total_infiltrated += stats['n_prey_escaped']

        # 2. Analyze Specific Events
        for p in sim.prey:
            if p.detected:
                # Security Margin: Distance from Base when caught
                dist_to_base = np.linalg.norm(p.pos - p.target_pos)
                intercept_distances.append(dist_to_base)
                
                # Time Efficiency
                catch_times.append(p.done_tick if p.done_tick else sim.max_ticks)

        # 3. Analyze Drone Behavior (Dispersion)
        # We check the final frame to see if they are spread out
        drone_positions = np.array([d.pos for d in sim.drones])
        # Standard Deviation of X and Y positions
        spread_x = np.std(drone_positions[:, 0])
        spread_y = np.std(drone_positions[:, 1])
        swarm_dispersions.append(np.mean([spread_x, spread_y]))

    # --- Print Report ---
    avg_security = np.mean(intercept_distances) if intercept_distances else 0
    avg_time = np.mean(catch_times) if catch_times else 0
    avg_spread = np.mean(swarm_dispersions)
    accuracy = (total_caught / (total_caught + total_infiltrated)) * 100

    print("\n" + "="*40)
    print("       SYSTEM PERFORMANCE REPORT       ")
    print("="*40)
    print(f"Accuracy (Capture Rate):   {accuracy:.1f}%")
    print(f"Intruders Caught:          {total_caught}")
    print(f"Intruders Infiltrated:     {total_infiltrated}")
    print("-" * 40)
    print("1. SECURITY MARGIN (Higher is Better)")
    print(f"   Avg Intercept Distance: {avg_security:.1f} meters")
    print("   (Distance from base when intruder was stopped)")
    print("-" * 40)
    print("2. TIME EFFICIENCY (Lower is Better)")
    print(f"   Avg Time to Catch:      {avg_time:.1f} ticks")
    print("-" * 40)
    print("3. SWARM DISPERSION (Higher is Better)")
    print(f"   Avg Spread Index:       {avg_spread:.1f}")
    print("   (>200 = Good Patrol, <100 = Clumping/Camping)")
    print("="*40)

if __name__ == "__main__":
    evaluate(runs=50)