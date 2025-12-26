import json
import numpy as np
from hunted_sim import HuntedSim, example_config, apply_drone_genome, decode_prey

def load_best_model():
    try:
        with open("best_model.json", "r") as f:
            data = json.load(f)
        return data['drone_genome'], data['prey_genome']
    except FileNotFoundError:
        print("WARNING: best_model.json not found. Using random parameters.")
        return None, None

def evaluate_accuracy(runs=100):
    print(f"--- Benchmarking System ({runs} Episodes) ---")
    
    drone_genome, prey_genome = load_best_model()
    
    total_caught = 0
    total_escaped = 0
    total_time_steps = 0
    
    for i in range(runs):
        # 1. Init Simulation
        # We use a different seed every time to randomize Base and Prey locations
        sim = HuntedSim(example_config(), seed=i)
        
        # 2. Apply the trained brains (if available)
        if drone_genome:
            apply_drone_genome(sim, drone_genome)
            sim.update_prey_list(decode_prey(prey_genome))
            
        # 3. Run until finished
        stats, _ = sim.run(record_trajectories=False)
        
        total_caught += stats['n_detected']
        total_escaped += stats['n_prey_escaped']
        # Estimate duration based on ticks (just for info)
        total_time_steps += sim.current_tick
        
        # Optional: Print progress every 10 runs
        if (i+1) % 10 == 0:
            print(f"Run {i+1}/{runs} completed...")

    # --- Calculate Final Metrics ---
    total_intruders = total_caught + total_escaped
    accuracy = (total_caught / total_intruders) * 100 if total_intruders > 0 else 0
    avg_ticks = total_time_steps / runs
    
    print("\n" + "="*30)
    print("FINAL SURVEILLANCE METRICS")
    print("="*30)
    print(f"Scenario:      Blind Search & Intercept")
    print(f"Total Runs:    {runs}")
    print(f"Total Prey:    {total_intruders}")
    print("-" * 30)
    print(f"Caught:        {total_caught}")
    print(f"Infiltrated:   {total_escaped}")
    print("-" * 30)
    print(f"ACCURACY:      {accuracy:.2f}%")
    print(f"Avg Time:      {avg_ticks:.1f} ticks")
    print("="*30)

    if accuracy < 50:
        print("\n[!] RECOMMENDATION: Accuracy is low. You should run 'ga_coevolution.py'")
        print("    to train the drones specifically for this Blind Search scenario.")
    else:
        print("\n[+] System is performing well.")

if __name__ == "__main__":
    evaluate_accuracy(runs=50) # Runs 50 simulations to get an average