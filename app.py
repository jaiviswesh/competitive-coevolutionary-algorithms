from flask import Flask, jsonify, render_template, request
import json
import numpy as np
from hunted_sim import HuntedSim, example_config, apply_drone_genome, decode_prey

app = Flask(__name__)

current_sim = None

def load_best_model():
    try:
        with open("best_model.json", "r") as f:
            data = json.load(f)
        return data['drone_genome'], data['prey_genome']
    except FileNotFoundError:
        print("WARNING: best_model.json not found. Using random behaviors.")
        return None, None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/reset', methods=['POST'])
def reset_game():
    global current_sim
    cfg = example_config()
    current_sim = HuntedSim(cfg)
    
    drone_genome, prey_genome = load_best_model()
    if drone_genome:
        apply_drone_genome(current_sim, drone_genome)
        current_sim.update_prey_list(decode_prey(prey_genome))
        print("Model Loaded and Applied.")
        
    return jsonify({"status": "Game Reset"})

@app.route('/step', methods=['GET'])
def step():
    global current_sim
    if not current_sim:
        return jsonify({"error": "Game not started"})

    current_sim.update()

    drones_data = [{"id": d.id, "x": d.pos[0], "y": d.pos[1], "angle": d.angle} for d in current_sim.drones]
    
    prey_data = []
    for p in current_sim.prey:
        prey_data.append({
            "id": p.id, 
            "x": p.pos[0], 
            "y": p.pos[1], 
            "alive": not p.detected and not p.escaped,
            "running": p.running,
            "escaped": p.escaped,
            "target": p.target_pos.tolist(),
            "is_player": p.is_player,  # <--- Send this to frontend
        })

    # INCLUDE BASE POSITION
    base_data = current_sim.base_pos.tolist()

    return jsonify({
        "drones": drones_data,
        "prey": prey_data,
        "base": base_data # Send base coords
    })

@app.route('/player_input', methods=['POST'])
def player_input():
    global current_sim
    if not current_sim: return jsonify({"error": "No game"})
    
    data = request.json
    target_x = data.get('target_x')
    target_y = data.get('target_y')
    
    if target_x is None or target_y is None:
        return jsonify({"error": "Missing target coordinates"})
    
    # Pass target position to the simulation
    current_sim.set_player_target(target_x, target_y)
    
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    app.run(debug=True, port=5000)