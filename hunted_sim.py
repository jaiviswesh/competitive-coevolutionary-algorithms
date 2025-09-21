import random
import numpy as np
import math
import matplotlib.pyplot as plt
from collections import defaultdict

# This script is now a library for ga_coevolution.py
# The if __name__ == '__main__' block is for simple, standalone testing.

class ChaoticRho:
    """Simple and stable chaotic number generator."""
    def __init__(self, seed=1):
        self.x = random.Random(seed).random()
    def next_rho(self):
        self.x = 4.0 * self.x * (1.0 - self.x)
        return float(self.x)

def clamp_pos(p, xmin, xmax, ymin, ymax):
    """Keeps agent positions within the map boundaries."""
    p[0] = np.clip(p[0], xmin, xmax)
    p[1] = np.clip(p[1], ymin, ymax)
    return p

class Drone:
    """Represents a predator agent with Boids behavior."""
    def __init__(self, id, typ, pos, angle, speed, detection_range, fov_deg):
        self.id = id
        self.typ = typ
        self.pos = np.array(pos, dtype=float)
        self.angle = angle
        self.speed = speed
        self.detection_range = detection_range
        self.fov = math.radians(fov_deg)
        self.velocity = np.array([math.cos(angle), math.sin(angle)]) * speed
        # Evolved Boids weights will be assigned by the GA
        self.sep_weight = 1.0
        self.ali_weight = 1.0
        self.coh_weight = 1.0

    def step_move(self, turn_angle):
        self.angle += turn_angle
        self.angle = (self.angle + math.pi) % (2 * math.pi) - math.pi # Normalize angle
        self.velocity = np.array([math.cos(self.angle), math.sin(self.angle)]) * self.speed
        self.pos += self.velocity

    def can_detect(self, target_pos):
        if np.linalg.norm(target_pos - self.pos) > self.detection_range:
            return False
        if self.fov >= 2 * math.pi:
            return True
        v_to_target = target_pos - self.pos
        ang = math.atan2(v_to_target[1], v_to_target[0])
        diff = (ang - self.angle + math.pi) % (2 * math.pi) - math.pi
        return abs(diff) <= self.fov / 2

class Prey:
    """Represents a prey agent trying to reach a central base."""
    def __init__(self, id, start_pos, target_base, speed, start_time, avoidance):
        self.id = id
        self.pos = np.array(start_pos, dtype=float)
        self.target_base = np.array(target_base, dtype=float)
        self.speed = speed
        self.start_time = start_time
        self.avoidance = avoidance
        self.running = False
        self.detected = False

    def step_move(self, drones_positions, dt=1.0):
        if not self.running: return
        
        dir_vec = self.target_base - self.pos
        dist_to_target = np.linalg.norm(dir_vec)
        if dist_to_target < 1e-6: return # Already at target
        
        attraction_vec = dir_vec / dist_to_target
        repulsion_vec = np.zeros(2)
        for dpos in drones_positions:
            dvec = self.pos - dpos
            d = np.linalg.norm(dvec)
            if 0 < d < 50.0: # Avoidance radius of 50m
                repulsion_vec += (dvec / d**2) # Inverse square law
        
        move_dir = attraction_vec + self.avoidance * repulsion_vec
        if np.linalg.norm(move_dir) == 0: return
        
        self.pos += (move_dir / np.linalg.norm(move_dir)) * self.speed * dt

class HuntedSim:
    """The main simulation engine."""
    def __init__(self, config, seed=0):
        self.rng = random.Random(seed)
        self.map_w = config.get('map_w', 400)
        self.map_h = config.get('map_h', 400)
        self.dt = config.get('dt', 1.0)
        self.max_ticks = int(config.get('sim_time', 600) / self.dt)
        self.drones = []
        self.prey = []
        self.chaos = ChaoticRho(seed=seed)
        
        # Define the four bases in the center
        center_x, center_y = self.map_w / 2, self.map_h / 2
        base_side = math.sqrt(40000) # 400m^2 area -> 20m side
        half_base = base_side / 2
        self.bases = [
            np.array([center_x - half_base, center_y - half_base]), # Bottom-left
            np.array([center_x + half_base, center_y - half_base]), # Bottom-right
            np.array([center_x - half_base, center_y + half_base]), # Top-left
            np.array([center_x + half_base, center_y + half_base])  # Top-right
        ]
        self.init_drones(config['drones'])
        self.init_prey(config['prey'])

    def init_drones(self, drones_cfg):
        did = 0
        center = np.array([self.map_w / 2.0, self.map_h / 2.0])
        for group_cfg in drones_cfg:
            for _ in range(group_cfg['count']):
                r = self.rng.uniform(0, 10)
                theta = self.rng.uniform(0, 2 * math.pi)
                pos = center + np.array([math.cos(theta) * r, math.sin(theta) * r])
                angle = self.rng.uniform(-math.pi, math.pi)
                drone = Drone(did, group_cfg['type'], pos, angle, group_cfg['speed'],
                              group_cfg['detection_range'], group_cfg['fov_deg'])
                self.drones.append(drone)
                did += 1

    def init_prey(self, prey_cfg):
        pid = 0
        for e_cfg in prey_cfg:
            edge = self.rng.choice(['N', 'S', 'E', 'W'])
            if edge == 'N':   start_pos = [self.rng.uniform(0, self.map_w), self.map_h]
            elif edge == 'S': start_pos = [self.rng.uniform(0, self.map_w), 0]
            elif edge == 'E': start_pos = [self.map_w, self.rng.uniform(0, self.map_h)]
            else:             start_pos = [0, self.rng.uniform(0, self.map_h)]
            distances = [np.linalg.norm(np.array(start_pos) - base) for base in self.bases]
            closest_base = self.bases[np.argmin(distances)]
            prey = Prey(pid, start_pos, closest_base, e_cfg['speed'],
                        e_cfg['escape_time'], e_cfg['avoidance'])
            self.prey.append(prey)
            pid += 1
            
    def update_prey_list(self, prey_cfg_list):
        """Helper to allow the GA to inject new prey configurations."""
        self.prey = []
        self.init_prey(prey_cfg_list)

    def run(self, record_trajectories=False):
        traj = {'drone': defaultdict(list), 'prey': defaultdict(list)}
        for t in range(self.max_ticks):
            time_s = t * self.dt
            
            for p in self.prey:
                if not p.running and time_s >= p.start_time:
                    p.running = True

            drone_pos = [v.pos.copy() for v in self.drones]
            for p in self.prey:
                if not p.detected:
                    p.step_move(drone_pos, dt=self.dt)
                    p.pos = clamp_pos(p.pos, 0, self.map_w, 0, self.map_h)

            # --- Boids Drone Movement ---
            NEIGHBOR_RADIUS, MAX_TURN = 50.0, math.pi/8
            new_angles = {}
            for v in self.drones:
                neighbors = [u for u in self.drones if u is not v and np.linalg.norm(u.pos - v.pos) < NEIGHBOR_RADIUS]
                if not neighbors:
                    turn = (self.chaos.next_rho() - 0.5) * MAX_TURN
                    new_angles[v.id] = v.angle + turn
                    continue
                
                # Calculate Boids vectors
                com = sum(n.pos for n in neighbors) / len(neighbors)
                avg_vel = sum(n.velocity for n in neighbors) / len(neighbors)
                sep_vec = sum((v.pos - n.pos) for n in neighbors if np.linalg.norm(n.pos - v.pos) < NEIGHBOR_RADIUS / 2)

                # Combine vectors using evolved weights
                final_vec = ((com - v.pos) * v.coh_weight + 
                             sep_vec * v.sep_weight + 
                             avg_vel * v.ali_weight)
                
                if np.linalg.norm(final_vec) < 1e-6:
                    new_angles[v.id] = v.angle
                else:
                    new_angles[v.id] = math.atan2(final_vec[1], final_vec[0])

            # Apply movement simultaneously
            for v in self.drones:
                turn_diff = (new_angles[v.id] - v.angle + math.pi) % (2 * math.pi) - math.pi
                v.step_move(np.clip(turn_diff, -MAX_TURN, MAX_TURN))
                v.pos = clamp_pos(v.pos, 0, self.map_w, 0, self.map_h)
            # --- End Boids Logic ---

            for p in self.prey:
                if p.detected: continue
                for v in self.drones:
                    if v.can_detect(p.pos):
                        p.detected = True
                        break
            
            if record_trajectories:
                for v in self.drones: traj['drone'][v.id].append(v.pos.copy())
                for p in self.prey: traj['prey'][p.id].append(p.pos.copy())
            
            if all(np.linalg.norm(p.pos - p.target_base) < 5.0 for p in self.prey): break

        # --- Fitness Calculation ---
        REACH_BONUS, total_score, reached = self.map_w * 2, 0.0, 0
        diag = math.sqrt(self.map_w**2 + self.map_h**2)
        for p in self.prey:
            score = 0
            final_dist = np.linalg.norm(p.pos - p.target_base)
            if final_dist < 5.0:
                reached += 1
                score += REACH_BONUS
            else:
                score += (diag - final_dist)
            if p.detected:
                score *= 0.5
            total_score += score
        
        F = total_score / len(self.prey) if self.prey else 0.0
        stats = {'F': F, 'n_prey_reached': reached, 'n_detected': sum(1 for p in self.prey if p.detected)}
        return stats, traj

def example_config():
    """Default configuration for a single test run."""
    drones = [{'type':'UAV', 'count':10, 'speed':1.5, 'detection_range':75.0, 'fov_deg':360}]
    prey = [{'speed':3.0, 'escape_time':0.0, 'avoidance':1.0}] * 4
    return {'drones':drones, 'prey':prey, 'map_w':400, 'map_h':400, 'sim_time':600}

if __name__ == '__main__':
    """This block is for simple, standalone testing of the simulator."""
    print("--- Running a simple standalone test of hunted_sim.py ---")
    cfg = example_config()
    sim = HuntedSim(cfg, seed=42)
    stats, traj = sim.run(record_trajectories=False)
    print("Simple Test Run Stats:", stats)
    print("\nNOTE: To run the evolution and see the final plot, run ga_coevolution.py")