import random
import numpy as np
import math
from collections import defaultdict

# Simulation constants
REACH_RADIUS = 5.0
CAPTURE_RADIUS = 5.0
REPULSION_RADIUS = 75.0  # how far prey sense and try to avoid drones
REPULSION_STRENGTH = 50.0  # scale for prey repulsion force (higher -> stronger evasion)

# --- Helper Functions Moved Here for Access by App ---
def apply_drone_genome(sim, drone_genome):
    if drone_genome is None: return
    for i, drone in enumerate(sim.drones):
        idx = i * 4
        if idx + 3 < len(drone_genome):
            drone.sep_weight = drone_genome[idx]
            drone.ali_weight = drone_genome[idx + 1]
            drone.coh_weight = drone_genome[idx + 2]
            drone.search_weight = drone_genome[idx + 3]

def decode_prey(prey_genome):
    # Default constants needed for decoding if not passed
    SPEED_MIN, SPEED_MAX = 1.0, 5.0
    TIME_MIN, TIME_MAX = 5.0, 600
    AVOID_MIN, AVOID_MAX = 0.0, 4.0
    
    if prey_genome is None: return example_config()['prey']
    prey_list = []
    for i in range(0, len(prey_genome), 3):
        prey_list.append({
            'speed': float(np.clip(prey_genome[i], SPEED_MIN, SPEED_MAX)),
            'escape_time': float(np.clip(prey_genome[i+1], TIME_MIN, TIME_MAX)),
            'avoidance': float(np.clip(prey_genome[i+2], AVOID_MIN, AVOID_MAX)),
        })
    return prey_list

def example_config():
    drones = [{'type':'UAV', 'count':8, 'speed':5, 'detection_range':25.0, 'fov_deg':360}]
    prey = [{'speed':3.0, 'escape_time':0.0, 'avoidance':1.0}] * 4
    return {'drones':drones, 'prey':prey, 'map_w':400, 'map_h':400, 'sim_time':1200}
# -----------------------------------------------------

class ChaoticRho:
    def __init__(self, seed=1):
        self.x = random.Random(seed).random()
    def next_rho(self):
        self.x = 4.0 * self.x * (1.0 - self.x)
        return float(self.x)

def clamp_pos(p, xmin, xmax, ymin, ymax):
    p[0] = np.clip(p[0], xmin, xmax)
    p[1] = np.clip(p[1], ymin, ymax)
    return p

class Drone:
    def __init__(self, id, typ, pos, angle, speed, detection_range, fov_deg):
        self.id = id
        self.typ = typ
        self.pos = np.array(pos, dtype=float)
        self.angle = angle
        self.speed = speed
        self.detection_range = detection_range
        self.fov = math.radians(fov_deg)
        self.velocity = np.array([math.cos(angle), math.sin(angle)]) * speed
        self.sep_weight = 1.0
        self.ali_weight = 0.4
        self.coh_weight = 0.4
        self.search_weight = 0.5 

    def step_move(self, turn_angle):
        self.angle += turn_angle
        self.angle = (self.angle + math.pi) % (2 * math.pi) - math.pi
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
    def __init__(self, id, start_pos, target_base, speed, start_time, avoidance):
        self.id = id
        self.pos = np.array(start_pos, dtype=float)
        self.target_base = np.array(target_base, dtype=float)
        self.speed = speed
        self.start_time = start_time
        self.avoidance = avoidance
        self.running = False
        self.detected = False
        self.caught = False
        self.reached = False

    def step_move(self, drones_positions, dt=1.0):
        # Do nothing if not active or already removed from the map
        if not self.running or self.caught or self.reached:
            return

        dir_vec = self.target_base - self.pos
        dist_to_target = np.linalg.norm(dir_vec)
        # If already very near, mark as reached and stop
        if dist_to_target <= REACH_RADIUS:
            self.pos = self.target_base.copy()
            self.running = False
            self.reached = True
            return

        attraction_vec = dir_vec / dist_to_target
        repulsion_vec = np.zeros(2)
        eps = 1e-6
        for dpos in drones_positions:
            dvec = self.pos - dpos
            d = np.linalg.norm(dvec)
            # React to drones within REPULSION_RADIUS; scale so prey actively
            # evade earlier and more strongly. Use a strength factor so the
            # repulsion can be comparable to the unit attraction vector.
            if 0 < d < REPULSION_RADIUS:
                # unit away vector times (REPULSION_STRENGTH / distance)
                repulsion_vec += (dvec / (d + eps)) * (REPULSION_STRENGTH / (d + eps))
        move_dir = attraction_vec + self.avoidance * repulsion_vec
        if np.linalg.norm(move_dir) == 0: return
        self.pos += (move_dir / np.linalg.norm(move_dir)) * self.speed * dt

        # After moving, check if prey has reached the base
        if np.linalg.norm(self.pos - self.target_base) <= REACH_RADIUS:
            self.pos = self.target_base.copy()
            self.running = False
            self.reached = True

class HuntedSim:
    def __init__(self, config, seed=0):
        self.rng = random.Random()
        self.map_w = config.get('map_w', 800)
        self.map_h = config.get('map_h', 800)
        self.dt = config.get('dt', 1.0)
        self.max_ticks = int(config.get('sim_time', 600) / self.dt)
        self.current_tick = 0
        self.drones = []
        self.prey = []
        self.chaos = ChaoticRho(seed=self.rng.randint(1, int(1e9)))
        center_x, center_y = self.map_w / 2, self.map_h / 2
        # Single base located at the center of the map
        self.bases = [np.array([center_x, center_y])]
        self.init_drones(config['drones'])
        self.init_prey(config['prey'])

    def init_drones(self, drones_cfg):
        did = 0
        # Drones start at the exact center of the map
        center = np.array([self.map_w / 2.0, self.map_h / 2.0])
        for group_cfg in drones_cfg:
            for _ in range(group_cfg['count']):
                pos = center.copy()
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

    def update_prey_list(self, prey_list):
        # Helper to rebuild prey list from config (used by GA and App)
        self.prey = []
        self.init_prey(prey_list)

    # --- THIS IS THE KEY NEW METHOD FOR THE WEB APP ---
    def update(self):
        """Advances the simulation by one time step."""
        time_s = self.current_tick * self.dt
        
        # 1. Activate Prey
        for p in self.prey:
            if not p.running and time_s >= p.start_time:
                p.running = True

        # 2. Move Prey
        drone_pos = [v.pos.copy() for v in self.drones]
        for p in self.prey:
            # Only move prey that are active and haven't been removed (caught/reached)
            if p.running and not p.caught and not p.reached:
                p.step_move(drone_pos, dt=self.dt)
                p.pos = clamp_pos(p.pos, 0, self.map_w, 0, self.map_h)

        # 3. Move Drones (Flocking Logic)
        NEIGHBOR_RADIUS, MAX_TURN = 50.0, math.pi/8
        new_angles = {}
        
        for v in self.drones:
            neighbors = [u for u in self.drones if u is not v and np.linalg.norm(u.pos - v.pos) < NEIGHBOR_RADIUS]
            
            if not neighbors:
                turn = (self.chaos.next_rho() - 0.5) * MAX_TURN
                new_angles[v.id] = v.angle + turn
                continue

            com = sum(n.pos for n in neighbors) / len(neighbors)
            avg_vel = sum(n.velocity for n in neighbors) / len(neighbors)
            
            # Simple vector sum for separation
            sep_vec = np.zeros(2)
            for n in neighbors:
                 if np.linalg.norm(n.pos - v.pos) < NEIGHBOR_RADIUS / 2:
                     sep_vec += (v.pos - n.pos)

            center_of_map = np.array([self.map_w / 2, self.map_h / 2])
            search_vec = v.pos - center_of_map # Go outward

            final_vec = ((com - v.pos) * v.coh_weight + 
                         sep_vec * v.sep_weight + 
                         avg_vel * v.ali_weight +
                         search_vec * v.search_weight)
            
            if np.linalg.norm(final_vec) < 1e-6:
                new_angles[v.id] = v.angle
            else:
                new_angles[v.id] = math.atan2(final_vec[1], final_vec[0])

        for v in self.drones:
            turn_diff = (new_angles[v.id] - v.angle + math.pi) % (2 * math.pi) - math.pi
            v.step_move(np.clip(turn_diff, -MAX_TURN, MAX_TURN))
            v.pos = clamp_pos(v.pos, 0, self.map_w, 0, self.map_h)

        # 4. Detection Logic (skip prey that have reached or been caught)
        for p in self.prey:
            if p.detected or p.caught or p.reached: continue
            for v in self.drones:
                if v.can_detect(p.pos):
                    p.detected = True
                    break

        # 5. Capture Logic: mark prey as caught when a drone comes within its
        # detection range. This allows a drone to 'catch' a prey without
        # needing to physically overlap it; being within the drone's
        # `detection_range` is sufficient.
        for p in self.prey:
            if p.caught or p.reached: continue
            for v in self.drones:
                if np.linalg.norm(p.pos - v.pos) <= v.detection_range:
                    p.caught = True
                    p.running = False
                    p.detected = True
                    break
        
        self.current_tick += 1

    def run(self, record_trajectories=False):
        """Legacy method for the GA to run the full sim at once."""
        traj = {'drone': defaultdict(list), 'prey': defaultdict(list)}
        
        for t in range(self.max_ticks):
            self.update() # Use the new update method

            if record_trajectories:
                for v in self.drones: traj['drone'][v.id].append(v.pos.copy())
                for p in self.prey: traj['prey'][p.id].append(p.pos.copy())

            # Stop early if there are no more prey on the map (all reached or caught)
            if all((p.reached or p.caught) for p in self.prey):
                break

        # Calculate Score
        REACH_BONUS = self.map_w * 2
        total_score, reached, caught = 0.0, 0, 0
        diag = math.sqrt(self.map_w**2 + self.map_h**2)
        for p in self.prey:
            score = 0
            if p.caught:
                # Caught prey give zero score (predator gets credit elsewhere)
                score = 0
                caught += 1
            else:
                final_dist = np.linalg.norm(p.pos - p.target_base)
                if p.reached:
                    reached += 1
                    score += REACH_BONUS
                else:
                    score += (diag - final_dist)
                # Detection penalizes prey that were detected but not caught/reached
                if p.detected and not p.reached:
                    score *= 0.5
            total_score += score
            
        F = total_score / len(self.prey) if self.prey else 0.0
        stats = {
            'F': F,
            'n_prey_reached': reached,
            'n_prey_caught': caught,
            'n_detected': sum(1 for p in self.prey if p.detected)
        }
        return stats, traj