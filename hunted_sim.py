import random
import numpy as np
import math
from collections import defaultdict

# --- Helper Functions ---
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
    # INTRUDER SEARCH SCENARIO
    # Drones need high speed to intercept since they don't know the destination
    drones = [{'type':'UAV', 'count':8, 'speed':6.0, 'detection_range':90.0, 'fov_deg':360}]
    prey = [{'speed':3.5, 'escape_time':0.0, 'avoidance':3.0}] * 6
    return {'drones':drones, 'prey':prey, 'map_w':1000, 'map_h':1000, 'sim_time':1000}

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
        
        # Search Weights (Drones must explore, not just sit)
        self.sep_weight = 2.0   # High separation to cover more ground
        self.ali_weight = 0.5   
        self.coh_weight = 0.1   # Low cohesion to prevent clumping
        self.search_weight = 1.0 # High chaos/wander

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
    def __init__(self, id, start_pos, target_pos, speed, start_time, avoidance, is_player=False):
        self.id = id
        self.pos = np.array(start_pos, dtype=float)
        self.target_pos = np.array(target_pos, dtype=float)
        self.speed = speed
        self.start_time = start_time
        self.avoidance = avoidance
        self.running = False
        
        self.is_player = is_player          # <--- NEW FLAG
        self.player_vel = np.array([0, 0])  # <--- NEW: Store player input velocity
        self.player_target = None           # <--- NEW: Target position for player movement
        
        # Status
        self.detected = False
        self.escaped = False
        self.done_tick = None 

        try:
            self.start_dist = float(np.linalg.norm(np.array(start_pos, dtype=float) - self.target_pos))
        except Exception:
            self.start_dist = 0.0

    def step_move(self, drones_positions, dt=1.0):
        if not self.running: return
        
        # --- NEW: PLAYER LOGIC ---
        if self.is_player:
            # If player has a target, check if reached
            if self.player_target is not None:
                dist_to_target = np.linalg.norm(self.player_target - self.pos)
                if dist_to_target < 10.0:  # Close enough to target (10 units)
                    self.running = False
                    self.player_target = None
                    self.player_vel = np.array([0, 0])
                    return
            
            # If player, move based on manual input only
            norm = np.linalg.norm(self.player_vel)
            if norm > 0:
                # Normalize to max speed if necessary, or just use input direction
                self.pos += (self.player_vel / norm) * self.speed * dt
            return 
        # -------------------------
        
        # 1. Attraction to Hidden Base
        dir_vec = self.target_pos - self.pos
        dist_to_target = np.linalg.norm(dir_vec)
        if dist_to_target < 1e-6: return
        attraction_vec = dir_vec / dist_to_target

        # 2. Repulsion from Drones
        repulsion_vec = np.zeros(2)
        AVOID_RANGE = 120.0 
        SCALING_FACTOR = 5000.0 

        for dpos in drones_positions:
            dvec = self.pos - dpos 
            d = np.linalg.norm(dvec)
            if 0 < d < AVOID_RANGE:
                strength = SCALING_FACTOR / (d**2)
                repulsion_vec += (dvec / d) * strength

        move_dir = attraction_vec + self.avoidance * repulsion_vec
        
        norm = np.linalg.norm(move_dir)
        if norm > 0:
            self.pos += (move_dir / norm) * self.speed * dt

class HuntedSim:
    def __init__(self, config, seed=0):
        self.rng = random.Random()
        self.map_w = config.get('map_w', 1000)
        self.map_h = config.get('map_h', 1000)
        self.dt = config.get('dt', 1.0)
        self.max_ticks = int(config.get('sim_time', 1500) / self.dt)
        self.current_tick = 0
        self.drones = []
        self.prey = []
        self.chaos = ChaoticRho(seed=self.rng.randint(1, int(1e9)))
        
        # --- NEW: Random Base Location (Unknown to Drones) ---
        # Padding to keep it off the immediate edge
        padding = 100
        self.base_pos = np.array([
            self.rng.uniform(padding, self.map_w - padding),
            self.rng.uniform(padding, self.map_h - padding)
        ])
        
        self.init_drones(config['drones'])
        self.init_prey(config['prey'])

    def init_drones(self, drones_cfg):
        did = 0
        # DRONES START AT MAP CENTER
        center_start = np.array([self.map_w/2, self.map_h/2])
        
        for group_cfg in drones_cfg:
            count = group_cfg['count']
            for i in range(count):
                # Start in a tight clump at center
                offset = np.array([self.rng.uniform(-10, 10), self.rng.uniform(-10, 10)])
                pos = center_start + offset
                
                # Random facing angle
                angle = self.rng.uniform(-math.pi, math.pi)
                
                drone = Drone(did, group_cfg['type'], pos, angle, group_cfg['speed'],
                              group_cfg['detection_range'], group_cfg['fov_deg'])
                self.drones.append(drone)
                did += 1

    def init_prey(self, prey_cfg):
        pid = 0
        for i, e_cfg in enumerate(prey_cfg): # Use enumerate to find index
            # SPAWN: Random Border Point
            edge = self.rng.choice(['N', 'S', 'E', 'W'])
            if edge == 'N':   start_pos = [self.rng.uniform(0, self.map_w), self.map_h]
            elif edge == 'S': start_pos = [self.rng.uniform(0, self.map_w), 0]
            elif edge == 'E': start_pos = [self.map_w, self.rng.uniform(0, self.map_h)]
            else:             start_pos = [0, self.rng.uniform(0, self.map_h)]

            # Make the FIRST prey (pid=0) the Player
            is_player = (pid == 0) 
            
            prey = Prey(pid, start_pos, self.base_pos, e_cfg['speed'],
                        e_cfg['escape_time'], e_cfg['avoidance'], is_player=is_player)
            self.prey.append(prey)
            pid += 1

    def update_prey_list(self, prey_list):
        self.prey = []
        self.init_prey(prey_list)

    # Add this helper function to HuntedSim to receive input
    def set_player_target(self, target_x, target_y):
        # Find the player prey (id 0)
        for p in self.prey:
            if p.is_player and not p.detected and not p.escaped:
                p.player_target = np.array([target_x, target_y])
                # Calculate direction to target
                direction = p.player_target - p.pos
                norm = np.linalg.norm(direction)
                if norm > 0:
                    p.player_vel = direction / norm  # Unit vector towards target
                    # Ensure player starts running immediately on input
                    if not p.running: p.running = True 
                break

    def update(self):
        time_s = self.current_tick * self.dt
        
        # 1. Activate Prey
        for p in self.prey:
            if not p.running and not p.done_tick and time_s >= p.start_time:
                p.running = True

        # 2. Move Prey
        drone_pos = [v.pos.copy() for v in self.drones]
        for p in self.prey:
            if not p.detected and not p.escaped: 
                p.step_move(drone_pos, dt=self.dt)
                p.pos = clamp_pos(p.pos, 0, self.map_w, 0, self.map_h)

                # Check Intrusion (Reached Base)
                # Capture radius 40
                if np.linalg.norm(p.pos - p.target_pos) < 40.0:
                    p.escaped = True
                    p.running = False
                    p.done_tick = self.current_tick 

        # 3. Move Drones (BLIND SEARCH LOGIC)
        # No "Target" or "Orbit" vectors because they don't know where the base is.
        # They rely on Flocking (spreading out) + Chaos (wandering).
        
        NEIGHBOR_RADIUS, MAX_TURN = 80.0, math.pi/6
        new_angles = {}
        
        for v in self.drones:
            neighbors = [u for u in self.drones if u is not v and np.linalg.norm(u.pos - v.pos) < NEIGHBOR_RADIUS]
            
            # CHAOS / SEARCH Vector (Random Wander)
            # Use Chaotic map to generate a pseudo-random turn preference
            rho = self.chaos.next_rho()
            wander_turn = (rho - 0.5) * MAX_TURN * 2.0
            
            # Create a vector from this wander angle relative to current heading
            wander_vec = np.array([
                math.cos(v.angle + wander_turn),
                math.sin(v.angle + wander_turn)
            ])

            com = np.zeros(2)
            avg_vel = np.zeros(2)
            sep_vec = np.zeros(2)
            
            if neighbors:
                com = sum(n.pos for n in neighbors) / len(neighbors)
                avg_vel = sum(n.velocity for n in neighbors) / len(neighbors)
                for n in neighbors:
                     if np.linalg.norm(n.pos - v.pos) < NEIGHBOR_RADIUS / 2:
                         sep_vec += (v.pos - n.pos)

                # Alignment + Cohesion + Separation + Search
                # Note: 'com - v.pos' is vector TO neighbors (Cohesion)
                final_vec = ((com - v.pos) * v.coh_weight + 
                             sep_vec * v.sep_weight + 
                             avg_vel * v.ali_weight +
                             wander_vec * (v.search_weight * 100.0)) # Strong wander influence
            else:
                # If alone, purely chaos/wander
                final_vec = wander_vec

            # Boundary Avoidance (Bounce off walls)
            # If close to wall, add strong vector inward
            wall_vec = np.zeros(2)
            margin = 50
            if v.pos[0] < margin: wall_vec[0] += 1
            elif v.pos[0] > self.map_w - margin: wall_vec[0] -= 1
            if v.pos[1] < margin: wall_vec[1] += 1
            elif v.pos[1] > self.map_h - margin: wall_vec[1] -= 1
            
            final_vec += wall_vec * 500.0

            if np.linalg.norm(final_vec) < 1e-6:
                new_angles[v.id] = v.angle
            else:
                new_angles[v.id] = math.atan2(final_vec[1], final_vec[0])

        for v in self.drones:
            turn_diff = (new_angles[v.id] - v.angle + math.pi) % (2 * math.pi) - math.pi
            v.step_move(np.clip(turn_diff, -MAX_TURN, MAX_TURN))
            v.pos = clamp_pos(v.pos, 0, self.map_w, 0, self.map_h)

        # 4. Detection Logic
        for p in self.prey:
            if p.detected or p.escaped: continue 
            for v in self.drones:
                if v.can_detect(p.pos):
                    p.detected = True
                    p.running = False
                    p.done_tick = self.current_tick 
                    break
        
        self.current_tick += 1

    def run(self, record_trajectories=False):
        traj = {'drone': defaultdict(list), 'prey': defaultdict(list)}
        for t in range(self.max_ticks):
            self.update()
            if record_trajectories:
                for v in self.drones: traj['drone'][v.id].append(v.pos.copy())
                for p in self.prey: traj['prey'][p.id].append(p.pos.copy())
            if all((p.detected or p.escaped) for p in self.prey): break 

        escaped_count = 0
        total_score = 0
        for p in self.prey:
            end_time = p.done_tick if p.done_tick is not None else self.max_ticks
            if p.escaped:
                escaped_count += 1
                score = 5000.0 + (self.max_ticks - end_time) * 5.0
            elif p.detected:
                score = float(end_time)
            else:
                dist_to_base = np.linalg.norm(p.pos - p.target_pos)
                start_dist = getattr(p, 'start_dist', 1.0)
                progress = 1.0 - (dist_to_base / start_dist)
                score = float(self.max_ticks) + (progress * 1000.0)
            total_score += score
                
        stats = {
            'F': total_score / len(self.prey) if self.prey else 0,
            'n_prey_escaped': escaped_count,
            'n_detected': len(self.prey) - escaped_count
        }
        return stats, traj