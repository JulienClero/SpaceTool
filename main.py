# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 17:33:28 2025

@author: clero
"""
import numpy as np
import random
from tqdm import tqdm
from UI import *

class Const:
    def __init__(self, G = 6.67430e-11, Gkm = 6.674184e-20, Gua = 1.48815931E-34):
        self.G = G
        self.Gkm = Gkm
        self.Gua = Gua

class Body:
    def __init__(self, initial_pos, initial_vel, mass = 1, radius = None,  name= "DefaultBodyName", color = (255,255,255), mesh = None):
        self.pos = initial_pos
        self.vel = initial_vel
        self.radius = radius
        self.mass = mass 
        self.name = name
        self.color = color
        
        if mesh is None:     # mesh is file path ie "./mesh"
            # — Générer une sphère en tant que surface (mesh) —
            sph_verts, sph_indices = generate_sphere_mesh(self.radius, 20,20)           
            # — Créer l'objet surface —
            self.tracer = SurfaceVBO(sph_verts, sph_indices, color=(random.uniform(0,1), random.uniform(0,1), random.uniform(0,1)))
            
        else : 
            
            pass #Implementer ça pour prendre le chemin mesh et lire un fichier .obj ou autre et le mettre en tant que tracer
        
        #Implementer un "tracer" qui est définie comme une SurfaceVBO avec la position du corps et les paramètres ( radius, color...)
        #ça peux être n'importe quoi puisque ObjectVBO fonctionne pareil ( Custom mesh) à condition de ré implementer OBJ et l'importation de .stl     
        
class Simulation:
    def __init__(self, span, dt, bodies, name= "DefaultSimulation"):
        self.name = name
        
        self.span = span
        self.dt = dt
        self.bodies = bodies
        self.const = Const()
        self.history = []
        
        self.positions = np.array([body.pos for body in self.bodies])
        self.velocities = np.array([body.vel for body in self.bodies])
        self.accelerations = np.array([np.array([0.0,0.0,0.0]) for body in self.bodies])
        self.masses = np.array([body.mass for body in self.bodies])  
        
        self.history = np.empty((span//dt, len(bodies), 2, 3), dtype=np.float64)
        self.current_step =0
 
    def store_history(self):
        self.history[self.current_step, :, 0, :] = self.positions
        self.history[self.current_step, :, 1, :] = self.velocities
        self.current_step += 1
       
 
    def compute_gravitational_accelerations(self, positions):
        
        diff = positions[None, :, :] - positions[:, None, :]
        dist_sq = np.sum(diff**2, axis=-1)
        dist = np.sqrt(dist_sq)
        direction = diff / dist[..., None]
    
        mass_matrix = self.masses[:, None] * self.masses[None, :]
        F_mag = self.const.G * mass_matrix / dist_sq
        F = F_mag[..., None] * direction
    
        np.fill_diagonal(F[:, :, 0], 0)
        np.fill_diagonal(F[:, :, 1], 0)
        np.fill_diagonal(F[:, :, 2], 0)
    
        F_total = np.sum(F, axis=1)
        return F_total / self.masses[:, None]  # (N, 3)

    
    def integrate(self):
        steps = int(self.span // self.dt)
        
        for _ in tqdm(range(steps)):
            # RK4
            p0 = self.positions
            v0 = self.velocities
    
            a0 = self.compute_gravitational_accelerations(p0)
            k1v = a0
            k1p = v0
    
            a1 = self.compute_gravitational_accelerations(p0 + 0.5 * self.dt * k1p)
            k2v = a1
            k2p = v0 + 0.5 * self.dt * k1v
    
            a2 = self.compute_gravitational_accelerations(p0 + 0.5 * self.dt * k2p)
            k3v = a2
            k3p = v0 + 0.5 * self.dt * k2v
    
            a3 = self.compute_gravitational_accelerations(p0 + self.dt * k3p)
            k4v = a3
            k4p = v0 + self.dt * k3v
    
            self.positions += self.dt / 6 * (k1p + 2 * k2p + 2 * k3p + k4p)
            self.velocities += self.dt / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
            
    def integrate_step(self):
        
        dt = self.dt
        dt2 = 0.5 * dt
        inv_6 = 1.0 / 6.0
    
        p0 = self.positions
        v0 = self.velocities
    
        # k1
        a0 = self.compute_gravitational_accelerations(p0)
        k1p = v0
        k1v = a0
    
        # k2
        p_temp = p0 + dt2 * k1p
        v_temp = v0 + dt2 * k1v
        a1 = self.compute_gravitational_accelerations(p_temp)
        k2p = v_temp
        k2v = a1
    
        # k3
        p_temp = p0 + dt2 * k2p
        v_temp = v0 + dt2 * k2v
        a2 = self.compute_gravitational_accelerations(p_temp)
        k3p = v_temp
        k3v = a2
    
        # k4
        p_temp = p0 + dt * k3p
        v_temp = v0 + dt * k3v
        a3 = self.compute_gravitational_accelerations(p_temp)
        k4p = v_temp
        k4v = a3
    
        # Final update
        self.positions += dt * inv_6 * (k1p + 2 * k2p + 2 * k3p + k4p)
        self.velocities += dt * inv_6 * (k1v + 2 * k2v + 2 * k3v + k4v)
    
        # Store in history if needed

        self.store_history()

          
    def build_scene(self,win): # Used to initilize a scene in the renderer
        print(np.shape(self.positions))
        scale = np.max(np.array([step[0] for step in self.history], dtype=np.float32))
        print(f"SCALE : {scale}")
        for body_index, body in enumerate(self.bodies):
            positions = np.array([step[body_index][0] for step in self.history], dtype=np.float32)*0.1  #Modifiy the scaling according to the largest pos
            positions = positions[:, [1,2,0]]
            traj = TrajectoryVBO(positions, color=(255, 255, 255), draw_points=False)
        
            # — Placer la sphère au bout de l'hélice —
            last_pos = positions[0]
            body.tracer.transform = lambda: glTranslatef(*last_pos)
        
            # — Lier la sphère à la trajectoire et ajouter à la scène —
            traj.tracer = body.tracer
            win.gl_widget.add_object(body.tracer)
            win.gl_widget.add_object(traj)


class App:
    def __init__(self, *simulations):
        self.Simulations = []
        for simulation in simulations:
            self.Simulations.append(simulation)

        self.Qapp = QApplication(sys.argv)
        self.win = MainWindow(self.Simulations)

        self.win.resize(1920, 1440)
        self.win.show()
        sys.exit(self.Qapp.exec_()) 
        
    def add_simulation(self,sim):
        self.Simulations.append(sim)

bodies = [
    Body(np.array([0.0, 0.0, 0.0]),           # Corps central
         np.array([0.0, 0.0, 0.0]),
         1.0e15,
         1),
    
    Body(np.array([100.0, 0.0, 0.0]),         # Corps en orbite 1
         np.array([0.0, 30.0, 0.0]),
         1.0e3,
         0.5),
    
    Body(np.array([-100.0, 0.0, 0.0]),        # Corps en orbite 2
         np.array([0.0, -30.0, 0.0]),
         1.0e3,
         0.5),
]

sim = Simulation(10000, 1, bodies, "3BodyTest")

app = App(sim)           
            
            
            
            
            
            
            
            
            
            
            
