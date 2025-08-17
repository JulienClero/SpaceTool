# -*- coding: utf-8 -*-
"""
Created on Wed May 21 23:42:46 2025

@author: clero
"""

import sys
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QLabel, QSlider, QStackedLayout, QSizePolicy,QGridLayout, QAction, QComboBox, QProgressBar
)
from PyQt5.QtCore import QTimer, Qt, QPoint, QThread, pyqtSignal, QObject
# QOpenGLWidget est dans QtWidgets
from PyQt5.QtCore import Qt, QTimer
try:
    from PyQt5.QtWidgets import QOpenGLWidget
except ImportError:
    from PyQt5.QtOpenGL import QGLWidget as QOpenGLWidget
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.arrays import vbo


# ---------- Utility ----------
def create_rotation_matrix(rx, ry, rz):
    sx, cx = np.sin(rx), np.cos(rx)
    sy, cy = np.sin(ry), np.cos(ry)
    sz, cz = np.sin(rz), np.cos(rz)
    Rz = np.array([[cz, -sz, 0, 0],
                   [sz,  cz, 0, 0],
                   [ 0,   0, 1, 0],
                   [ 0,   0, 0, 1]], dtype=np.float32)
    Ry = np.array([[ cy, 0, sy, 0],
                   [  0, 1,  0, 0],
                   [-sy, 0, cy, 0],
                   [  0, 0,  0, 1]], dtype=np.float32)
    Rx = np.array([[1,   0,   0, 0],
                   [0,  cx, -sx, 0],
                   [0,  sx,  cx, 0],
                   [0,   0,   0, 1]], dtype=np.float32)
    return Rz @ Ry @ Rx

def generate_sphere(radius, n_lat, n_lon):
    theta = np.linspace(0, np.pi, n_lat + 1)
    phi   = np.linspace(0, 2*np.pi, n_lon, endpoint=False)
    pts = []
    for th in theta:
        for ph in phi:
            x = radius * np.sin(th)*np.cos(ph)
            y = radius * np.cos(th)
            z = radius * np.sin(th)*np.sin(ph)
            pts.append([x, y, z])
    return np.array(pts, dtype=np.float32)

def generate_sphere_mesh(radius, n_lat, n_lon):
    """
    Returns:
      vertices: (N,3) float32 array
      indices:  (M,3) int32 array of triangle indices
    """
    if radius is None:
        radius = 0.5
    
    # 1) build vertices
    thetas = np.linspace(0, np.pi,   n_lat + 1)      # from north pole to south
    phis   = np.linspace(0, 2*np.pi, n_lon, endpoint=False)
    verts = []
    for i, th in enumerate(thetas):
        for j, ph in enumerate(phis):
            x = radius * np.sin(th) * np.cos(ph)
            y = radius * np.cos(th)
            z = radius * np.sin(th) * np.sin(ph)
            verts.append((x,y,z))
    vertices = np.array(verts, dtype=np.float32)

    # 2) build triangle indices
    # latitude bands: 0..n_lat, longitude: 0..n_lon-1
    indices = []
    for i in range(n_lat):
        for j in range(n_lon):
            p0 = i   * n_lon + j
            p1 = p0 + n_lon
            p2 = p1 + 1 if (j+1)<n_lon else p1 - (n_lon-1)
            p3 = p0 + 1 if (j+1)<n_lon else p0 - (n_lon-1)
            # skip degenerate at poles:
            if i>0:
                indices.append((p0, p1, p3))
            if i< n_lat-1:
                indices.append((p3, p1, p2))
    indices = np.array(indices, dtype=np.int32)

    return vertices, indices

# ---------- VBO-based Renderables ----------
class PointCloudVBO:
    def __init__(self, coords, color=(1,1,1), size=5):
        """
        coords: Nx3 float32 array
        """
        self.color = color
        self.size  = size
        self.n     = len(coords)
        self.vbo   = vbo.VBO(coords)

    def draw(self):
        glColor3f(*self.color)
        glPointSize(self.size)
        self.vbo.bind()
        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, self.vbo)
        glDrawArrays(GL_POINTS, 0, self.n)
        glDisableClientState(GL_VERTEX_ARRAY)
        self.vbo.unbind()


class TrajectoryVBO:
    def __init__(self, coords, color=(255,255,255), width=2, draw_points=False, anim_speed=50.0):
        """
        coords:        Nx3 float32 array in path order
        anim_speed:    points per second
        """
        self.color       = color
        self.width       = width
        self.draw_points = draw_points
        self.draw_lines = True

        # full trajectory data
        self.coords      = coords
        self.n           = len(coords)

        # VBOs
        self.vbo_line    = vbo.VBO(coords)
        self.vbo_pts = vbo.VBO(coords)

        # animation state
        self.anim_speed  = anim_speed
        self.playhead    = 0.0      # float so we can interpolate or speed‐scale
        self.loop        = True
        
        #Object at the head of the trajectory (i.e Sphere to represent planet in its orbit)
        self.tracer = None

    def update(self, dt):
        self.playhead += self.anim_speed * dt
        if self.loop:
            self.playhead %= self.n
        else:
            self.playhead = min(self.playhead, self.n - 1)
    
        if self.tracer:
            i0 = int(np.floor(self.playhead))
            i1 = (i0 + 1) % self.n
            t = self.playhead - i0
            pos = (1 - t) * self.coords[i0] + t * self.coords[i1]
            self.tracer.translation = np.array([[1,0,0,pos[0]],
                                                [0,1,0,pos[1]],
                                                [0,0,1,pos[2]],
                                                [0,0,0,1]], dtype=np.float32)


    def draw(self):
        # determine how many vertices to draw this frame
        count = int(self.playhead)
        if count < 2:
            return  # nothing or just a point
        
        if self.draw_lines:
            # draw line up to current playhead
            glColor3f(*self.color)
            glLineWidth(self.width)
            self.vbo_line.bind()
            glEnableClientState(GL_VERTEX_ARRAY)
            glVertexPointer(3, GL_FLOAT, 0, self.vbo_line)
            glDrawArrays(GL_LINE_STRIP, 0, count)
            glDisableClientState(GL_VERTEX_ARRAY)
            self.vbo_line.unbind()

        # optionally draw points
        if self.draw_points:
            glPointSize(4)
            self.vbo_pts.bind()
            glEnableClientState(GL_VERTEX_ARRAY)
            glVertexPointer(3, GL_FLOAT, 0, self.vbo_pts)
            glDrawArrays(GL_POINTS, 0, count)
            glDisableClientState(GL_VERTEX_ARRAY)
            self.vbo_pts.unbind()



class Object3DVBO:
    def __init__(self, vertices, faces, position=(0,0,0), color=(1,1,1), line_width=2):
        """
        vertices: list of (x,y,z)
        faces:    list of tuples of vertex-indices
        """
        self.color      = color
        self.line_width = line_width
        # build transformation components
        tx, ty, tz = position
        self.translation = np.array([[1,0,0,tx],
                                     [0,1,0,ty],
                                     [0,0,1,tz],
                                     [0,0,0,1]], dtype=np.float32)
        self.rotation = [0.0, 0.0, 0.0]
        # prepare flat line-list coords
        lines = []
        for face in faces:
            L = len(face)
            for i in range(L):
                v0 = vertices[face[i]]
                v1 = vertices[face[(i+1)%L]]
                lines.append(v0)
                lines.append(v1)
        coords = np.array(lines, dtype=np.float32)
        self.n      = len(coords)
        self.vbo    = vbo.VBO(coords)

    def _model_matrix(self):
        R = create_rotation_matrix(*self.rotation)
        M = self.translation @ R
        return M

    def draw(self):
        glColor3f(*self.color)
        glLineWidth(self.line_width)
        M = self._model_matrix().T  # transpose for column-major
        glPushMatrix()
        glMultMatrixf(M)
        self.vbo.bind()
        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, self.vbo)
        glDrawArrays(GL_LINES, 0, self.n)
        glDisableClientState(GL_VERTEX_ARRAY)
        self.vbo.unbind()
        glPopMatrix()

class SurfaceVBO:
    def __init__(self, vertices, indices, position=(0, 0, 0), color=(0.3, 0.6, 0.9)):
        self.vertices = vertices
        self.indices = indices
        self.color = color
        self.position = position
        self.initialized = False

    def init_gl(self):
        if self.initialized:
            return
        self.vertex_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vertex_vbo)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)

        self.index_vbo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.index_vbo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.indices.nbytes, self.indices, GL_STATIC_DRAW)

        self.initialized = True


    def _model_matrix(self):
        R = create_rotation_matrix(*self.rotation)
        return self.translation @ R

    def draw(self):
        self.init_gl()
    
        glPushMatrix()
        if hasattr(self, 'translation'):
            glMultMatrixf(self.translation.T)  # attention au .T (transpose)
    
        glColor3f(*self.color)
        glEnableClientState(GL_VERTEX_ARRAY)
    
        glBindBuffer(GL_ARRAY_BUFFER, self.vertex_vbo)
        glVertexPointer(3, GL_FLOAT, 0, None)
    
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.index_vbo)
        glDrawElements(GL_TRIANGLES, len(self.indices)*3, GL_UNSIGNED_INT, None)
    
        glDisableClientState(GL_VERTEX_ARRAY)
        glPopMatrix()




# ---------- Camera & Renderer (unchanged except background) ----------
class Camera:
    def __init__(self, radius=50, theta=np.pi/4, phi=np.pi/6):
        self.target = np.zeros(3, dtype=np.float32)
        self.radius, self.theta, self.phi = radius, theta, phi
        self.up = np.array([0,1,0], dtype=np.float32)
        self._update_pos()

    def _update_pos(self):
        x = self.radius*np.cos(self.phi)*np.sin(self.theta)
        y = self.radius*np.sin(self.phi)
        z = self.radius*np.cos(self.phi)*np.cos(self.theta)
        self.position = self.target + np.array([x,y,z], dtype=np.float32)

    def orbit(self, dx, dy):
        self.theta -= 0.003*dx
        self.phi = np.clip(self.phi - 0.003*dy, -np.pi/2+0.01, np.pi/2-0.01)
        self._update_pos()

    def pan(self, dx, dy):
        forward = self.target - self.position
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, np.array([0,1,0],dtype=np.float32))
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        up /= np.linalg.norm(up)
        pan_speed = self.radius * 0.001
        offset = pan_speed * (right*dx + up*dy)
        self.position += offset
        self.target   += offset

    def zoom(self, d):
        self.radius = max(0.5, self.radius*(1-0.1*d))
        self._update_pos()

    def look(self):
        gluLookAt(*self.position, *self.target, *self.up)

class GLWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.StrongFocus)
        self.camera = Camera()
        self.objects = []
        self.rotating = False
        self.panning = False
        self.last_pos = QPoint()
        # Timer pour la boucle de rendu (~60 FPS)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.on_timeout)
        self.timer.start(16)

    def add_object(self, obj):
        self.objects.append(obj)

    def initializeGL(self):
        glEnable(GL_DEPTH_TEST)
        glClearColor(0.05, 0.05, 0.05, 1)
        glMatrixMode(GL_PROJECTION)
        gluPerspective(45, self.width() / self.height() if self.height() else 1, 0.1, 1e4)
        glMatrixMode(GL_MODELVIEW)

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, w / h if h else 1, 0.1, 1e4)
        glMatrixMode(GL_MODELVIEW)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        self.camera.look()
        self.draw_axes()
        for obj in self.objects:
            obj.draw()

    def on_timeout(self):
        dt = self.timer.interval() / 1000.0
        for obj in self.objects:
            if hasattr(obj, 'update'):
                obj.update(dt)
        self.update()  # déclenche paintGL

    def draw_axes(self):
        glLineWidth(2)
        glBegin(GL_LINES)
        glColor3f(1,0,0); glVertex3f(0,0,0); glVertex3f(5,0,0)
        glColor3f(0,1,0); glVertex3f(0,0,0); glVertex3f(0,5,0)
        glColor3f(0,0,1); glVertex3f(0,0,0); glVertex3f(0,0,5)
        glEnd()

    # ---------------------
    # Gestion de la souris
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.rotating = True
            self.last_pos = event.pos()
        elif event.button() == Qt.RightButton:
            self.panning = True
            self.last_pos = event.pos()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.rotating = False
        elif event.button() == Qt.RightButton:
            self.panning = False

    def mouseMoveEvent(self, event):
        dx = event.x() - self.last_pos.x()
        dy = event.y() - self.last_pos.y()
        self.last_pos = event.pos()
        if self.rotating:
            self.camera.orbit(dx, -dy)
        elif self.panning:
            self.camera.pan(-dx, dy)
        self.update()


    def wheelEvent(self, event):
        delta = event.angleDelta().y() / 120  # 120 est l'unité par défaut
        self.camera.zoom(delta)
        self.update()


class MainWindow(QMainWindow):
    def __init__(self, sims=None):
        super().__init__()

        self.sims = sims
        self.current_sim = None
        self.setWindowTitle("SpaceTool : Default Page")

        # Create central widget and main layout
        container = QWidget()
        self.setCentralWidget(container)
        main_layout = QHBoxLayout(container)  # Use horizontal layout for left/right split
        
        # --- Left column: Controls ---
        self.controls_widget = QWidget()
        controls_layout = QVBoxLayout(self.controls_widget)
        
        # Configurable controls width
        self.controls_width = 200  # Default width
        self.controls_widget.setFixedWidth(self.controls_width)
        
        # Add buttons
        buttons = [
            ("Recharger Scène", self.reload_scene),
            ("Trajectories Points Toggle", self.toggle_traj_points),
            ("Trajectories Toggle", self.toggle_traj),
        ]
        
        for label, callback in buttons:
            btn = QPushButton(label)
            btn.clicked.connect(callback)
            controls_layout.addWidget(btn)

        if sims is None:
            # Menu déroulant avec QComboBox
            self.combo_label = QLabel("Sélectionner une simulation :")
            self.combo_box = QComboBox()
            self.combo_box.addItems(["Aucune Simulation en stock"])

            controls_layout.addWidget(self.combo_label)
            controls_layout.addWidget(self.combo_box)
        else :
            # Menu déroulant avec QComboBox
            self.combo_label = QLabel("Sélectionner une simulation :")
            self.combo_box = QComboBox()
            self.combo_box.addItems(["Select a Simulation"] + [sim.name for sim in sims])
            self.combo_box.currentTextChanged.connect(self.on_simulation_selected)

            controls_layout.addWidget(self.combo_label)
            controls_layout.addWidget(self.combo_box)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        controls_layout.addWidget(self.progress_bar)

        # Add slider with label
        self.slider_label = QLabel("Vitesse animation : 50")
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(1, 1000)
        self.slider.setValue(50)
        self.slider.valueChanged.connect(self.on_slider_change)
        
        controls_layout.addWidget(self.slider_label)
        controls_layout.addWidget(self.slider)
        
        # Add stretch to push controls to top
        controls_layout.addStretch()
        
        # --- Right side: OpenGL widget ---
        self.gl_widget = GLWidget()
        
        # --- Right column: Big vertical slider ---
        self.right_widget = QWidget()
        right_layout = QVBoxLayout(self.right_widget)
        
        # Configurable right panel width
        self.right_width = 100  # Default width for right panel
        self.right_widget.setFixedWidth(self.right_width)
        
        # Big vertical slider
        self.vertical_slider_label = QLabel("Position n°50")
        self.vertical_slider_label.setAlignment(Qt.AlignCenter)
        
        self.vertical_slider = QSlider(Qt.Vertical)
        self.vertical_slider.setRange(0, 100)
        self.vertical_slider.setValue(50)
        self.vertical_slider.valueChanged.connect(self.on_vertical_slider_change)
        self.vertical_slider.setMinimumHeight(1080)  # Make it big
        
        right_layout.addWidget(self.vertical_slider_label)
        right_layout.addWidget(self.vertical_slider, 1, Qt.AlignCenter)  # Give slider stretch factor
        right_layout.addStretch()  # Add stretch at bottom
        
        # Add widgets to main layout: Left - Center - Right
        main_layout.addWidget(self.controls_widget)
        main_layout.addWidget(self.gl_widget, 1)  # Give OpenGL widget stretch factor of 1
        main_layout.addWidget(self.right_widget)
        
    def toggle_traj_points(self):
        for obj in self.gl_widget.objects:
            if isinstance(obj, TrajectoryVBO):
                if not obj.draw_points:
                    obj.draw_points = True
                else : 
                    obj.draw_points = False
                    
    def toggle_traj(self):
        for obj in self.gl_widget.objects:
            if isinstance(obj, TrajectoryVBO):
                if not obj.draw_lines:
                    obj.draw_lines = True
                else : 
                    obj.draw_lines = False    

    def on_vertical_slider_change(self, value):
        """Handle vertical slider change"""
        self.vertical_slider_label.setText(f"Position n°{value}")
        # Add your vertical slider handling logic here
                   
    def on_slider_change(self, value):
        self.animation_speed = value
        self.slider_label.setText(f"Vitesse animation : {value}")
        # Appliquer la vitesse à tous les objets qui ont anim_speed
        for obj in self.gl_widget.objects:
            if hasattr(obj, 'anim_speed'):
                obj.anim_speed = value

    def build_scene(self):
        
        # — Hélice comme trajectoire —
        t = np.linspace(0, 4*np.pi, 200)
        helix = np.column_stack([np.cos(t)*3, np.sin(t)*3, t*0.5]).astype(np.float32)
        traj = TrajectoryVBO(helix, color=(255, 255, 255), draw_points=True)
        
        # — Générer une sphère en tant que surface (mesh) —
        sph_verts, sph_indices = generate_sphere_mesh(0.2, 20, 20)
        
        # — Créer l'objet surface —
        tracer = SurfaceVBO(sph_verts, sph_indices, color=(1, 1, 0))
        
        # — Placer le tracer au début de la trajectoire —
        last_pos = helix[0]
        tracer.transform = lambda: glTranslatef(*last_pos)
        
        # — Lier la sphère à la trajectoire et ajouter à la scène —
        traj.tracer = tracer
        self.gl_widget.add_object(tracer)
        self.gl_widget.add_object(traj)


    def reload_scene(self):
        if self.current_sim == None:
            print("Aucune scène n'est sélectionée")
        else:
            print("Rechargement de la scène...")
            # Vider la liste des objets dans le GLWidget (supposons qu'ils sont dans self.gl_widget.objects)
            self.gl_widget.objects.clear()
            # Reconstruire la scène dans ce même widget
            self.current_sim.integrate()
            self.current_sim.build_scene(self)
            # Forcer le rafraîchissement
            self.gl_widget.update()
            # logique pour reconstruire ou mettre à jour la scène
            # e.g., vider self.gl_widget.objects et rappeler build_scene()

    def on_simulation_selected(self, text):
        for sim in self.sims:
            if sim.name == text:
                self.current_sim = sim
                self.load_scene(self.current_sim)

    def update_progress_bar(self, value):
        self.progress_bar.setValue(value)


    def load_scene(self, sim):
        print(f"Chargement de {sim.name}...")

        self.setWindowTitle(f"SpaceTool : {sim.name}")

        # Vider la liste des objets dans le GLWidget (supposons qu'ils sont dans self.gl_widget.objects)
        self.gl_widget.objects.clear() 
        # Reconstruire la scène dans ce même widget
        
        #Worker thread to compute seperatly from GUI
        self.worker = SimulationWorker(sim, self)
        self.thread = QThread()
        
        self.worker.moveToThread(self.thread)
        
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        
        self.worker.progress.connect(self.update_progress_bar)  # Optional
        self.thread.start()
        
        #Scene setup (GUI)
        #sim.build_scene(self)       #COMME LES CALCULS SONT SUR UN AUTRE THREAD LES RESULTATS N'EXISTENT PAS ET DONC ON NE PEUX PAS 
                                    #DEFINIR "scale".
        # Forcer le rafraîchissement     
        self.gl_widget.update()

class SimulationWorker(QObject):
    finished = pyqtSignal()
    progress = pyqtSignal(int)  # Optional: emit steps or percentage

    def __init__(self, simulation, win):
        super().__init__()
        self.simulation = simulation
        self.win = win
        self._is_running = True

    def run(self):
        steps = int(self.simulation.span // self.simulation.dt)
        emit_every = max(1, steps // 100)  # Emit at most 100 updates
    
        for i in range(steps):
            if not self._is_running:
                break
    
            self.simulation.integrate_step()
    
            # Emit only every N steps
            if i % emit_every == 0 or i == steps - 1:
                self.progress.emit(int((i + 1) / steps * 100))
    
        self.finished.emit()
        self.simulation.build_scene(self.win)

    def stop(self):
        self._is_running = False

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MainWindow()
    win.resize(1920, 1080)
    win.show()
    sys.exit(app.exec_())
