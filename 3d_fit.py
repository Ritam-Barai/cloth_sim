import sys
import cv2
import numpy as np
import trimesh
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from scipy.spatial import Delaunay

class ShirtMesh:
    def __init__(self):
        # Create a basic t-shirt mesh with known measurements
        self.base_measurements = {
            'chest': 50.0,  # cm
            'length': 70.0,  # cm
            'shoulders': 45.0  # cm
        }
        
        # Create vertices for a basic t-shirt shape
        self.create_base_mesh()
        
    def create_base_mesh(self):
        """Create a basic t-shirt mesh with known dimensions"""
        # Define key points for t-shirt shape (normalized coordinates)
        vertices = np.array([
            # Collar
            [0.5, 0.0, 0.0],    # Center top
            [0.35, 0.05, 0.0],  # Left collar
            [0.65, 0.05, 0.0],  # Right collar
            
            # Shoulders
            [0.2, 0.1, 0.0],    # Left shoulder
            [0.8, 0.1, 0.0],    # Right shoulder
            
            # Armpits
            [0.15, 0.3, 0.0],   # Left armpit
            [0.85, 0.3, 0.0],   # Right armpit
            
            # Waist
            [0.2, 0.6, 0.0],    # Left waist
            [0.8, 0.6, 0.0],    # Right waist
            
            # Bottom
            [0.25, 1.0, 0.0],   # Left bottom
            [0.75, 1.0, 0.0],   # Right bottom
        ])
        
        # Scale vertices to real measurements
        vertices[:, 0] *= self.base_measurements['shoulders']
        vertices[:, 1] *= self.base_measurements['length']
        
        # Create triangulation
        points_2d = vertices[:, :2]
        tri = Delaunay(points_2d)
        
        # Store mesh data
        self.vertices = vertices
        self.faces = tri.simplices
        self.original_vertices = vertices.copy()
        
        # Create trimesh object for 3D operations
        self.mesh = trimesh.Trimesh(vertices=vertices, faces=self.faces)
        
    def deform_to_silhouette(self, silhouette_points, view='front'):
        """Deform mesh to match detected silhouette points"""
        if view == 'front':
            # Match x and y coordinates while preserving z
            for i, point in enumerate(silhouette_points):
                if i < len(self.vertices):
                    self.vertices[i, 0] = point[0]
                    self.vertices[i, 1] = point[1]
        else:  # side view
            # Match y and z coordinates while preserving x
            for i, point in enumerate(silhouette_points):
                if i < len(self.vertices):
                    self.vertices[i, 1] = point[1]
                    self.vertices[i, 2] = point[0]  # x in image maps to z in 3D
        
        # Update mesh
        self.mesh.vertices = self.vertices
        self.mesh.fix_normals()
        
    def get_measurements(self):
        """Calculate measurements from deformed mesh"""
        measurements = {}
        
        # Chest measurement (distance between armpit points)
        left_armpit = self.vertices[5]
        right_armpit = self.vertices[6]
        measurements['chest'] = np.linalg.norm(right_armpit - left_armpit)
        
        # Length measurement (top to bottom)
        top_point = self.vertices[0]
        bottom_point = np.mean(self.vertices[9:11], axis=0)  # Average of bottom points
        measurements['length'] = np.linalg.norm(bottom_point - top_point)
        
        # Shoulder measurement
        left_shoulder = self.vertices[3]
        right_shoulder = self.vertices[4]
        measurements['shoulders'] = np.linalg.norm(right_shoulder - left_shoulder)
        
        return measurements

class ShirtDetector:
    def __init__(self):
        self.shirt_mesh = ShirtMesh()
        
    def detect_shirt_silhouette(self, frame):
        """Detect shirt silhouette in frame"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply edge detection
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return frame, None
            
        # Find the largest contour (assumed to be the shirt)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Simplify contour to match mesh points
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # Ensure we have the right number of points
        target_points = len(self.shirt_mesh.vertices)
        if len(approx_contour) > target_points:
            # Reduce points by taking equally spaced samples
            indices = np.linspace(0, len(approx_contour)-1, target_points, dtype=int)
            silhouette_points = approx_contour[indices].reshape(-1, 2)
        else:
            # Interpolate to get more points
            silhouette_points = self.interpolate_points(approx_contour.reshape(-1, 2), target_points)
        
        # Draw detected points and mesh
        self.draw_detection(frame, silhouette_points, largest_contour)
        
        return frame, silhouette_points
        
    def interpolate_points(self, points, target_count):
        """Interpolate between points to get desired number of points"""
        # Calculate cumulative distances
        dists = np.zeros(len(points))
        for i in range(1, len(points)):
            dists[i] = dists[i-1] + np.linalg.norm(points[i] - points[i-1])
            
        # Create evenly spaced points
        total_dist = dists[-1]
        new_dists = np.linspace(0, total_dist, target_count)
        
        # Interpolate x and y coordinates
        new_points = np.zeros((target_count, 2))
        for i in range(2):
            new_points[:, i] = np.interp(new_dists, dists, points[:, i])
            
        return new_points
        
    def draw_detection(self, frame, points, contour):
        """Draw detection visualization"""
        # Draw contour
        cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
        
        # Draw key points
        for point in points:
            cv2.circle(frame, tuple(point.astype(int)), 3, (255, 0, 0), -1)
            
        # Draw mesh lines
        for face in self.shirt_mesh.faces:
            for i in range(3):
                pt1 = self.shirt_mesh.vertices[face[i], :2].astype(int)
                pt2 = self.shirt_mesh.vertices[face[(i+1)%3], :2].astype(int)
                cv2.line(frame, tuple(pt1), tuple(pt2), (0, 0, 255), 1)

class ShirtSizeFitter(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Mesh-Based Shirt Size Fitter")
        self.setGeometry(100, 100, 1200, 800)
        
        self.camera = cv2.VideoCapture(0)
        self.detector = ShirtDetector()
        self.current_view = 'front'
        
        self.setup_ui()
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)
        
    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Camera view
        self.camera_label = QLabel()
        self.camera_label.setMinimumSize(800, 600)
        layout.addWidget(self.camera_label)
        
        # Controls
        controls_layout = QHBoxLayout()
        
        self.view_button = QPushButton("Switch to Side View")
        self.view_button.clicked.connect(self.toggle_view)
        controls_layout.addWidget(self.view_button)
        
        self.measure_button = QPushButton("Calculate Measurements")
        self.measure_button.clicked.connect(self.show_measurements)
        controls_layout.addWidget(self.measure_button)
        
        layout.addLayout(controls_layout)
        
        # Measurements display
        self.measurements_label = QLabel()
        self.measurements_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.measurements_label)
        
    def update_frame(self):
        ret, frame = self.camera.read()
        if ret:
            # Process frame
            processed_frame, silhouette_points = self.detector.detect_shirt_silhouette(frame)
            
            if silhouette_points is not None:
                # Update mesh with detected silhouette
                self.detector.shirt_mesh.deform_to_silhouette(silhouette_points, self.current_view)
            
            # Convert to Qt format and display
            rgb_image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.camera_label.setPixmap(QPixmap.fromImage(qt_image))
            
    def toggle_view(self):
        self.current_view = 'side' if self.current_view == 'front' else 'front'
        self.view_button.setText(f"Switch to {'Side' if self.current_view == 'front' else 'Front'} View")
        
    def show_measurements(self):
        measurements = self.detector.shirt_mesh.get_measurements()
        measurements_text = (
            f"Chest: {measurements['chest']:.1f} cm\n"
            f"Length: {measurements['length']:.1f} cm\n"
            f"Shoulders: {measurements['shoulders']:.1f} cm"
        )
        self.measurements_label.setText(measurements_text)
        
    def closeEvent(self, event):
        self.camera.release()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ShirtSizeFitter()
    window.show()
    sys.exit(app.exec_())
