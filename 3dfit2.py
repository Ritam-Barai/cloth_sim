import sys
import cv2
import numpy as np
import trimesh
import json
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from scipy.spatial import Delaunay,distance

class ShirtMesh:
    def __init__(self, frame_width, frame_height):
        # Create a basic t-shirt mesh with known measurements
        self.base_measurements = {
            'chest': 50.0,  # cm
            'length': 70.0,  # cm
            'shoulders': 45.0  # cm
        }
        
        # Store frame dimensions for scaling
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.template = ShirtTemplate(frame_width, frame_height)
        
        # Create vertices for a basic t-shirt shape
        self.create_base_mesh()
        
    def create_base_mesh(self):
        """Create a basic t-shirt mesh aligned with template"""
        # Define template width and height relative to frame
        template_width = int(self.frame_width * 0.6)
        template_height = int(self.frame_height * 0.8)
        
        # Calculate center position
        center_x = self.frame_width // 2
        center_y = self.frame_height // 2
        '''
        # Define normalized vertices matching template points
        normalized_vertices = np.array([
            [0.5, 0.15, 0.0],    # Center top (neck)
            [0.4, 0.1, 0.0],  # Left collar
            [0.2, 0.15, 0.0],    # Left shoulder
            [0.05, 0.25, 0.0],   # Left sleeve
            [0.2, 0.4, 0.0],    # Left armpit
            [0.2, 0.7, 0.0],    # Left waist
            [0.2, 0.9, 0.0],   # Left bottom
            [0.45, 0.95, 0.0],   # Left bottom center
            [0.55, 0.95, 0.0],    # Right bottom center
            [0.8, 0.9, 0.0],    # Right bottom
            [0.8, 0.7, 0.0],    # Right waist
            [0.8, 0.4, 0.0],    # Right armpit
            [0.95, 0.25, 0.0],   # Right sleeve
            [0.8, 0.15, 0.0],    # Right shoulder
            [0.6, 0.1, 0.0]  # Right collar
            
            
            
            
            
            
            
        ])
        '''
        with open('norm_vertex_data.json') as file:
            data = json.load(file)

        # Loop through each vertex and access x, y, z
        #for vertex in data:
        # Convert to list if it's a tuple
        #    location = list(vertex['normalized_location'])
        # Extract x and z components from normalized_location
        normalized_vertices = np.array([
            [round(vertex['normalized_location'][0],2)+0.50,0.50- round(vertex['normalized_location'][2],2),0.0] 
            for vertex in data
        ])
       
        # Scale vertices to template size and position
        vertices = normalized_vertices.copy()
        vertices[:, 0] = (vertices[:, 0] * template_width) + (center_x - template_width/2)
        vertices[:, 1] = (vertices[:, 1] * template_height) + (center_y - template_height/2)
        
        # Create triangulation
        points_2d = vertices[:, :2]
        tri = Delaunay(points_2d)
        
        # Store mesh data
        self.vertices = vertices
        self.faces = tri.simplices
        self.original_vertices = vertices.copy()
        
        # Create trimesh object for 3D operations
        self.mesh = trimesh.Trimesh(vertices=vertices, faces=self.faces)
        
        # Store template points for alignment
        self.template_points = vertices[:, :2].astype(np.int32)
        
    def deform_to_silhouette(self, silhouette_points, view='front'):
        """Deform mesh to match detected silhouette points"""
        #print(f"Original vertices:\n{self.vertices}")
        #print(f"Silhouette points:\n{silhouette_points}")
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
        left_armpit = self.vertices[4]
        right_armpit = self.vertices[11]
        measurements['chest'] = np.linalg.norm(right_armpit - left_armpit)
        
        # Length measurement (top to bottom)
        top_point = np.mean([self.vertices[1],self.vertices[14]], axis=0)
        bottom_point = np.mean(self.vertices[7:8], axis=0)  # Average of bottom points
        measurements['length'] = np.linalg.norm(bottom_point - top_point)
        
        # Shoulder measurement
        left_shoulder = self.vertices[2]
        right_shoulder = self.vertices[13]
        measurements['shoulders'] = np.linalg.norm(right_shoulder - left_shoulder)
        
        return measurements

class ShirtTemplate:
    def __init__(self, frame_width, frame_height):
        # Define template size relative to frame
        self.width = int(frame_width * 0.5)  # 40% of frame width
        self.height = int(frame_height * 0.7)  # 60% of frame height
        
        # Calculate center position
        self.center_x = frame_width // 2
        self.center_y = frame_height // 2
        
        self.padding_factor = 0.2
        
        
        '''
        # Define normalized points for a t-shirt shape
        # Points are defined clockwise from top center
        self.template_points = np.array([
            [0.5, 0.15],     # Top center (neck)
            [0.4, 0.1],    # Left collar
            [0.22, 0.15],   # Left shoulder
            [0.05, 0.25],    # Left armpit
            [0.2, 0.4],    # Left upper body
            [0.2, 0.7],    # Left waist
            [0.2, 0.9],     # Left bottom
            [0.45, 0.95],    # Bottom left center
            [0.55, 0.95],    # Bottom right center
            [0.8, 0.9],     # Right bottom
            [0.8, 0.7],    # Right waist
            [0.8, 0.4],    # Right upper body
            [0.95, 0.25],    # Right armpit
            [0.78, 0.15],   # Right shoulder
            [0.6, 0.1],    # Right collar
        ])
        '''
        with open('norm_vertex_data.json') as file:
            data = json.load(file)

        x_values = [round(vertex['normalized_location'][0],2) for vertex in data]
        y_values = [round(vertex['normalized_location'][2],2) for vertex in data]
        x_avg = round((max(x_values) - min(x_values))/2.00,2)# - (0.5-max(x_values))
        y_avg = round((max(y_values) - min(y_values))/2.00,2) #- (0.5-max(y_values))
        self.template_points = np.array([
            [round(vertex['normalized_location'][0],2)+x_avg,y_avg  - round(vertex['normalized_location'][2],2)] 
            for vertex in data
        ])
        print(len(self.template_points),self.template_points)
        # Generate template points
        self.generate_template(frame_width, frame_height)
        
    def generate_template(self,frame_width, frame_height):
        """Generate template points for a t-shirt shape"""
        
        
        # Scale and position the template
        scaled_points = self.template_points.copy()
        scaled_points[:, 0] = (scaled_points[:, 0] * self.width) + (self.center_x - self.width/2)
        scaled_points[:, 1] = (scaled_points[:, 1] * self.height) + (self.center_y - self.height/2)
        
        self.points = scaled_points.astype(np.int32)
        print(self.points)
        
        # Create template mask
        #self.mask = np.zeros((self.height * 2, self.width * 2), dtype=np.uint8)
        self.mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
        # Create closed polygon
        cv2.fillPoly(self.mask, [self.points], 255)
        
        # Create a new mask to apply the padding
        self.padded_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)

        # Create a padded polygon by dilating the original polygon
        kernel_size = int(self.padding_factor * np.max((frame_height, frame_width)))  # Determine kernel size for dilation
        kernel = np.ones((kernel_size, kernel_size), np.uint8)  # Create a square kernel

        # Dilate the original mask to create a padded effect
        self.padded_mask = cv2.dilate(self.mask, kernel, iterations=1)
        cv2.imshow("",self.padded_mask)
        
        # Define edges for visualization
        self.edges = [
            # Neckline
            (0, 1), (0, 14),
            # Left side
            (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7),
            # Bottom
            (7, 8),
            # Right side
            (8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14),
            # Additional details
            (2, 4),  # Left sleeve
            (11, 13)  # Right sleeve
        ]

class ShirtDetector:
    def __init__(self, frame_width, frame_height):
        self.shirt_mesh = ShirtMesh(frame_width, frame_height)
        self.template = ShirtTemplate(frame_width, frame_height)
        self.f_width = frame_width
        self.f_height = frame_height
        self.is_aligned = False
        self.alignment_threshold = 0.080
        self.stable_frames = 0
        self.required_stable_frames = 10
        self.adjustment_factor = 0.04
        self.tolerance = 20
        self.adjusted_points = []
        self.og_points = self.template.template_points
        self.edges_near_tp = {i: [] for i in range(len(self.template.points))}
        
    '''
    def process_frame(self, frame):
        """Process frame and check alignment with template"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        
        # Threshold to get binary image
        #binary = cv2.adaptiveThreshold(
        #gray, 255, 
        #cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        #cv2.THRESH_BINARY, 5, 2
        #)
        _, binary = cv2.threshold(
        gray, 0, 255, 
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        
        # Optionally combine edges with binary image using bitwise OR
        binary = cv2.bitwise_or(binary, edges)
        # Calculate overlap with template
        overlap = self.calculate_overlap(binary)
        
        # Draw template and feedback
        frame = self.draw_feedback(frame, overlap)
        
        if overlap >= self.alignment_threshold:
            self.stable_frames += 1
            if self.stable_frames >= self.required_stable_frames:
                self.is_aligned = True
                # Use template points for mesh fitting
                return frame, self.template.points
        else:
            self.stable_frames = 0
            self.is_aligned = False
            
        return frame, None'''
        
    def process_frame(self, frame):
        """Process frame and check alignment with template"""
        
        # Reset adjusted points for the new frame
        self.adjusted_points = []
        shirt_contour = 255
        

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Select the largest contour as the shirt contour
            all_inside = True
            shirt_contour = max(contours, key=cv2.contourArea).reshape(-1, 2)
            print("Shh",len(shirt_contour))
            for point in shirt_contour:
                if self.template.padded_mask[point[1],point[0]] == 0:  # Check if point is outside the mask
                    all_inside = False
                    break  # No need to check further points for this contour
            if all_inside:
            
                # Create a blank mask with the same dimensions as the frame
                mask = np.zeros(frame.shape[:2], dtype=np.uint8)

                # Fill the mask with the largest contour
                cv2.fillPoly(mask, [shirt_contour], 255)  # Fill the contour area with white (255)
                colored_area = cv2.bitwise_and(frame, frame, mask=mask)

                # Blend the colored area with the frame using a specified alpha
                alpha = 0.7
                frame = cv2.addWeighted(colored_area, alpha, frame, 1 - alpha, 0)
                
                # Adjust template points based on the nearest shirt edges
                for idx, tp in enumerate(self.template.points):
                    # Find all contour points within tolerance of tp
                    nearby_edges = []
                    for i in range(len(shirt_contour) - 1):
                        start, end = shirt_contour[i], shirt_contour[i + 1]
                        edge_midpoint = (start + end) / 2

                        # Check if the midpoint of the edge is within tolerance distance
                        if distance.euclidean(tp, edge_midpoint) <= self.tolerance:
                            nearby_edges.append((start, end))

                    # Update edges near this template point if better edges are found
                    if nearby_edges:
                        # Store the edge if it's closer than any previously stored edges
                        self.edges_near_tp[idx] = nearby_edges

                    # Calculate the adjusted point by finding the closest point on the closest edge
                    if self.edges_near_tp[idx]:
                        closest_edge = min(
                            self.edges_near_tp[idx],
                            key=lambda edge: min(
                                distance.euclidean(tp, edge[0]),
                                distance.euclidean(tp, edge[1])
                            )
                        )
                        closest_point = min(closest_edge, key=lambda pt: distance.euclidean(tp, pt))

                        # Adjust the template point towards the closest point on the closest edge
                        if distance.euclidean(tp, closest_point) > self.tolerance:
                            new_point = tp + self.adjustment_factor * (closest_point - tp)
                            self.adjusted_points.append(new_point.astype(int))
                        else:
                            self.adjusted_points.append(tp)
                    else:
                        self.adjusted_points.append(tp)

        """ Adjust template points towards shirt edges where they are non-aligned. """
        
        '''
        for tp in self.template.points:
            # Find the closest contour point to each template point
            distances = np.linalg.norm(shirt_contour - tp, axis=1)
            closest_point = shirt_contour[np.argmin(distances)]
        
            # Adjust template point towards the closest shirt contour point
            if distance.euclidean(tp, closest_point) > self.tolerance:
                new_point = tp + self.adjustment_factor * (closest_point - tp)
                self.adjusted_points.append(new_point.astype(int))
            else:
                self.adjusted_points.append(tp)'''
        
         # Check if more than 4 of the original template points are in adjusted points
        count_valid_points = sum(1 for tp in self.template.points if any(np.array_equal(tp, ap) for ap in self.adjusted_points))
        
        
        # Check alignment score to determine fit
        alignment_score = np.mean([distance.euclidean(p1, p2) for p1, p2 in zip(self.template.points, self.adjusted_points)]) 
        #overlap = alignment_score/100 
        frame = self.draw_feedback(frame, self.stable_frames)
        #if not self.is_aligned:
            #print(self.adjusted_points,self.template.points,alignment_score) 
        
        if count_valid_points <= 5 and not self.is_aligned:
            # Not enough points for alignment; reset adjusted points to original template points
            #self.adjusted_points = self.template.points.copy()  # Or handle as needed
            print(alignment_score)
            cv2.putText(frame, "Not enough aligned points for fitting.", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return frame, None  # Skip this frame or return default values
         
        #else:
        self.template.points = np.array(self.adjusted_points)
        
        #self.template.generate_template(self.f_width, self.f_height)
        #self.adjusted_points = np.array(self.adjusted_points).reshape((-1, 1, 2))  # Reshape to (num_points, 1, 2
        polygon_points = np.array(self.template.points).reshape((-1,1,2))
        
        
        if (alignment_score < self.tolerance) and not self.is_aligned:
            self.stable_frames += 1
            #print("Yaay")
            # Segmentation when alignment score is within tolerance
            #mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            
            #cv2.imshow("Segmented Shirt", segmented_shirt)
            cv2.putText(frame, "Aligning template to shirt...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if self.stable_frames >= self.required_stable_frames and not self.is_aligned:
                self.is_aligned = True
                # Use template points for mesh fitting
                cv2.fillPoly(self.template.mask, [polygon_points], 1)
                segmented_shirt = cv2.bitwise_and(frame, frame, mask=self.template.mask)
                cv2.imshow("Segmented Shirt", segmented_shirt)
                return frame, self.template.points
        else:
            self.stable_frames = 0
            self.is_aligned = False
            
            
        return frame,None
        
        
        
        
    def draw_feedback(self, frame, overlap):
        """Draw template visualization and feedback"""
        overlay = frame.copy()
        
        # Draw template edges
        color = (0, 255, 0) if self.is_aligned else (0, 0, 255)
        points = self.template.points
        
        for start_idx, end_idx in self.template.edges:
            start_point = tuple(points[start_idx])
            end_point = tuple(points[end_idx])
            cv2.line(overlay, start_point, end_point, color, 2)
        
        # Draw template points
        for point in points:
            cv2.circle(overlay, tuple(point), 4, color, -1)
        
        # Draw 3D mesh preview if aligned
        if self.is_aligned:
            self.draw_3d_preview(overlay)
        
        # Add alignment percentage and instructions
        text = f"Alignment: {overlap*10:.1f}%"
        cv2.putText(overlay, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        if not self.is_aligned:
            instruction = "Please align shirt with template"
            cv2.putText(overlay, instruction, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(overlay, "Aligned! Fitting mesh...", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Blend with original frame
        alpha = 0.7
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        return frame
        
    def draw_3d_preview(self, frame):
        """Draw preview of 3D mesh fitting"""
        if hasattr(self.shirt_mesh, 'mesh'):
            # Project 3D mesh edges onto 2D
            for face in self.shirt_mesh.faces:
                for i in range(3):
                    pt1 = self.shirt_mesh.vertices[face[i], :2].astype(int)
                    pt2 = self.shirt_mesh.vertices[face[(i+1)%3], :2].astype(int)
                    cv2.line(frame, tuple(pt1), tuple(pt2), (255, 165, 0), 1)
    
    def calculate_overlap(self, binary_image):
        """Calculate overlap percentage between template and detected shirt"""
        shirt_mask = cv2.bitwise_and(binary_image, self.template.mask)
        template_pixels = cv2.countNonZero(self.template.mask)
        overlap_pixels = cv2.countNonZero(shirt_mask)
        return overlap_pixels / template_pixels if template_pixels > 0 else 0

class ShirtSizeFitter(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Mesh-Based Shirt Size Fitter")
        self.setGeometry(100, 100, 1200, 800)
        
        self.camera = cv2.VideoCapture(0)
        _, frame = self.camera.read()
        self.frame_height, self.frame_width = frame.shape[:2]
        
        self.detector = ShirtDetector(self.frame_width, self.frame_height)
        self.current_view = 'front'
        
        self.setup_ui()
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def update_frame(self):
        ret, frame = self.camera.read()
        if ret:
            # Process frame with template matching
            processed_frame, silhouette_points = self.detector.process_frame(frame)
            
            if silhouette_points is not None and self.detector.is_aligned:
                # Update mesh with detected silhouette
                self.detector.shirt_mesh.deform_to_silhouette(silhouette_points, self.current_view)
            
            # Convert to Qt format and display
            rgb_image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.camera_label.setPixmap(QPixmap.fromImage(qt_image))
            
            # Update measurements if aligned
            if self.detector.is_aligned:
                self.show_measurements()
                
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
        if self.camera.isOpened():
            self.camera.release()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ShirtSizeFitter()
    window.show()
    sys.exit(app.exec_())
