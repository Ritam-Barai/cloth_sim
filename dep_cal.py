import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QLabel, QProgressBar, QPushButton
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap, QPalette, QColor


class DepthCalibration:
    def __init__(self):
        # Known shirt measurements for calibration (in cm)
        self.reference_height = 70  # Standard T-shirt height
        self.reference_width = 50   # Standard T-shirt width
        
        # Store multiple z-positions and corresponding pixel measurements
        self.z_measurements = []
        
        # Calibration state
        self.is_calibrated = False
        self.scale_factor = None
        self.focal_length = None
        
    def add_measurement(self, pixel_height, pixel_width, relative_z=None):
        """Add a measurement at current z position"""
        if relative_z is None:
            relative_z = len(self.z_measurements)
            
        self.z_measurements.append({
            'pixel_height': pixel_height,
            'pixel_width': pixel_width,
            'relative_z': relative_z
        })
        
        if len(self.z_measurements) >= 2:
            self._calculate_calibration()
            
    def _calculate_calibration(self):
        """Calculate calibration parameters using multiple measurements"""
        if len(self.z_measurements) < 2:
            return
            
        # Get measurements at different z positions
        z1 = self.z_measurements[0]
        z2 = self.z_measurements[-1]
        
        # Calculate focal length using similar triangles principle
        height_ratio = z2['pixel_height'] / z1['pixel_height']
        z_ratio = z2['relative_z'] / max(z1['relative_z'], 1)  # Avoid division by zero
        
        self.focal_length = (z_ratio * z1['pixel_height'] * z2['relative_z'] - 
                           z1['pixel_height'] * z1['relative_z']) / (z_ratio - 1)
        
        # Calculate scale factor (pixels to cm)
        self.scale_factor = self.reference_height / z1['pixel_height']
        
        self.is_calibrated = True
        
    def get_depth(self, pixel_height):
        """Estimate depth from pixel height using calibrated parameters"""
        if not self.is_calibrated:
            return None
            
        return (self.focal_length * self.reference_height) / pixel_height
        
    def get_real_measurements(self, pixel_height, pixel_width):
        """Convert pixel measurements to real-world measurements using depth"""
        if not self.is_calibrated:
            return None, None
            
        depth = self.get_depth(pixel_height)
        
        # Adjust measurements based on perspective projection
        real_height = (pixel_height * depth) / self.focal_length
        real_width = (pixel_width * depth) / self.focal_length
        
        return real_width, real_height


class ShirtSizeFitter(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Shirt Size Fitter with Z-Calibration")
        self.setGeometry(100, 100, 1000, 800)
        
        self.camera = cv2.VideoCapture(0)
        self.calibration = DepthCalibration()
        self.setup_ui()
        
        # Camera update timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)
        
        self.measurement_count = 0  # Counter for measurements
        self.bbox_width = 80        # Initial bounding box width
        self.bbox_height = 100       # Initial bounding box height
    
    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Camera view
        self.camera_label = QLabel()
        self.camera_label.setMinimumSize(800, 600)
        layout.addWidget(self.camera_label)
        
        # Calibration progress
        self.calibration_bar = QProgressBar()
        layout.addWidget(self.calibration_bar)
        
        # Instructions
        self.instruction_label = QLabel(
            "Move camera slowly up and down to calibrate depth measurement"
        )
        self.instruction_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.instruction_label)

        # Button to capture measurement
        self.capture_button = QPushButton("Capture Measurement")
        self.capture_button.clicked.connect(self.capture_measurement)
        layout.addWidget(self.capture_button)

        # Button to reset calibration
        self.reset_button = QPushButton("Reset Calibration")
        self.reset_button.clicked.connect(self.reset_calibration)
        layout.addWidget(self.reset_button)
        
    def update_frame(self):
        ret, frame = self.camera.read()
        if ret:
            processed_frame = frame.copy()
        
            # Center the bounding box
            frame_height, frame_width = processed_frame.shape[:2]
            box_x = int((frame_width - self.bbox_width) // 2)  # Ensure box_x is an integer
            box_y = int((frame_height - self.bbox_height) // 2)  # Ensure box_y is an integer

            # Ensure bounding box dimensions are integers and valid
            bbox_width = int(self.bbox_width)
            bbox_height = int(self.bbox_height)
        
            # Draw a bounding box around the detected object
            cv2.rectangle(processed_frame, (box_x, box_y), (box_x + bbox_width, box_y + bbox_height), (0, 255, 0), 2)

            # Update calibration progress
            if self.calibration.is_calibrated:
                self.calibration_bar.setValue(100)
                self.instruction_label.setText("Calibration complete! Continue moving to refine measurements")

                # Calculate z_depth for the object
                z_depth = self.calibration.get_depth(bbox_height)
            
                # Display the z_depth on the frame
                cv2.putText(processed_frame, f"Z-Depth: {z_depth:.2f} cm", (box_x, box_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            else:
                progress = min(len(self.calibration.z_measurements) * 50, 99)
                self.calibration_bar.setValue(progress)

            # Convert to Qt format and display
            rgb_image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.camera_label.setPixmap(QPixmap.fromImage(qt_image))

    
    def capture_measurement(self):
        """Capture current frame's measurements and add to calibration"""
        ret, frame = self.camera.read()
        if ret:
            # Using actual pixel measurements from the bounding box
            pixel_height = self.bbox_height
            pixel_width = self.bbox_width
            
            # Increment measurement count for z position
            relative_z = self.measurement_count * 10  # Adjust the increment based on your setup
            
            # Add measurement to calibration
            self.calibration.add_measurement(pixel_height, pixel_width, relative_z)
            self.measurement_count += 1
            
            # Update bounding box dimensions based on current measurements
            self.bbox_width = pixel_width * self.calibration.scale_factor if self.calibration.is_calibrated else self.bbox_width
            self.bbox_height = pixel_height * self.calibration.scale_factor if self.calibration.is_calibrated else self.bbox_height
            
            # Update instruction label
            self.instruction_label.setText(f"Measurement {self.measurement_count} captured!")

    def reset_calibration(self):
        """Reset the calibration process"""
        self.calibration = DepthCalibration()
        self.measurement_count = 0
        self.bbox_width = 80  # Reset to initial width
        self.bbox_height = 100 # Reset to initial height
        self.calibration_bar.setValue(0)
        self.instruction_label.setText("Calibration reset. Move camera to start.")
    
    def closeEvent(self, event):
        self.camera.release()
        event.accept()





if __name__ == "__main__":
    
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    window = ShirtSizeFitter()
    window.show()
    sys.exit(app.exec_())


