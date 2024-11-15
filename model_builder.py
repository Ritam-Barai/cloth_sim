import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import cv2
import numpy as np
from typing import List, Dict, Tuple
import albumentations as A
from PIL import Image
import time
import os
import csv
from pathlib import Path
from datetime import datetime


def write_to_file(epoch, train_loss,train_pck, val_loss,val_pck, file_path='training_log.csv'):
    """Append the epoch, train_loss, and val_loss to a CSV file."""
    
    # Get the current timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # Open the file in append mode and write the data
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Write the epoch, train_loss, and val_loss as a new row
        writer.writerow([timestamp, epoch, train_loss,train_pck, val_loss, val_pck])
'''
class TshirtKeypointNet(nn.Module):
    def __init__(self, num_keypoints: iimport csvnt):
        super(TshirtKeypointNet, self).__init__()
        
        # Use ResNet18 as backbone
        self.backbone = torch.hub.load('pytorch/vision:v0.10.0', 
                                     'resnet18', pretrained=True)
        
        # Modify final layers for keypoint detection
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_keypoints * 2)  # x,y coordinates for each keypoint
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        output = self.backbone(x)
        return output.view(batch_size, -1, 2)  # Reshape to (batch_size, num_keypoints, 2)
'''
class TshirtKeypointNet(nn.Module):
    def __init__(self, category_to_keypoints: Dict[str, int]):
        super(TshirtKeypointNet, self).__init__()
        #self.category_to_keypoints = category_to_keypoints
        
        # Use ResNet18 as backbone
        self.backbone = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        
        # Modify final layers for dynamic keypoint detection
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.keypoint_heads = nn.ModuleList()
        category_list = list(category_to_keypoints.keys())
        for category, num_keypoints in category_to_keypoints.items():
            #category_index = category_list.index(category)
            self.keypoint_heads.append ( nn.Sequential(
                nn.Linear(num_features, 512),
                #nn.InstanceNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.5),
                #nn.Linear(512, 256),
                #nn.InstanceNorm1d(256),
                #nn.ReLU(),
                #nn.Dropout(0.5),
                nn.Linear(512, num_keypoints * 2),  # *2 for x,y coordinates
                #nn.InstanceNorm1d(num_keypoints * 2),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(num_keypoints * 2, 40 *2),  # *2 for x,y coordinates
            ))
    
    def forward(self, x, category_index: torch.Tensor):
        '''
        # Ensure `category_index` is an integer
        category_index = category_index.item() if isinstance(category_index, torch.Tensor) else category_index

        batch_size = x.size(0)
        features = self.backbone(x)
        
        keypoint_head = self.keypoint_heads[category_index]
        output = keypoint_head(features)
        # Reshape output to (batch_size, num_keypoints, 2) where 2 corresponds to (x, y)
        num_keypoints = keypoint_head[-1].out_features // 2
        return output.view(x.size(0), num_keypoints, 2)  # Reshape to (batch_size, num_keypoints, 2)
        '''
        batch_size = x.size(0)
        print("Image shape",x.shape)
    
        #resnet18_feature_extractor = torch.nn.Sequential(*list(self.backbone.children())[:-1])
        features = self.backbone(x)
        print(features.shape)
        features = features.view(batch_size, -1)  # Flatten if needed

        # Initialize an empty list to store the outputs for each element in the batch
        outputs = []

        for i in range(batch_size):
        
            # Get the features from the backbone (ResNet18 in your case)
            
            # Get the category index for the current element in the batch
            category_idx = category_index[i].item()
        
            # Select the keypoint head corresponding to the category
            keypoint_head = self.keypoint_heads[category_idx]
            #print(keypoint_head)
        
            # Get current sample's features
            current_features = features[i:i+1]  # Keep batch dimension: [1, 1000]
            
            # Pass the features through the selected keypoint head
            output = keypoint_head(current_features)  # Should output [1, num_keypoints * 2]
        
            # Get the number of keypoints
            #num_keypoints = keypoint_head[-1].out_features // 2
        
            # Reshape the output to (1, num_keypoints, 2)
            outputs.append(output.view(1, 40, 2))
    
        # Stack the outputs to get a tensor of shape (batch_size, num_keypoints, 2)
        return torch.cat(outputs, dim=0)
        
        
class TshirtKeypointDetector:
    def __init__(self, model_path: str, num_keypoints: int = 40):
        """
        Initialize the keypoint detector.
        
        Args:
            model_path: Path to trained model weights
            num_keypoints: Number of keypoints to detect
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = TshirtKeypointNet(num_keypoints).to(self.device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
        self.transform = A.Compose([
            A.Resize(256, 256),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ], keypoint_params=A.KeypointParams(format='xy'))
        
    def detect_keypoints(self, frame: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Detect t-shirt keypoints in a frame.
        
        Args:
            frame: RGB image as numpy array
            
        Returns:
            Dictionary mapping vertex_id to normalized keypoint coordinates
        """
        # Prepare image
        transformed = self.transform(image=frame)
        image = torch.FloatTensor(transformed['image']).unsqueeze(0)
        image = image.to(self.device)
        
        # Get predictions
        with torch.no_grad():
            predictions = self.model(image)
        
        # Convert predictions to normalized coordinates
        keypoints = predictions[0].cpu().numpy()
        
        # Create result dictionary
        result = {}
        for i, (x, y) in enumerate(keypoints):
            # Add dummy z-coordinate (can be refined with depth estimation)
            result[i] = np.array([x, y, 0.0])
            
        return result

class RealTimeTshirtAnalysis:
    def __init__(self, model_path: str, template_points: List[Dict]):
        """
        Initialize the real-time t-shirt analysis system.
        
        Args:
            model_path: Path to trained keypoint detection model
            template_points: Template points for t-shirt adjustment
        """
        self.detector = TshirtKeypointDetector(model_path)
        self.template_adjuster = TshirtTemplateAdjuster(template_points)
        
    def process_frame(self, frame: np.ndarray) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
        """
        Process a single frame.
        
        Args:
            frame: RGB image as numpy array
            
        Returns:
            Tuple of (detected keypoints, adjusted template points)
        """
        # Detect keypoints
        keypoints = self.detector.detect_keypoints(frame)
        
        # Adjust template
        adjusted_template = self.template_adjuster.adjust_template(keypoints)
        
        return keypoints, adjusted_template
    
    def run_realtime(self):
        """Run real-time analysis using webcam."""
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame
            keypoints, adjusted_template = self.process_frame(frame)
            
            # Visualize results
            self._visualize_frame(frame, keypoints, adjusted_template)
            
            # Display
            cv2.imshow('T-shirt Analysis', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
        
    def _visualize_frame(self, frame: np.ndarray, 
                        keypoints: Dict[int, np.ndarray],
                        adjusted_template: Dict[int, np.ndarray]):
        """
        Visualize detected keypoints and adjusted template on frame.
        
        Args:
            frame: RGB image
            keypoints: Detected keypoints
            adjusted_template: Adjusted template points
        """
        # Draw detected keypoints
        for vertex_id, point in keypoints.items():
            x, y = int(point[0] * frame.shape[1]), int(point[1] * frame.shape[0])
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(frame, str(vertex_id), (x + 5, y + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Draw adjusted template connections
        connections = [
            (436, 367), (367, 532), (532, 300), (300, 164),
            (164, 101), (101, 153), (153, 155), (155, 222),
            (222, 413), (413, 387), (387, 506), (506, 328),
            (328, 528), (528, 364), (364, 436)
        ]
        
        for start_id, end_id in connections:
            if start_id in adjusted_template and end_id in adjusted_template:
                start = adjusted_template[start_id]
                end = adjusted_template[end_id]
                
                start_point = (int(start[0] * frame.shape[1]), 
                             int(start[1] * frame.shape[0]))
                end_point = (int(end[0] * frame.shape[1]), 
                           int(end[1] * frame.shape[0]))
                
                cv2.line(frame, start_point, end_point, (255, 0, 0), 2)
                
def train_model(train_loader: DataLoader, 
                val_loader: DataLoader,
                #num_keypoints: int,
                category_to_keypoints:Dict[str, int],
                num_epochs: int = 50):
    """
    Train the keypoint detection model.
    
    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        num_keypoints: Number of keypoints to detect
        num_epochs: Number of training epochs
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Define the category-to-keypoints mapping
    
    criterion = nn.MSELoss()
    threshold = 0.05 * 256
    best_val_loss = float('inf')
    
    # Initialize model with a default num_keypoints value (e.g., for short_sleeve_top)
    #default_category = 'short_sleeve_top'
    model = TshirtKeypointNet(category_to_keypoints).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.05)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_pck = 0.0
        train_kp = 0
        for batch in train_loader:
            images = batch['image'].to(device)
            keypoints = batch['keypoints'].to(device)
            bbox = batch['bbox'].to(device)
            num_keypoints = batch['num_keypoints'].to(device)
            category_index = batch['category_index'].to(device)
            visibility = batch['visibility'].to(device)
            #print(visibility.shape)
            #category = batch['category'] # Assuming category is provided as a string in the batch
            #print(f'Category: {category.shape}')
            # Handle the case where category is a list
            #if isinstance(category, list):
            #    category = category[0]  # Take the first category if it's a list
            
            # Map category to num_keypoints
            #num_keypoints = category_to_keypoints.get(category, 0)
            
            #if num_keypoints == 0:
            #    raise ValueError(f"Unknown category: {category}")
            
            # Instantiate or update the model based on num_keypoints
            #model = TshirtKeypointNet(num_keypoints).to(device)
            
            #optimizer = optim.Adam(model.parameters(), lr=0.001)
            #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
            
            optimizer.zero_grad()
            #print(images)
            outputs = model(images,category_index)
            
            # Initialize batch loss to 0
            batch_loss = 0.0
            
            
    
            # Iterate through each image in the batch
            for i in range(images.size(0)):  # Iterate over batch size
                #num_kp = num_keypoints[i].item()  # Get number of keypoints for this image
        
                # Slice the outputs and keypoints based on num_keypoints[i]
                output_slice = outputs[i, :] #*256 # Slicing model output
                #outputs[i, :] = output_slice
                visibility_expand = visibility[i, :].unsqueeze(-1)  # Shape [40, 1]
                
                # Create a mask where visibility == 0
                mask = ~torch.isclose(visibility_expand,torch.zeros(40, 1).to(device),atol=1e-6)  # Shape [40, 1], True where visibility is 0, False otherwise


                # Multiply only where visibility == 0 (we use the mask)
                #output_slice = output_slice * mask.float()  # Convert mask to float to allow multiplication
                #outputs[i, :] = output_slice
                keypoints_slice = keypoints[i, :]  # Slicing ground truth keypoints
                keypoints_slice  = keypoints_slice * mask.float()
                keypoints[i, :] = keypoints_slice 
                #print(output_slice.shape,keypoints_slice.shape,mask.float().shape )
                # Compute the Euclidean distance between the corresponding points in output_slice and keypoint_slice
                
                #
        
                #print(output_slice,keypoints_slice)
                # Compute the loss for this image
            #outputs = outputs * 256
            loss = criterion(outputs/256.0, keypoints/256.0)
            #print(outputs,outputs.shape,keypoints,keypoints.shape)
            #batch_loss += image_loss  # Accumulate loss for the batch
    
            # Average loss for the batch
            #batch_loss = batch_loss / images.size(0)  # Normalize by batch size
            '''
            loss = criterion(outputs, keypoints)
            loss.backward()
            
            for i, category in enumerate(categories):
                outputs = model(images[i:i+1], category)
                loss = criterion(outputs, keypoints[i:i+1])
                loss.backward()
                '''
            
            # Initialize the PCK tensor with zeros, shape [batch_size, 1]
            pck_tensor = torch.zeros(images.size(0), 2).to(device)
            for i in range(images.size(0)):  # Iterate over batch size
                num_kp = num_keypoints[i].item()  # Get number of keypoints for this image
                outputs_slice = outputs[i, :num_kp, :]  # Shape [30, 5, 2]
                keypoints_slice = keypoints[i, :num_kp, :]  # Shape [30, 5, 2]
                print(outputs_slice.shape, keypoints_slice.shape)
                euclidean_distances = torch.norm((outputs_slice - keypoints_slice), p =2, dim=1)
                # Count keypoints within the threshold
                correct_keypoints = (euclidean_distances < threshold).float()
                pck_tensor[i,0] = correct_keypoints.mean()  # Convert to percentage
                pck_tensor[i,1] = len(correct_keypoints)
                print(outputs_slice, keypoints_slice, correct_keypoints, pck_tensor[i,:], euclidean_distances)
            
            train_loss += loss.item()
            train_pck += (pck_tensor[:, 0] * pck_tensor[:, 1]).sum().item()
            train_kp += pck_tensor[:, 1].sum().item()  # Total valid keypoints
            #print('Batch PCK:',pck_tensor.sum())

            # Backpropagation
            loss.backward()
            optimizer.step()
            
        train_loss /= len(train_loader)
        train_pck = train_pck / train_kp if train_kp >0 else 0
        
        print('Train Accuracy:',train_pck*100)
        # Validation
        model.eval()
        val_loss = 0.0
        val_pck = 0.0
        val_kp = 0
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                keypoints = batch['keypoints'].to(device)
                bbox = batch['bbox'].to(device)
                num_keypoints = batch['num_keypoints'].to(device)
                category_index = batch['category_index'].to(device)
                visibility = batch['visibility'].to(device)
                # Ensure `category_index` is an integer
                #num_keypoints = num_keypoints.item() if isinstance(num_keypoints, torch.Tensor) else num_keypoints


                '''
                # Map category to num_keypoints
                #num_keypoints = category_to_keypoints.get(category, 0)
                
                #if num_keypoints == 0:
                #    raise ValueError(f"Unknown category: {category}")
                
                # Use the model with the appropriate num_keypoints for validation
                outputs = model(images)
                print(keypoints)
                loss = criterion(outputs,keypoints)
                #loss = criterion(outputs, list(map(lambda coord: [coord[0] * images.shape[0], coord[1] * images.shape[1]], keypoints)))
                
                for i, category in enumerate(categories):
                    outputs = model(images[i:i+1], category)
                    loss = criterion(outputs, keypoints[i:i+1])
                val_loss += loss.item()
                '''
                outputs = model(images,category_index)
                
                # Initialize batch loss to 0
                batch_val_loss = 0.0
                
        
                # Iterate through each image in the batch
                for i in range(images.size(0)):  # Iterate over batch size
                    #num_kp = num_keypoints[i].item()  # Get number of keypoints for this image
                    # Slice the outputs and keypoints based on num_keypoints[i]
                    output_slice = outputs[i, :] #*256 # Slicing model output
                    #outputs[i, :] = output_slice
                    visibility_expand = visibility[i, :].unsqueeze(-1)  # Shape [40, 1]
                
                    # Create a mask where visibility == 0
                    mask = ~torch.isclose(visibility_expand,torch.zeros(40, 1).to(device),atol=1e-6)  # Shape [40, 1], True where visibility is 0, False otherwise


                    # Multiply only where visibility == 0 (we use the mask)
                    #output_slice = output_slice * mask.float()  # Convert mask to float to allow multiplication
                    #outputs[i, :] = output_slice
                    keypoints_slice = keypoints[i, :]  # Slicing ground truth keypoints
                    keypoints_slice  = keypoints_slice * mask.float()
                    keypoints[i, :] = keypoints_slice 
                    
                    # Compute the Euclidean distance between the corresponding points in output_slice and keypoint_slice
                    
                    #print(correct_keypoints,val_pck_tensor[i]/100,euclidean_distances)
                    '''
                    # Slice the outputs and keypoints based on num_keypoints[i]
                    output_slice = outputs[i, :num_keypoints_for_image]  # Slicing model output
                    keypoints_slice = keypoints[i, :num_keypoints_for_image]  # Slicing ground truth keypoints
            
                    # Compute the loss for this image
                    image_val_loss = criterion(output_slice, keypoints_slice)
                    batch_val_loss += image_val_loss  # Accumulate loss for the batch
                    '''
        
                # Average loss for the batch
                #batch_val_loss = batch_val_loss / images.size(0)  # Normalize by batch size
                batch_val_loss=criterion(outputs/256.0, keypoints/256.0)
                val_loss += batch_val_loss.item()
                
                # Initialize the PCK tensor with zeros, shape [batch_size, 1]
                val_pck_tensor = torch.zeros(images.size(0), 2).to(device)
                for i in range(images.size(0)):  # Iterate over batch size
                    num_kp = num_keypoints[i].item()  # Get number of keypoints for this image
                    euclidean_distances = torch.norm(output_slice[:num_kp] - keypoints_slice[:num_kp], dim=1)
                    # Count keypoints within the threshold
                    correct_keypoints = (euclidean_distances < threshold).float()
                    val_pck_tensor[i,0] = correct_keypoints.mean()  # Convert to percentage
                    val_pck_tensor[i,1] = len(correct_keypoints)
                
                val_pck += (val_pck_tensor[:,0] * val_pck_tensor[:,1]).sum().item()
                val_kp += val_pck_tensor[:,1].sum().item()
                print('Batch Val PCK:',val_pck_tensor)
        val_loss /= len(val_loader)
        val_pck = val_pck / val_kp if val_kp > 0 else 0
        
        # Write epoch, train_loss, val_loss to file
        write_to_file(epoch, train_loss,train_pck, val_loss,val_pck,file_path='train/training_log.csv')
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'train/best_model.pth')
            
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Train PCK: {train_pck:.4f}')
        print(f'Val Loss: {val_loss:.4f}')
        print(f'Val PCK: {val_pck:.4f}')
