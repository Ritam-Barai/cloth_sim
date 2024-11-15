import bpy
import json
import os
import mathutils
import math

# Function to get the bounding box dimensions of the mesh
def get_mesh_dimensions(obj):
    bbox_corners = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]
    dimensions = {
        'length_x': (bbox_corners[0] - bbox_corners[4]).length* 1.05,
        'length_z': (bbox_corners[0] - bbox_corners[1]).length* 1.05,
        'length_y': (bbox_corners[0] - bbox_corners[3]).length* 1.05,
        'eu_distance': 0,
    }
    return dimensions

def normalize_location(location, dimensions):
    if dimensions['length_x'] != 0:
        location[0] = float(f"{(location[0] / dimensions['length_x']):.2f}")
    if dimensions['length_y'] != 0:
        location[1] = float(f"{(location[1] / dimensions['length_y']):.2f}")
    if dimensions['length_z'] != 0:
        location[2] = float(f"{location[2] / dimensions['length_z']:.2f}")
    return location


# Save normalized vertex data to a JSON file
def save_normalized_vertex_data(vertex_data, json_file_path):
    with open(json_file_path, 'w') as file:
        json.dump(vertex_data, file, indent=4)
        
# Load vertex data from JSON file
def load_vertex_data(json_file_path):
    if os.path.exists(json_file_path):
        with open(json_file_path, 'r') as file:
            return json.load(file)
    else:
        print(f"JSON file not found: {json_file_path}")
        return []
        
def euclidean_distance(point1, point2):
    return round(math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2 + (point1[2] - point2[2])**2), 2)
        
# Save measurements to a text file
def save_measurements_to_txt(dimensions):
    txt_file_path = bpy.path.abspath("/home/ritam/cloth_sim/mesh_dimensions.txt")
    with open(txt_file_path, 'w') as file:
        for key, value in dimensions.items():
            file.write(f"{key}: {value}\n")

# Main execution
active_obj = bpy.context.active_object
# Define the JSON file path
json_file_path = bpy.path.abspath("/home/ritam/cloth_sim/vertex_data.json")
norm_json_path = bpy.path.abspath("/home/ritam/cloth_sim/norm_vertex_data.json")




if active_obj and active_obj.type == 'MESH':
    dimensions = get_mesh_dimensions(active_obj)
    
    
    normalized_vertices = []
    
    # Load existing data if the file exists
    if os.path.exists(json_file_path):
        with open(json_file_path, 'r') as file:
            existing_data = json.load(file)
    else:
       existing_data = []
       
    vertex2_location = existing_data[2]['location']
    vertex13_location = existing_data[13]['location']
    
    # Calculate Euclidean distance
    eu_distance = euclidean_distance(vertex2_location, vertex13_location)
    dimensions['eu_distance'] = eu_distance
    save_measurements_to_txt(dimensions)
    
    # Get vertex locations and normalize them
    for vertex in existing_data:
        vertex_id = vertex['vertex_id']
        original_location = bpy.context.active_object.data.vertices[vertex_id].co.copy()  # Get the original vertex location from the mesh
        
        normalized_location = normalize_location(original_location, dimensions)
        
        # Create a dictionary to store vertex ID and normalized location
        vertex_info = {
            "vertex_id": vertex_id,
            "normalized_location": normalized_location[:]
        }
        normalized_vertices.append(vertex_info)

    # Save the normalized vertex data to norm_vertex.json
    #json_file_path = bpy.path.abspath("//norm_vertex.json")
    save_normalized_vertex_data(normalized_vertices, norm_json_path)
    print(f"Normalized vertex data saved to: {norm_json_path}")
else:
    print("Please select a mesh object.")


