import bpy
import json
import os
import mathutils

# Function to get the bounding box dimensions of the mesh
def get_mesh_dimensions(obj):
    bbox_corners = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]
    dimensions = {
        'length_x': (bbox_corners[0] - bbox_corners[4]).length,
        'length_y': (bbox_corners[0] - bbox_corners[1]).length,
        'length_z': (bbox_corners[0] - bbox_corners[3]).length,
    }
    return dimensions

# Normalize the vertex location based on mesh dimensions
def normalize_location(location, dimensions):
    if dimensions['length_x'] != 0:
        location[0] /= dimensions['length_x']
    if dimensions['length_y'] != 0:
        location[1] /= dimensions['length_y']
    if dimensions['length_z'] != 0:
        location[2] /= dimensions['length_z']
    return location

# Load vertex data from JSON file
def load_vertex_data(json_file_path):
    if os.path.exists(json_file_path):
        with open(json_file_path, 'r') as file:
            return json.load(file)
    else:
        print(f"JSON file not found: {json_file_path}")
        return []
        
# Save normalized vertex data to a JSON file
def save_normalized_vertex_data(vertex_data, json_file_path):
    with open(json_file_path, 'w') as file:
        json.dump(vertex_data, file, indent=4)

# Update vertex locations in the mesh
def update_vertex_locations(obj, vertex_data, dimensions):
    bpy.ops.object.mode_set(mode='EDIT')  # Switch to Edit Mode
    bpy.ops.mesh.select_all(action='DESELECT')  # Deselect all vertices

    for entry in vertex_data:
        vertex_id = entry["vertex_id"]
        normalized_location = entry["normalized_location"]

        # Find the vertex and set its new location
        if vertex_id < len(obj.data.vertices):
            vertex = obj.data.vertices[vertex_id]
            # Normalize back to original scale before updating the location
            original_location = [coord * dimensions[dim] for coord, dim in zip(normalized_location, ['length_x', 'length_y', 'length_z'])]
            vertex.co = original_location

            # Select the vertex for visual feedback
            vertex.select = True

    bpy.ops.object.mode_set(mode='OBJECT')  # Switch back to Object Mode

# Main execution
active_obj = bpy.context.active_object

if active_obj and active_obj.type == 'MESH':
    dimensions = get_mesh_dimensions(active_obj)

    # Load vertex data from JSON
    json_file_path = bpy.path.abspath("//vertex_data.json")
    vertex_data = load_vertex_data(json_file_path)

    # Update vertex locations
    update_vertex_locations(active_obj, vertex_data, dimensions)
else:
    print("Please select a mesh object.")
    
    
    
    
# Function to get the active vertex
def get_selected_vertex_info(obj):
    if obj and obj.type == 'MESH':
        bpy.ops.object.mode_set(mode='OBJECT')
        
        # Get the selected vertices
        selected_vertices = [v for v in obj.data.vertices if v.select]
        
        # Check if exactly one vertex is selected
        if len(selected_vertices) == 1:
            vertex = selected_vertices[0]
            vertex_id = vertex.index
            vertex_location = list(vertex.co)

            print(f"Vertex ID: {vertex_id}, Location: {vertex_location}")

            return vertex_id, vertex_location
        else:
            print("Please select exactly one vertex.")
            return None, None
    else:
        print("Select a mesh object.")
        return None, None

# Normalize the vertex location based on mesh dimensions
def normalize_location(location, dimensions):
    if dimensions['length_x'] != 0:
        location[0] /= dimensions['length_x']
    if dimensions['length_y'] != 0:
        location[1] /= dimensions['length_y']
    if dimensions['length_z'] != 0:
        location[2] /= dimensions['length_z']
    return location

# Save measurements to a text file
def save_measurements_to_txt(dimensions):
    txt_file_path = bpy.path.abspath("/home/ritam/cloth_sim/mesh_dimensions.txt")
    with open(txt_file_path, 'w') as file:
        for key, value in dimensions.items():
            file.write(f"{key}: {value}\n")

# Save to JSON
def save_to_json(vertex_id, normalized_location):
    if vertex_id is not None:
        # Define the JSON file path
        json_file_path = bpy.path.abspath("/home/ritam/cloth_sim/vertex_data.json")
        norm_json_path = bpy.path.abspath("/home/ritam/cloth_sim/norm_vertex_data.json")

        # Load existing data if the file exists
        if os.path.exists(json_file_path):
            with open(json_file_path, 'r') as file:
                existing_data = json.load(file)
        else:
            existing_data = []

        # Prepare data to append to JSON
        data = {
            "vertex_id": vertex_id,
            "normalized_location": normalized_location
        }
        existing_data.append(data)

        # Write updated data back to JSON file
        with open(norm_json_path, 'w') as file:
            json.dump(existing_data, file, indent=4)

# Main execution
active_obj = bpy.context.active_object
dimensions = get_mesh_dimensions(active_obj)
vertex_id, vertex_location = get_selected_vertex_info(active_obj)

if vertex_id is not None and vertex_location is not None:
    normalized_location = normalize_location(vertex_location.copy(), dimensions)
    save_measurements_to_txt(dimensions)
    save_to_json(vertex_id, normalized_location)

