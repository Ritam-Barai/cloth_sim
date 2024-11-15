import bpy
import json
import os

# Function to get the active vertex
def get_selected_vertex_info():
    obj = bpy.context.active_object

    # Ensure we are working with a mesh object
    if obj and obj.type == 'MESH':
        # Switch to Object Mode to access vertex data
        bpy.ops.object.mode_set(mode='OBJECT')
        
        # Get the selected vertices
        selected_vertices = [v for v in obj.data.vertices if v.select]
        
        # Check if exactly one vertex is selected
        if len(selected_vertices) == 1:
            vertex = selected_vertices[0]
            vertex_id = vertex.index
            vertex_location = list(vertex.co)

            print(f"Vertex ID: {vertex_id}, Location: {vertex_location}")

            # Prepare data to append to JSON
            data = {
                "vertex_id": vertex_id,
                "location": vertex_location
            }
            return data
        else:
            print("Please select exactly one vertex.")
            return None
    else:
        print("Select a mesh object.")
        return None

# Save to JSON
def save_to_json(data):
    if data:
        # Define the JSON file path
        json_file_path = bpy.path.abspath("/home/ritam/cloth_sim/vertex_data.json")

        # Load existing data if the file exists
        if os.path.exists(json_file_path):
            with open(json_file_path, 'r') as file:
                existing_data = json.load(file)
        else:
            existing_data = []

        # Append the new data
        existing_data.append(data)

        # Write updated data back to JSON file
        with open(json_file_path, 'w') as file:
            json.dump(existing_data, file, indent=4)

# Execute the functions
vertex_info = get_selected_vertex_info()
save_to_json(vertex_info)


