import bpy

# Replace this with the ID (or index) of the vertex you want to highlight
vertex_id = 10

# Ensure we're in Object Mode to select the mesh object first
bpy.ops.object.mode_set(mode='OBJECT')

# Get the active object
obj = bpy.context.active_object

# Make sure we are working with a mesh
if obj and obj.type == 'MESH':
    # Go to Edit Mode
    bpy.ops.object.mode_set(mode='EDIT')
    
    # Access the object's vertex data in Edit Mode
    bpy.ops.mesh.select_all(action='DESELECT')  # Deselect all vertices first
    bpy.ops.object.mode_set(mode='OBJECT')  # Switch to Object mode to access vertices by ID
    
    # Select and highlight the specific vertex by its index
    obj.data.vertices[vertex_id].select = True  # Select the vertex
    obj.data.vertices[vertex_id].co  # Access the coordinates if needed

    # Return to Edit Mode to view the selection
    bpy.ops.object.mode_set(mode='EDIT')
    
    print(f"Vertex with ID {vertex_id} is now selected and highlighted.")
else:
    print("No active mesh object selected.")

