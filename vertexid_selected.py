import bpy

# Make sure you're in Edit mode and have a mesh object selected
obj = bpy.context.active_object
if obj and obj.type == 'MESH':
    bpy.ops.object.mode_set(mode='EDIT')
    # Switch to object mode to access vertex data
    bpy.ops.object.mode_set(mode='OBJECT')
    
    for vertex in obj.data.vertices:
        if vertex.select:
            print("Selected Vertex ID:", vertex.index)

