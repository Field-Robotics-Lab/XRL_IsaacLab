#import omni.usd
from pxr import Usd, UsdGeom, Gf, Sdf, UsdShade

# Create a new USD stage in memory
stage = Usd.Stage.CreateInMemory()
#stage = Usd.Stage.CreateNew('dirt_field.usd')

# Define the root Prim at path '/World', which is a common practice
root_prim = stage.DefinePrim("/World", "Xform")
stage.SetDefaultPrim(root_prim)

# Define a large empty dirt field
field = UsdGeom.Mesh.Define(stage, '/World/Field')
#field = stage.DefinePrim('/World/ground','Mesh')

# Define points for the field
points = [
    Gf.Vec3f(-100, 0, -100),
    Gf.Vec3f(100, 0, -100),
    Gf.Vec3f(100, 0, 100),
    Gf.Vec3f(-100, 0, 100)
]
field.CreatePointsAttr(points)

# Define face vertex indices and counts for the field
face_vertex_indices = [0, 1, 2, 3]
face_vertex_counts = [4]
field.CreateFaceVertexIndicesAttr(face_vertex_indices)
field.CreateFaceVertexCountsAttr(face_vertex_counts)

# Define a simple material for the field
#material = UsdGeom.Mesh.Define(stage, '/World/Material')
material = UsdShade.Material.Define(stage, '/World/Looks/DirtMaterial')
#shader = UsdShade.Shader.Define(stage, '/World/Material/Shader')
shader = UsdShade.Shader.Define(stage, '/World/Looks/DirtMaterial/Shader')
shader.CreateIdAttr('UsdPreviewSurface')
shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.5, 0.3, 0.1))  # Brown color

# Connect shader to material
#material.CreateSurfaceOutput().ConnectToSource(shader, 'surface')
shader_out = shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
material.CreateSurfaceOutput().ConnectToSource(shader_out)

# Bind the material to the field
UsdShade.MaterialBindingAPI(field).Bind(material)
#binding_api = UsdShade.MaterialBindingAPI(field)
#binding_api.Bind(material)
#print(field.GetPrim().GetTypeName())

# Print the resulting USD stage
print(stage.GetRootLayer().ExportToString())

# Save the stage to a file
stage.GetRootLayer().Export("dirt_field.usda")