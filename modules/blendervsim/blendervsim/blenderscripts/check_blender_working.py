import bpy
import numpy as np

print(f"{np.zeros((10, 10))}")

print("Working.")

# Set the render engine and output settings
bpy.context.scene.render.engine = 'CYCLES'  # Or 'BLENDER_EEVEE'
bpy.context.scene.cycles.device = 'CPU'
bpy.context.scene.render.image_settings.file_format = 'PNG'
bpy.context.scene.cycles.samples = 1  # Adjust sample count for quality
bpy.context.scene.cycles.use_adaptive_sampling = True
bpy.context.scene.cycles.use_denoising = False


output_path = "/tmp/render_result.png"  # Adjust to your desired location
bpy.context.scene.render.image_settings.file_format = 'PNG'
bpy.context.scene.render.filepath = output_path

# Render and save to file
bpy.ops.render.render(write_still=True)

from PIL import Image
import numpy as np

# Load the rendered image
image = Image.open(output_path)

# Convert the image to a NumPy array
image_array = np.array(image)
print("Image loaded as NumPy array with shape:", image_array.shape)
