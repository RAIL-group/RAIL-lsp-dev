# Add to the path to import local packages
import bpy
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from utils.manager import BlenderManager
from utils.build import add_map_data


class BlenderManagerOverhead(BlenderManager):
    def render_overhead(self, map_data, render_settings=None):
        objects = add_map_data(map_data)
        image = self.render_image(render_settings=render_settings)

        # Clean up: clear newly-added objects
        [bpy.data.objects.remove(obj, do_unlink=True) for obj in objects]
        [bpy.data.meshes.remove(mesh) for mesh in bpy.data.meshes if mesh.users == 0]
        [bpy.data.curves.remove(curve) for curve in bpy.data.curves if curve.users == 0]

        return image


def main():
    with BlenderManagerOverhead() as manager:
        while manager.alive:
            manager.listen()


if __name__ == "__main__":
    main()
