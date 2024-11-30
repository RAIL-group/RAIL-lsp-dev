from blendervsim import BlenderVSim
import numpy as np

import socket
import subprocess
import struct
import pickle
import environments

import matplotlib
matplotlib.use("TkAgg")  # Use TkAgg backend
import matplotlib.pyplot as plt

def get_grid():
    # Set the arguments (usually done via the command line)
    args = lambda: None
    args.current_seed = 2005
    args.save_dir = './'

    # Robot Arguments
    args.step_size = 1.8
    args.num_primitives = 32
    args.laser_scanner_num_points = 1024
    args.field_of_view_deg = 360
    args.map_type = 'office2'

    known_map, map_data, pose, goal = environments.generate.map_and_poses(args)
    return map_data


def main():
    # Start Blender as a subprocess
    scene = '/resources/blender_scenes/render_overhead.blend'
    with BlenderVSim(blender_scene_path=scene, verbose=True, debug=False) as blender:
        blender.echo('Hello from blender!')
        blender.echo(message='Hello from blender 2!')
        blender.echo(message='Hello from blender 2!')
        image = blender.render_image(
            render_settings={'samples': 4, 'resolution_x': 512, 'resolution_y': 512})

        # grid = np.random.rand(100, 100) > 0.3
        # sample_map_data = {
        #     'resolution': 1.0,
        #     'semantic_labels': {'background': 0, 'hallway': 1},
        #     'semantic_grid': grid,
        #     'occ_grid': grid
        # }
        # image = blender.render_overhead(map_data=sample_map_data,
        #     render_settings={'samples': 64, 'resolution_x': 2400, 'resolution_y': 2400})
        # plt.imshow(image)
        # plt.show()


if __name__ == '__main__':
    main()
