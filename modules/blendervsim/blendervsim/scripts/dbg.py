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
    args.save_dir = "./"

    # Robot Arguments
    args.step_size = 1.8
    args.num_primitives = 32
    args.laser_scanner_num_points = 1024
    args.field_of_view_deg = 360
    args.map_type = "maze"

    known_map, map_data, pose, goal = environments.generate.map_and_poses(args)
    try:
        map_data['semantic_labels']['background'] = map_data['semantic_labels'].pop('wall')
        map_data['semantic_labels']['free'] = map_data['semantic_labels'].pop('hallway')
    except:
        pass

    return map_data


def main():
    # Start Blender as a subprocess
    scene = "/resources/blender_scenes/render_overhead.blend"
    with BlenderVSim(blender_scene_path=scene, verbose=True) as blender:
        blender.echo("Hello from blender!")
        blender.echo(message="Hello from blender 2!")
        # image = blender.render_image(
        #     render_settings={"samples": 4, "resolution_x": 512, "resolution_y": 512}
        # )
        # blender.error()

        # blender.error(message='Hello from blender 2!')
        # grid = np.random.rand(100, 100) > 0.3
        # sample_map_data = {
        #     'resolution': 1.0,
        #     'semantic_labels': {'background': 0, 'hallway': 1},
        #     'semantic_grid': grid,
        #     'occ_grid': grid
        # }
        sample_map_data = get_grid()
        sample_map_data['resolution'] = 0.3
        image, im_data = blender.render_overhead(map_data=sample_map_data,
                                                 pixels_per_meter=10,
                                                 edge_buffer_meters=2.5,
                                                 render_settings={'samples': 16, 'use_denoising': True})
        print(image.shape)
        plt.imshow(image, extent=im_data['extent'])
        plt.show()


if __name__ == "__main__":
    main()
