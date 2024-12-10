import numpy as np
import pytest
import time

from blendervsim import BlenderVSim


@pytest.mark.timeout(15)
def test_blendervsim_runs_without_crashing():
    with BlenderVSim() as blender:
        pass


@pytest.mark.timeout(15)
def test_blendervsim_supports_comms():
    with BlenderVSim() as blender:
        blender.echo("Hello from blender!")
        blender.echo(message="Hello from blender!")


@pytest.mark.timeout(15)
def test_blendervsim_can_render_image():
    with BlenderVSim() as blender:
        image = blender.render_image(
            render_settings={"samples": 1, "resolution_x": 64, "resolution_y": 32}
        )

    assert image is not None
    assert image.shape[0] == 32
    assert image.shape[1] == 64
    assert np.mean(image) > 0
    assert np.std(image) > 0


@pytest.mark.timeout(15)
def test_blendervsim_closes_when_errors():
    """When blender crashes, it should result in a runtime error and then close gracefully."""
    with BlenderVSim() as blender:
        with pytest.raises(RuntimeError):
            blender.error("This call should force an error in Blender")


@pytest.mark.timeout(15)
def test_blendervsim_render_overhead_fully_known():
    grid = np.zeros((100, 100))
    grid[10:90, 10:90] = 1
    grid[15:85, 15:85] = 2
    map_data = {
        'semantic_labels': {'background': 1, 'free': 2},
        'resolution': 0.1,
        'x_offset': 0,
        'y_offset': 0,
    }

    scene = "/resources/blender_scenes/render_overhead.blend"
    with BlenderVSim(blender_scene_path=scene) as blender:
        map_data['semantic_grid'] = np.ones_like(grid)
        map_data['occ_grid'] = np.zeros_like(grid)
        bkd_image, _ = blender.render_overhead(
            map_data=map_data,
            pixels_per_meter=25,
            render_settings={'samples': 32, 'use_denoising': True})

        map_data['semantic_grid'] = grid
        map_data['occ_grid'] = (grid > 1.5).astype(int)
        sq_image, _ = blender.render_overhead(
            map_data=map_data,
            pixels_per_meter=25,
            render_settings={'samples': 32, 'use_denoising': True})

    s = bkd_image.shape
    bkd_image_subset = bkd_image[int(0.25*s[0]):int(0.75*s[0]),
                                int(0.25*s[1]):int(0.75*s[1])][:,:, :3]
    sq_image_subset = sq_image[int(0.25*s[0]):int(0.75*s[0]),
                            int(0.25*s[1]):int(0.75*s[1])][:,:, :3]

    assert np.mean(bkd_image_subset) < 200
    assert np.mean(sq_image_subset) > 200


def test_blendervsim_render_gpu_cpu_speed():
    grid = np.zeros((100, 100))
    grid[10:90, 10:90] = 1
    grid[15:85, 15:85] = 2
    map_data = {
        'semantic_labels': {'background': 1, 'free': 2},
        'resolution': 0.1,
        'x_offset': 0,
        'y_offset': 0,
        'semantic_grid': grid,
        'occ_grid': (grid > 1.5).astype(int),

    }
    render_settings = {
    }

    scene = "/resources/blender_scenes/render_overhead.blend"

    with BlenderVSim(blender_scene_path=scene) as blender:
        stime = time.time()
        sq_image, _ = blender.render_overhead(
            map_data=map_data,
            pixels_per_meter=25,
            render_settings={
                'device': 'CPU',
                'resolution_x': 1024,
                'resolution_y': 1024,
                'samples': 1024,
                'use_denoising': False,
                "use_adaptive_sampling": False,
            })
        cpu_time = time.time() - stime

    with BlenderVSim(blender_scene_path=scene) as blender:
        stime = time.time()
        sq_image, _ = blender.render_overhead(
            map_data=map_data,
            pixels_per_meter=25,
            render_settings={
                'device': 'GPU',
                'resolution_x': 1024,
                'resolution_y': 1024,
                'samples': 1024,
                'use_denoising': False,
                "use_adaptive_sampling": False,
            })
        gpu_time = time.time() - stime


    print(f"CPU time: {cpu_time}")
    print(f"GPU time: {gpu_time}")
    assert False
