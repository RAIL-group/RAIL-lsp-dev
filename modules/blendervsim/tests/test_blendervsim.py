import numpy as np
import pytest

from blendervsim import BlenderVSim


@pytest.mark.timeout(15)
def test_blendervsim_runs_without_crashing():
    with BlenderVSim() as blender:
        pass


@pytest.mark.timeout(15)
def test_blendervsim_supports_comms():
    with BlenderVSim() as blender:
        blender.echo('Hello from blender!')
        blender.echo(message='Hello from blender!')


@pytest.mark.timeout(15)
def test_blendervsim_can_render_image():
    with BlenderVSim() as blender:
        image = blender.render_image(
            render_settings={'samples': 1, 'resolution_x': 64, 'resolution_y': 32})

    assert image is not None
    assert image.shape[0] == 32
    assert image.shape[1] == 64
    assert np.mean(image) > 0
    assert np.std(image) > 0
