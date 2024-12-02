# Blender Visual Sim Module: `blendervsim`

This package is designed to make it easy to render images and figures from within Blender via interprocess communication from within the core RAIL code base. This simple minimal example opens a Blender scene file and then renders an image, returned as a numpy array:

```python
scene = "/resources/blender_scenes/render_overhead.blend"
with BlenderVSim(blender_scene_path=scene) as blender:
	image = blender.render_image(
		render_settings={"samples": 1, "resolution_x": 64, "resolution_y": 32}
	)
```

## Quick Start

The downloading of Blender and installation of relevant packages is done via the `blender-build` target. This target runs Blender within the docker container, and so passing `USE_GPU=false` may be advisable.

This set of commands will build the repository, download Blender, install packages within it, and then run tests specific to this module:

```python
make build
make blender-build USE_GPU=false
make test USE_GPU=false PYTEST_FILTER=blendervsim
```

[The tests have minimal examples](./tests/test_blendervsim.py) showing some of the basic functionality of the code. 

## Code Structure

The code exists in two parts:
1. The core `blendervsim` code (and the `BlenderVSim` class) that runs within the main repository. This code launches Blender as a subprocess and handles communication to it and error handling.
2. The Blender-run code in the `blenderscripts` folder. This code is run from within Blender, which receives commands sent via the parent code. The `render_main.py` script runs a `BlenderManager` class and listens on a loop until killed.

The `getattr` function of the `BlenderVSim` class has been overloaded so that functions (and their arguments) on an instance are passed to the `BlenderManager` within the Blender subprocess and run there. This means that functions need only be implemented on the Blender side and can then be run transparently via calling it on the parent. For example, the `echo` command is implemented from within blender yet can be called from outside:

```python
# Within manage.py & run within Blender
class BlenderManager(object):
	...
    def echo(self, message):
        print(f"Printing data: {message}")

# Call on the parent process will call the 
# above `echo` function:
with BlenderVSim() as blender:
	blender.echo("Hello from blender!")
```

Extending this module should thus be done from within the `blenderscipts` directory, which handles the Blender side of things.
