# Blender Visual Sim Module: `blendervsim`

This package is designed to make it easy to render images and figures from within Blender via interprocess communication from within the core RAIL code base. This simple minimal example opens a Blender scene file and then renders an image, returned as a numpy array:

```python
scene = "/resources/blender_scenes/render_overhead.blend"
with BlenderVSim(blender_scene_path=scene) as blender:
	image = blender.render_image(
		render_settings={"samples": 1, "resolution_x": 64, "resolution_y": 32}
	)
```

## Quick Start & Minimal Example

The downloading of Blender and installation of relevant packages is done via the `blender-build` target. This target runs Blender within the docker container, and so passing `USE_GPU=false` may be advisable.

This set of commands will build the repository, download Blender, install packages within it, and then run tests specific to this module:

```python
make build
make blender-build USE_GPU=false
make test USE_GPU=false PYTEST_FILTER=blendervsim
```
