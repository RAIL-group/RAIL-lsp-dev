import bpy


DEFAULT_RENDER_SETTINGS = {
    "engine": "CYCLES",
    "device": "CPU",
    "resolution_x": 256,
    "resolution_y": 256,
    "samples": 1,
    "use_adaptive_sampling": True,
    "use_denoising": False,
}


def apply_render_settings(upd_render_settings):
    """Apply render settings from a dictionary."""
    render_settings = DEFAULT_RENDER_SETTINGS.copy()
    if upd_render_settings:
        for k, v in upd_render_settings.items():
            render_settings[k] = v

    scene = bpy.context.scene
    render = scene.render  # Render settings
    render.image_settings.file_format = 'PNG'

    # Set general render settings
    for key, value in render_settings.items():
        if hasattr(render, key):
            setattr(render, key, value)
        elif render.engine == "CYCLES" and hasattr(bpy.context.scene.cycles, key):
            setattr(bpy.context.scene.cycles, key, value)
        else:
            print(f"Warning: Unknown setting {key}")
