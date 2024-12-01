import sys

sys.path.append("/Users/gjstein/.local/lib/python3.11/site-packages")

# Important for running within Jupyter
import os

os.environ["MPLBACKEND"] = "Agg"  # Set a non-GUI backend


import bpy
import bmesh
import matplotlib.pyplot
import numpy as np
import pickle
import shapely
import shapely.ops
import scipy.signal


def _apply_transforms(obj):
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.transform_apply(scale=True)


def _add_robot_path(robot_poses, radius, height=0.0):
    coordinates = [(pose[1], pose[0], height) for pose in robot_poses]

    # Create a new curve object
    curve_data = bpy.data.curves.new(name="MyCurve", type="CURVE")
    curve_data.dimensions = "3D"  # Set to '2D' if you want a flat curve

    # Create a new spline within the curve and add points to it
    spline = curve_data.splines.new(
        type="POLY"
    )  # 'POLY' creates a polyline, 'BEZIER' creates a smooth curve
    spline.points.add(len(coordinates) - 1)  # Add the correct number of points

    # Assign the coordinates to the spline points
    for i, coord in enumerate(coordinates):
        x, y, z = coord
        spline.points[i].co = (
            x,
            y,
            z,
            1,
        )  # The fourth value is the weight, set to 1 for default

    curve_data.bevel_depth = radius
    curve_data.bevel_resolution = 8

    # Create an object with the curve data and link it to the scene
    curve_object = bpy.data.objects.new("robot_path", curve_data)
    bpy.context.collection.objects.link(curve_object)

    return curve_object


def make_obj_from_grid(grid, resolution, z_floor=0.0, extrude_height=1.0):
    # Create a new mesh and a new object
    mesh = bpy.data.meshes.new("PolygonMesh")
    obj = bpy.data.objects.new("PolygonObject", mesh)

    # Link the object to the current scene
    bpy.context.collection.objects.link(obj)

    # Set the object as active and enter edit mode
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")

    # Use a BMesh to create geometry
    bm = bmesh.new()

    # Get coords (and flip x and x), then scale and offset.
    coordinates = [(x, y) for (y, x), val in np.ndenumerate(grid) if val > 0.5]
    coordinates = resolution * np.array(coordinates) - 0.5 * resolution

    r = resolution
    for x, y in coordinates:
        verts = [
            bm.verts.new((x, y, z_floor))
            for x, y in [(x, y), (x + r, y), (x + r, y + r), (x, y + r)]
        ]
        bm.faces.new(verts)

    # Update mesh and exit edit mode
    bmesh.ops.remove_doubles(
        bm, verts=bm.verts, dist=0.001
    )  # Remove any duplicate vertices
    bpy.ops.object.mode_set(mode="OBJECT")
    bm.to_mesh(mesh)
    bm.free()

    if extrude_height > 0.0:
        # Add Solidify modifier to extrude the polygon
        solidify_modifier = obj.modifiers.new(name="Extrude", type="SOLIDIFY")
        solidify_modifier.thickness = (
            extrude_height  # Set the extrude thickness (adjust as needed)
        )
        solidify_modifier.offset = 1.0
        bpy.ops.object.modifier_apply(modifier=solidify_modifier.name)

    return obj


def _add_frontier(
    frontier_points, point_radius, success_probability, colormap="viridis"
):

    # Make the object
    # raise ValueError(frontier_points.T[0])
    spheres = [
        _add_sphere(point[1], point[0], 0, point_radius, name="f")
        for point in frontier_points.T
    ]
    frontier = _join_objects(spheres)
    frontier.name = "frontier"

    # Set the material and color
    bsdf = _add_material_to_object(
        frontier, material_name=f"frontier_mat_{success_probability}"
    )
    cmap = matplotlib.pyplot.get_cmap(colormap)
    color = list(cmap(success_probability))
    bsdf.inputs["Alpha"].default_value = 0.5
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Emission Color"].default_value = color
    bsdf.inputs["Emission Strength"].default_value = 0.0

    return frontier


def _add_material_to_object(obj, material_name):
    # Check if the material already exists
    if material_name in bpy.data.materials:
        # Retrieve the existing material
        material = bpy.data.materials[material_name]
    else:
        # If the material doesn't exist, create a new one
        material = bpy.data.materials.new(name=material_name)
        material.use_nodes = True  # Enable nodes if needed

    # Customize the material (for example, set a base color)
    bsdf = material.node_tree.nodes.get("Principled BSDF")

    # Assign the material to the object
    obj.data.materials.clear()  # Clear existing materials if needed
    obj.data.materials.append(material)

    return bsdf


def set_object_properties(obj):
    if "frontier" in obj.name:
        return

    if "door" in obj.name:
        obj.scale.z = 0.01
    if "room" in obj.name:
        obj.scale.z = 0.01
    if "hallway" in obj.name:
        obj.scale.z = 0.01
    if "free" in obj.name:
        obj.scale.z = 0.01
    if "background" in obj.name:
        obj.scale.z = 1.0
        if "unseen" in obj.name:
            obj.scale.z = 0.1
    if "goal_path" in obj.name:
        obj.scale.z = 0.01
    if "clutter" in obj.name:
        obj.scale.z = 0.5
        if "unseen" in obj.name:
            obj.scale.z = 0.05

    _add_material_to_object(obj, obj.name)
    _apply_transforms(obj)


def add_map_data(map_data, robot_poses=None, observed_grid=None, subgoal_data=None, do_partial_walls=True):

    objects = []

    for semantic_class, value in map_data["semantic_labels"].items():
        if semantic_class == "background" and do_partial_walls:
            inflated_free = scipy.signal.convolve(
                (map_data["occ_grid"] < 0.5).astype(int), np.ones((5, 5)), mode="same"
            )
            grid_region = ((inflated_free > 0.1) & (map_data["occ_grid"] > 0.5)).astype(
                int
            )
        else:
            grid_region = map_data["semantic_grid"] == value

        if grid_region.sum() == 0:
            continue

        if observed_grid is not None:
            grid_region_seen = (observed_grid >= 0) & grid_region
            object = make_obj_from_grid(
                grid_region_seen.astype(int), map_data["resolution"], extrude_height=1.0
            )
            object.name = semantic_class
            objects.append(object)

            grid_region_unseen = (observed_grid < 0) & grid_region
            object = make_obj_from_grid(
                grid_region_unseen.astype(int),
                map_data["resolution"],
                extrude_height=1.0,
            )
            object.name = semantic_class + "_unseen"
            objects.append(object)
        else:
            object = make_obj_from_grid(
                grid_region.astype(int), map_data["resolution"], extrude_height=1.0
            )
            object.name = semantic_class
            objects.append(object)

    if robot_poses is not None:
        objects.append(
            _add_robot_path(robot_poses, map_data["resolution"] * 1.5)
        )
        objects.append(
            _add_sphere(
                robot_poses[-1][1],
                robot_poses[-1][0],
                0.0,
                radius=3 * map_data["resolution"],
                name="robot_pose_orb",
            )
        )

    if subgoal_data is not None:
        objects += [
            _add_frontier(
                f["points"] * map_data["resolution"],
                0.65 * map_data["resolution"],
                f["prob_feasible"],
            )
            for f in subgoal_data
        ]

    [set_object_properties(object) for object in objects]

    return objects


def _add_sphere(x, y, z, radius, name):
    bpy.ops.mesh.primitive_ico_sphere_add(
        subdivisions=3, radius=radius, location=(x, y, z)
    )
    obj = bpy.context.active_object
    obj.name = name
    return obj


def _join_objects(objects):
    # Ensure all objects are deselected first
    bpy.ops.object.select_all(action="DESELECT")

    # Select and set the first object in the list as active
    [obj.select_set(True) for obj in objects]

    # Set the first object in the list as the active object
    bpy.context.view_layer.objects.active = objects[0]

    bpy.ops.object.join()

    return objects[0]


def render_map_data(map_data):
    # Populate the scene
    objects = add_map_data(map_data)
    # Render the image
    bpy.ops.render.render(write_still=True)

    # Clear newly-added objects
    [bpy.data.objects.remove(obj, do_unlink=True) for obj in objects]
    [bpy.data.meshes.remove(mesh) for mesh in bpy.data.meshes if mesh.users == 0]
    [bpy.data.curves.remove(curve) for curve in bpy.data.curves if curve.users == 0]
