import argparse
import hashlib
import math
import os
import sys

import bpy
import numpy as np

from mathutils import Vector, Euler

from infinigen.core import init


def feature_rng_from_file(seed: int, suffix: str = ""):
    script_name = os.path.basename(sys.argv[0])
    tag = f"{seed}_{script_name}{suffix}"
    h = int(hashlib.sha256(tag.encode("utf-8")).hexdigest(), 16) % (2**32)
    return np.random.RandomState(h)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Input .blend file", default=None)
    parser.add_argument("--output", required=True, help="Output .blend file")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--perturb", type=float, default=5.0)
    parser.add_argument("--theta_range", type=float, default=22.5, help="Theta half-range in degrees")
    parser.add_argument(
        "--phi_range",
        type=float,
        nargs=2,
        default=[-5.0, 30.0],
        metavar=("PHI_MIN", "PHI_MAX"),
        help="Phi range in degrees",
    )
    parser.add_argument("--fov_x", type=float, required=True)
    parser.add_argument("--fov_y", type=float, required=True)
    args = init.parse_args_blender(parser)
    rng = feature_rng_from_file(seed=args.seed)

    if args.input:
        bpy.ops.wm.open_mainfile(filepath=args.input)
    else:
        for obj in bpy.data.objects:
            bpy.data.objects.remove(obj, do_unlink=True)

    # Remove all objects from camera_rigs and cameras collections
    for collection_name in ["CameraRigs", "cameras", "camera_rigs"]:
        if collection_name in bpy.data.collections:
            collection = bpy.data.collections[collection_name]
            # First remove all objects in the collection
            for obj in list(collection.objects):
                bpy.data.objects.remove(obj, do_unlink=True)

    # Remove any existing camrig and camera objects
    for obj in list(bpy.data.objects):
        if obj.name.startswith("camrig.") or obj.name.startswith("camera_"):
            bpy.data.objects.remove(obj, do_unlink=True)

    # Keep option-4 behavior for camera distance and expose angular ranges directly.
    r_min, r_max = 6, 12
    theta_range = np.radians(args.theta_range)
    phi_range = np.radians(args.phi_range[0]), np.radians(args.phi_range[1])

    cam_data_lens = 22.473
    cam_data_sensor_width = 2 * cam_data_lens * np.tan(np.radians(args.fov_x)/2)
    cam_data_sensor_height = 2 * cam_data_lens * np.tan(np.radians(args.fov_y)/2)

    # Sample 8 cameras in spherical coordinates
    num_cameras = 8
    for i in range(num_cameras):
        # Sample spherical coordinates
        r = rng.uniform(r_min, r_max)
        theta = rng.uniform(-theta_range, theta_range)
        phi = rng.uniform(*phi_range)
        
        # Convert spherical to Cartesian (convention: theta=0, phi=0 -> (r, 0, 0))
        x = r * np.cos(phi) * np.cos(theta)
        y = r * np.cos(phi) * np.sin(theta)
        z = r * np.sin(phi)
        cam_location = Vector((x, y, z))

        # Create rig (parent) at the sampled pose
        rig_name = f"camrig.{i}"
        rig = bpy.data.objects.new(rig_name, None)
        rig.empty_display_type = 'PLAIN_AXES'
        rig.location = cam_location
        bpy.context.scene.collection.objects.link(rig)

        # Point rig at origin
        direction = Vector((0, 0, 0)) - cam_location
        rot_quat = direction.to_track_quat('-Z', 'Y')
        rig.rotation_euler = rot_quat.to_euler()

        # Create camera as child with identity local transform
        cam_data = bpy.data.cameras.new(name=f"Camera_{i}")

        cam_data.sensor_width = cam_data_sensor_width
        cam_data.sensor_height = cam_data_sensor_height
        cam_data.lens = cam_data_lens

        cam_obj = bpy.data.objects.new(name=f"camera_{i}_0", object_data=cam_data)
        bpy.context.scene.collection.objects.link(cam_obj)
        cam_obj.parent = rig
        cam_obj.matrix_parent_inverse = rig.matrix_world.inverted()
        cam_obj.location = Vector((0, 0, 0))
        cam_obj.rotation_euler = Euler((0.0, 0.0, 0.0), 'XYZ')

        
        # Apply perturbation to rig (after camera is parented)
        if args.perturb > 0:
            max_rad = math.radians(args.perturb)
            rx = rng.uniform(-max_rad, max_rad)
            ry = rng.uniform(-max_rad, max_rad)
            rz = rng.uniform(-max_rad, max_rad)
            rot = Euler((rx, ry, rz), 'XYZ').to_matrix().to_4x4()
            bpy.context.view_layer.update()
            rig.matrix_world = rig.matrix_world @ rot
        
 
    bpy.ops.wm.save_as_mainfile(filepath=args.output)

if __name__ == "__main__":
    main()