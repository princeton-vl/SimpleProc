import argparse
import hashlib
import os
import sys

import bpy
import numpy as np
from mathutils import Vector

from infinigen.core import init


def feature_rng_from_file(seed: int, suffix: str = ""):
    script_name = os.path.basename(sys.argv[0])
    tag = f"{seed}_{script_name}{suffix}"
    h = int(hashlib.sha256(tag.encode("utf-8")).hexdigest(), 16) % (2**32)
    return np.random.RandomState(h)


def get_world_bbox(obj):
    mat = obj.matrix_world
    return [mat @ Vector(corner) for corner in obj.bound_box]


def infer_camera_ids():
    inferred = []
    for obj in bpy.data.objects:
        if obj.name.startswith("camrig."):
            suffix = obj.name.split("camrig.", 1)[1]
            if suffix:
                inferred.append(suffix)

    # Preserve deterministic behavior: numeric ids first by numeric value, then lexical.
    return sorted(set(inferred), key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else x))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--input", required=True, help="Input .blend file")
    parser.add_argument("--output", required=True, help="Output .blend file")
    parser.add_argument("--random", type=int, default=False)
    args = init.parse_args_blender(parser)

    bpy.ops.wm.open_mainfile(filepath=args.input)
    rng = feature_rng_from_file(seed=args.seed)
    camera_ids = infer_camera_ids()

    flag = not args.random or rng.rand() < 0.5

    if flag:
        all_points = []
        for obj in bpy.data.objects:
            if obj.name.startswith("Object_"):
                all_points.extend(get_world_bbox(obj))

        for camera_id in camera_ids:
            cam = bpy.data.objects.get(f"camrig.{camera_id}")
            if cam is not None:
                all_points.append(cam.location)

        if not all_points:
            print("No Object_* meshes or camrig.* cameras found; skipping room generation")
            bpy.ops.wm.save_as_mainfile(filepath=args.output)
            return 0

        min_corner = Vector((min(p.x for p in all_points),
                         min(p.y for p in all_points),
                         min(p.z for p in all_points)))
        max_corner = Vector((max(p.x for p in all_points),
                            max(p.y for p in all_points),
                            max(p.z for p in all_points)))

        max_corner.z = max(max_corner.z, 10.5)
        max_corner += Vector((0.5, 0.5, 0.5))
        min_corner -= Vector((0.5, 0.5, 0.5))

        print("Global bounding box:")
        print("  Min:", min_corner)
        print("  Max:", max_corner)
        center = (min_corner + max_corner) / 2
        size = max_corner - min_corner

        lights = [x for x in bpy.data.objects if x.name.startswith("Area")]
        len_lights = len(lights)
        if len_lights > 0:
            min_num = max(1, int(round(len_lights * ((max_corner - min_corner).x * (max_corner - min_corner).y / 66 ** 2))))
            min_num = min(min_num, len_lights)
            total = rng.randint(min_num, len_lights + 1)
            lights = rng.choice(lights, total, replace=False)
        else:
            lights = []

        for obj in lights:
            obj.location.x = min(max_corner.x - 0.5, obj.location.x)
            obj.location.x = max(min_corner.x + 0.5, obj.location.x)
            obj.location.y = min(max_corner.y - 0.5, obj.location.y)
            obj.location.y = max(min_corner.y + 0.5, obj.location.y)

        def create_face_cube(name, location, scale):
            bpy.ops.mesh.primitive_cube_add(size=1, location=location)
            cube = bpy.context.active_object
            cube.name = name
            cube.scale = scale
            return cube

        # -X face
        create_face_cube("plane_0",
            (min_corner.x, center.y, center.z),
            (0.01, size.y, size.z))

        # +X face
        create_face_cube("plane_1",
            (max_corner.x, center.y, center.z),
            (0.01, size.y, size.z))

        # -Y face
        create_face_cube("plane_2",
            (center.x, min_corner.y, center.z),
            (size.x, 0.01, size.z))

        # +Y face
        create_face_cube("plane_3",
            (center.x, max_corner.y, center.z),
            (size.x, 0.01, size.z))

        # -Z face
        create_face_cube("plane_4",
            (center.x, center.y, min_corner.z),
            (size.x, size.y, 0.01))

        # +Z face
        create_face_cube("plane_5",
            (center.x, center.y, max_corner.z),
            (size.x, size.y, 0.01))

    bpy.ops.wm.save_as_mainfile(filepath=args.output)

if __name__ == "__main__":
    sys.exit(main())
