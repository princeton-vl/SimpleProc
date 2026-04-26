import argparse

import bpy
import numpy as np
from mathutils import Vector

from infinigen.assets.scatters import grass
from infinigen.core import init
from infinigen.core.placement import density
from infinigen.core.util.blender import apply_transform

def link(obj, col):
    for c in obj.users_collection:
        c.objects.unlink(obj)
    col.objects.link(obj)
    return obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--input", required=True, help="Input .blend file")
    parser.add_argument("--output", required=True, help="Output .blend file")
    parser.add_argument("--random", type=int, default=False)
    parser.add_argument("--height", type=float, default=-2)
    args = init.parse_args_blender(parser)
    bpy.ops.wm.open_mainfile(filepath=args.input)
    rng = np.random.RandomState(args.seed)
    for obj in bpy.data.objects:
        if obj.name == "Ground":
            bpy.data.objects.remove(obj, do_unlink=True)
    if (not args.random) or rng.randint(0, 2):
        if "Ground" not in bpy.data.objects.keys():
            col = bpy.data.collections["Collection"]
            ground_center1 = (0, 0, args.height)
            ground_scale1 = (33, 33, 1)
            bpy.ops.mesh.primitive_plane_add(size=2, location=ground_center1)
            ground = bpy.context.active_object
            ground.scale = Vector(ground_scale1)
            apply_transform(ground, loc=True, rot=True, scale=True)
            ground.name = "Ground"
            link(ground, col)

            np.random.seed(1)
            select_max = 0.3
            selection = density.placement_mask(
                normal_dir=(0, 0, 1),
                scale=0.1,
                return_scalar=True,
                select_thresh=np.random.uniform(select_max / 2, select_max),
            )
            grass.apply(ground, selection=selection)
            for key in ["scatters"]:
                sca = bpy.data.collections[key]
                sca.hide_viewport = True

    bpy.ops.wm.save_as_mainfile(filepath=args.output)

if __name__ == "__main__":
    main()


