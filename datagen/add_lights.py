
import argparse
import hashlib
import os
import sys

import bpy
import numpy as np
from infinigen.core import init

def exponential_uniform(start, end, rng):
    return np.exp(np.log(end / start) * rng.random() + np.log(start))


def feature_rng_from_file(seed: int, suffix: str = ""):
    script_name = os.path.basename(sys.argv[0])
    tag = f"{seed}_{script_name}{suffix}"
    h = int(hashlib.sha256(tag.encode("utf-8")).hexdigest(), 16) % (2**32)
    return np.random.RandomState(h)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input .blend file")
    parser.add_argument("--output", required=True, help="Output .blend file")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--num", type=int, default=80)
    args = init.parse_args_blender(parser)


    bpy.ops.wm.open_mainfile(filepath=args.input)

    rng = feature_rng_from_file(seed=args.seed)


    def random_natural_color(min_v=0.0):
        h = rng.random()
        s = rng.uniform(0.0, 0.8)
        v = rng.uniform(min_v, 1.0)
        from colorsys import hsv_to_rgb
        return hsv_to_rgb(h, s, v)

    bpy.ops.object.select_all(action='DESELECT')

    for obj in bpy.data.objects:
        if obj.type == 'LIGHT' or obj.type == 'LIGHT_PROBE' or obj.name.startswith("Sphere"):
            obj.select_set(True)
            bpy.data.objects.remove(obj, do_unlink=True)

    light_num = args.num
    height = 10
    for _ in range(light_num):
        bpy.ops.object.light_add(type='AREA', location=(rng.uniform(-30, 30), rng.uniform(-30, 30), height))
        area_light = bpy.context.object
        area_light.data.energy = exponential_uniform(500, 3000, rng) * 3 * 10 / light_num
        area_light.data.size = rng.uniform(1.0, 3.0)
        area_light.data.color = random_natural_color(min_v=1.0)
        area_light.rotation_euler = (0, 0, 0)


    bpy.ops.wm.save_as_mainfile(filepath=args.output)


if __name__ == "__main__":
    main()




