import argparse

import numpy as np
import bpy
from infinigen.core import init
from tqdm import tqdm

from infinigen.core.util.blender import SelectObjects, ViewportMode
from nurbs_profile_sampler import compute_maximum_edge_length


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input .blend file")
    parser.add_argument("--output", required=True, help="Output .blend file")
    args = init.parse_args_blender(parser)

    bpy.ops.wm.open_mainfile(filepath=args.input)

    for obj in tqdm(bpy.data.objects):
        if not obj.name.startswith("Object_"):
            continue

        mesh = obj.data
        verts = np.array([v.co[:] for v in mesh.vertices])
        edges = np.array([[e.vertices[0], e.vertices[1]] for e in mesh.edges])
        max_length = compute_maximum_edge_length(verts, edges)
        number_cuts = int(np.ceil(max_length / 0.06 - 1))
        print(max_length, number_cuts)

        number_cuts = min(number_cuts, 7)
        with SelectObjects(obj), ViewportMode(obj, "EDIT"):
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.subdivide(number_cuts=number_cuts, smoothness=1)

    bpy.ops.wm.save_as_mainfile(filepath=args.output)

if __name__ == "__main__":
    main()

