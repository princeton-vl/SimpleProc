import argparse
import hashlib
import os
import sys

import bpy
import numpy as np

from infinigen.core import init

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
    parser.add_argument("--use_bump", type=int, default=True)
    parser.add_argument("--min_detail", type=float, default=10)
    parser.add_argument("--max_detail", type=float, default=10)
    parser.add_argument("--scale", type=float, default=0.02)
    parser.add_argument("--freq", type=float, default=30)
    args = init.parse_args_blender(parser)

    bpy.ops.wm.open_mainfile(filepath=args.input)
    rng = feature_rng_from_file(seed=args.seed)

    for obj in bpy.data.objects:
        if not obj.name.startswith("Object_"):
            continue

        mat = obj.active_material
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links

        scale_node2 = nodes.new("ShaderNodeVectorMath")
        scale_node2.operation = 'SCALE'
        output_node = next(
            (n for n in nodes if n.type == 'OUTPUT_MATERIAL'),
            None
        )
        bpy.context.scene.render.engine = 'CYCLES'
        if args.use_bump:
            mat.displacement_method = 'BOTH'
        else:
            mat.displacement_method = 'DISPLACEMENT'

        def exponential_uniform(start, end, rng):
            return np.exp(np.log(end / start) * rng.random() + np.log(start))

        tex_node = nodes.new("ShaderNodeTexNoise")
        tex_node.inputs["Detail"].default_value = rng.uniform(args.min_detail, args.max_detail)
        tex_node.inputs["Scale"].default_value = args.freq
        if rng.random() < 0.2:
            scale_node2.inputs[3].default_value = 0
        else:
            scale_node2.inputs[3].default_value = exponential_uniform(0.1, 1, rng) * args.scale

        scale_node = nodes.new("ShaderNodeVectorMath")
        scale_node.operation = 'SCALE'
        geo_node = nodes.new("ShaderNodeNewGeometry")
        links.new(geo_node.outputs["Normal"], scale_node.inputs[0])
        links.new(geo_node.outputs["Position"], tex_node.inputs["Vector"])

        subtract_node = nodes.new("ShaderNodeMath")
        subtract_node.operation = 'SUBTRACT'
        subtract_node.inputs[1].default_value = 0.5
        links.new(tex_node.outputs["Fac"], subtract_node.inputs[0])
        links.new(subtract_node.outputs["Value"], scale_node.inputs[3])

        links.new(scale_node.outputs["Vector"], scale_node2.inputs[0])

        disp_input = output_node.inputs["Displacement"]
        add_node = nodes.new("ShaderNodeVectorMath")
        add_node.operation = 'ADD'
        links.new(scale_node2.outputs["Vector"], add_node.inputs[1])
        links.new(add_node.outputs["Vector"], disp_input)


    bpy.ops.wm.save_as_mainfile(filepath=args.output)

if __name__ == "__main__":
    main()

