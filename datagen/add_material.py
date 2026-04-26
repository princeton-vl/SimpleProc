import argparse
from colorsys import hsv_to_rgb
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
    parser.add_argument("--max_s", type=float, default=1)
    parser.add_argument("--max_v", type=float, default=1)
    parser.add_argument("--room_only", type=int, default=False)
    args = init.parse_args_blender(parser)

    bpy.ops.wm.open_mainfile(filepath=args.input)

    suffix = f"{args.room_only}"
    rng = feature_rng_from_file(seed=args.seed, suffix=suffix)

    def random_natural_color():
        h = rng.random()
        s = rng.uniform(0.0, args.max_s)
        v = rng.uniform(0.0, args.max_v)
        return hsv_to_rgb(h, s, v)

    for obj in bpy.data.objects:
        collections = [col.name for col in obj.users_collection]
        if "GrassSources" in collections:
            continue

        if args.room_only:
            if not obj.name.startswith("plane"):
                continue
        else:
            if (
                not obj.name.startswith("Object_")
                and not obj.name.startswith("loft_nurbs")
                and not obj.name.startswith("Ground")
                and not obj.name.startswith("plane")
            ):
                continue

        obj.data.materials.clear()

        mat = bpy.data.materials.new(name=f"RandomNaturalMaterial_{obj.name}")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links

        for node in nodes:
            nodes.remove(node)

        output_node = nodes.new(type='ShaderNodeOutputMaterial')
        bsdf_node = nodes.new(type='ShaderNodeBsdfPrincipled')
        links.new(bsdf_node.outputs['BSDF'], output_node.inputs['Surface'])
        bsdf_node.inputs['Metallic'].default_value = rng.choice([0, rng.uniform(0, 0.8)], p=[0.7, 0.3])
        roughness = rng.choice([0.2, rng.uniform(0.2, 1)], p=[0.3, 0.7])
        bsdf_node.inputs['Roughness'].default_value = roughness
        bsdf_node.inputs['Coat Weight'].default_value = max(0, rng.uniform(-1, 1))
        bsdf_node.inputs['Coat Roughness'].default_value = roughness
        bsdf_node.inputs['Subsurface Weight'].default_value = max(0, rng.uniform(-1, 1))

        noise_type = rng.choice(["noise", "wave", "pure", "brick"])
        if noise_type in ["noise", "wave"]:
            if noise_type == "noise":
                tex_node = nodes.new(type='ShaderNodeTexNoise')
                tex_node.inputs['Scale'].default_value = rng.uniform(1, 5)
            elif noise_type == "wave":
                tex_node = nodes.new(type='ShaderNodeTexWave')
                tex_node.inputs['Scale'].default_value = rng.uniform(1, 5)
                tex_node.inputs['Distortion'].default_value = 0.0 if args.room_only else rng.uniform(0, 10)
                tex_node.bands_direction = rng.choice(['X', 'Y'])
            mix_node = nodes.new(type='ShaderNodeMixRGB')
            mix_node.blend_type = 'MIX'
            math_node = nodes.new(type='ShaderNodeMath')
            math_node.operation = 'GREATER_THAN'
            math_node.inputs[1].default_value = 0.5
            color1 = random_natural_color()
            color2 = random_natural_color()
            mix_node.inputs['Color1'].default_value = (*color1, 1)
            mix_node.inputs['Color2'].default_value = (*color2, 1)
            links.new(math_node.outputs['Value'], mix_node.inputs['Fac'])
            links.new(tex_node.outputs['Fac'], math_node.inputs[0])
            links.new(mix_node.outputs['Color'], bsdf_node.inputs['Base Color'])
        elif noise_type == "brick":
            tex_node = nodes.new(type='ShaderNodeTexBrick')
            tex_node.inputs['Scale'].default_value = rng.uniform(1, 5) * (10.0 if args.room_only else 1.0)
            tex_node.inputs['Color1'].default_value = (*random_natural_color(), 1)
            tex_node.inputs['Color2'].default_value = (*random_natural_color(), 1)
            tex_node.inputs['Mortar'].default_value = (*random_natural_color(), 1)
            links.new(tex_node.outputs['Color'], bsdf_node.inputs['Base Color'])
        elif noise_type == "pure":
            color = random_natural_color()
            bsdf_node.inputs['Base Color'].default_value = (*color, 1)

        if noise_type != "pure":
            if rng.choice([True, False]):
                tex_coords = nodes.new(type='ShaderNodeTexCoord')
                uv_offset = nodes.new(type='ShaderNodeVectorMath')
                uv_offset.operation = 'ADD'
                uv_offset.inputs[1].default_value = (
                    rng.uniform(0.0, 1.0),
                    rng.uniform(0.0, 1.0),
                    0.0,
                )
                links.new(tex_coords.outputs['UV'], uv_offset.inputs[0])
                links.new(uv_offset.outputs['Vector'], tex_node.inputs['Vector'])

        obj.data.materials.append(mat)

    bpy.ops.wm.save_as_mainfile(filepath=args.output)

if __name__ == "__main__":
    main()

