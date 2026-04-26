import argparse
import hashlib
import os
import sys

import bpy
import numpy as np
from infinigen.core import init
from infinigen.core.util.blender import SelectObjects

def feature_rng_from_file(seed: int, suffix: str = ""):
    script_name = os.path.basename(sys.argv[0])
    tag = f"{seed}_{script_name}{suffix}"
    h = int(hashlib.sha256(tag.encode("utf-8")).hexdigest(), 16) % (2**32)
    return np.random.RandomState(h)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input .blend file")
    parser.add_argument("--output", required=True, help="Output .blend file")
    parser.add_argument("--min_detail", type=float, default=10)
    parser.add_argument("--max_detail", type=float, default=10)
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    args = init.parse_args_blender(parser)

    bpy.ops.wm.open_mainfile(filepath=args.input)
    rng = feature_rng_from_file(seed=args.seed)


    for obj in bpy.data.objects:
        if not obj.name.startswith("Object_") and not obj.name.startswith("loft_nurbs"):
            continue


        for modifier in list(obj.modifiers):
            if modifier.name != "ScatterGeoNodes":
                obj.modifiers.remove(modifier)
        
        with SelectObjects(obj):
            bpy.ops.object.modifier_add(type='NODES')
        mod = obj.modifiers[-1]
        mods = obj.modifiers
        mods.move(mods.find(mod.name), 0)

        gn_tree = bpy.data.node_groups.new("Geometry Nodes", "GeometryNodeTree")
        gn_tree.interface.new_socket(
            name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry"
        )
        gn_tree.interface.new_socket(
            name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry"
        )
        gn_tree.nodes.new("NodeGroupInput")
        output_node = gn_tree.nodes.new("NodeGroupOutput")
        output_node.is_active_output = True
        mod.node_group = gn_tree
        nodes = gn_tree.nodes
        links = gn_tree.links
        group_input = nodes.get("Group Input")
        group_output = nodes.get("Group Output")
        set_pos = nodes.new("GeometryNodeSetPosition")
        normal_node = nodes.new("GeometryNodeInputNormal")
        pos_node = nodes.new("GeometryNodeInputPosition")
        scale_node2 = nodes.new("ShaderNodeVectorMath")
        scale_node2.operation = 'SCALE'

        def exponential_correlated_uniform(start1, end1, start2, end2, rng, range=2):
            u = rng.random()
            v1 = np.exp(np.log(end1 / start1) * u + np.log(start1))
            v2 = np.exp(np.log(end2 / start2) * u + np.log(start2))
            v2 = max(start2, min(v2 * rng.uniform(1/range, range), end2))
            return v1, v2
            

        noise_type = rng.choice(["noise", "wave", "brick", "pure"], p=[0.2, 0.2, 0.2, 0.4])
        if noise_type == "noise":
            tex_node = nodes.new("ShaderNodeTexNoise")
            tex_node.inputs["Detail"].default_value = rng.uniform(args.min_detail, args.max_detail)
            base_scale, inv_scale = exponential_correlated_uniform(0.03, 1.2, 1, 5, rng)
            tex_node.inputs["Scale"].default_value = 5 / inv_scale
            scale_node2.inputs[3].default_value = base_scale
        elif noise_type == "wave":
            tex_node = nodes.new("ShaderNodeTexWave")
            tex_node.inputs["Distortion"].default_value = rng.uniform(0, 10)
            base_scale, inv_scale = exponential_correlated_uniform(0.0025, 0.1, 1, 5, rng)
            tex_node.inputs["Scale"].default_value = 5 / inv_scale
            scale_node2.inputs[3].default_value = base_scale
        elif noise_type == "brick":
            tex_node = nodes.new("ShaderNodeTexBrick")
            tex_node.inputs["Mortar Size"].default_value = rng.uniform(0.01, 0.1)
            base_scale, inv_scale = exponential_correlated_uniform(0.0025, 0.1, 1, 5, rng)
            tex_node.inputs["Scale"].default_value = 5 / inv_scale
            scale_node2.inputs[3].default_value = base_scale
        elif noise_type == "pure":
            scale_node2.inputs[3].default_value = 0

        scale_node = nodes.new("ShaderNodeVectorMath")
        scale_node.operation = 'SCALE'
        links.new(group_input.outputs["Geometry"], set_pos.inputs["Geometry"])
        links.new(set_pos.outputs["Geometry"], group_output.inputs["Geometry"])
        if noise_type != "pure": links.new(pos_node.outputs["Position"], tex_node.inputs["Vector"])
        links.new(normal_node.outputs["Normal"], scale_node.inputs[0])
        if noise_type != "pure": links.new(tex_node.outputs["Fac"], scale_node.inputs[3])
        links.new(scale_node2.outputs["Vector"], set_pos.inputs["Offset"])
        links.new(scale_node.outputs["Vector"], scale_node2.inputs[0])

        for node in nodes:
            if node.type == 'TEX_NOISE':
                fac_output = node.outputs.get("Fac")
                if fac_output and fac_output.links:
                    has_offset = False
                    for link in fac_output.links:
                        to_node = link.to_node
                        if (to_node.type == 'MATH' and 
                            to_node.operation == 'SUBTRACT' and 
                            len(to_node.inputs) > 1 and
                            abs(to_node.inputs[1].default_value - 0.5) < 0.001):
                            has_offset = True
                            break
                    
                    if not has_offset:
                        links_to_redirect = list(fac_output.links)

                        offset_node = nodes.new("ShaderNodeMath")
                        offset_node.operation = 'SUBTRACT'
                        offset_node.inputs[1].default_value = 0.5

                        links.new(fac_output, offset_node.inputs[0])

                        for link in links_to_redirect:
                            links.new(offset_node.outputs["Value"], link.to_socket)


    bpy.ops.wm.save_as_mainfile(filepath=args.output)

if __name__ == "__main__":
    main()

