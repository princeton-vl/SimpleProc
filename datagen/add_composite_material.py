import argparse
import hashlib
import os
import sys

import numpy as np
import bpy

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
    parser.add_argument("--room_only", type=int, default=False)
    args = init.parse_args_blender(parser)

    bpy.ops.wm.open_mainfile(filepath=args.input)
    rng = feature_rng_from_file(seed=args.seed)

    materials = []
    if args.room_only:
        for obj in bpy.data.objects:
            if obj.name.startswith("plane"):
                for mat in obj.data.materials:
                    materials.append(mat)
    else:
        for mat in bpy.data.materials:
            if mat.node_tree:
                materials.append(mat)
    for mat in materials:
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        for node in mat.node_tree.nodes:
            if node.type == 'BSDF_PRINCIPLED':
                principled_node = node
                break
        original_noise_type = principled_node.inputs['Base Color'].links[0].from_node.type if principled_node.inputs['Base Color'].links else "NONE"
        if original_noise_type == "NONE": continue
        noise_type = rng.choice(["noise", "wave"])
        if noise_type == "noise":
            tex_node = nodes.new(type='ShaderNodeTexNoise')
            tex_node.inputs['Scale'].default_value = rng.uniform(1, 5)
        elif noise_type == "wave":
            tex_node = nodes.new(type='ShaderNodeTexWave')
            tex_node.inputs['Scale'].default_value = rng.uniform(1, 5)
            tex_node.inputs['Distortion'].default_value = 0.0 if args.room_only else rng.uniform(0, 10)
            tex_node.bands_direction = rng.choice(['X', 'Y'])
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
        math_node = nodes.new(type='ShaderNodeMath')
        math_node.operation = 'GREATER_THAN'
        math_node.inputs[1].default_value = 0.5
        links.new(tex_node.outputs['Fac'], math_node.inputs[0])
        if original_noise_type == "MIX_RGB":
            mix_node = principled_node.inputs['Base Color'].links[0].from_node
            original_fac_link = mix_node.inputs['Fac'].links[0]
            
            # Use math node with boolean operations instead of non-existent ShaderNodeBoolean
            mix_type = rng.choice(['AND', 'OR', 'XOR'])
            composite_node = nodes.new(type='ShaderNodeMath')
            
            # Set the math operation to the selected boolean operation
            if mix_type == 'AND':
                composite_node.operation = 'MULTIPLY'  # AND can be approximated with multiply for 0/1 values
                links.new(math_node.outputs['Value'], composite_node.inputs[0])
                links.new(original_fac_link.from_socket, composite_node.inputs[1])
                links.new(composite_node.outputs['Value'], mix_node.inputs['Fac'])
            elif mix_type == 'OR':
                composite_node.operation = 'MAXIMUM'   # OR can be approximated with maximum for 0/1 values
                links.new(math_node.outputs['Value'], composite_node.inputs[0])
                links.new(original_fac_link.from_socket, composite_node.inputs[1])
                links.new(composite_node.outputs['Value'], mix_node.inputs['Fac'])
            elif mix_type == 'XOR':
                # XOR = A + B - 2*A*B, requires multiple nodes
                add_node = nodes.new(type='ShaderNodeMath')
                add_node.operation = 'ADD'
                
                multiply_node = nodes.new(type='ShaderNodeMath')
                multiply_node.operation = 'MULTIPLY'
                
                multiply2_node = nodes.new(type='ShaderNodeMath')
                multiply2_node.operation = 'MULTIPLY'
                multiply2_node.inputs[1].default_value = 2.0
                
                composite_node.operation = 'SUBTRACT'
                
                # Connect: A + B
                links.new(math_node.outputs['Value'], add_node.inputs[0])
                links.new(original_fac_link.from_socket, add_node.inputs[1])
                
                # Connect: A * B
                links.new(math_node.outputs['Value'], multiply_node.inputs[0])
                links.new(original_fac_link.from_socket, multiply_node.inputs[1])
                
                # Connect: 2 * (A * B)
                links.new(multiply_node.outputs['Value'], multiply2_node.inputs[0])
                
                # Connect: (A + B) - 2*(A * B)
                links.new(add_node.outputs['Value'], composite_node.inputs[0])
                links.new(multiply2_node.outputs['Value'], composite_node.inputs[1])
                
                links.new(composite_node.outputs['Value'], mix_node.inputs['Fac'])
        
        elif original_noise_type == "TEX_BRICK":
            # Get the original brick texture node
            brick_node = principled_node.inputs['Base Color'].links[0].from_node
            
            # Create a copy of the original brick node with mortar size 0
            brick_copy = nodes.new(type='ShaderNodeTexBrick')
            brick_copy.inputs['Scale'].default_value = brick_node.inputs['Scale'].default_value * (10.0 if args.room_only else 1.0)
            brick_copy.inputs['Mortar Size'].default_value = 0.0  # No mortar
            brick_copy.inputs['Mortar Smooth'].default_value = brick_node.inputs['Mortar Smooth'].default_value
            brick_copy.inputs['Bias'].default_value = brick_node.inputs['Bias'].default_value
            brick_copy.inputs['Brick Width'].default_value = brick_node.inputs['Brick Width'].default_value
            brick_copy.inputs['Row Height'].default_value = brick_node.inputs['Row Height'].default_value
            brick_copy.inputs['Color1'].default_value = brick_node.inputs['Color1'].default_value
            brick_copy.inputs['Color2'].default_value = brick_node.inputs['Color2'].default_value
            
            # Copy texture coordinate connections if any
            if brick_node.inputs['Vector'].links:
                links.new(brick_node.inputs['Vector'].links[0].from_socket, brick_copy.inputs['Vector'])
            
            # Boolean operation between brick fac and new math node
            mix_type = rng.choice(['AND', 'OR', 'XOR'])
            
            if mix_type == 'AND':
                bool_node = nodes.new(type='ShaderNodeMath')
                bool_node.operation = 'MULTIPLY'
                links.new(brick_node.outputs['Fac'], bool_node.inputs[0])
                links.new(math_node.outputs['Value'], bool_node.inputs[1])
            elif mix_type == 'OR':
                bool_node = nodes.new(type='ShaderNodeMath')
                bool_node.operation = 'MAXIMUM'
                links.new(brick_node.outputs['Fac'], bool_node.inputs[0])
                links.new(math_node.outputs['Value'], bool_node.inputs[1])
            elif mix_type == 'XOR':
                # XOR = A + B - 2*A*B
                add_node = nodes.new(type='ShaderNodeMath')
                add_node.operation = 'ADD'
                
                multiply_node = nodes.new(type='ShaderNodeMath')
                multiply_node.operation = 'MULTIPLY'
                
                multiply2_node = nodes.new(type='ShaderNodeMath')
                multiply2_node.operation = 'MULTIPLY'
                multiply2_node.inputs[1].default_value = 2.0
                
                bool_node = nodes.new(type='ShaderNodeMath')
                bool_node.operation = 'SUBTRACT'
                
                # Connect: A + B
                links.new(brick_node.outputs['Fac'], add_node.inputs[0])
                links.new(math_node.outputs['Value'], add_node.inputs[1])
                
                # Connect: A * B
                links.new(brick_node.outputs['Fac'], multiply_node.inputs[0])
                links.new(math_node.outputs['Value'], multiply_node.inputs[1])
                
                # Connect: 2 * (A * B)
                links.new(multiply_node.outputs['Value'], multiply2_node.inputs[0])
                
                # Connect: (A + B) - 2*(A * B)
                links.new(add_node.outputs['Value'], bool_node.inputs[0])
                links.new(multiply2_node.outputs['Value'], bool_node.inputs[1])
            
            # Create mix node: if boolean result is 1, use mortar color, otherwise use brick copy color
            final_mix = nodes.new(type='ShaderNodeMixRGB')
            final_mix.blend_type = 'MIX'
            
            # Connect boolean result to mix factor
            links.new(bool_node.outputs['Value'], final_mix.inputs['Fac'])
            
            # Connect brick copy (no mortar) to Color1 input (when fac = 0)
            links.new(brick_copy.outputs['Color'], final_mix.inputs['Color1'])
            
            # Create a constant color node for the mortar color (extracted from original brick node)
            mortar_color_node = nodes.new(type='ShaderNodeRGB')
            mortar_color_node.outputs['Color'].default_value = brick_node.inputs['Mortar'].default_value
            
            # Connect constant mortar color to Color2 input (when fac = 1)
            links.new(mortar_color_node.outputs['Color'], final_mix.inputs['Color2'])
            
            # Connect final mix to principled BSDF
            links.new(final_mix.outputs['Color'], principled_node.inputs['Base Color'])

    bpy.ops.wm.save_as_mainfile(filepath=args.output)


if __name__ == "__main__":
    main()






