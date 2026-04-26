import argparse
import os
import sys
import time

import numpy as np
import bpy
import cv2
import subprocess
import OpenEXR
import matplotlib.pyplot as plt


def list_of_ints(arg):
    return list(map(int, arg.split(',')))



def configure_compositor_output(
    node_tree,
    frames_folder,
    image_denoised,
    image_noisy,
    passes_to_save,
    saving_ground_truth,
):
    nodes = node_tree.nodes
    links = node_tree.links

    file_output_node_png = nodes.new(type="CompositorNodeOutputFile")
    file_output_node_png.base_path = str(frames_folder)
    file_output_node_png.format.file_format = 'PNG'
    file_output_node_png.format.color_mode = 'RGB'

    file_output_node_exr = nodes.new(type="CompositorNodeOutputFile")
    file_output_node_exr.base_path = str(frames_folder)
    file_output_node_exr.format.file_format = 'OPEN_EXR'
    file_output_node_exr.format.color_mode = 'RGB'

    default_file_output_node = file_output_node_exr if saving_ground_truth else file_output_node_png

    viewlayer = bpy.context.scene.view_layers["ViewLayer"]
    render_layers = nodes.new(type="CompositorNodeRLayers")

    file_slot_list = []

    for viewlayer_pass, socket_name in passes_to_save:
        if hasattr(viewlayer, f"use_pass_{viewlayer_pass}"):
            setattr(viewlayer, f"use_pass_{viewlayer_pass}", True)
        else:
            setattr(viewlayer.cycles, f"use_pass_{viewlayer_pass}", True)

        file_output_node = (
            default_file_output_node
            if viewlayer_pass != "material_index"
            else file_output_node_exr
        )

        slot_input = file_output_node.file_slots.new(socket_name)
        render_socket = render_layers.outputs[socket_name]

        if viewlayer_pass == "vector":
            separate_color = nodes.new(type="CompositorNodeSepRGBA")
            links.new(render_socket, separate_color.inputs[0])

            combine_color = nodes.new(type="CompositorNodeCombRGBA")
            combine_color.inputs[1].default_value = separate_color.outputs[3].default_value
            combine_color.inputs[2].default_value = separate_color.outputs[2].default_value

            links.new(combine_color.outputs[0], slot_input)

        elif viewlayer_pass == "normal":
            mix = nodes.new(type="CompositorNodeMixRGB")
            mix.blend_type = 'ADD'
            mix.inputs[2].default_value = (0.0, 0.0, 0.0, 0.0)
            links.new(render_socket, mix.inputs[1])
            links.new(mix.outputs[0], slot_input)

        else:
            links.new(render_socket, slot_input)

        file_slot_list.append(file_output_node.file_slots[slot_input.name])

    slot_input = default_file_output_node.file_slots["Image"]
    image = image_denoised if image_denoised is not None else image_noisy
    links.new(image, default_file_output_node.inputs["Image"])

    if saving_ground_truth:
        slot_input.path = "UniqueInstances"
    else:
        links.new(image, file_output_node_exr.inputs["Image"])
        file_slot_list.append(file_output_node_exr.file_slots[slot_input.path])

    file_slot_list.append(default_file_output_node.file_slots[slot_input.path])

    return file_slot_list



def compositor_postprocessing(node_tree, source, show=True, color_correct=True):
    nodes = node_tree.nodes
    links = node_tree.links

    def link(new_node, input_name="Image"):
        links.new(source, new_node.inputs[input_name])
        return new_node

    if color_correct:
        cc_node = nodes.new(type="CompositorNodeBrightContrast")
        cc_node.inputs["Bright"].default_value = 1.0
        cc_node.inputs["Contrast"].default_value = 4.0
        source = link(cc_node).outputs["Image"]

    if show:
        composite = nodes.new(type="CompositorNodeComposite")
        links.new(source, composite.inputs["Image"])

    return source if hasattr(source, "outputs") else source



def load_single_channel(p):
    file = OpenEXR.InputFile(str(p))
    channel, channel_type = next(iter(file.header()["channels"].items()))
    match str(channel_type.type):
        case "FLOAT":
            np_type = np.float32
        case _:
            np_type = np.uint8
    data = np.frombuffer(file.channel(channel, channel_type.type), np_type)
    dw = file.header()["dataWindow"]
    sz = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)
    return data.reshape(sz)


def load_depth(p):
    return load_single_channel(p)


def colorize_depth(depth, scale_vmin=1.0):
    valid = (depth > 1e-3) & (depth < 1e4)
    if not valid.any():
        vmin, vmax = 0, 1
    else:
        vmin = depth[valid].min() * scale_vmin
        vmax = depth[valid].max()
    cmap = plt.cm.jet
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    depth = cmap(norm(depth))
    depth[~valid] = 1
    return np.ascontiguousarray(depth[..., :3] * 255, dtype=np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input .blend file")
    parser.add_argument("--output", required=True)
    parser.add_argument('--camera_ids', type=list_of_ints, default="0,1,2,3,4,5,6,7")
    parser.add_argument("--samples", type=int, default=64, help="Number of samples for rendering")
    parser.add_argument("--denoise_image", type=int, default=False)
    parser.add_argument("--colorize_depth", type=int, default=False)
    parser.add_argument("--raytracing", type=int, default=False)
    parser.add_argument("--total_pixels", type=int, default=576*768)

    parser.add_argument("--use_exr", type=int, default=False, help="Output EXR format (1) or PNG format (0)")
    args = sys.argv[sys.argv.index("--") + 1:]
    args = parser.parse_args(args)

    tic = time.time()
    bpy.ops.wm.open_mainfile(filepath=args.input)
    toc = time.time()
    print(f"Time to open file: {toc - tic:.2f} seconds")
    tic = time.time()

    cmd = f"mkdir -p {args.output}"
    subprocess.call(cmd, shell=True)



    def round8(x):
        return int(np.round(x / 8) * 8)

    # Infer FOV from an existing camera: fov = 2 * atan(sensor_dim / (2 * lens)).
    first_cam_obj = next((obj for obj in bpy.data.objects if obj.type == 'CAMERA'), None)
    if first_cam_obj is None:
        raise RuntimeError("No camera found in the scene to infer fov_x and fov_y")

    first_cam = first_cam_obj.data
    if first_cam.lens <= 0:
        raise RuntimeError("Camera lens must be > 0 to infer fov_x and fov_y")
    if first_cam.sensor_height <= 0 or first_cam.sensor_width <= 0:
        raise RuntimeError("Camera sensor dimensions must be > 0 to infer FOV")

    inferred_fov_x = np.degrees(2.0 * np.arctan(first_cam.sensor_width / (2.0 * first_cam.lens)))
    inferred_fov_y = np.degrees(2.0 * np.arctan(first_cam.sensor_height / (2.0 * first_cam.lens)))

    fov_x_rad = np.radians(inferred_fov_x)
    fov_y_rad = np.radians(inferred_fov_y)
    H = int(np.sqrt(args.total_pixels * np.tan(fov_y_rad / 2) / np.tan(fov_x_rad / 2)))
    W = int(args.total_pixels // H)
    H = round8(H)
    W = round8(W)

    bpy.context.scene.render.engine = 'BLENDER_EEVEE_NEXT'
    bpy.context.scene.eevee.taa_render_samples = args.samples
    bpy.context.scene.render.resolution_x = W
    bpy.context.scene.render.resolution_y = H

    if args.raytracing:
        bpy.context.scene.eevee.use_raytracing = True
        bpy.context.scene.eevee.ray_tracing_options.use_denoise = False


    if args.denoise_image:
        folder = f"{args.output}/images_denoised"
        os.makedirs(folder, exist_ok=True)
        bpy.context.scene.cycles.use_denoising = True
        for camera_id in args.camera_ids:
            obj = bpy.data.objects.get(f"camera_{camera_id}_0")
            bpy.context.scene.camera = obj
            output_path = f"{folder}/{camera_id:08d}.png"
            if os.path.exists(output_path): continue
            bpy.context.scene.render.filepath = output_path
            bpy.ops.render.render(write_still=True)

    bpy.context.scene.use_nodes = True
    compositor_node_tree = bpy.context.scene.node_tree
    render_layers = compositor_node_tree.nodes.new(type="CompositorNodeRLayers")

    final_image_noisy = compositor_postprocessing(
        compositor_node_tree, source=render_layers.outputs["Image"]
    )

    tmp_folder = f"{args.output}/{args.camera_ids[0]}"
    configure_compositor_output(
        compositor_node_tree,
        tmp_folder,
        image_denoised=None,
        image_noisy=final_image_noisy,
        passes_to_save=[['z', 'Depth']],
        saving_ground_truth=True,
    )

    bpy.context.scene.cycles.use_denoising = False
    
    # Set render output format based on argument
    if args.use_exr:
        bpy.context.scene.render.image_settings.file_format = 'OPEN_EXR'
        bpy.context.scene.render.image_settings.color_mode = 'RGBA'
        bpy.context.scene.render.image_settings.color_depth = '32'
        file_extension = '.exr'
    else:
        bpy.context.scene.render.image_settings.file_format = 'PNG'
        bpy.context.scene.render.image_settings.color_mode = 'RGBA'
        bpy.context.scene.render.image_settings.color_depth = '8'
        file_extension = '.png'
    
    folder = f"{args.output}/images"
    os.makedirs(folder, exist_ok=True)
    depth_folder = f"{args.output}/depths"
    os.makedirs(depth_folder, exist_ok=True)
    cam_folder = f"{args.output}/cams"
    os.makedirs(cam_folder, exist_ok=True)

    toc = time.time()
    print(f"Time to set up compositor: {toc - tic:.2f} seconds")


    for i, camera_id in enumerate(args.camera_ids):
        tic = time.time()
        obj = bpy.data.objects.get(f"camera_{camera_id}_0")

        bpy.context.scene.camera = obj
        output_path = f"{folder}/{i:08d}{file_extension}"
        output_finish_tag = f"{folder}/{i:08d}.finish"
        if os.path.exists(output_finish_tag):
            continue
        bpy.context.scene.render.filepath = output_path
        bpy.ops.render.render(write_still=True)
        depth_dst_path = f"{tmp_folder}/Depth{1:04d}.exr"
        depth_array = load_depth(depth_dst_path)
        np.save(f"{depth_folder}/{i:08d}.npy", depth_array)
        cmd = f"rm -rf {tmp_folder}"
        subprocess.call(cmd, shell=True)
        if args.colorize_depth:
            cv2.imwrite(f"{depth_folder}/{i:08d}_colored.png", colorize_depth(depth_array))
        # cmd = f"cp /u/zeyum/s/cams/{camera_id:08d}_cam.txt {cam_folder}/{i:08d}_cam.txt"
        # subprocess.call(cmd, shell=True)
        blender_pose = obj.parent.matrix_world
        blender_pose[2][3] += 538
        cv_to_blender = np.array([
            [1,  0,  0, 0],
            [0, -1,  0, 0],
            [0,  0, -1, 0],
            [0,  0,  0, 1]
        ])
        cam_pose = np.array(blender_pose) @ np.linalg.inv(cv_to_blender)
        cam = obj.data
        scene = bpy.context.scene
        render = scene.render
        res_x = render.resolution_x * render.resolution_percentage / 100
        res_y = render.resolution_y * render.resolution_percentage / 100
        aspect_ratio = res_x / res_y
        sensor_width = cam.sensor_width
        sensor_height = cam.sensor_height
        if cam.sensor_fit == 'VERTICAL':
            sensor_width = sensor_height * aspect_ratio
        elif cam.sensor_fit == 'HORIZONTAL':
            sensor_height = sensor_width / aspect_ratio
        f_in_mm = cam.lens
        f_x = (res_x * f_in_mm) / sensor_width
        f_y = (res_y * f_in_mm) / sensor_height
        c_x = res_x / 2.0
        c_y = res_y / 2.0
        with open(f"{cam_folder}/{i:08d}_cam.txt", "w") as f:
            f.write("extrinsic\n")
            for row in np.linalg.inv(cam_pose):
                f.write(" ".join([f"{v:.7f}" for v in row]) + "\n")
            f.write("\nintrinsic\n")
            f.write(f"{f_x:.7f} 0.0 {c_x:.7f}\n")
            f.write(f"0.0 {f_y:.7f} {c_y:.7f}\n")
            f.write("0.0 0.0 1.0\n")
        cmd = f"touch {output_finish_tag}"
        subprocess.call(cmd, shell=True)
        toc = time.time()
        print(f"Time to render camera {i}:{camera_id}: {toc - tic:.2f} seconds")
    



if __name__ == "__main__":
    main()
