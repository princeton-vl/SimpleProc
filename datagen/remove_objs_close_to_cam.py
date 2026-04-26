import argparse

import bpy
import numpy as np

from infinigen.core import init


def infer_camera_ids():
    camera_ids = []
    for obj in bpy.data.objects:
        if obj.name.startswith("camera_") and obj.name.endswith("_0"):
            camera_id = obj.name[len("camera_"):-len("_0")]
            if camera_id:
                camera_ids.append(camera_id)
    return sorted(set(camera_ids), key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else x))



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input .blend file")
    parser.add_argument("--output", required=True, help="Output .blend file")
    parser.add_argument('--threshold', type=float, default=3)
    args = init.parse_args_blender(parser)

    bpy.ops.wm.open_mainfile(filepath=args.input)
    camera_ids = infer_camera_ids()

    cameras = []
    for cam_id in camera_ids:
        cam = bpy.data.objects.get(f"camera_{cam_id}_0")
        if cam is not None:
            cameras.append(cam)

    if not cameras:
        print("No camera_<id>_0 objects found; skipping object removal")
        bpy.ops.wm.save_as_mainfile(filepath=args.output)
        return 0

    for obj in bpy.data.objects:
        if obj.type != 'MESH':
            continue
        if obj.name.startswith("Object_") or obj.name.startswith("Prim_"):
            verts = [obj.matrix_world @ v.co for v in obj.data.vertices]
            verts = np.array([v[:] for v in verts])
            close_flag = False
            for vert in verts:
                for cam in cameras:
                    cam_location = cam.matrix_world.translation
                    if np.linalg.norm(vert - cam_location) < args.threshold:
                        close_flag = True
                        break
                if close_flag:
                    break
            if close_flag:
                print("remove ", obj.name)
                bpy.data.objects.remove(obj)

    bpy.ops.wm.save_as_mainfile(filepath=args.output)


if __name__ == "__main__":
    main()
