import argparse
import hashlib
import os
import re
import sys
import bpy
import bpy_extras
import numpy as np
from pathlib import Path
from mathutils import Vector

from infinigen.core import init

sys.path.append(str(Path(__file__).resolve().parent))
from nurbs_shape_sampler import CompoSampler
from nurbs_profile_sampler import (
    UniformSampler,
    UniformIntSampler,
    GaussianSampler,
    RandomWalkSampler,
    StarfishSampler,
    KnotVectorSampler,
    MixedSampler,
    ReptileSampler,
)

from tqdm import tqdm


def feature_rng_from_file(seed: int, suffix: str = ""):
    script_name = os.path.basename(sys.argv[0])
    tag = f"{seed}_{script_name}{suffix}"
    h = int(hashlib.sha256(tag.encode("utf-8")).hexdigest(), 16) % (2**32)
    return np.random.RandomState(h)


def random_location_from_first_n_bbox(n_base: int, rng):
    points = []
    for j in range(n_base):
        obj = bpy.data.objects.get(f"Object_{j}")
        if obj is None:
            continue
        points.extend([obj.matrix_world @ Vector(corner) for corner in obj.bound_box])

    if not points:
        return None

    min_corner = Vector((
        min(p.x for p in points),
        min(p.y for p in points),
        min(p.z for p in points),
    ))
    max_corner = Vector((
        max(p.x for p in points),
        max(p.y for p in points),
        max(p.z for p in points),
    ))
    return Vector((
        rng.uniform(min_corner.x, max_corner.x),
        rng.uniform(min_corner.y, max_corner.y),
        rng.uniform(min_corner.z, max_corner.z),
    ))



def is_point_in_camera(camera_obj, point_world, scene=None):
    """
    Check if a world-space point is inside the camera's view frustum.
    
    Args:
        camera_obj: bpy.types.Object, must be a camera
        point_world: mathutils.Vector, world-space location
        scene: bpy.types.Scene, defaults to current scene

    Returns:
        bool
    """
    if scene is None:
        scene = bpy.context.scene

    co_ndc = bpy_extras.object_utils.world_to_camera_view(scene, camera_obj, point_world)


    return (0.0 <= co_ndc.x <= 1.0 and
            0.0 <= co_ndc.y <= 1.0 and
            0.0 <= co_ndc.z)




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Input .blend file", required=True)
    parser.add_argument("--output", required=True, help="Output .blend file")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument('--n_objects', type=int, default=8)
    parser.add_argument("--child_density", type=int, default=40)
    parser.add_argument("--min_child_scale", type=float, default=0.2)
    parser.add_argument("--max_child_scale", type=float, default=0.8)
    args = init.parse_args_blender(parser)

    rng = feature_rng_from_file(seed=args.seed)

    bpy.ops.wm.open_mainfile(filepath=args.input)

    
    pert_sampler = GaussianSampler(0, 1, rng=rng)
    cyclic_knotvector_sampler = KnotVectorSampler(cyclic=True, rng=rng)
    clamped_knotvector_sampler = KnotVectorSampler(clamped=True, rng=rng)
    degree_sampler = UniformIntSampler(1, 3, rng=rng)
    nctrlpts_sampler = UniformIntSampler(3, 10, rng=rng)

    stem_nctrlpts_sampler = MixedSampler(
        [UniformIntSampler(2, 3, rng=rng), UniformIntSampler(4, 10, rng=rng)],
        [0.5, 0.5],
        rng=rng,
    )
    stem_sampler = RandomWalkSampler(degree_sampler, stem_nctrlpts_sampler, pert_sampler, clamped_knotvector_sampler, dim=3)
    radpert_sampler = GaussianSampler(0, 0.5, rng=rng)
    tanpert_sampler = GaussianSampler(0, 0.2, rng=rng)
    profile_sampler = StarfishSampler(degree_sampler, nctrlpts_sampler, radpert_sampler, tanpert_sampler, cyclic_knotvector_sampler)
    nstempts_sampler = UniformIntSampler(4, 10, rng=rng)
    composampler1 = CompoSampler(profile_sampler, stem_sampler, nstempts_sampler, rng=rng)


    base_sampler = RandomWalkSampler(degree_sampler, nctrlpts_sampler, pert_sampler, clamped_knotvector_sampler, dim=2)
    radius_sampler = UniformSampler(0.2, 0.4, rng=rng)
    rept_pert_sampler = GaussianSampler(0, 0.1, rng=rng)
    profile_sampler = ReptileSampler(base_sampler, radius_sampler, rept_pert_sampler)
    composampler2 = CompoSampler(profile_sampler, stem_sampler, nstempts_sampler, rng=rng)
    

    max_dim_dict = {}


    n_objects = [args.n_objects, args.child_density * args.n_objects]
    attach_object_id = np.zeros(sum(n_objects), dtype=int) - 1
    cnt = n_objects[0]
    for i in range(1, len(n_objects)):
        attach_object_id[cnt:cnt + n_objects[i]] = rng.randint(cnt - n_objects[i - 1], cnt, size=n_objects[i])
        cnt += n_objects[i]

    pattern = re.compile(r"^camera_(\d+)_0$")
    camera_ids = sorted(
        [m.group(1) for obj in bpy.data.objects if (m := pattern.match(obj.name))],
        key=int,
    )

    for i in tqdm(range(len(attach_object_id))):
        if attach_object_id[i] == -1:
            target_max_dim = 10
            if i == 0:
                location = Vector((0, 0, 0))
            else:
                c = 0
                while True:
                    c += 1
                    location = Vector((rng.uniform(-30, 30), rng.uniform(-30, 30), 0))
                    if not camera_ids:
                        break
                    visibilities = []
                    for camera_id in camera_ids:
                        camera_obj = bpy.data.objects[f"camera_{camera_id}_0"]
                        visibilities.append(is_point_in_camera(camera_obj, location))
                    if np.mean(visibilities) > 0.5:
                        break
                    if c > 1e3:
                        location = Vector((0, 0, 0))
                        break
        else:
            attach_object = bpy.data.objects[f"Object_{attach_object_id[i]}"]
            mesh = attach_object.data
            if rng.rand() < 0.5:
                random_index = rng.randint(0, len(mesh.vertices) - 1)
                v = mesh.vertices[random_index]
                location = attach_object.matrix_world @ v.co
            else:
                location = random_location_from_first_n_bbox(args.n_objects, rng)
                if location is None:
                    raise RuntimeError(
                        f"Failed to sample bbox location from first {args.n_objects} objects"
                    )
            target_max_dim = max_dim_dict[attach_object_id[i]] * rng.uniform(args.min_child_scale, args.max_child_scale)
        max_dim_dict[i] = target_max_dim
        if rng.randint(2) == 0:
            obj = composampler1.sample(target_max_dim=target_max_dim, location=location, face_size=0.1)
        else:
            obj = composampler2.sample(target_max_dim=target_max_dim, location=location, face_size=0.1)
        obj.name = f"Object_{i}"

    bpy.ops.wm.save_as_mainfile(filepath=args.output)

    


if __name__ == "__main__":
    main()
