
import argparse
import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import bpy
from tqdm import tqdm

from infinigen.core import init
from nurbs_profile_sampler import (
    GaussianSampler,
    KnotVectorSampler,
    MixedSampler,
    RandomWalkSampler,
    StarfishSampler,
    UniformIntSampler,
)
from nurbs_shape_sampler import CompoSampler

sys.path.insert(0, str(Path(__file__).resolve().parent))


def link(obj, col):
    for c in obj.users_collection:
        c.objects.unlink(obj)
    col.objects.link(obj)
    return obj


def feature_rng_from_file(seed: int, suffix: str = ""):
    script_name = os.path.basename(sys.argv[0])
    tag = f"{seed}_{script_name}{suffix}"
    h = int(hashlib.sha256(tag.encode("utf-8")).hexdigest(), 16) % (2**32)
    return np.random.RandomState(h)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input .blend file")
    parser.add_argument("--output")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    args = init.parse_args_blender(parser)

    bpy.ops.wm.open_mainfile(filepath=args.input)

    rng = feature_rng_from_file(seed=args.seed)

    n_predefined_objs = rng.choice([1, 2, 4, 8, 16, 32, 64], p=np.array([0.05, 0.05, 0.1, 0.2, 0.2, 0.2, 0.2]))
    predefined_objs = []
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
    composampler = CompoSampler(profile_sampler, stem_sampler, nstempts_sampler, rng=rng)

    for i in tqdm(range(n_predefined_objs)):
        obj = composampler.sample(target_max_dim=5, location=(0, 0, 0), face_size=2)
        obj.hide_render = True
        predefined_objs.append(obj)
    source_coll = bpy.data.collections.new("GrassSources")
    bpy.context.scene.collection.children.link(source_coll)
    for obj in predefined_objs:
        source_coll.objects.link(obj)

    scatter_mod = None
    for obj in bpy.context.scene.objects:
        for mod in obj.modifiers:
            if mod.name == "geo_instance_scatter":
                scatter_mod = mod
                break

        if scatter_mod is not None:
            break

    if scatter_mod is not None:
        node_group = scatter_mod.node_group
        node_group.nodes["Collection Info"].inputs[0].default_value = source_coll
        node_group.nodes["Vector Math.002"].inputs[1].default_value[0] /= 100
        node_group.nodes["Vector Math.002"].inputs[1].default_value[1] /= 100
        node_group.nodes["Vector Math.002"].inputs[1].default_value[2] /= 20
        node1 = node_group.nodes["Instance on Points"]
        translate_node = node_group.nodes.new("GeometryNodeSetPosition")
        translate_node.inputs["Offset"].default_value = (0, 0, 0.1)
        node_group.links.new(node1.outputs[0], translate_node.inputs[0])
        node1 = translate_node
        node_set_mat = node_group.nodes.new("GeometryNodeSetMaterial")
        node_group.links.new(node1.outputs[0], node_set_mat.inputs[0])
        node_set_mat.inputs[2].default_value = bpy.data.materials["RandomNaturalMaterial_Object_0"]
        node_group.links.new(node_set_mat.outputs[0], node_group.nodes["Group Output"].inputs[0])
        scatter_mod["Socket_2"] *= 10

    bpy.ops.wm.save_as_mainfile(filepath=args.output)


if __name__ == "__main__":
    sys.exit(main())
