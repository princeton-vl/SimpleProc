from datetime import datetime

import bmesh
import bpy
import numpy as np
from mathutils import Vector

from infinigen.assets.utils.geometry.lofting import loft, Skin
from infinigen.core.util.math import lerp_sample
from nurbs_profile_sampler import check_self_intersection, evaluate_curve_safely


def compute_signed_area(verts):
    assert verts.shape[1] == 2, "Vertices must be 2D points"
    area = 0.0
    n = len(verts)
    for i in range(n):
        j = (i + 1) % n
        area += verts[i][0] * verts[j][1]
        area -= verts[j][0] * verts[i][1]
    return area / 2.0



class Timer:
    def __init__(self, desc="Timer", disable_timer=False):
        self.desc = desc
        self.disable_timer = disable_timer

    def __enter__(self):
        if self.disable_timer:
            return
        self.start = datetime.now()
        print(f"[{self.desc}] started")

    def __exit__(self, exc_type, exc_val, traceback):
        if self.disable_timer:
            return
        end = datetime.now()
        duration = end - self.start
        if exc_type is None:
            print(f"[{self.desc}] finished in {duration}")
        else:
            print(f"[{self.desc}] failed with {exc_type}")

class CompoSampler:
    def __init__(
        self,
        profile_sampler,
        stem_sampler,
        nstempts_sampler,
        normalize_profiles=True,
        cone_prob=0.5,
        rng=None,
    ):
        self.profile_sampler = profile_sampler
        self.stem_sampler = stem_sampler
        self.nstempts_sampler = nstempts_sampler
        self.normalize_profiles = normalize_profiles
        self.cone_prob = cone_prob
        if rng is None:
            raise ValueError("CompoSampler requires an explicit rng")
        self.rng = rng

    def sample(
        self,
        debug=False,
        target_max_dim=5,
        location=(0, 0, 0),
        face_size=0.02,
        return_metadata=False,
    ):

        with Timer("Generating Stem"):
            while True:
                stem = self.stem_sampler.sample()
                stem_pts, _, stem_rads = evaluate_curve_safely(stem, num_samples=300, with_circumrad=True)
                check = check_self_intersection(stem_pts, closed=False)
                if not check: break
            if stem_pts.shape[1] == 2:
                stem_pts = np.concatenate([stem_pts, np.zeros((len(stem_pts), 1))], axis=1)

        with Timer("Generating Profiles"):
            N = self.nstempts_sampler.sample()
            profiles = []

            if hasattr(self.profile_sampler, "knotvector_sampler"):
                closed = self.profile_sampler.knotvector_sampler.cyclic
            else:
                closed = True
            for i in range(N):
                while True:
                    profile = self.profile_sampler.sample()
                    verts = evaluate_curve_safely(profile, num_samples=300)
                    verts = np.concatenate([verts, np.zeros((len(verts), 1))], axis=1)
                    check = check_self_intersection(verts, closed=closed)
                    if not check: break
                verts = verts[:, :2]
                verts_center = np.mean(verts, axis=0)
                verts = verts - verts_center
                if self.normalize_profiles:
                    verts_scale = np.linalg.norm(verts, axis=1, keepdims=True).max(axis=0)
                    verts /= verts_scale
                else:
                    verts_scale = 1.0
                if closed:
                    verts = verts[:-1]
                    area = compute_signed_area(verts[:, :2])
                    if area < 0: verts = verts[::-1]
                    angles = np.abs(np.mod(np.arctan2(verts[:, 1], verts[:, 0]), np.pi*2) - np.pi / 2)
                    min_index = np.argmin(angles)
                    verts = np.roll(verts, -min_index, axis=0)
                else:
                    if i > 0:
                        distance_to_prev = np.linalg.norm(profiles[-1][0] - verts, axis=1).sum()
                        distance_to_prev_reverse = np.linalg.norm(profiles[-1][0][::-1] - verts, axis=1).sum()
                        if distance_to_prev_reverse < distance_to_prev:
                            verts = verts[::-1]
                profiles.append((verts, profile, {'center': verts_center, 'scale': verts_scale}))
            profiles_2d = [(p[0].copy(), p[1], p[2]) for p in profiles]
            first_profile_curve = profiles_2d[0][1] if len(profiles_2d) > 0 else None
            first_profile_transform = profiles_2d[0][2] if len(profiles_2d) > 0 else None

        with Timer("Setting up Skin"):
            is_cone = self.rng.uniform() < self.cone_prob
            if is_cone:
                if self.rng.randint(2) == 0:
                    profiles[0] = (np.zeros_like(profiles[0][0]), profiles[0][1])
                else:
                    profiles[-1] = (np.zeros_like(profiles[-1][0]), profiles[-1][1])
            verts_only = np.array([p[0] for p in profiles])
            profiles = verts_only
            ts = np.linspace(0, 1, N)
            stem_rads = self.rng.uniform(0, 1, size=np.asarray(stem_rads).shape)
            stem_rads = np.asarray(stem_rads).reshape(-1)
            profile_rads = lerp_sample(stem_rads.reshape((-1, 1)), ts * (len(stem_rads) - 1)).reshape(-1, 1, 1)
            profiles = profiles * profile_rads
            profiles = np.concatenate([np.zeros((profiles.shape[0], profiles.shape[1], 1)), profiles], axis=-1)
            skeleton = stem_pts
            skin = Skin(ts=ts, profiles=profiles, profile_as_points=True)

        with Timer("Call Lofting"):
            obj = loft(
                skeleton,
                skin,
                method="geomdl",
                debug=debug,
                cyclic_v=closed,
                face_size=min(0.2, face_size * 5 / target_max_dim),
            )

        with Timer("Scaling and Centering"):
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.mode_set(mode='EDIT')
            bm = bmesh.from_edit_mesh(obj.data)
            bm.verts.ensure_lookup_table()
            center = sum((v.co for v in bm.verts), start=bm.verts[0].co.copy()) / len(bm.verts)
            for v in bm.verts:
                v.co -= center
            bmesh.update_edit_mesh(obj.data)
            bpy.ops.object.mode_set(mode='OBJECT')
            bbox = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
            min_corner = Vector((min(c[i] for c in bbox) for i in range(3)))
            max_corner = Vector((max(c[i] for c in bbox) for i in range(3)))
            size = max_corner - min_corner
            max_dim = max(size)
            scale_factor = target_max_dim / max_dim
            obj.scale *= scale_factor
            obj.location = Vector(location)
            bpy.ops.object.select_all(action='DESELECT')
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.transform_apply(location=True, scale=True)

        if return_metadata:
            return obj, {
                "first_profile": profiles_2d[0][0] if len(profiles_2d) > 0 else None,
                "first_profile_curve": first_profile_curve,
                "first_profile_transform": first_profile_transform,
                "stem_curve": stem,
                "stem_points": stem_pts.copy(),
            }
        return obj

