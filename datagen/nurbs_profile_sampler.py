import os

from ctypes import CDLL, POINTER, RTLD_LOCAL, c_float, c_int32
from pathlib import Path

import numpy as np
if not hasattr(np, 'float'):
    np.float = float
from geomdl import NURBS, fitting
from mathutils import Vector
from numpy import ascontiguousarray as AC

def compute_circumradius(A, B, C):
    if not isinstance(A, Vector):
        A = Vector(A)
    if not isinstance(B, Vector):
        B = Vector(B)
    if not isinstance(C, Vector):
        C = Vector(C)

    AB = B - A
    BC = C - B
    CA = A - C

    a = AB.length
    b = BC.length
    c = CA.length

    if len(A) == 2:
        area = abs(((B - A).cross(C - A))) / 2
    else:
        area = ((B - A).cross(C - A)).length / 2

    if area == 0:
        return float('inf')

    R = (a * b * c) / (4 * area)
    return R
    
def evaluate_curve_safely(curve, num_samples=300, with_derivatives=False, with_circumrad=False, closed=False):
    eps = 1e-8
    delta_u = 1e-3
    degree = curve.degree
    kv = curve.knotvector
    u_start = kv[degree] + eps
    u_end = kv[-(degree + 1)] - eps
    
    u_vals = np.linspace(u_start, u_end, num_samples)
    points = []
    derivatives = []
    circumrads = []
    if closed:
        u_vals = u_vals[:-1]
    for i_u, u in enumerate(u_vals):
        pt = curve.evaluate_single(u)
        points.append(np.array(pt))
        if with_derivatives:
            if i_u != num_samples - 1:
                pt_prev = curve.evaluate_single(u)
                pt_nxt = curve.evaluate_single(u+delta_u)
            else:
                pt_prev = curve.evaluate_single(u-delta_u)
                pt_nxt = curve.evaluate_single(u)
            derivatives.append(np.array(pt_nxt) - np.array(pt_prev))
        if with_circumrad:
            if i_u != num_samples - 1:
                pt_prev = curve.evaluate_single(u)
                pt = curve.evaluate_single(u+delta_u)
                pt_nxt = curve.evaluate_single(u+2*delta_u)
            else:
                pt_prev = curve.evaluate_single(u-2*delta_u)
                pt = curve.evaluate_single(u-delta_u)
                pt_nxt = curve.evaluate_single(u)
            circumrads.append(compute_circumradius(pt_prev, pt, pt_nxt))
    if closed:
        points.append(points[0])
        if with_derivatives:
            derivatives.append(derivatives[0])
        if with_circumrad:
            circumrads.append(circumrads[0])

    if not with_derivatives and not with_circumrad:
        return np.array(points)
    return np.array(points), np.array(derivatives) if with_derivatives else None, np.array(circumrads) if with_circumrad else None

def ASINT(x):
    return x.ctypes.data_as(POINTER(c_int32))


def ASFLOAT(x):
    return x.ctypes.data_as(POINTER(c_float))

def load_cdll(path):
    return CDLL(path, mode=RTLD_LOCAL)


def resolve_nurbs_utils_path():
    env_path = os.environ.get("NURBS_UTILS_SO")
    if env_path and os.path.exists(env_path):
        return env_path

    local_path = str(Path(__file__).resolve().parent / "nurbs_utils.so")
    if os.path.exists(local_path):
        return local_path

    raise FileNotFoundError(
        "Could not locate nurbs_utils.so. Set NURBS_UTILS_SO or place nurbs_utils.so next to nurbs_profile_sampler.py"
    )


def check_self_intersection(points, eps=1e-6, closed=False):
    path = resolve_nurbs_utils_path()
    dll = load_cdll(path)
    self_intersection = dll.self_intersection
    self_intersection.argtypes = [
        c_int32,
        POINTER(c_float),
        c_int32,
        c_float,
    ]
    self_intersection.restype = c_int32
    return self_intersection(
        int(len(points)),
        ASFLOAT(AC(points.reshape(-1).astype(np.float32))),
        int(closed),
        float(eps),
    )

def compute_maximum_edge_length(verts, edges):
    path = resolve_nurbs_utils_path()
    dll = load_cdll(path)
    maximum_edge_length = dll.maximum_edge_length
    maximum_edge_length.argtypes = [
        c_int32,
        c_int32,
        POINTER(c_float),
        POINTER(c_int32),
    ]
    maximum_edge_length.restype = c_float
    return maximum_edge_length(
        int(len(verts)),
        int(len(edges)),
        ASFLOAT(AC(verts.reshape(-1).astype(np.float32))),
        ASINT(edges.reshape(-1).astype(np.int32)),
    )

class UniformSampler:
    def __init__(self, low, high, rng=None):
        self.low = low
        self.high = high
        if rng is None:
            raise ValueError("UniformSampler requires an explicit rng")
        self.rng = rng


    def sample(self, size=None):
        if size is None:
            return self.rng.uniform(self.low, self.high)
        else:
            return self.rng.uniform(self.low, self.high, size)


class UniformIntSampler:
    def __init__(self, low, high, rng=None):
        self.low = low
        self.high = high
        if rng is None:
            raise ValueError("UniformIntSampler requires an explicit rng")
        self.rng = rng

    def sample(self, size=None):
        if size is None:
            return self.rng.randint(self.low, self.high + 1)
        else:
            return self.rng.randint(self.low, self.high + 1, size)


class GaussianSampler:
    def __init__(self, mean, std, rng=None):
        self.mean = mean
        self.std = std
        if rng is None:
            raise ValueError("GaussianSampler requires an explicit rng")
        self.rng = rng

    def sample(self, size=None):
        if size is None:
            return self.rng.normal(self.mean, self.std)
        else:
            return self.rng.normal(self.mean, self.std, size)


class MixedSampler:
    def __init__(self, sampler_list, weights=None, rng=None):
        self.sampler_list = sampler_list
        if rng is None:
            raise ValueError("MixedSampler requires an explicit rng")
        self.rng = rng
        if weights is None:
            self.weights = np.ones(len(sampler_list)) / len(sampler_list)
        else:
            assert(len(weights) == len(sampler_list))
            self.weights = np.array(weights)
            self.weights /= np.sum(self.weights)

    def sample(self, size=None):
        if size is None:
            idx = self.rng.choice(len(self.sampler_list), p=self.weights)
            return self.sampler_list[idx].sample()
        else:
            idxs = self.rng.choice(len(self.sampler_list), size=size, p=self.weights)
            return np.array([self.sampler_list[i].sample() for i in idxs])


class KnotVectorSampler:
    def __init__(self, clamped=False, cyclic=False, uniform=False, rng=None):
        self.clamped = clamped
        self.cyclic = cyclic
        self.uniform = uniform
        if rng is None:
            raise ValueError("KnotVectorSampler requires an explicit rng")
        self.rng = rng
        assert not (clamped and cyclic), "Cannot have both clamped and cyclic knot vectors"
    
    def sample(self, n, degree):
        if self.clamped:
            n -= (degree + 1) * 2
        assert(n >= 0)
        if n == 0:
            assert(self.clamped)
            knotvector_base = []
        else:
            if not self.uniform:
                m = self.rng.randint(max(1, (n + degree - 1) // degree), n + 1)
                while True:
                    bins = np.zeros(m, dtype=int)
                    bins[0] = 1
                    for i in range(n-1):
                        bins[self.rng.randint(m)] += 1
                    if np.any(bins > degree):
                        continue
                    if np.all(bins >= 1):
                        break
                knotvector_base = [0] * (bins[0]-1)
                for i in range(1, m):
                    knotvector_base.append(1)
                    knotvector_base.extend([0] * (bins[i] - 1))
                knotvector_base = [1, *knotvector_base]
            else:
                knotvector_base = [1] * n
        if self.cyclic:
            nknotvector = n + 2*degree + 1
            knotvector = []
            offset = self.rng.randint(len(knotvector_base))
            for i in range(nknotvector):
                knotvector.append(knotvector_base[(i+offset) % len(knotvector_base)])
            knotvector = np.array(knotvector)
        elif self.clamped:
            knotvector = [0] * (degree + 1) + knotvector_base + [1] + [0] * degree
            knotvector = np.array(knotvector)
        else:
            knotvector = np.array(knotvector_base)
        knotvector = np.cumsum(knotvector)
        knotvector -= knotvector[0]
        return knotvector


class StarfishSampler:
    def __init__(self, degree_sampler, nctrlpts_sampler, radpert_sampler, tanpert_sampler, knotvector_sampler):
        self.degree_sampler = degree_sampler
        self.nctrlpts_sampler = nctrlpts_sampler
        self.radpert_sampler = radpert_sampler
        self.tanpert_sampler = tanpert_sampler
        self.knotvector_sampler = knotvector_sampler
        self.dim = 2

    def sample(self, degree_override=None):
        curve = NURBS.Curve()
        curve.degree = self.degree_sampler.sample() if degree_override is None else degree_override
        nctrlpts = self.nctrlpts_sampler.sample()
        control_points = np.array([(np.cos(theta), np.sin(theta)) for theta in np.linspace(0, 2 * np.pi, nctrlpts, endpoint=False)])
        control_points_tangent = np.array([(np.cos(theta + np.pi / 2), np.sin(theta + np.pi / 2)) for theta in np.linspace(0, 2 * np.pi, nctrlpts, endpoint=False)])
        control_points += control_points * self.radpert_sampler.sample(nctrlpts).reshape((-1, 1)) + control_points_tangent * self.tanpert_sampler.sample(nctrlpts).reshape((-1, 1))
        curve.ctrlpts = np.concatenate((control_points, control_points[:curve.degree]), axis=0)
        knotvector = self.knotvector_sampler.sample(nctrlpts, curve.degree)
        curve.knotvector = knotvector
        return curve


class RandomWalkSampler:
    def __init__(self, degree_sampler, nctrlpts_sampler, pert_sampler, knotvector, dim=2):
        self.degree_sampler = degree_sampler
        self.nctrlpts_sampler = nctrlpts_sampler
        self.pert_sampler = pert_sampler
        self.knotvector = knotvector
        self.dim = dim
    
    def sample(self):
        curve = NURBS.Curve()
        curve.degree = self.degree_sampler.sample()
        nctrlpts = max(curve.degree + 1, self.nctrlpts_sampler.sample())
        control_points = np.zeros((nctrlpts, self.dim))
        for i in range(nctrlpts - 1):
            control_points[i + 1] = control_points[i] + self.pert_sampler.sample(self.dim)
        curve.ctrlpts = control_points
        knotvector = self.knotvector.sample(nctrlpts + curve.degree + 1, curve.degree)
        curve.knotvector = knotvector
        return curve


class ReptileSampler:
    def __init__(self, base_sampler, radius_sampler, pert_sampler):
        self.base_sampler = base_sampler
        self.radius_sampler = radius_sampler
        self.pert_sampler = pert_sampler
        self.dim = base_sampler.dim

    def sample(self):
        radius = self.radius_sampler.sample()
        fit_ctrlpts_size = 20

        curve = self.base_sampler.sample()
        points = evaluate_curve_safely(curve, num_samples=300)
        tangents = np.gradient(points, axis=0)

        def compute_normal(tangent):
            return np.array([-tangent[1], tangent[0]]) / np.linalg.norm(tangent)

        normals = np.array([compute_normal(t) for t in tangents])
        outer = points + radius * normals
        inner = points - radius * normals
        inner = inner[::-1]

        contour_loop = np.vstack([outer, inner, outer[0:1]])

        fit_curve = fitting.approximate_curve(contour_loop.tolist(), degree=3, ctrlpts_size=fit_ctrlpts_size)
        fit_curve.closed = True
        ctrlpts = np.array(fit_curve.ctrlpts) + self.pert_sampler.sample((fit_ctrlpts_size, self.dim))
        ctrlpts[-1] = ctrlpts[0]
        fit_curve.ctrlpts = ctrlpts.tolist()
        fit_curve.closed = True
        return fit_curve

