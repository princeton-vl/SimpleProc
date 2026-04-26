import argparse
import hashlib
import math
import os
import shlex
import shutil
import subprocess
import sys

import numpy as np


def run_cmd(cmd, cwd=None):
    print("Running:", " ".join(shlex.quote(x) for x in cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def ensure_parent(path):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def feature_rng_from_file(seed: int, suffix: str = ""):
    script_name = os.path.basename(sys.argv[0])
    tag = f"{seed}_{script_name}{suffix}"
    h = int(hashlib.sha256(tag.encode("utf-8")).hexdigest(), 16) % (2**32)
    return np.random.RandomState(h)


def run_py_script(script_name, pipeline_dir, infinigen_repo, args):
    """Run a Python script from the pipeline with given arguments."""
    run_cmd(
        ["python", os.path.join(pipeline_dir, script_name), *args],
        cwd=infinigen_repo,
    )


def run_blender_script(blender_bin, script_name, pipeline_dir, infinigen_repo, args):
    """Run a Blender script from the pipeline with given arguments."""
    run_cmd(
        [blender_bin, "-b", "-P", os.path.join(pipeline_dir, script_name), "--", *args],
        cwd=infinigen_repo,
    )


def main():
    parser = argparse.ArgumentParser(description="Run the scene generation pipeline locally.")
    parser.add_argument("--output-scene", required=True, help="Final output .blend file path")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--work-scene", type=str, default=None, help="Optional temp .blend path")
    parser.add_argument("--render-output", type=str, default=None, help="Optional render output directory")
    parser.add_argument("--total-pixels", type=int, default=576 * 768)
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    pipeline_dir = os.path.join(project_root, "datagen")
    infinigen_repo = os.path.join(project_root, "infinigen")
    blender_bin = os.path.join(project_root, "blender", "blender")

    if not os.path.isdir(pipeline_dir):
        raise FileNotFoundError(f"Pipeline folder not found: {pipeline_dir}")
    if not os.path.isdir(infinigen_repo):
        raise FileNotFoundError(f"Infinigen repo not found: {infinigen_repo}")

    output_scene = os.path.abspath(args.output_scene)
    ensure_parent(output_scene)
    work_scene = args.work_scene or f"/tmp/scene_{args.seed}.blend"
    should_generate = not os.path.isfile(output_scene)

    rng = feature_rng_from_file(seed=args.seed)
    fov_x_0 = 30.92
    fov_x_1 = 96.88
    r1 = rng.random()
    fov_x = fov_x_0 + r1 * (fov_x_1 - fov_x_0)
    fov_x_rad = math.radians(fov_x)
    fov_y_rad = 2 * math.atan((3 / 4) * math.tan(fov_x_rad / 2))
    fov_y = math.degrees(fov_y_rad)

    if should_generate:
        # Create initial scene on work_scene
        run_py_script(
            "add_cameras.py", pipeline_dir, infinigen_repo,
            ["--output", work_scene, "--seed", str(args.seed), "--fov_x", str(fov_x), "--fov_y", str(fov_y)],
        )

        # All subsequent stages modify work_scene inplace
        seed_str = str(args.seed)
        io_args = ["--input", work_scene, "--output", work_scene, "--seed", seed_str]
        io_args_no_seed = ["--input", work_scene, "--output", work_scene]

        # Stage commands pipeline: (script, extra_args)
        stage_scripts = [
            ("add_shapes.py", []),
            ("add_material.py", []),
            ("add_grass.py", ["--random", "1"]),
            ("add_nurbs_grass.py", []),
            ("add_composite_material.py", []),
            ("add_displacement.py", []),
            ("add_lights.py", []),
            ("add_displacement_shader.py", []),
            ("add_room.py", ["--random", "1"]),
            ("add_material.py", ["--room_only", "1"]),
            ("add_composite_material.py", ["--room_only", "1"]),
            ("remove_objs_close_to_cam.py", []),
        ]

        for script, extra_args in stage_scripts:
            stage_io_args = io_args_no_seed if script == "remove_objs_close_to_cam.py" else io_args
            run_py_script(script, pipeline_dir, infinigen_repo, stage_io_args + extra_args)

        # Copy final work_scene to output_scene
        shutil.copy2(work_scene, output_scene)
    else:
        print("Found existing scene, skipping generation stage:", output_scene)

    if args.render_output:
        render_scene = f"/tmp/scene_{args.seed}.blend"
        run_py_script(
            "subdivide_obj.py", pipeline_dir, infinigen_repo,
            ["--input", output_scene, "--output", render_scene],
        )

        run_blender_script(
            blender_bin, "render_scene.py", pipeline_dir, infinigen_repo,
            ["--input", render_scene, "--output", os.path.abspath(args.render_output), "--total_pixels", str(args.total_pixels)],
        )

    print("Done. Final scene:", output_scene)
    if args.render_output:
        print("Rendered outputs:", os.path.abspath(args.render_output))


if __name__ == "__main__":
    main()
