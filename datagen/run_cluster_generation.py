import argparse
import math
import os
import shlex
import subprocess
import time

import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument(
    "--start-index", type=int, default=10000, help="Start scene index (inclusive)"
)
parser.add_argument(
    "--end-index", type=int, default=11000, help="End scene index (exclusive)"
)
args = parser.parse_args()
if args.end_index <= args.start_index:
    raise ValueError("--end-index must be greater than --start-index")

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
tools_folder = os.path.join(project_root, "datagen")
user_name = os.environ.get("USER", "user")
suffix = ""
dataset_spec = os.environ.get("DATASET_FOLDER", "final_vFeb21")
if os.path.isabs(dataset_spec):
    dataset_path = os.path.normpath(os.path.expanduser(dataset_spec))
else:
    dataset_path = os.path.normpath(os.path.join(project_root, dataset_spec))
output_root = os.path.dirname(dataset_path)
dataset_folder = os.path.basename(dataset_path)
infinigen_repo = os.path.join(project_root, "infinigen")
blender_bin = os.path.join(project_root, "blender", "blender")

os.makedirs(f"{output_root}/logs", exist_ok=True)

submitted_jobs = set()
submitted_jobs_render = set()


def run_cmd(cmd, cwd=None):
    print("Running:", " ".join(shlex.quote(x) for x in cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def get_running_jobs():
    """Get set of currently running job names that contain dataset_folder."""
    try:
        result = subprocess.run(
            ["squeue", "-u", user_name, "--format=%j,%t", "--noheader"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            all_jobs = result.stdout.strip().split("\n")
            # Filter jobs that contain the dataset_folder string and exclude state "CG" (completing)
            filtered_jobs = {}
            for job_line in all_jobs:
                if "," in job_line:
                    job_name, state = job_line.rsplit(",", 1)
                    if dataset_folder in job_name and "train" not in job_name and state != "CG":
                        filtered_jobs[job_name] = state
            return filtered_jobs
        return {}
    except Exception:
        return {}


def submit_jobs(running_jobs, start, end):
    jobs_submitted = 0
    for i in range(start, end):
        # Check if job is already running
        job_name = f"{dataset_folder}_{i}{suffix}"
        if job_name in running_jobs:
            print(f"Skipping {job_name} - already running")
            continue
            
        exists = os.path.exists(f"{output_root}/{dataset_folder}/scene_{i}/finish")
        if exists:
            continue

        rng = np.random.default_rng(i)

        file_name = f"{output_root}/logs/{job_name}.sh"
        final_scene = f"{output_root}/{dataset_folder}/scene_{i}/scene.blend"
        tmp_scene = f"/tmp/{job_name}.blend"

        fov_x_0 = 30.92
        fov_x_1 = 96.88
        r1 = rng.random()
        fov_x = fov_x_0 + r1 * (fov_x_1 - fov_x_0)
        fov_x_rad = math.radians(fov_x)
        fov_y_rad = 2 * math.atan((3 / 4) * math.tan(fov_x_rad / 2))
        fov_y = math.degrees(fov_y_rad)
        
        if i not in submitted_jobs:
            account = "allcs"
            mem = "64000"
            timelimit = "00-01:00:00"
        else:
            account = "pvl"
            mem = "64000"
            timelimit = "00-05:00:00"

        with open(file_name, "w") as f:
            f.write(f'''#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={output_root}/logs/{job_name}.out
#SBATCH --error={output_root}/logs/{job_name}.err
#SBATCH --mail-user=zeyum@cs.princeton.edu
#SBATCH --mail-type=FAIL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem={mem}
#SBATCH --time={timelimit}
#SBATCH --account={account}
#SBATCH --exclude=node014,node016,node007,node901,node902,node906,node907,node908,node909,node703,node805,node013,node017,node018,node019,node020,node021,node022,node023,node024,node025,node026,node027,node028,node029,node030,node031,node403

cd {infinigen_repo}

mkdir -p {output_root}/{dataset_folder}/scene_{i}

if [ ! -f "{output_root}/{dataset_folder}/scene_{i}/add_cameras.finish" ]; then
    python {tools_folder}/add_cameras.py --output {final_scene} --seed {i} --fov_x {fov_x} --fov_y {fov_y}
    touch {output_root}/{dataset_folder}/scene_{i}/add_cameras.finish
fi
if [ ! -f "{output_root}/{dataset_folder}/scene_{i}/add_shapes.finish" ]; then
    python {tools_folder}/add_shapes.py --input {final_scene} --output {final_scene} --seed {i}
    touch {output_root}/{dataset_folder}/scene_{i}/add_shapes.finish
fi


if [ ! -f "{output_root}/{dataset_folder}/scene_{i}/others.finish" ]; then
    cp {final_scene} {tmp_scene}
    python {tools_folder}/add_material.py --input {tmp_scene} --output {tmp_scene} --seed {i}
    

    python {tools_folder}/add_grass.py --input {tmp_scene} --output {tmp_scene} --seed {i} --random 1
    python {tools_folder}/add_nurbs_grass.py --input {tmp_scene} --output {tmp_scene} --seed {i}
    
    python {tools_folder}/add_composite_material.py --input {tmp_scene} --output {tmp_scene} --seed {i}

    python {tools_folder}/add_displacement.py --input {tmp_scene} --output {tmp_scene} --seed {i}


    python {tools_folder}/add_lights.py --input {tmp_scene} --output {tmp_scene} --seed {i}
    
    python {tools_folder}/add_displacement_shader.py --input {tmp_scene} --output {tmp_scene} --seed {i}
    
    python {tools_folder}/add_room.py --input {tmp_scene} --output {tmp_scene} --random 1 --seed {i}
    python {tools_folder}/add_material.py --input {tmp_scene} --output {tmp_scene} --seed {i} --room_only 1
    python {tools_folder}/add_composite_material.py --input {tmp_scene} --output {tmp_scene} --seed {i} --room_only 1

    python {tools_folder}/remove_objs_close_to_cam.py --input {tmp_scene} --output {tmp_scene}

    cp {tmp_scene} {final_scene}
    touch {output_root}/{dataset_folder}/scene_{i}/others.finish
fi

rm -rf {output_root}/{dataset_folder}/scene_{i}/*.blend1
touch {output_root}/{dataset_folder}/scene_{i}/finish

''')
        run_cmd(["sbatch", file_name])
        submitted_jobs.add(i)
        jobs_submitted += 1
        
        # Small delay to avoid submission congestion
        time.sleep(0.5)
    
    return jobs_submitted



def submit_jobs_render(running_jobs, start, end):
    jobs_submitted = 0
    
    for i in range(start, end):
        # Check if job is already running
        job_name = f"{dataset_folder}_{i}{suffix}_render"
        if job_name in running_jobs:
            print(f"Skipping {job_name} - already running")
            continue
            
        exists = True
        for j in range(8):
            if not os.path.exists(f"{output_root}/{dataset_folder}/renders{suffix}/scene_{i}/images/{j:08d}.finish"):
                exists = False
        if exists:
            continue

        file_name = f"{output_root}/logs/{job_name}.sh"
        beginning_scene = f"{output_root}/{dataset_folder}/scene_{i}/scene.blend"
        final_scene = f"/tmp/{job_name}.blend"

        total_pixels = 576 * 768

        if i not in submitted_jobs_render:
            account = "allcs"
            mem = "20000"
            timelimit = "00-01:00:00"
        else:
            account = "pvl"
            mem = "64000"
            timelimit = "00-05:00:00"
        

        with open(file_name, "w") as f:
            f.write(f'''#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={output_root}/logs/{job_name}.out
#SBATCH --error={output_root}/logs/{job_name}.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --mem={mem}
#SBATCH --time={timelimit}
#SBATCH --account={account}
#SBATCH --gres=gpu:1
#SBATCH --exclude=node014,node016,node007,node902,node906,node909,node703,node805,node013,node103,node104,node017,node018,node019,node020,node021,node022,node023,node024,node025,node026,node027,node028,node029,node030,node031

cd {infinigen_repo}


set -e

python {tools_folder}/subdivide_obj.py --input {beginning_scene} --output {final_scene}
{blender_bin} -b -P {tools_folder}/render_scene.py -- --input {final_scene} --output {output_root}/{dataset_folder}/renders{suffix}/scene_{i} --total_pixels {total_pixels}


rm -rf {final_scene}*
rm -rf {output_root}/{dataset_folder}/scene_{i}/*.blend*


''')
        run_cmd(["sbatch", file_name])
        submitted_jobs_render.add(i)
        jobs_submitted += 1
        
        # Small delay to avoid submission congestion
        time.sleep(0.5)
    
    return jobs_submitted


start = args.start_index
end = args.end_index

while start < end:
    batch_start = start
    batch_end = min(start + 1000, end)
    print(f"\nProcessing batch: [{batch_start}, {batch_end})")

    while True:
        print("\n" + "=" * 50)
        print("Step1 Checking job status and submitting remaining work...")
        
        # Get currently running jobs
        running_jobs = get_running_jobs()
        print(f"Currently running jobs: {len(running_jobs)}")
        
        total_jobs_submitted = 0
        

        jobs_submitted = submit_jobs(running_jobs, batch_start, batch_end)
        total_jobs_submitted += jobs_submitted
        print(f"Submitted {jobs_submitted} jobs")
        
        print(f"\nTotal jobs submitted this round: {total_jobs_submitted}")
        
        # If no jobs were submitted, check if any are still running
        if total_jobs_submitted == 0:
            if len(running_jobs) == 0:
                print("No jobs submitted and no jobs running. All work complete!")
                break
            else:
                print("No new jobs to submit, but jobs are still running. Waiting...")
        
        # Wait before next check - shorter interval for more responsive monitoring
        print("Waiting 30 seconds before next check...")
        time.sleep(30)  # Check every 30 seconds

    print("Step1 Job submission monitoring complete!")



    while True:
        print("\n" + "=" * 50)
        print("Step2 Checking job status and submitting remaining work...")
        
        # Get currently running jobs
        running_jobs = get_running_jobs()
        print(f"Currently running jobs: {len(running_jobs)}")
        
        total_jobs_submitted = 0
        

        jobs_submitted = submit_jobs_render(running_jobs, batch_start, batch_end)
        total_jobs_submitted += jobs_submitted
        print(f"Submitted {jobs_submitted} jobs")
        
        print(f"\nTotal jobs submitted this round: {total_jobs_submitted}")
        
        # If no jobs were submitted, check if any are still running
        if total_jobs_submitted == 0:
            if len(running_jobs) == 0:
                print("No jobs submitted and no jobs running. All work complete!")
                break
            else:
                print("No new jobs to submit, but jobs are still running. Waiting...")
        
        # Wait before next check - shorter interval for more responsive monitoring
        print("Waiting 30 seconds before next check...")
        time.sleep(30)  # Check every 30 seconds

    print("Step2 Job submission monitoring complete!")
    start = batch_end