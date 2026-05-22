# SimpleProc: Fully Procedural Synthetic Data from Simple Rules for Multi-View Stereo


We explore the design space of procedural rules for multi-view stereo (MVS). We demonstrate that we can generate effective training data using SimpleProc: a fully procedural generator driven by a very small set of rules using Non-Uniform Rational Basis Splines (NURBS), as well as basic displacement and texture patterns. At a modest scale of 8,000 images, our approach achieves superior results compared to manually curated images (at the same scale) sourced from games and real-world objects. When scaled to 352,000 images, our method yields performance comparable to—and in several benchmarks, exceeding—models trained on over 692,000 manually curated images.


![Alternative Text](teaser.jpg)
*Figure 1: Fully procedural synthetic data from simple rules (top) is as effective as curated data from artists or 3D scans (bottom) for training multi-view stereo models.*

If you find our work useful for your work, please consider citing our academic paper:

<h3 align="center">
    <a href="https://arxiv.org/abs/2604.04925">
        SimpleProc: Fully Procedural Synthetic Data from Simple Rules for Multi-View Stereo
    </a>
</h3>
<p align="center">
    <a href="https://mazeyu.github.io/">Zeyu Ma</a>, 
    <a href="https://araistrick.github.io/">Alexander Raistrick</a>, 
    <a href="http://www.cs.princeton.edu/~jiadeng">Jia Deng</a><br>
</p>

```
@misc{ma2026fullyproceduralsyntheticdata,
      title={Fully Procedural Synthetic Data from Simple Rules for Multi-View Stereo}, 
      author={Zeyu Ma and Alexander Raistrick and Jia Deng},
      year={2026},
      eprint={2604.04925},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2604.04925}, 
}
```

# Data Generator

## 1) Create Conda Environment + Prepare Infinigen

```bash
conda create -n datagen python=3.11 -y
conda activate datagen

git clone git@github.com:princeton-vl/infinigen.git
cd infinigen
pip install -e ".[dev,terrain,vis]"
cd ..
cd datagen
g++ -O3 -std=c++17 -shared -fPIC nurbs_utils.cpp -o nurbs_utils.so
cd ..
```

Notes: This project uses Infinigen APIs (not Infinigen assets).

## 2) Install Blender 4.5 for EEVEE

```bash
wget https://download.blender.org/release/Blender4.5/blender-4.5.0-linux-x64.tar.xz
tar -xf blender-4.5.0-linux-x64.tar.xz
mv blender-4.5.0-linux-x64 blender
rm -rf blender-4.5.0-linux-x64.tar.xz
./blender/4.5/python/bin/python3.11 -m pip install opencv-python OpenEXR==3.3.5 matplotlib
```


## 3) Run Locally (single-scene style)

```bash
# Generate one local scene
python datagen/run_local_scene_generation.py \
  --output-scene output/test_scene.blend \
  --seed 0

# render pass, GPU is required because EEVEE is used
python datagen/run_local_scene_generation.py \
  --output-scene output/test_scene.blend \
  --seed 0 \
  --render-output output/test_renders
```



## 4) Run on Cluster

```bash
export DATASET_FOLDER=output/dataset

python datagen/run_cluster_generation.py --start-index 0 --end-index 1
```


# Training MVSAnywhere with Generated Data
For our 44k scenes used in the paper, please download from Hugging Face:
https://huggingface.co/datasets/princeton-vl/SimpleProc

The release is hosted as WebDataset shards (`shard-*.tar`). Download and unpack to scene folders with:

```bash
pip install -U huggingface_hub

export hf_data_root=/path/to/simpleproc_hf
export dst_folder=/path/to/dataset

huggingface-cli download princeton-vl/SimpleProc \
  --repo-type dataset \
  --include "shard-*.tar" \
  --local-dir "$hf_data_root" \
  --local-dir-use-symlinks False

python scripts/unpack_data.py
```


Clone our fork of MVSAnywhere and install its environment: https://github.com/mazeyu/mvsanywhere.
Run the command below to train a MVSAnywhere on these data. GPU Memory requirement: 48G.

In `configs/data/infinigen_cubism/44k_scenes.yaml`, please update "dataset_path: path/to/dataset". 

```bash
cd mvsanywhere

python src/mvsanywhere/train.py \
  --log_dir /path/to/training/dir \
  --name NAME \
  --config_file configs/models/mvsanywhere_model.yaml \
  --data_config configs/data/infinigen_cubism/44k_scenes.yaml \
  --val_data_config "" \
  --batch_size 6 \
  --da_weights_path weights/depth_anything_v2_vitb.pth \
  --gpus 1 \
  --max_steps 1600000 \
  --val_interval 5000
```

You can download our trained model here: https://huggingface.co/princeton-vl/mvsanywhere-simpleproc/tree/main (3 checkpoints for 3 different runs).
