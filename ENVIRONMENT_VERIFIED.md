# NSG-VD verified environment

Verified on 2026-10-12 (Asia/Shanghai) for evaluation with the released
Deep-MMD checkpoints. This setup does not retrain the kernel.

## Repository and hardware

- Repository commit: `9a60f57fcb464b418899110a4717c8c8fd7687bc`
- Branch: `main`
- OS architecture: Linux x86_64
- GPU: NVIDIA GeForce RTX 3090, 24 GiB
- NVIDIA driver: 570.133.07
- Driver-supported CUDA: 12.8
- PyTorch CUDA runtime: 12.1
- cuDNN reported by PyTorch: 90100
- Conda environment: `nsg-vd`
- Python: 3.10.20

The CUDA 12.8 value printed by `nvidia-smi` is the maximum CUDA version
supported by the driver. PyTorch's CUDA 12.1 runtime is compatible with this
newer driver.

## Key package versions

- torch 2.4.1+cu121
- torchvision 0.19.1+cu121
- torchaudio 2.4.1+cu121
- numpy 1.26.4
- opencv-python 4.10.0.84
- timm 0.9.16
- transformers 4.57.6
- OpenAI CLIP commit `d05afc436d78f1c48dc0dbf8e5980a9d471f35f6`
- hydra-core 1.3.7
- omegaconf 2.3.0
- moviepy 2.2.1
- ffmpeg 8.1.2
- OpenMPI 4.1.6
- mpi4py 4.1.2
- aria2 1.37.0

The complete Python package snapshot is in `requirements-verified.txt`.

## Environment creation

```bash
conda create -n nsg-vd python=3.10 pip -y
conda activate nsg-vd

python -m pip install --upgrade pip setuptools wheel
conda install -c conda-forge ffmpeg openmpi mpi4py aria2 -y

python -m pip install \
  torch==2.4.1 \
  torchvision==0.19.1 \
  torchaudio==2.4.1 \
  --index-url https://download.pytorch.org/whl/cu121

python -m pip install \
  "numpy==1.26.4" \
  "opencv-python==4.10.0.84" \
  "timm==0.9.16" \
  "transformers<5"

grep -v 'github.com/openai/CLIP' requirements.txt \
  > /tmp/nsg-requirements-no-clip.txt
python -m pip install -r /tmp/nsg-requirements-no-clip.txt

python -m pip install \
  "git+https://github.com/openai/CLIP.git@d05afc436d78f1c48dc0dbf8e5980a9d471f35f6" \
  blobfile pandas pillow pyyaml scipy tqdm

# Re-assert the NumPy/OpenCV pair after installing unconstrained requirements.
python -m pip install --force-reinstall \
  "numpy==1.26.4" \
  "opencv-python==4.10.0.84"

python -m pip check
```

## Model files

```text
a37c32fffd316cd494cf3f35b339936debdc1576dad13fe57c42399a5dbc78b1  ../Checkpoints/256x256_diffusion_uncond.pt
c1d2a9121f5172e79c47ceb15df29987c9828ee03e1fd8302477a8a12bf7599f  ckpts/standard-Pika-mp.pth
a7c6d2100353c20e00b355c97b878b357bff05374e899413fef8ddfa40e99310  ckpts/standard-SEINE-mp.pth
```

## Evaluation data

Data root: `/root/autodl-tmp/Data/GenVideo`

- 8,286 fake MP4 files across the ten released evaluation generators
- 10,000 MSR-VTT MP4 files
- 49 split files
- Default evaluation subset coverage: complete, zero missing files
- Fake evaluation samples used by the default config: 4,551
- MSR-VTT reference samples: first 100 entries of `val_ids.txt`
- MSR-VTT real test samples: 500 entries from offsets 200:700 of
  `test_ids.txt`
- Kinetics-400 is not required for released-checkpoint evaluation.

## Verification results

- `python -m pip check`: passed
- CUDA available: true, one RTX 3090 detected
- CUDA matrix multiplication: passed
- torchvision NMS operator: passed
- All training/evaluation entry-point imports: passed
- All baseline model imports: passed
- Released Deep-MMD checkpoint strict load: passed
- Deep-MMD GPU forward: passed; output `(1, 2)`, feature `(1, 300)`
- Diffusion checkpoint load: passed
- Diffusion score GPU forward: passed; output `(1, 6, 224, 224)`
- Diffusion test peak allocated memory: approximately 4.16 GiB
- ffmpeg and MPI smoke checks: passed

The `torch.meshgrid` deprecation warning and the `torch.load(weights_only=False)`
future warning originate in repository/upstream code and did not affect the
verification.

## Evaluation command

```bash
conda activate nsg-vd
cd /root/autodl-tmp/NSG-VD

python test_dMMD.py \
  --config-path configs/nsg-vd-224x224 \
  --config-name test.yaml \
  ckpt_path=./ckpts/standard-Pika-mp.pth \
  model.is_yy_zero=true \
  log_path=./results/test/nsg-vd \
  save_csv_file=standard-Pika-mmd-mp.csv
```

For an MMD-D checkpoint, use `model.is_yy_zero=false`. For an MMD-MP
checkpoint, use `model.is_yy_zero=true`.
