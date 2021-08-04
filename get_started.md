# Get Started

## Usage

### Install

- Clone this repo:

```bash
git clone https://github.com/kevin-ssy/ViP.git
cd ViP
```

- Create a conda virtual environment and activate it:

```bash
conda create -n vip python=3.7 -y
conda activate vip
```

- Install `CUDA==10.1` with `cudnn7` following
  the [official installation instructions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
- Install `PyTorch==1.7.1` and `torchvision==0.8.2` with `CUDA==10.1`:

```bash
conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=10.1 -c pytorch
```

- Install `timm==0.3.4, einops, pyyaml`:

```bash
pip3 install timm=0.3.4, einops, pyyaml
```

- Install `Apex`:

```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

### Data preparation

We use standard ImageNet dataset, you can download it from http://image-net.org/. We provide the following two ways to
load data:

- For standard folder dataset, move validation images to labeled sub-folders. The file structure should look like:
  ```bash
  $ tree data
  imagenet
  ├── train
  │   ├── class1
  │   │   ├── img1.jpeg
  │   │   ├── img2.jpeg
  │   │   └── ...
  │   ├── class2
  │   │   ├── img3.jpeg
  │   │   └── ...
  │   └── ...
  └── val
      ├── class1
      │   ├── img4.jpeg
      │   ├── img5.jpeg
      │   └── ...
      ├── class2
      │   ├── img6.jpeg
      │   └── ...
      └── ...
 
  ```
- To boost the slow speed when reading images from massive small files, we also support zipped ImageNet, which includes
  four files:
    - `train.zip`, `val.zip`: which store the zipped folder for train and validate splits.
    - `train_map.txt`, `val_map.txt`: which store the relative path in the corresponding zip file and ground truth
      label. Make sure the data folder looks like this:

  ```bash
  $ tree data
  data
  └── ImageNet-Zip
      ├── train_map.txt
      ├── train.zip
      ├── val_map.txt
      └── val.zip
  
  $ head -n 5 data/ImageNet-Zip/val_map.txt
  ILSVRC2012_val_00000001.JPEG	65
  ILSVRC2012_val_00000002.JPEG	970
  ILSVRC2012_val_00000003.JPEG	230
  ILSVRC2012_val_00000004.JPEG	809
  ILSVRC2012_val_00000005.JPEG	516
  
  $ head -n 5 data/ImageNet-Zip/train_map.txt
  n01440764/n01440764_10026.JPEG	0
  n01440764/n01440764_10027.JPEG	0
  n01440764/n01440764_10029.JPEG	0
  n01440764/n01440764_10040.JPEG	0
  n01440764/n01440764_10042.JPEG	0
  ```

### Evaluation

To evaluate a pre-trained `ViP` on ImageNet val, run:

```bash
python3 main.py <data-root> --model <model-name> -b <batch-size> --eval_checkpoint <path-to-checkpoint>
```

### Training from scratch

To train a `ViP` on ImageNet from scratch, run:

```bash
bash ./distributed_train.sh <job-name> <config-path> <num-gpus>

```


For example, to train `ViP` with 8 GPU on a single node, run:

`ViP-Tiny`:

```bash
bash ./distributed_train.sh vip-t-001 configs/vip_t_bs1024.yaml 8
```

`ViP-Small`:

```bash
bash ./distributed_train.sh vip-s-001 configs/vip_s_bs1024.yaml 8
```

`ViP-Medium`:

```bash
bash ./distributed_train.sh vip-m-001 configs/vip_m_bs1024.yaml 8
```

`ViP-Base`:

```bash
bash ./distributed_train.sh vip-b-001 configs/vip_b_bs1024.yaml 8
```

### Profiling the model

To measure the throughput, run:

```bash
python3 test_throughput.py <model-name>
```

For example, if you want to get the test speed of `Vip-Tiny` on your device, run:
```bash
python3 test_throughput.py vip-tiny
```


To measure the FLOPS and number of parameters, run:

```bash
python3 test_flops.py <model-name>
```