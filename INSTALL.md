# Installation of DSANet

This document contains detailed instructions for installing dependencies for DSANet. We recommand using the [install.sh](install.sh). The code is tested on an Ubuntu 16.04 system with Nvidia GPU (We recommand 1080TI / TITAN XP).

### Requirments
* Conda with Python 3.7.
* Nvidia GPU.
* PyTorch 1.4.0
* yacs
* pyyaml
* matplotlib
* tqdm
* OpenCV

## Step-by-step instructions

#### Create environment and activate
```bash
conda create --name pysot python=3.7
conda activate pysot
```

#### Install numpy/pytorch/opencv
```
conda install numpy
conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch
pip install opencv-python
```

#### Install other requirements
```
pip install pyyaml yacs tqdm colorama matplotlib cython tensorboardX
```

#### Build extensions
```
python setup.py build_ext --inplace
cd pysot/models/head/dcn/ && python setup.py build_ext --inplace
```


## Try with scripts
```
bash install.sh /path/to/your/conda pysot
```
