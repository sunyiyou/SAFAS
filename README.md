# Rethinking Domain Generalization for Face Anti-spoofing: Separability and Alignment

This is the source code for CVPR 2023 paper [Rethinking Domain Generalization for Face Anti-spoofing:
Separability and Alignment](https://arxiv.org/abs/2303.13662) 
by Yiyou Sun, Yaojie Liu, Xiaoming Liu, Yixuan Li and Wen-Sheng Chu.

## Preliminaries
It is tested under Ubuntu Linux 20.04 and Python 3.8 environment, and requries some packages to be installed:
* [PyTorch](https://pytorch.org/)
* [scipy](https://github.com/scipy/scipy)
* [numpy](http://www.numpy.org/)
* [sklearn](https://scikit-learn.org/stable/)
* [MTCNN](https://pypi.org/project/mtcnn/)
* [ylib](https://github.com/sunyiyou/ylib) (Manually download and copy to the current folder)

## Usage

### 1. Dataset Preparation

Download the OULU-NPU, CASIA-FASD, Idiap Replay-Attack, and MSU-MFSD datasets. Put datasets into the directory of `datasets/FAS`.

### 2. Prepocessing 

Run `./preposess.py`.

### 3. Demo 

Run `./train.py --protocol [O_C_I_to_M/O_M_I_to_C/O_C_M_to_I/I_C_M_to_O]`.

## Citation

If you use our codebase, please cite our work:

```
@article{sun2023safas,
  title={Rethinking Domain Generalization for Face Anti-spoofing:
Separability and Alignment},
  author={Sun, Yiyou and Liu, Yaojie and Liu, Xiaoming and Li, Yixuan and Chu Wen-Sheng},
  journal={CVPR},
  year={2023}
}
```

