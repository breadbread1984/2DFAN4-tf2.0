# 2DFAN4 

an implement of 2DFAN4 proposed in the paper ["How far are we from solving the 2D & 3D face alignment problem?"](https://arxiv.org/abs/1703.07332) facial landmarker algorithm with Tensorflow 2.0. This project is a reimplement of my project at [link](https://github.com/breadbread1984/2DFAN4)

### how to train

download 300W-LP and place its directory under the root directory of this project. run the following command.

```Bash
python3 train_2DFAN4.py
```

### how to test

test the model with class Landmarker.py. you can test the model on a given image by running the following command.

```Bash
python3 Landmarker.py
```

