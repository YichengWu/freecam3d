# FreeCam3D

### [Project](https://yichengwu.github.io/freecam3d/) | [1-min Summary](https://www.youtube.com/watch?v=QCOzSyEeniw) | [10-min Talk](https://www.youtube.com/watch?v=-6AhHc_1sOs) | [Paper](https://drive.google.com/file/d/1P3_ZJYdp_VDuWOQaPuf_2xt0VGWCt7Jg/view?usp=sharing)

This repository contains TensorFlow implementation for the ECCV2020 paper *PhaseCam3D: Snapshot Structured Light 3D with Freely-moving Cameras* by [Yicheng Wu](https://yichengwu.github.io), [Vivek Boominathan](https://vivekboominathan.com/), [Xuan Zhao](https://www.linkedin.com/in/xuan-zhao-94308991/), [Jacob T. Robinson](https://www.robinsonlab.com/jacob-t-robinson), [Hiroshi Kawasaki](http://www.cvg.ait.kyushu-u.ac.jp), [Aswin Sankaranarayanan](http://imagesci.ece.cmu.edu/index.html), and [Ashok Veeraraghavan](https://computationalimaging.rice.edu/).

![Overview](/docs/data/teaser_fullsize.jpg)


## Installation
Clone this repo.
```bash
git clone https://github.com/YichengWu/freecam3D
cd freecam3D/
```
The code is developed using Python 3.6.8 and TensorFlow 1.14.0. The GPU we used is NVIDIA GTX 2080 Ti (11G). Change `batch_size` accordingly if you run the code with different GPU memory.

## Dataset

The dataset is generated from Blender, which contains both the depth map from the project and camera view of a same scene, as well as the relative pose tranformation.
The pre-processed TFrecord files can be downloaded from [Google Drive](https://drive.google.com/drive/u/0/folders/1g-0gVUvasoL4LlOwqlDS_cEfoksAaiWa). It contains 4850 training elements, 912 validation elements, and 201 test elements.

## Train

The PSFs used here are captured from an experiemntal prototype. To train the network to optimize the network, simply run the following code.
```
python train.py
```
Inside the code, `DATA_PATH_root` is the directory of the downloaded dataset, `results_dir` is the output result directory.

### Logging

We use Tensorboard for logging training progress. Recommended: execute `tensorboard --logdir /path/to/save_dir --port 9999` and visit `localhost:9999` in the browser.

## Evaluation

Once the network is trained, the performance can be evaluated using the testing dataset. 
```
python test.py
```
Change `results_dir` to the place you save your model. Once the testing is finished, a new folder called `test` will be created inside the model directory.

## Citation
If you use this code for your research, please cite our papers.
```
@inproceedings{wu2020freecam3d,
  title={FreeCam3D: Snapshot Structured Light 3D with Freely-Moving Cameras},
  author={Wu, Yicheng and Boominathan, Vivek and Zhao, Xuan and Robinson, Jacob T and Kawasaki, Hiroshi and Sankaranarayanan, Aswin and Veeraraghavan, Ashok},
  booktitle={European Conference on Computer Vision},
  pages={309--325},
  year={2020},
  organization={Springer}
}
```
## Contributions
If you have any questions/comments/bug reports, feel free to open a github issue or pull a request or e-mail to the author Yicheng Wu (wuyichengg@gmail.com).
