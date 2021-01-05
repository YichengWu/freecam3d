# FreeCam3D

### [Project](https://yichengwu.github.io/freecam3D/) | [1-min Summary](https://www.youtube.com/watch?v=QCOzSyEeniw) | [10-min TalkSummary](https://www.youtube.com/watch?v=-6AhHc_1sOs) | [Paper](https://drive.google.com/file/d/1P3_ZJYdp_VDuWOQaPuf_2xt0VGWCt7Jg/view?usp=sharing)

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
The pre-processed TFrecord files can be downloaded from [Google Drive](https://drive.google.com/drive/folders/18b1CamTQd6wf2o3kxfL5aqWtWopIDuVG?usp=sharing). It contains 5077 training patches, 553 validation patches, and 419 test patches.

## Train

To train the entire framework, simply run the following code.
```
python depth_estimation.py
```
Inside the code, `results_dir` is the output result directory, `DATA_PATH` is the directory of the downloaded dataset. The learning rate of the optical layer and digital network can be set individually using `lr_optical` and `lr_digital`. Detailed instruction about choosing the learning rate can be found in paper Sec. IIID(d).

### Logging

We use Tensorboard for logging training progress. Recommended: execute `tensorboard --logdir /path/to/save_dir --port 9001` and visit `localhost:9001` in the browser.

## Evaluation

Once the network is trained, the performance can be evaluated using the testing dataset. 
```
python depth_estimation_test.py
```
Change `results_dir` to the place you save your model. Once the testing is finished, a new folder called `test_all` will be created inside the model directory. It contains 400 scenes, and each one includes a clean image `sharp.png`, a coded image `blur.png`, an estimated disparity map `phiHat.png`, and a ground truth disparity map `phiGT.png`.

If you want to see the performance of our best result, please download from [Google Drive](https://drive.google.com/drive/folders/12zqZjkllc9IllSIloOSfJNlkDvHL46Hb?usp=sharing). Sample images are shown below.

<p align="center">
  <img width="500" src="/figures/PhaseCam3D_sim_results.png">
</p>

### Ablation study
We did a comprehensive ablation study by varying the learning rate, initialization, and loss. All the results are listed here. Please let us know if you outperform our results!

Exp.       | Learn mask   | Initialization       | Loss                  | Error (RMS)   |
-----------|--------------|----------------------|-----------------------|---------------|
A          | No           | No mask              | RMS                   | 2.69          |
B          | Yes          | Random               | RMS                   | 1.07          |
C          | No           | Fisher mask          | RMS                   | 0.97          |
D          | Yes          | Random               | RMS+CRLB              | 0.88          |
E          | Yes          | Fisher mask          | RMS                   | 0.74          |
F          | Yes          | Fisher mask          | RMS+CRLB              | 0.85          |
**G**      | **Yes**      | **Fisher mask**      | **RMS+gradient**      | **0.56**      |

## Citation
If you use this code for your research, please cite our papers.
```
@inproceedings{wu2019phasecam3d,
  title={PhaseCam3D—Learning Phase Masks for Passive Single View Depth Estimation},
  author={Wu, Yicheng and Boominathan, Vivek and Chen, Huaijin and Sankaranarayanan, Aswin and Veeraraghavan, Ashok},
  booktitle={2019 IEEE International Conference on Computational Photography (ICCP)},
  pages={1--12},
  year={2019},
  organization={IEEE}
}
```
## Contributions
If you have any questions/comments/bug reports, feel free to open a github issue or pull a request or e-mail to the author Yicheng Wu (wuyichengg@gmail.com).
