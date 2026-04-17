# Sparsh: Self-supervised touch representations for vision-based tactile sensing


<p align="center">
Carolina Higuera<sup>*</sup>,
Akash Sharma<sup>*</sup>,
Chaithanya Krishna Bodduluri,
Taosha Fan,
Patrick Lancaster,
Mrinal Kalakrishnan,
Michael Kaess,
Byron Boots,
Mike Lambeta,
Tingfan Wu,
Mustafa Mukadam
</p>

<p align="center">
<sup>*</sup>Equal contribution
</p>

<p align="center">
    <a href=https://ai.facebook.com/research/ai-systems>AI at Meta, FAIR</a>;
    <a href=https://ri.cmu.edu/>The Robotics Institute, CMU</a>;
    <a href=https://www.washington.edu/>University of Washington</a>
</p>

<p align="center">
    <a href="https://ai.facebook.com/research/publications/sparsh-self-supervised-touch-representations-for-vision-based-tactile-sensing"><img src="http://img.shields.io/badge/Paper-PDF-red.svg"></img></a>
    <a href="https://arxiv.org/abs/2410.24090"><img src="https://img.shields.io/badge/arXiv-2410.24090-b31b1b.svg"></img></a>
    <a href="https://sparsh-ssl.github.io"><img src="http://img.shields.io/badge/Project-Page-blue.svg"></img></a>
    <a href="https://youtu.be/8q2BI5HePq0"><img src="http://img.shields.io/badge/Video-Link-green.svg"></img></a>
    <a href="https://huggingface.co/collections/facebook/sparsh-67167ce57566196a4526c328"><img src="https://img.shields.io/badge/Models%20and%20datasets-Link-yellow?logo=huggingface"></img></a>
    <a href="#-citing-sparsh"><img src="http://img.shields.io/badge/Cite-Us-orange.svg"></img></a>


</p>

<p align="center">
<img src="./assets/teaser.png" alt="drawing" width="700"/>
</p>
Sparsh is a family of general touch representations trained via self-supervision algorithms such as MAE, DINO and JEPA. Sparsh is able to generate useful representations for DIGIT, Gelsight'17 and Gelsight Mini. It outperforms end-to-end models in the downstream tasks proposed in TacBench by a large margin, and can enable data efficient training for new downstream tasks.


This repository contains the pytorch implementation, pre-trained models, and datasets released with Sparsh.

<p align="center">
<img src="assets/tacbench.gif" alt="animated" />
</p>


## 🛠️Installation and setup

Clone this repository:
```bash
git clone https://github.com/facebookresearch/sparsh.git
cd sparsh
```
and create a conda environment with dependencies:
```bash
mamba env create -f environment.yml
mamba activate tactile_ssl
```

## 🚀 Pretrained models

Pretrained model weights are available for download from our Hugging Face: [facebook/sparsh](https://huggingface.co/facebook/sparsh)

<table style="margin: auto">
  <thead>
    <tr>
      <th>model</th>
      <th>small</th>
      <th>base</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Sparsh (MAE)</td>
      <td><a href="https://huggingface.co/facebook/sparsh-mae-small">backbone</a></td>
      <td><a href="https://huggingface.co/facebook/sparsh-mae-base">backbone only</a></td>
    </tr>
    <tr>
      <td>Sparsh (DINO)</td>
      <td><a href="https://huggingface.co/facebook/sparsh-dino-small/">backbone</a></td>
      <td><a href="https://huggingface.co/facebook/sparsh-dino-base/">backbone</a></td>
    </tr>
    <tr>
      <td>Sparsh (DINOv2)</td>
      <td>:x:</td>
      <td><a href="https://huggingface.co/facebook/sparsh-dinov2-base/">backbone</a></td>
    </tr>
    <tr>
      <td>Sparsh (IJEPA)</td>
      <td><a href="https://huggingface.co/facebook/sparsh-ijepa-small/">backbone</a></td>
      <td><a href="https://huggingface.co/facebook/sparsh-ijepa-base/">backbone</a></td>
    </tr>
    <tr>
      <td>Sparsh (VJEPA)</td>
      <td><a href="https://huggingface.co/facebook/sparsh-vjepa-small/">backbone</a></td>
      <td><a href="https://huggingface.co/facebook/sparsh-vjepa-base/">backbone</a></td>
    </tr>
  </tbody>
</table>


## 📥 Datasets

### Pretraining datasets

For pretraining, we curate datasets from multiple sources containing unlabeled data from DIGIT and GelSight sensors.

For DIGIT, the dataset is a mixture of the [YCB-Slide dataset](https://github.com/rpl-cmu/YCB-Slide) and in-house collected data: [Touch-Slide](https://huggingface.co/datasets/facebook/touch-slide). It contains approximately 360k samples of tactile images with a diverse set of no-contact images or backgrounds.

For GelSight, we use open source datasets available online, [Touch and Go](https://touch-and-go.github.io/) and [ObjectFolder-Real](https://objectfolder.stanford.edu/objectfolder-real-download).

<!-- - [ ] @Carolina: Update with correct scripts -->

#### DIGIT

To download the dataset, please edit `path_dataset` in the bash script `scripts/download_digitv1_dataset.sh`. This will download and extract the data in `path_dataset` for both YCB-Slide and Touch-Slide datasets.

The structure of the dataset is:

```bash
digitv1/Object-Slide
├── object_0 # eg: 004_sugar_box
│   ├── dataset_0.pkl
    ├── dataset_1.pkl
    ├── dataset_2.pkl
    ├── dataset_3.pkl
    ├── dataset_4.pkl
├── object_1 # eg: bread
...
├── bgs
    ├── bg_0.jpg
    ...
    ├── bg_18.jpg
```

In the `bgs/` folder there are images of several no-contact images or backgrounds from different DIGIT sensors. This is necessary for pre-processing the data. Please add to these folder background images from your sensor in case you're adding new tactile data.

To load this dataset, use `tactile_ssl/data/vision_tactile.py`

#### GelSight dataset

We use [Touch and Go](https://touch-and-go.github.io/) to pretrain on GelSight'17 (with markers). The dataset consists of short videoclips making contact with in-the-wild objects. We use all frames from those videoclips, including no-contact frames. We do not perform any preprocessing since the markers contain relevant static shear information.

We also use sequences from the [ObjectFolder-Real](https://objectfolder.stanford.edu/objectfolder-real-download) dataset for pre-training. We preprocess the data by extracting only the tactile images (GelSight Mini), as we do not use the other modalities.

We provide a script to download the preprocessed and compatible version of these datasets with our pipeline. To do so, run the bash script `scripts/download_gelsight_dataset.sh`. This will download and extract the data. Don't forget to edit `path_dataset` in the script.


The structure of the dataset is:

```bash
gelsight/touch_go
    ├── 20220410_031604.pkl
    ├── 20220410_031843.pkl
    ├── ...
    ├── 20220607_133934.pkl
gelsight/object_folder
    ├── 001.pkl
    ├── 002.pkl
    ...
    ├── 051.pkl
```

If you would like to download the data directly from Touch and Go and ObjectFolder-Real, you can also do so. Data can be downloaded by running the bash scripts `scripts/download_datasets_scratch/download_gelsight_object_folder.sh` and `scripts/download_datasets_scratch/download_gelsight_touchgo.sh`. Then, you can process the data to make it compatible with our pipeline by running the Python scripts `scripts/download_datasets_scratch/compress_object_folder.py` and `scripts/download_datasets_scratch/compress_touch_go.py`. Please modify the corresponding paths in all scripts accordingly.

To load this dataset, use `tactile_ssl/data/vision_tactile.py`

### Downstream task datasets

We open-source the data that we collected in-house for force estimation, slip detection and pose estimation downstream tasks. The datasets can be downloaded from the Sparsh collection in Hugging Face:
- Force estimation and Slip detection: [DIGIT](https://huggingface.co/datasets/facebook/digit-force-estimation), [GelSight Mini](https://huggingface.co/datasets/facebook/gelsight-force-estimation)
- Pose estimation: [DIGIT](https://huggingface.co/datasets/facebook/digit-pose-estimation)

Please locate these datasets in a directory designated for hosting all downstream task datasets.

#### T1 Force estimation and T2 slip detection
This dataset contains paired tactile and force data, intended for use in predicting 3-axis normal and shear forces applied to the sensor's elastomer. We used three different indenter shapes to collect force-labeled data: hemisphere, sharp, and flat. To measure force ground truths, we employed the ATI nano17 force/torque sensor. The protocol consisted of applying a random normal load followed by a shear load, achieved by sliding the probe 2mm on the sensor's elastomer.

The dataset consists a collection of normal/shear load trajectories for each probe. The structure is as follows (example for DIGIT dataset):

```bash
T1_force/digit/sphere
├── batch_1
│   ├── dataset_digit_00.pkl
│   ├── ...
│   ├── dataset_digit_03.pkl
│   ├── dataset_slip_forces.pkl
├── batch_2
│   ├── ...
T1_force/digit/flat
├── batch_1
│   ├── dataset_digit_00.pkl
│   ├── ...
│   ├── dataset_digit_03.pkl
│   ├── dataset_slip_forces.pkl
│   ...
T1_force/digit/sharp
├── ....
```
For each batch:
- `dataset_digit_xy.pkl`: contains the binarized tactile images only.
- `dataset_slip_forces.pkl`: it's a dictionary where each key represents a sliding trajectory. Each trajectory has the corresponding force and slip labels.

To load this dataset (DIGIT and GelSight Mini), use `tactile_ssl/data/vision_based_forces_slip_probes.py`

#### T3 Pose estimation

This dataset contains time-synchronized pairs of DIGIT images and SE(3) object poses. In our setup, the robot hand is stationary with its palm facing downwards and pressing against the object on a table. The robot hand has DIGIT sensors mounted on the index, middle, and ring fingertips, all of which are in contact with the object. A human manually perturbs the object's pose by translating and rotating it in SE(2). We use tag tracking to obtain the object's pose. We collect data using two objects: a Pringles can and the YCB sugar box, both of which have a tag fixed to their top surfaces.

The dataset is a collection of sequences where a human manually perturbs the object's pose. We collect data using two objects: a Pringles can and the YCB sugar box. Each sequence corresponds to a pickle file containing the following labeled data:
- DIGIT tactile images for index, middle and ring fingers
- Object pose tracked from tag in format (x, y, z, qw, qx, qy, qz)
- Robot hand joint positions
- `object_index_rel_pose_n5`: the pose change within the last 5 samples as a transformation matrix. The object pose is with respect to the index finger.
- `object_middle_rel_pose_n5`: the pose change within the last 5 samples as a transformation matrix. The object pose is with respect to the middle finger.
- `object_ring_rel_pose_n5`: the pose change within the last 5 samples as a transformation matrix. The object pose is with respect to the ring finger.
We also provide reference (no contact) images for each of the DIGITs to facilitate pre-processing such as background subtraction.

```bash
T3_pose/digit/train
├── pringles
│   ├── bag_00.pkl
│   ├── ...
│   ├── bag_37.pkl
│   ├── bag_38.pkl
├── sugar
│   ├── ...
T3_pose/digit/test
├── pringles
│   ├── bag_00.pkl
│   ├── ...
│   ├── bag_05.pkl
│   ├── bag_06.pkl
├── sugar
│   ├── ...
T3_pose/digit/bgs
├── digit_index.png
├── digit_index.png
├── digit_index.png
```

To load this dataset use `tactile_ssl/data/vision_based_pose_probes.py`

#### T4 Grasp stability
We use the [Feeling of Success](https://sites.google.com/view/the-feeling-of-success/) dataset. It contains approximately 9k grasp trials over 100k objects using GelSight'17 sensors mounted on a parallel gripper.

You can download the data directly from the webpage or run the bash script `scripts/download_datasets_scratch/download_gelsight_feeling_success.sh` to download the data and the Python script `scripts/download_datasets_scratch/compress_feeling_success.py` to preprocess the dataset compatible with our pipeline. Please update the paths in the scripts accordingly.

#### T5 Textile recognition

Please download the [Clothing dataset](http://data.csail.mit.edu/active_clothing/Data_ICRA18.tar). The dataset consist of 4467 short video clips (10-25 frames), of a robot with a GelSight'17 grasping several types of textile (20  classes), such as leather, cotton, polyester, etc.

## 🏋️‍♂️ Training Sparsh

We use hydra for configuration management in this repository. The configuration files are located in `config/`.

The config folder is organized as follows:

```bash
├── config
│   ├── data # contains dataset configs
│   ├── experiment # contains full config for a specific experiment (eg. Sparsh(DINO) or downstream task)
│   ├── model # contains configs for each ssl algorithm
│   ├── paths # add your config here with paths to datasets / checkpoint / outputs / etc.
│   ├── task # contains downstream_task configs for each downstream task in TacBench
│   ├── wandb # add your wandb config here for experiment tracking
│   ├── default.yaml # The SSL training default config is overridden by experiment/dino_vit.yaml and the like
```

Following are the instructions to train a Sparsh model:

- Setup the pretraining datasets according to the instructions [above](#pretraining-datasets)
- add `paths/${YOUR_PATHS}.yaml` similar to existing examples and point to the data root accordingly
- similarly add `wandb/${YOUR_CONFIG}.yaml`
- Then choose an experiment, for example: `dino_vit.yaml` and use the following script

You may need to adjust batch size according to your GPU. All training experiments were done with 8 A100-80GB GPUs.

```bash
python train.py +experiment=${YOUR_EXP_NAME} paths=${YOUR_PATHS} wandb=${YOUR_CONFIG}
```

### Training downstream tasks

For training downstream tasks, in our paper we largely follow frozen evaluation, where we freeze the weights of the Sparsh encoder, and only train a lightweight decoder for each downstream task.
Training downstream tasks is quite similar to the above instructions but additionally requires a pre-trained model checkpoint `checkpoint_encoder` which can be specified by updating the `task.checkpoint_encoder` field in the config. Downstream tasks also need a labeled dataset for the corresponding downstream task.

Use the following script to train downstream tasks:
```bash
python train_task.py --config-name=experiment/downstream_task/${EXPERIMENT} paths=${YOUR_PATH_CONFIG} wandb=${YOUR_WANDB_CONFIG}
```

Once you complete training downstream task decoders for each task, you can also test the checkpoints using the `test_task.py` script which essentially follows the same format as above. For convenience, we also provide a `submit_task.sh` bash script to train and test downstream tasks if you're using a SLURM based cluster.

Finally, we also `tacbench_report.ipynb` where we compute metrics for all the downstream tasks, once the data is computed from the `test_task.py` script.

## 🤹‍♀️ Sparsh demo: force field visualization

<p align="center">
<img src="assets/demo_digit.gif" alt="animated" />
</p>

For testing Sparsh(DINO) + force field decoder live, you only need one DIGIT or GelSight Mini sensor. Follow these steps to run the demo:

1. Create a folder for downloading the task checkpoints. For example, `${YOUR_PATH}/outputs_sparsh/checkpoints`.
<!-- UPDATE THIS -->
2. Download the decoder checkpoints from Hugging Face for [DIGIT](https://huggingface.co/facebook/sparsh-digit-forcefield-decoder) and [GelSight Mini](https://huggingface.co/facebook/sparsh-gelsight-forcefield-decoder).
3. Connect the sensor to your PC. In case of DIGIT, please make sure you have [digit-interface](https://github.com/facebookresearch/digit-interface) installed.
4. Make sure the device is recognized by the OS (you can use Cheese in Linux to see the video that the sensor is streaming).

5. Running the demo for DIGIT:

```bash
python demo_forcefield.py +experiment=downstream_task/forcefield/digit_dino paths=${YOUR_PATH_CONFIG} paths.output_dir=${YOUR_PATH}/outputs_sparsh/checkpoints/ test.demo.digit_serial=${YOUR_DIGIT_SERIAL}`
```
The DIGIT serial number is printed on the back of the sensor and has the format `DXXXXX`.

6. Running the demo for GelSight Mini:

```bash
python demo_forcefield.py +experiment=downstream_task/forcefield/gelsight_dino paths=${YOUR_PATH_CONFIG} paths.output_dir=${YOUR_PATH}/outputs_sparsh/checkpoints/ test.demo.gelsight_device_id=${YOUR_GELSIGHT_VIDEO_ID}`
```

The GelSight Mini is recognized as a webcam. You can get the video ID by checking in a terminal `ls -l /dev/video*`.

7. Take the sensor and slide it across the edge of a table, or across objects with interesting textures! Look at the normal field to localize where you're making contact on the sensor's surface. Look at the shear field to gather an intuition about the direction of the shear force that you applied while sliding the sensor. For example, slide the sensor over an edge up and down to get translational shear or rotate the sensor in place to see torsional slip!


## License
This project is licensed under [LICENSE](LICENSE).


## 📚 Citing Sparsh

If you find this repository useful, please consider giving a star :star: and citation:
```
@inproceedings{
    higuera2024sparsh,
    title={Sparsh: Self-supervised touch representations for vision-based tactile sensing},
    author={Carolina Higuera and Akash Sharma and Chaithanya Krishna Bodduluri and Taosha Fan and Patrick Lancaster and Mrinal Kalakrishnan and Michael Kaess and Byron Boots and Mike  Lambeta and Tingfan Wu and Mustafa Mukadam},
    booktitle={8th Annual Conference on Robot Learning},
    year={2024},
    url={https://openreview.net/forum?id=xYJn2e1uu8}
}
```

## 🤝 Acknowledgements


We thank Ishan Misra, Mahmoud Assran for insightful discussions on SSL for vision that informed this work, and Changhao Wang, Dhruv Batra, Jitendra Malik, Luis Pineda, Tess Hellebrekers for helpful discussions on the research.


We also thank the team behind datasets like [YCB-Slide](https://github.com/rpl-cmu/YCB-Slide), [Touch and Go](https://touch-and-go.github.io/), [ObjectFolder-Real](https://objectfolder.stanford.edu/objectfolder-real-download), [Feeling of Success](https://sites.google.com/view/the-feeling-of-success/)  and [Clothing dataset](http://data.csail.mit.edu/active_clothing/Data_ICRA18.tar) for contributing to the research community by open-sourcing their tactile data.
