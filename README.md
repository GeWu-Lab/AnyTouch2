<h1 align="center"> (ICLR 2026) <a href="https://gewu-lab.github.io/AnyTouch2/" style="color:#9C276A">
AnyTouch 2: General Optical Tactile Representation Learning For Dynamic Tactile Perception</a></h1>



<h5 align="center"> 🚀 Welcome to the repo of AnyTouch 2! If our project helps you, please give us a star ⭐ on GitHub to support us. 🙏 </h5>

<h5 align="center">


[![arXiv](https://img.shields.io/badge/ICLR-<OpenReview>-<COLOR>.svg)](https://openreview.net/pdf?id=ndilONnABZ) [![hf_checkpoint](https://img.shields.io/badge/🤗-AnyTouch_2_Model-9C276A.svg)](https://huggingface.co/xxuan01/AnyTouch2-Model) [![hf_data](https://img.shields.io/badge/🤗-ToucHD_dataset-9C276A.svg)](https://huggingface.co/collections/BAAI/touchd) [![arXiv](https://img.shields.io/badge/Arxiv-2602.09617-AD1C18.svg?logo=arXiv)](https://arxiv.org/abs/2602.09617) [![Webpage](https://img.shields.io/badge/Webpage-AnyTouch_2-<COLOR>.svg)](https://gewu-lab.github.io/AnyTouch2/) <br>

<img src="https://gewu-lab.github.io/AnyTouch2/asset/second.png" width="800" />



## 📑To Do

- [x] Quick Start Demo Code
- [x] Dataset Pre-processing
- [x] Sparsh Evaluation Code
- [ ] Real-world Code

**[2026/4/17]** We have updated the Sparsh evaluation code. Please pull the latest changes from the repository using `git pull`.



## 🛠️ Requirements and Installation

1. Create Environment

   ```
   conda create -n anytouch2 python=3.9
   conda activate anytouch2
   ```

2. Install PyTorch 2.4.0 + Cuda 12.4

   ```
   pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu124
   ```

3. Install other required packages:

   ```
   git clone https://github.com/GeWu-Lab/AnyTouch2.git
   cd AnyTouch2
   pip install -r requirements.txt
   ```
   
   

## 🚀 Quick Start

1. Download AnyTouch 2 Model Checkpoints into `checkpoints/` **(Complete the [form](https://huggingface.co/xxuan01/AnyTouch2-Model) to get access first)**

   ```
   huggingface-cli download --repo-type model xxuan01/AnyTouch2-Model --local-dir checkpoints
   ```

   Checkpoint Performance:

   |                |            |    TAG    |   Cloth   |               Slip / Delta Force (Sparsh)                |             Force (Sparsh)             |             Force (ToucHD)              |
   | :------------: | :--------: | :-------: | :-------: | :------------------------------------------------------: | :------------------------------------: | :-------------------------------------: |
   | **num_frames** | **stride** |   Acc ↑   |   Acc ↑   |                   F1 Score ↑ / RMSE ↓                    |                 RMSE ↓                 |                 RMSE ↓                  |
   |       4        |     2      | **76.97** | **42.31** | **86.66** / 87.80 (DG)<br />**97.96** / **80.83** (Mini) | **624.26** (DG)<br />**202.14** (Mini) | **894.32** (DG)<br />**1051.03** (Mini) |
   |       2        |     6      |   74.15   |   40.76   |     86.60 / **83.15** (DG)<br />97.85 / 89.21 (Mini)     |     643.91 (DG)<br />208.41 (Mini)     |    1076.33 (DG)<br />1311.27 (Mini)     |

2. Run `quick_start.sh` (Coming Soon)

   ```
   bash scripts/quick_start.sh
   ```

   



## 🤖 Downstream Evaluation

### ToucHD Bench and Object Bench

1. Download [ToucHD (Force)](https://huggingface.co/datasets/BAAI/ToucHD-Force) **(Complete the [form](https://huggingface.co/datasets/BAAI/ToucHD-Force) to get access first)**, [Touch and Go](https://github.com/fredfyyang/Touch-and-Go/tree/main/Visuo-tactile%20contrastive%20learning) an [Cloth](http://data.csail.mit.edu/active_clothing/Data_ICRA18.tar) into `datasets/`

   ```
   ### Download ToucHD (Force). Please complete the form to get access first.
   huggingface-cli download --repo-type dataset xxuan01/BAAI/ToucHD-Force --local-dir datasets
   ```

2. Pre-process the datasets

   ```sh
   cd datasets/ToucHD-Force
   for f in *.zip; do
     unzip "$f" -d "${f%.zip}"
   done
   ```

3. Run scripts to start downstream training and evaluation

   ```
   bash scripts/run_probe_tag.sh
   bash scripts/run_probe_cloth.sh
   bash scripts/run_probe_touchd.sh
   ```



### Sparsh Bench

1. Download [Sparsh datasets](https://huggingface.co/collections/facebook/sparsh) into `datasets/`

   ```
   huggingface-cli download --repo-type dataset facebook/gelsight-force-estimation --local-dir datasets
   huggingface-cli download --repo-type dataset facebook/digit-force-estimation --local-dir datasets
   huggingface-cli download --repo-type dataset facebook/digit-pose-estimation --local-dir datasets
   ```

2. Rename the dataset folders by removing '-estimation' (e.g. gelsight-force-estimation -> gelsight-force)

3. Run scripts to start downstream training and evaluation

   ```
   bash sparsh/run_task.sh
   ```




## 📑 Citation

```
@inproceedings{fenganytouch2,
  title={AnyTouch 2: General Optical Tactile Representation Learning For Dynamic Tactile Perception},
  author={Feng, Ruoxuan and Zhou, Yuxuan and Mei, Siyu and Zhou, Dongzhan and Wang, Pengwei and Cui, Shaowei and Fang, Bin and Yao, Guocai and Hu, Di},
  booktitle={The Fourteenth International Conference on Learning Representations}
}
```



# Coming Soon!
