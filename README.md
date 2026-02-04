<h3 align="center"> (ICLR 2026) <a href="https://gewu-lab.github.io/AnyTouch2/" style="color:#9C276A">
AnyTouch 2: General Optical Tactile Representation Learning For Dynamic Tactile Perception</a></h3>


<h5 align="center"> 🚀 Welcome to the repo of AnyTouch 2! If our project helps you, please give us a star ⭐ on GitHub to support us. 🙏 </h5>

<h5 align="center">


[![ArXiv](https://img.shields.io/badge/ICLR-<OpenReview>-<COLOR>.svg)](https://openreview.net/pdf?id=ndilONnABZ) [![hf_checkpoint](https://img.shields.io/badge/🤗-AnyTouch_2_Model-9C276A.svg)](https://huggingface.co/xxuan01/AnyTouch2-Model) [![hf_data](https://img.shields.io/badge/🤗-ToucHD_dataset-9C276A.svg)](https://huggingface.co/collections/BAAI/touchd) [![arXiv](https://img.shields.io/badge/Arxiv_(Coming_Soon)-AD1C18.svg?logo=arXiv)]() [![Webpage](https://img.shields.io/badge/Webpage-AnyTouch_2-<COLOR>.svg)](https://gewu-lab.github.io/AnyTouch2/) <br>

<img src="https://gewu-lab.github.io/AnyTouch2/asset/second.png" width="800" />



## 📑To Do

- [ ] Quick Start Demo Code
- [ ] Dataset Pre-processing
- [ ] Sparsh Evaluation Code



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
   pip install -r requirements.txt
   ```

   

## 🚀 Quick Start

1. Download AnyTouch 2 Model Checkpoints into `checkpoints/` **(Complete the [form](https://huggingface.co/xxuan01/AnyTouch2-Model) to get access first)**

   ```
   huggingface-cli download --repo-type model xxuan01/AnyTouch2-Model --local-dir checkpoints
   ```

2. Run `quick_start.sh` (Coming Soon)



## 🤖 Downstream Evaluation

1. Download [ToucHD (Force)](https://huggingface.co/datasets/BAAI/ToucHD-Force) **(Complete the [form](https://huggingface.co/datasets/BAAI/ToucHD-Force) to get access first)**, [Touch and Go](https://github.com/fredfyyang/Touch-and-Go/tree/main/Visuo-tactile%20contrastive%20learning) an [Cloth](http://data.csail.mit.edu/active_clothing/Data_ICRA18.tar) into `datasets/`

   ```
   ### Download ToucHD (Force). Please complete the form to get access first.
   huggingface-cli download --repo-type dataset xxuan01/BAAI/ToucHD-Force --local-dir datasets
   ```

2. Pre-process the datasets (Coming Soon)

3. Run scripts to start downstream training and evaluation

   ```
   ./run_probe_tag.sh
   ./run_probe_cloth.sh
   ./run_probe_touchd.sh
   ```

   





# Coming Soon!