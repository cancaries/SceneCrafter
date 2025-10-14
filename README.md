<a id="readme-top"></a>

<div align="center">
  <img src="assets/scenecrafter_logo.png" alt="Logo" width="300">
  <h1 align="center">Unraveling the Effects of Synthetic Data on <br>End-to-End Autonomous Driving Humanoid Robots</h1>

<p align="center">
    <a href="https://scholar.google.com/citations?user=2oeqemsAAAAJ&hl=en"><strong>Junhao Ge*</strong></a><sup>1</sup>
    ,
    <a href="https://scholar.google.com/citations?user=0S38F-EAAAAJ&hl=en"><strong>Zuhong Liu*</strong></a><sup>1</sup>
    ,
    <a href="https://scholar.google.com/citations?user=nM5_jUIAAAAJ&hl=en"><strong>Longteng Fan*</strong></a><sup>1</sup>
    <a href="https://scholar.google.com/citations?user=2oeqemsAAAAJ&hl=zh-CN"><strong>Yifan Jiang*</strong></a><sup>1</sup>
    <br>
    <a href="https://scholar.google.com/citations?user=2oeqemsAAAAJ&hl=zh-CN"><strong>Jiaqi Su</strong></a><sup>1</sup>
    ,
    <a href="https://yimingli-page.github.io/"><strong>Yiming Li</strong></a><sup>2</sup>
    ,
    <a href="https://zhejz.github.io/"><strong>Zhejun Zhang</strong></a><sup>3</sup>
    ,
    <a href="https://siheng-chen.github.io/"><strong>Siheng Chen</strong></a><sup>1,4</sup>
    <br>
    <sup>1</sup>Shanghai Jiao Tong University <sup>2</sup>New York University 
    <br>
    <sup>3</sup>ETH Zurich <sup>4</sup>Shanghai Artificial Intelligence Laboratory
</p>
  <p>
    <!-- <a href="https://xdimlab.github.io/HUGSIM/">
      <img src="https://img.shields.io/badge/Project-Page-green?style=for-the-badge" alt="Project Page" height="20">
    </a> -->
    <a href="https://arxiv.org/abs/">
      <img src="https://img.shields.io/badge/arXiv-Paper-red?style=for-the-badge" alt="arXiv Paper" height="20">
    </a>
  </p>

  <img src="assets/scenecrafter_teaser.png" width="800" style="display: block; margin: 0 auto;">

  <br>

  <p align="left">
    This is the official project repository of the paper <b>Unraveling the Effects of Synthetic Data on End-to-End Autonomous Driving</b>.
  </p>
  
</div>

## TODOs
- [X] Arxiv Link
- [ ] Code Release
  - [X] Data Processing
  - [X] Data Generation
  - [X] Data Rendering
  - [ ] Closed-Loop Evaluation
- [X] Demo Data Relase
- [ ] Installation of this repo
- [ ] Demo script for demo
- [ ] Comment for Code
 
# SceneCrafter

A comprehensive scene generation and simulation framework for autonomous driving research.

## Overview

SceneCrafter is a powerful toolkit for generating realistic driving scenarios, simulating traffic behaviors, and rendering high-quality scene data for autonomous driving research and development.

## Installation

### Prerequisites

- Python 3.8+
- CUDA-enabled GPU (for rendering)
- pytorch 2.4.1+cu121

### Set up the conda environment
```
# git clone the repository
git clone https://github.com/cancaries/SceneCrafter.git
cd SceneCrafter

# Set conda environment
conda create -n scenecrafter python=3.8
conda activate scenecrafter

# Install torch (corresponding to your CUDA version)
pip install torch==2.4.1+cu121 torchvision==0.19.1+cu121 torchaudio==2.4.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# Install requirements
pip install -r requirements.txt

# Install SceneRenderer submodules(Street Gaussian)
pip install ./SceneRenderer/street-gaussian/submodules/diff-gaussian-rasterization
pip install ./SceneRenderer/street-gaussian/submodules/simple-knn
pip install ./SceneRenderer/street-gaussian/submodules/simple-waymo-open-dataset-reader
```

### Demo Data
We provide a demo data in [demo link](https://drive.google.com/file/d/1MEWgGKGfl3bf7MIQAwg_jhc-uMK4hWYU/view?usp=drive_link), please unzip it in the `root` folder. You can try the demo by running the following command:

```
python ./scripts/generate_data.sh
```


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use SceneCrafter in your research, please cite our work:

```bibtex
@misc{ge2025unravelingeffectssyntheticdata,
      title={Unraveling the Effects of Synthetic Data on End-to-End Autonomous Driving}, 
      author={Junhao Ge and Zuhong Liu and Longteng Fan and Yifan Jiang and Jiaqi Su and Yiming Li and Zhejun Zhang and Siheng Chen},
      year={2025},
      eprint={2503.18108},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2503.18108}, 
}
```

## Acknowledgements
SceneRenderer is built based on Street Gaussian. More details can be found in [Street Gaussian](https://github.com/zju3dv/street_gaussians).