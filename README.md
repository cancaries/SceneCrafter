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
  - [ ] Data Rendering
  - [ ] Closed-Loop Evaluation
- [ ] Demo Data Relase
- [ ] Installation of this repo
- [ ] Demo script for demo
- [ ] Comment for Code
 
# SceneCrafter

A comprehensive scene generation and simulation framework for autonomous driving research.

## Overview

SceneCrafter is a powerful toolkit for generating realistic driving scenarios, simulating traffic behaviors, and rendering high-quality scene data for autonomous driving research and development.

## Features

- **Scene Generation**: Create diverse driving scenarios with configurable parameters
- **Traffic Simulation**: Simulate realistic traffic flow and vehicle behaviors
- **3D Rendering**: High-quality scene rendering using advanced graphics techniques
- **Data Processing**: Utilities for processing and preparing simulation data
- **Waymo Integration**: Support for Waymo Open Dataset format

## Installation

### Prerequisites

- Python 3.8+
- CARLA Simulator (for full simulation capabilities)
- CUDA-enabled GPU (for rendering)

### Setup

```bash
# Clone the repository
git clone https://github.com/your-username/SceneCrafter.git
cd SceneCrafter

# Install dependencies
pip install -r requirements.txt

# Setup submodules (if applicable)
git submodule update --init --recursive
```

## Usage

### Basic Scene Generation

```python
from SceneController.scene import SceneGenerator
from SceneController.agents.navigation.tools.misc import build_transform_path_from_ego_pose_data

# Initialize scene generator
generator = SceneGenerator(config_path="config/scene_config/default.yaml")

# Generate a basic scene
scene = generator.generate_scene()

# Process ego vehicle data
ego_transforms = build_transform_path_from_ego_pose_data(scene.ego_pose_data)
```

### Traffic Simulation

```python
from SceneController.agents.map import MapManager
from SceneController.agents.navigation.tools.misc import detect_route_interaction

# Load map data
map_manager = MapManager("path/to/map/data")

# Simulate traffic flow
traffic_flow = map_manager.generate_traffic_flow()

# Check route interactions
interaction = detect_route_interaction(test_path, reference_path)
```

### Rendering

```python
from SceneRenderer.street_gaussian import SceneRenderer

# Initialize renderer
renderer = SceneRenderer(config_path="config/render_config/default.yaml")

# Render scene
rendered_scene = renderer.render(scene_data)
```

## Project Structure

```
SceneCrafter/
├── SceneController/          # Scene control and simulation
│   ├── agents/              # Autonomous agents
│   ├── config/              # Configuration files
│   ├── simulation/          # Simulation utilities
│   └── scripts/            # Utility scripts
├── SceneRenderer/           # Rendering components
│   ├── street-gaussian/    # Gaussian splatting renderer
│   └── street-gaussian_e2e/# End-to-end rendering pipeline
├── data_utils/             # Data processing utilities
│   ├── box_utils.py        # Bounding box operations
│   ├── cal_utils.py        # Calibration utilities
│   └── waymo_utils.py      # Waymo dataset utilities
└── demo/                   # Demo assets and examples
```

## Configuration

Configuration files are located in `SceneController/config/`:

- `agent_config/`: Agent behavior configurations
- `scene_config/`: Scene generation parameters

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use SceneCrafter in your research, please cite our work:

```bibtex
@software{scenecrafter2024,
  title = {SceneCrafter: A Comprehensive Scene Generation Framework},
  author = {SceneCrafter Team},
  year = {2024},
  url = {https://github.com/your-username/SceneCrafter}
}
```

## Support

For questions and support:

- Open an issue on GitHub
- Check the documentation (coming soon)
- Join our discussion forum

## Acknowledgments

- CARLA Simulator team for the excellent simulation platform
- Waymo Open Dataset for providing valuable autonomous driving data
- Gaussian Splatting community for advanced rendering techniques