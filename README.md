<a id="readme-top"></a>

<div align="center">
  <h1 style="font-size:3vw;padding:0;margin:0;display:inline">SNAP: Sigmoidal-Neuronal-Adaptive-Plasticity</h1>
  <h3 style="margin:0">SNAP is an approximation to Long-Term Potentiation for Artificial Neural Networks to reduce catastrophic forgetting.</h3>
  <a href="https://arxiv.org/abs/2410.15318"><strong>Read the paper»</strong></a>
</div>

<br />

<div align="center">

<a href="">[![Contributors][contributors-shield]][contributors-url]</a>
<a href="">[![Issues][issues-shield]][issues-url]</a>
<a href="">[![MIT License][license-shield]][license-url]</a>

</div>


## Getting Started

### Prerequisites

The project uses [uv](https://docs.astral.sh/uv/) to manage and lock project dependencies for a consistent and reproducible environment. If you do not have `uv` installed on your system, visit [this page](https://docs.astral.sh/uv/getting-started/installation/) for installation instructions.

**Note**: If you have `pip` you can just invoke:

```sh
pip install uv
```

### Installation

```sh
# Clone the repo
git clone git@github.com:sebastian9991/SNAP-Sigmoidal-Neuronal-Adaptive-Plasticity.git

# Enter the repo directory
cd SNAP-Sigmoidal-Neuronal-Adaptive-Plasticity

# Install core dependencies into an isolated environment
uv sync
```

## Usage

### Running Full SNAP Experiments
```sh
./run_all_experiments
```

## License

Distributed under the MIT License. See `LICENSE.txt` for more information.


## Citation

```
@article{xu2024snapstoppingcatastrophicforgetting,
      title={SNAP: Stopping Catastrophic Forgetting in Hebbian Learning with Sigmoidal Neuronal Adaptive Plasticity}, 
      author={Tianyi Xu and Patrick Zheng and Shiyan Liu and Sicheng Lyu and Isabeau Prémont-Schwarz},
      year={2024},
      eprint={2410.15318},
      archivePrefix={arXiv},
      primaryClass={cs.NE},
      url={https://arxiv.org/abs/2410.15318}, 
}
```


[contributors-shield]: https://img.shields.io/github/contributors/sebastian9991/SNAP-Sigmoidal-Neuronal-Adaptive-Plasticity.svg?style=for-the-badge
[contributors-url]: https://github.com/sebastian9991/SNAP-Sigmoidal-Neuronal-Adaptive-Plasticity/graphs/contributors
[issues-shield]: https://img.shields.io/github/issues/sebastian9991/SNAP-Sigmoidal-Neuronal-Adaptive-Plasticity.svg?style=for-the-badge
[issues-url]: https://github.com/sebastian9991/SNAP-Sigmoidal-Neuronal-Adaptive-Plasticity/issues
[license-shield]: https://img.shields.io/github/license/sebastian9991/SNAP-Sigmoidal-Neuronal-Adaptive-Plasticity.svg?style=for-the-badge
[license-url]: https://github.com/sebastian9991/SNAP-Sigmoidal-Neuronal-Adaptive-Plasticity/blob/master/LICENSE.txt
