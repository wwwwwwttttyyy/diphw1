# Assignment 2 - DIP with PyTorch

### In this assignment, you will implement traditional DIP (Poisson Image Editing) and deep learning-based DIP (Pix2Pix) with PyTorch.

### Resources:
- [Assignment Slides](https://pan.ustc.edu.cn/share/index/66294554e01948acaf78)
- [Paper: Poisson Image Editing](https://www.cs.jhu.edu/~misha/Fall07/Papers/Perez03.pdf)
- [Paper: Image-to-Image Translation with Conditional Adversarial Nets](https://phillipi.github.io/pix2pix/)
- [Paper: Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1411.4038)
- [PyTorch Installation & Docs](https://pytorch.org/)

---

### 1. Implement Poisson Image Editing with PyTorch.

Fill the [Polygon to Mask function](run_blending_gradio.py#L95) and the [Laplacian Distance Computation](run_blending_gradio.py#L115) in `run_blending_gradio.py`.

Run:

```bash
python run_blending_gradio.py
```

Usage:
1. Upload a foreground image.
2. Click several points to define a polygon region.
3. Click `Close Polygon`.
4. Upload a background image.
5. Adjust `dx` and `dy` to move the selected region.
6. Click `Blend Images`.

Example image pairs are provided in `data_poisson/`.

---

### 2. Pix2Pix implementation.

Fill the network definition in `Pix2Pix/FCN_network.py`, then prepare a dataset and run training in the `Pix2Pix/` folder.

Recommended workflow:

```bash
cd Pix2Pix
python train.py
```

The default training script reads:
- `train_list.txt`
- `val_list.txt`

The provided dataset loader expects paired images stored as a single image, with the input on the left half and the target on the right half.

Datasets:
- `facades` can be used for a quick baseline.
- `maps` or other official pix2pix datasets can be used for more training data.

Training outputs:
- `Pix2Pix/train_results/`
- `Pix2Pix/val_results/`
- `Pix2Pix/checkpoints/`

See the [Pix2Pix subfolder](Pix2Pix/) for dataset preparation details.

---

### Requirements:
- Please configure the environment by yourself. Using a [conda environment](https://docs.anaconda.com/miniconda/) is recommended.
- Install PyTorch according to your CUDA version, then install other required packages such as `opencv-python`, `gradio`, `pillow`, and `numpy`.
- Please submit your code, result images, and a short markdown report.

---

### Suggested submission contents:
- Source code with completed functions.
- Poisson blending results on at least one or two examples.
- Pix2Pix training and validation result images.
- A short report including method, setup, results, and brief analysis.
