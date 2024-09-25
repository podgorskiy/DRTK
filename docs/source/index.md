---
myst:
  html_meta:
    "description lang=en": |
      Top-level documentation for DRTK.
html_theme.sidebar_secondary.remove: true
---

# DRTK – Differentiable Rendering Toolkit

DRTK is a PyTorch library that provides functionality for differentiable rasterization.

A typical flow looks like this:

{bdg-primary}`transform` → {bdg-primary}`rasterize` → {bdg-primary}`render` → {bdg-primary}`interpolate` → {bdg-warning}`CUSTOM SHADING` → {bdg-primary}`edge_grad` → {bdg-warning}`LOSS FUNCTION`

where DRTK package provides:
- {bdg-primary}`transform`  : projects the vertex positions from camera space to image space
- {bdg-primary}`rasterize` : performs rasterization, where pixels in the output image are associated with triangles
- {bdg-primary}`render` : computes depth and baricentric image
- {bdg-primary}`interpolate` : interpolates arbitrary vertex attributes
- {bdg-primary}`edge_grad` : module for computing gradients at discontinuities. For details refer to [**Rasterized Edge Gradients: Handling Discontinuities Differentiably**](https://arxiv.org/abs/2405.02508)

While the following, typically implemented by the User:
- {bdg-warning}`CUSTOM SHADING` : user implemented shading
- {bdg-warning}`LOSS FUNCTION` : user implemented loss function

## Hello Triangle

The "Hello Triangle" with DRTK would look like this:
```python
import drtk
import torch as th
from torchvision.utils import save_image  # to save images

# create vertex buffer of shape [1 x n_vertices x 3], here for triangle `n_vertices` == 3
v = th.as_tensor([[[0, 511, 1], [255, 0, 1], [511, 511, 1]]]).float().cuda()

# create index buffer
vi = th.as_tensor([[0, 1, 2]]).int().cuda()

# rasterize
index_img = drtk.rasterize(v, vi, height=512, width=512)

# compute baricentrics
_, bary = drtk.render(v, vi, index_img)

# we won't do shading, we'll just save the baricentrics and filter out the empty region
# which is marked with `-1` in `index_img`
img = bary * (index_img != -1)

save_image(img, "render.png")
```

![hello triangle](/_static/hellow_triangle.png)

## Dependencies
Cure dependencies:
* PyTorch >= 2.1.0
* numpy

Some examples and tests, may additionally require:
* torchvision
* opencv-python

## Building
To build a wheel and install it:
```
pip install wheel
python setup.py  bdist_wheel
pip install dist/drtk-<wheel_name>.whl
```

To build inplace, which is useful for package development:
```
python setup.py build_ext --inplace -j 1
```

## Contributing
See the [CONTRIBUTING](https://github.com/facebookresearch/DRTK//blob/main/CONTRIBUTING.md) file for how to help out.

## License
DRTK is CC-BY-NC 4.0 licensed, as found in the [LICENSE](https://github.com/facebookresearch/DRTK//blob/main/LICENSE) file.

## Citation
When using DRTK in academic projects, please cite:
```bibtex
@article{pidhorskyi2024rasterized,
  title={Rasterized Edge Gradients: Handling Discontinuities Differentiably},
  author={Pidhorskyi, Stanislav and Simon, Tomas and Schwartz, Gabriel and Wen, He and Sheikh, Yaser and Saragih, Jason},
  journal={arXiv preprint arXiv:2405.02508},
  year={2024}
}
```

```{toctree}
:glob:
:maxdepth: 2
:hidden:

installation/index
```

```{toctree}
:glob:
:maxdepth: 2
:hidden:

examples/index
```

```{toctree}
:glob:
:maxdepth: 2
:hidden:

api_reference/index
```

```{toctree}
:hidden:

GitHub <https://github.com/facebookresearch/DRTK>
```

