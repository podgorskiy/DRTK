
:github_url: https://github.com/facebookresearch/DRTK

DRTK – Differentiable Rendering Toolkit
=======================================

DRTK is a PyTorch library that provides functionality for differentiable rasterization.

.. The package consists of five main modules:
..
.. * :doc:`api_reference/transform`
.. * :doc:`api_reference/rasterize`
.. * :doc:`api_reference/render`
.. * :doc:`api_reference/interpolate`
.. * :doc:`api_reference/edge_grad_estimator`
..
.. There are other many other components, e.g. :doc:`api_reference/msi`, :doc:`api_reference/mipmap_grid_sampler`, :doc:`api_reference/grid_scatter`, etc, but they may

A typical flow looks like this:

**transform** → **rasterize** → **render** → **interpolate** → **CUSTOM SHADING** → **edge_grad** → **LOSS FUNCTION**

where:
- *transform*: projects the vertex positions from camera space to image space
- *rasterize*: performs rasterization, where pixels in the output image are associated with triangles
- *render*: computes depth and baricentric image
- *interpolate*: interpolates arbitrary vertex attributes
- *CUSTOM SHADING*: user implemented shading
- *edge_grad*: special module that computes gradients for the **rasterize** step, which is not differentiable on its own. For details, please see [**Rasterized Edge Gradients: Handling Discontinuities Differentiably**](https://arxiv.org/abs/2405.02508)

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

![hello triangle](docs/hellow_triangle.png)

## Dependencies
* PyTorch >= 2.1.0

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

See the [CONTRIBUTING](CONTRIBUTING.md) file for how to help out.

License
--------
DRTK is CC-BY-NC 4.0 licensed, as found in the [LICENSE](LICENSE) file.

Citation
--------

..  code-block:: none

    @article{Pidhorskyi2024RasterizedEG,
      title={Rasterized Edge Gradients: Handling Discontinuities Differentiably},
      author={Stanislav Pidhorskyi and Tomas Simon and Gabriel Schwartz and He Wen and Yaser Sheikh and Jason Saragih},
      journal={ArXiv},
      year={2024},
      volume={abs/2405.02508},
      url={https://api.semanticscholar.org/CorpusID:269604831}
    }


.. toctree::
   :glob:
   :maxdepth: 2
   :hidden:

   installation/index


.. toctree::
   :glob:
   :maxdepth: 2
   :hidden:

   examples/index

.. toctree::
   :glob:
   :maxdepth: 2
   :hidden:

   api_reference/index

.. toctree::
   :hidden:

    GitHub <https://github.com/facebookresearch/DRTK>

