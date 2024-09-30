
:github_url: https://github.com/facebookresearch/DRTK

Installation
===================================

At the moment, we do not distribute pre-compiled binaries. Current DRTK version is |version|.

Prerequisites:

* PyTorch >= 2.1.0
* CUDA Toolkit

Optionally, we would also recommend installing the following packages in order to run tests and examples:

* torchvision
* opencv_python

Installation from github using pip
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The easiest way to install, is using ``pip`` directly from github repo:

.. code-block:: shell

    # To install latest
    pip install git+https://github.com/facebookresearch/DRTK.git

.. code-block:: shell

    # To install stable
    pip install git+https://github.com/facebookresearch/DRTK.git@stable

It may take significant amount of time to compile. In most cases this should be enough, given that PyTorch,
CUDA Toolkit, and Build Essentials for your platform are installed and environment is correctly configured.

Specifying architectures
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If the CUDA arch of the device where the code will be running is known, then it would be better to specify it directly, e.g.:

.. code-block:: shell

    # TORCH_CUDA_ARCH_LIST can use "named" architecture, see table below
    TORCH_CUDA_ARCH_LIST="Ampere" install git+https://github.com/facebookresearch/DRTK.git

or

.. code-block:: shell

    # TORCH_CUDA_ARCH_LIST can combine several architectures separated with semicolon or space.
    # Add `+PTX` if you want also to save intermediate byte code for better compatibility.
    TORCH_CUDA_ARCH_LIST="8.0;8.6+PTX" install git+https://github.com/facebookresearch/DRTK.git

which is the same.

If ``TORCH_CUDA_ARCH_LIST`` is not specified, the following architectures will be built by default: 7.2, 7.5, 8.0, 8.6, 9.0.

``TORCH_CUDA_ARCH_LIST`` can either take one or more values (combined with plus `+` symbol) from a list of named architecture:

.. list-table:: Named architectures
   :header-rows: 1

   * - Name
     - Arch
   * - Kepler+Tesla
     - 3.7
   * - Kepler
     - 3.5+PTX
   * - Maxwell+Tegra
     - 5.3
   * - Maxwell
     - 5.0;5.2+PTX
   * - Pascal
     - 6.0;6.1+PTX
   * - Volta+Tegra
     - 7.2
   * - Volta
     - 7.0+PTX
   * - Turing
     - 7.5+PTX
   * - Ampere+Tegra
     - 8.7
   * - Ampere
     - 8.0;8.6+PTX
   * - Ada
     - 8.9+PTX
   * - Hopper
     - 9.0+PTX

or can take one or more values (combined into a one string with semicolon `;`) from the list of supported architecture list: ``'3.5', '3.7', '5.0', '5.2', '5.3', '6.0', '6.1', '6.2','7.0', '7.2', '7.5', '8.0', '8.6', '8.7', '8.9', '9.0', '9.0a'``.

See more information about ``TORCH_CUDA_ARCH_LIST`` see `PyTorch docs <https://pytorch.org/docs/stable/cpp_extension.html#torch.utils.cpp_extension.CUDAExtension>`_ and  `source code on github <https://github.com/pytorch/pytorch/blob/c9653bf2ca6dd88b991d71abf836bd9a7a1d9dc3/torch/utils/cpp_extension.py#L1980>`_

Installing from a cloned repository
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Alternatively, you can install the package from a local clone of the repository.
This can be handy, in case some adjustments are needed to the code of the package.

Clone the repository and ``cd`` into it:

.. code-block:: shell

    git clone https://github.com/facebookresearch/DRTK
    cd DRTK

Then install package from path using ``pip``. Note the ``--no-build-isolation`` flag, it is needed for modern build
system to disable building in a separate clean python environment.
The reason is that it will install a default ``torch`` version from pip, which likely will not match the one already installed in the system (due to usage of ``--index-url ``).

.. code-block:: shell

    pip install . --no-build-isolation


Building and installing a wheel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To build a wheel run:

.. code-block:: shell

    # You might need to install build first
    # pip install --upgrade build
    python -m build --wheel --no-isolation

Alternatively, you can use the deprecated CLI of setuptools:

.. code-block:: shell

    # You might need to install wheel first, though newer versions of setuptools do not require it anymore.
    # pip install --upgrade wheel
    python setup.py bdist_wheel

Then you will find a wheel in the ``dist/`` folder. You can install this wheel by running:

.. code-block:: shell

    pip install dist/drtk-<tags>.whl

where ``<tags>`` are compatibility tags. You can figure them out by listing the ``dist/`` directory. E.g.:

.. code-block:: shell

    pip install dist/drtk-0.1.0-cp310-cp310-linux_x86_64.wh

Reinstalling the wheel
^^^^^^^^^^^^^^^^^^^^^^

If you had already previously installed pip, unless the version was increase, ``pip``` will not reinstall the package.
If you are modifing package locally, that would be undesired behaivour, and in order to force reinstall you would need to add
``-upgrade --force-reinstall --no-deps`` arguments, e.g.:

.. code-block:: shell

    pip install --upgrade --force-reinstall --no-deps .

Inplace build
^^^^^^^^^^^^^

For package development, it can be very useful to do an inplace build with:

.. code-block:: shell

    # There can be issues with concurrent build jobs, it is safer to specify `-inplace -j 1`
    python setup.py build_ext --inplace -j 1

then you can use the root of the cloned repository as a working directory, and you should be able to do ``import drtk``, run tests and examples.

Trouble shooting
^^^^^^^^^^^^^^^^

If after installation, during the attempt of using the package the following error occurs:

    RuntimeError: CUDA error: no kernel image is available for execution on the device

that means that the CUDA code was not build for the arch where the code was running. The best way to resolve it using
``TORCH_CUDA_ARCH_LIST`` as in example above.

------------------

If during the build or the attempt of using the package the following or similar error occurs:

    ImportError: libcudnn.so.9: cannot open shared object file: No such file or directory

then it is likely due to build isolation. Since we do not ditribute binaries at the moment, it is hard to get version mismatch otherwise.
Make sure that you include ``--no-build-isolation`` argument.

------------------

Errors like

    error: no suitable conversion function from "const __half" to "unsigned short" exists

Typically means that there is compiler version mismatch, it is likely that the version is too old.

------------------

Errors like

    ... aten/src/ATen/core/boxing/impl/boxing.h:41:105: error: expected primary-expression before ‘>’ token

are related to some problematic SFINAE logic in templates. This issue persisted in some recent PyTorch versions,
and there was suggested remidation in https://github.com/pytorch/pytorch/issues/122169 which is to add ``-std=c++20`` to ``nvcc`` arguments.

That is why there is this line in ``setup.py``:

.. code-block:: python

    nvcc_args.append("-std=c++20")

With some older versions of CUDA Toolkit, that may cause ``unrecognized command line option '-std=c++20'`` error.
