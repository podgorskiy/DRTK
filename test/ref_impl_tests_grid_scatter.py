import logging
import os
import unittest
from typing import Callable, Dict, Optional, Tuple

import cv2 as cv
import numpy as np
import torch as th

from drtk import (grid_scatter, grid_scatter_ref, interpolate, rasterize,
                  render, transform)
from utils import load_obj


class RenderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.logger = logging.getLogger("drtk.render")

        mesh = load_obj(os.path.join(os.path.dirname(__file__), "bunny.obj"))
        v, vi, vt, vti = mesh["v"], mesh["vi"], mesh["vt"], mesh["vti"]
        b = 3
        image_height, image_width = 512, 512

        camera_distance = 0.5
        a1 = np.pi
        a2 = 1
        a3 = 0
        offset = np.array([0, 0.1, 0])

        def rodrigues(x):
            x, _ = cv.Rodrigues(np.asarray(x, dtype=np.float64))
            return np.float64(x)

        # camera extrinsics
        t = np.array([0, 0, camera_distance], dtype=np.float64)
        R = np.matmul(
            np.matmul(rodrigues([a1, 0, 0]), rodrigues([0, a2, 0])),
            rodrigues([0, 0, a3]),
        )

        cam_pos = (
            th.as_tensor(
                (R.transpose(-1, -2) @ -t[..., None])[..., 0] + offset, dtype=th.float32
            )
            .cuda()[None]
            .expand(b, -1)
        )
        cam_rot = th.as_tensor(R, dtype=th.float32).cuda()[None].expand(b, -1, -1)

        # camera intrinsics
        focal = (
            th.as_tensor(
                [[2 * image_width, 0.0], [0.0, 2 * image_height]], dtype=th.float32
            )
            .cuda()[None]
            .expand(b, -1, -1)
        )
        princpt = (
            th.as_tensor(
                [image_width / 2, image_height / 2],
                dtype=th.float32,
            )
            .cuda()[None]
            .expand(b, -1)
        )

        v_pix = transform(v, cam_pos, cam_rot, focal, princpt)
        index_img = rasterize(v_pix, vi, image_height, image_width)
        _, bary_img = render(v_pix, vi, index_img)

        # compute vt image
        vt_img = interpolate(
            vt.mul(2.0).sub(1.0)[None].expand(b, -1, -1), vti, index_img, bary_img
        )

        # mask image
        mask: th.Tensor = (index_img != -1)[:, None]

        self.vt_img = vt_img.permute(0, 2, 3, 1)
        self.input = th.rand_like(vt_img)
        self.vt_img.requires_grad_(True)
        self.input.requires_grad_(True)

        self.mask = mask
        self.vi = vi

    def _test(
        self,
        error_tolerance: Dict[str, float],
    ) -> None:
        """Test single config set"""

        compare(
            self.logger,
            error_tolerance,
            self.vt_img,
            self.mask,
            self.input,
            grid_scatter_ref,
            grid_scatter,
        )

    def test_interpolate(self) -> None:
        error_tolerance = {
            "forward_error_max": 5e-6,
            "forward_error_median": 1e-9,
            "backward_error_max": 5e-6,
            "backward_error_median": 1e-8,
        }
        self._test(error_tolerance)


def get_error(x: th.Tensor, ref: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
    mean = (x - ref).abs().mean() / (ref.amax() - ref.amin() + 1e-12)
    median = (x - ref).abs().median() / (ref.amax() - ref.amin() + 1e-12)
    max = (x - ref).abs().amax() / (ref.amax() - ref.amin() + 1e-12)
    return mean, median, max


def combine_error_str(mean: th.Tensor, median: th.Tensor, max: th.Tensor) -> str:
    return ", ".join(
        [
            f"{s}: {x.item():.4}"
            for x, s in zip([mean, median, max], ["mean", "median", "max"])
        ]
    )


def check(
    median: th.Tensor,
    max: th.Tensor,
    error_tolerance: Dict[str, float],
    prefix: str,
) -> None:
    assert (
        max < error_tolerance[f"{prefix}_error_max"]
    ), f"got max error {max} > {error_tolerance[f'{prefix}_error_max']}"
    assert (
        median < error_tolerance[f"{prefix}_error_median"]
    ), f"got median error {median} > {error_tolerance[f'{prefix}_error_median']}"


def run_forward_and_backward(
    vt_img: th.Tensor,
    mask: th.Tensor,
    input: th.Tensor,
    func: Callable[
        [th.Tensor, th.Tensor, int, int, Optional[str], Optional[str], Optional[bool]],
        th.Tensor,
    ],
    name: str,
) -> Tuple[th.Tensor, Optional[th.Tensor], Optional[th.Tensor], float, float]:
    start = th.cuda.Event(enable_timing=True)
    end = th.cuda.Event(enable_timing=True)

    vt_img.grad = None
    input.grad = None

    th.cuda.synchronize()
    start.record()

    with th.profiler.record_function(f"[{name}] render"):
        out = func(input, vt_img, 512, 512, align_corners=False) * mask

    end.record()

    th.cuda.synchronize()
    fwd_time: float = start.elapsed_time(end)

    th.cuda.manual_seed(1)
    out_grad = th.randn_like(out)

    start.record()
    with th.profiler.record_function(f"[{name}] render backward"):
        out.backward(out_grad)
    end.record()
    th.cuda.synchronize()
    bwd_time: float = start.elapsed_time(end)

    return out, vt_img.grad, input.grad, fwd_time, bwd_time


def compare(
    logger: logging.Logger,
    error_tolerance: Dict[str, float],
    vt_img: th.Tensor,
    mask: th.Tensor,
    input: th.Tensor,
    func_torch: Callable[
        [th.Tensor, th.Tensor, th.Tensor], Tuple[th.Tensor, th.Tensor]
    ],
    func_cuda: Callable[[th.Tensor, th.Tensor, th.Tensor], Tuple[th.Tensor, th.Tensor]],
) -> None:
    n_steps = 3
    skip_first = 1

    timings = []

    out_pytorch = None
    grid_grad_pytorch = None
    input_grad_pytorch = None
    out_cuda = None
    grid_grad_cuda = None
    input_grad_cuda = None

    for i in range(n_steps):
        (
            out_pytorch,
            grid_grad_pytorch,
            input_grad_pytorch,
            pytorch_fwd_time,
            pytorch_bwd_time,
        ) = run_forward_and_backward(vt_img, mask, input, func_torch, "PyTorch")

        (
            out_cuda,
            grid_grad_cuda,
            input_grad_cuda,
            cuda_fwd_time,
            cuda_bwd_time,
        ) = run_forward_and_backward(vt_img, mask, input, func_cuda, "CUDA")

        if i >= skip_first:
            timings.append(
                (pytorch_fwd_time, pytorch_bwd_time, cuda_fwd_time, cuda_bwd_time)
            )

    assert out_pytorch is not None and grid_grad_pytorch is not None
    assert out_cuda is not None and grid_grad_cuda is not None

    logger.info("\tTimings: ")
    timings = np.median(np.asarray(timings), axis=0)
    logger.info(
        f"\t\t forward: pytorch {timings[0]:0.6}ms,  cuda {timings[2]:0.6}ms, relative speedup: {timings[0] / timings[2]:0.3f}"
    )
    logger.info(
        f"\t\t backward: pytorch {timings[1]:0.6}ms,  cuda {timings[3]:0.6}ms, relative speedup: {timings[1] / timings[3]:0.3f}"
    )

    logger.info("\tErrors: ")
    mean, median, max = get_error(out_cuda, out_pytorch)
    logger.info(f"\t\t forward error: {combine_error_str(mean, median, max)}")
    check(median, max, error_tolerance, "forward")

    logger.info("\tBackward: ")
    assert grid_grad_cuda is not None
    mean, median, max = get_error(grid_grad_cuda, grid_grad_pytorch)
    logger.info(f"\t\t backward error wrt grid: {combine_error_str(mean, median, max)}")
    check(median, max, error_tolerance, "backward")

    logger.info("\tBackward: ")
    assert grid_grad_cuda is not None
    mean, median, max = get_error(input_grad_cuda, input_grad_pytorch)
    logger.info(
        f"\t\t backward error wrt input: {combine_error_str(mean, median, max)}"
    )
    check(median, max, error_tolerance, "backward")


if __name__ == "__main__":
    import sys

    logging.basicConfig(stream=sys.stderr)
    logging.getLogger("drtk.render").setLevel(logging.INFO)
    unittest.main()
