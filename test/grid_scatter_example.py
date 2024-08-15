import cv2
from typing import Optional
import numpy as np
import torch as th
from drtk import grid_scatter, grid_scatter_ref, interpolate, rasterize, render, transform
from drtk.utils import vert_normals
from torch.nn import functional as thf
from typedio.typed.file.obj_io_nonprod import load_obj
from torchvision.utils import save_image

mesh = load_obj("bunny.obj")
v, vi, vt, vti = mesh["v"], mesh["vi"], mesh["vt"], mesh["vti"]
v = th.as_tensor(v, dtype=th.float32)[None, ...].cuda()
vi = th.as_tensor(vi, dtype=th.int32).cuda()
vt = th.as_tensor(vt, dtype=th.float32).cuda()
vti = th.as_tensor(vti, dtype=th.int32).cuda()

image_height, image_width = int(1024 * 1.5), 1024

camera_distance = 0.46
a1 = np.pi
a2 = 1
a3 = 0
offset = np.array([0, 0.1, -0.02])


def rodrigues(x):
    x, _ = cv2.Rodrigues(np.asarray(x, dtype=np.float64))
    return np.float64(x)


# camera extrinsics
t = np.array([0, 0, camera_distance], dtype=np.float64)
R = np.matmul(np.matmul(rodrigues([a1, 0, 0]), rodrigues([0, a2, 0])), rodrigues([0, 0, a3]))
cam_pos = th.as_tensor((R.transpose(-1, -2) @ -t[..., None])[..., 0] + offset, dtype=th.float32).cuda()[None]
cam_rot = th.as_tensor(R, dtype=th.float32).cuda()[None]

# camera intrinsics
focal = th.as_tensor([[2 * image_height, 0.0], [0.0, 2 * image_height]], dtype=th.float32).cuda()[None]
princpt = th.as_tensor(
        [image_width / 2, image_height / 2],
        dtype=th.float32,
).cuda()[None]

# compute normals
normals = vert_normals(v, vi[None].long())

def shade(vn_img: th.Tensor, light_dir: th.Tensor, ambient_intensity: float = 1.0, direct_intensity: float = 1.0, shadow_img: Optional[th.Tensor] = None):
    ambient = (vn_img[:, 1:2] * 0.5 + 0.5) * th.as_tensor([0.45, 0.5, 0.7]).cuda()[None, :, None, None]
    direct = th.sum(vn_img.mul(thf.normalize(light_dir, dim=1)), dim=1, keepdim=True).clamp(min=0.0) * th.as_tensor([0.65, 0.6, 0.5]).cuda()[None, :, None, None]
    if shadow_img is not None:
        direct = direct * shadow_img
    return th.pow(ambient * ambient_intensity + direct * direct_intensity, 1 / 2.2)

v_pix = transform(v, cam_pos, cam_rot, focal, princpt)
index_img = rasterize(v_pix, vi, image_height, image_width)
_, bary_img = render(v_pix, vi, index_img)

# mask image
mask: th.Tensor = (index_img != -1)[:, None]

# compute vt image
vt_img = interpolate(vt.mul(2.0).sub(1.0)[None], vti, index_img, bary_img)

# compute normals
vn_img = interpolate(normals, vi, index_img, bary_img)

diffuse = shade(vn_img, th.as_tensor([0.5, 0.5, 0.0]).cuda()[None, :, None, None])* mask

save_image(diffuse, "diffuse.png")

texel_weight = grid_scatter(mask.float(), vt_img.permute(0, 2, 3, 1), upsampling_factor=32, output_width=512, output_height=512, mode="bilinear", padding_mode="border", align_corners=False)
threshold = 0.5  # texel_weight is proportional to how much pixel are the texel covers. We can specify a threshold of how much covered pixel area counts as visible.
visibility = texel_weight.clamp(max=1.0).float()

save_image(visibility.expand(-1, 3, -1, -1), "visibility.png")

camera_distance = 0.5
a1 = np.pi + 0.5
a2 = 1 - 1.5
a3 = 0
offset = np.array([-0.01, 0.1, 0.03])

t = np.array([0, 0, camera_distance], dtype=np.float64)
R = np.matmul(np.matmul(rodrigues([a1, 0, 0]), rodrigues([0, a2, 0])), rodrigues([0, 0, a3]))
cam_pos_new = th.as_tensor((R.transpose(-1, -2) @ -t[..., None])[..., 0] + offset, dtype=th.float32).cuda()[None]
cam_rot_new = th.as_tensor(R, dtype=th.float32).cuda()[None]

v_pix = transform(v, cam_pos_new, cam_rot_new, focal, princpt)
index_img = rasterize(v_pix, vi, image_height, image_width)
_, bary_img = render(v_pix, vi, index_img)

# mask image
mask: th.Tensor = (index_img != -1)[:, None]

# compute vt image
vt_img = interpolate(vt.mul(2.0).sub(1.0)[None], vti, index_img, bary_img)

# compute v image (for near-field)
v_img = interpolate(v, vi, index_img, bary_img)

# shadow
shadow_img = thf.grid_sample(visibility, vt_img.permute(0, 2, 3, 1), mode="bilinear", padding_mode="border", align_corners=False)

# compute normals
vn_img = interpolate(normals, vi, index_img, bary_img)

diffuse = shade(vn_img, cam_pos[:, :, None, None] - v_img, 0.05, 0.4, shadow_img) * mask

save_image(diffuse, "diffuse_w_shadow.png")
