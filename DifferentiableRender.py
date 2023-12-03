import math
import torch
import torchvision
import torch.nn.functional as F
from torchvision.transforms import v2
import torch.optim as optim


def get_default_device():
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    return "cuda"
    #return "cpu"


def is_tensor(v) -> bool:
    if isinstance(v, torch.Tensor):
        return True
    return False


def is_tensor_with_autograd(v) -> bool:
    if not is_tensor(v):
        return False

    return v.requires_grad


def is_tensor_shaped_1x3(v) -> bool:
    if not is_tensor(v):
        return False

    shape = v.shape
    if len(shape) != 2:
        return False

    if shape[0] != 1:
        return False

    if shape[1] != 3:
        return False
    return True


def is_tensor_shaped_mxnx3(v) -> bool:
    if not is_tensor(v):
        return False

    shape = v.shape
    if len(shape) != 3:
        return False

    if shape[2] != 3:
        return False
    return True


def is_tensor_shaped_mxnx1(v) -> bool:
    if not is_tensor(v):
        return False

    shape = v.shape
    if len(shape) != 3:
        return False

    if shape[2] != 1:
        return False
    return True


def is_tensor_shaped_nx3(v) -> bool:
    if not is_tensor(v):
        return False

    shape = v.shape
    if len(shape) != 2:
        return False

    if shape[1] != 3:
        return False
    return True


def is_tensor_shaped_nx1(v) -> bool:
    if not is_tensor(v):
        return False

    shape = v.shape
    if len(shape) != 2:
        return False

    if shape[1] != 1:
        return False
    return True


def assert_is_tensor(v):
    if is_tensor(v):
        return
    raise ValueError


def assert_is_tensor_with_autograd(v):
    if is_tensor_with_autograd(v):
        return
    raise ValueError


def assert_is_tensor_without_autograd(v):
    if not is_tensor_with_autograd(v):
        return
    raise ValueError


def assert_is_tensor_shaped_mxnx3(v):
    if is_tensor_shaped_mxnx3(v):
        return
    raise ValueError


def assert_is_tensor_shaped_mxnx1(v):
    if is_tensor_shaped_mxnx1(v):
        return
    raise ValueError


def assert_is_tensor_shaped_nx3(v):
    if is_tensor_shaped_nx3(v):
        return
    raise ValueError


def assert_is_tensor_shaped_1x3(v):
    if is_tensor_shaped_1x3(v):
        return
    raise ValueError


def assert_is_tensor_shaped_nx1(v):
    if is_tensor_shaped_nx1(v):
        return
    raise ValueError


def vec3(x, y, z):
    return torch.tensor([[x, y, z]]).to(dtype=torch.float32, device=get_default_device())


def scalar(x):
    return torch.tensor(x).to(device=get_default_device())


def pi():
    return scalar(3.14159265359)


# 'promote' 2d tensor shaped (N,M) to a 3d tensor shaped(num, N, M)
# by repeat it multiple times
def promote_2d_to_3d(v, num):
    assert_is_tensor(v)
    if num == 1:
        return v.unsqueeze(0)
    v_repeated = v.unsqueeze(0).repeat(num, 1, 1)
    return v_repeated


def normalize_vec3(v):
    assert_is_tensor_shaped_nx3(v)
    v = v / v.norm(dim=1, keepdim=True)
    return v


# get tensor shaped as (Z,N,M) slice it along Z axis and return tensor shaped (N,M)
def slice_3d(v, index):
    assert_is_tensor(v)
    # check if 3d tensor
    if len(v.shape) != 3:
        raise ValueError

    # check slice index
    if index >= v.shape[0]:
        raise ValueError

    return v[index]


def _abs(v):
    assert_is_tensor(v)
    return torch.abs(v)


def _saturate(v):
    assert_is_tensor(v)
    return v.clamp(0.0, 1.0)


def _clamp(v, v_min, v_max):
    assert_is_tensor(v)
    return v.clamp(v_min, v_max)


def _normalize_as_vec3(v):
    assert_is_tensor_shaped_mxnx3(v)
    norms = torch.norm(v, p=2, dim=2, keepdim=True)
    res = v / norms
    return res


def _dot_as_vec3(a, b):
    assert_is_tensor_shaped_mxnx3(a)
    assert_is_tensor_shaped_mxnx3(b)
    if a.shape[0] != b.shape[0]:
        raise ValueError
    if a.shape[1] != b.shape[1]:
        raise ValueError
    # shape(N,M)
    dp = (a * b).sum(dim=2)
    # Reshape to (N, M, 1)
    return dp.unsqueeze(2)


#
# Image I/O helpers
#
############################################################################################################

# load RGB image from disk and return it as a tensor with shape (w*h, 3)
# RGB values are normalized in range 0..1
def load_as_rgb(path):
    raw_image = torchvision.io.read_image(path)
    # num_channels = raw_image.shape[0]
    height = raw_image.shape[1]
    width = raw_image.shape[2]
    depth = raw_image.shape[0]

    # drop alpha if needed
    if depth == 4:
        raw_image = raw_image[0:3]

    rgb = raw_image.permute(1, 2, 0).reshape(-1, 3).to(dtype=torch.float32, device=get_default_device())
    rgb = rgb / 255.0
    return rgb, width, height


# load RGB image from disk and return R channel  as a tensor with shape (w*h, 1)
# grayscale values are normalized in range 0..1
def load_as_grayscale(path):
    raw_image = torchvision.io.read_image(path)
    height = raw_image.shape[1]
    width = raw_image.shape[2]
    depth = raw_image.shape[0]

    # drop alpha if needed
    if depth == 4:
        raw_image = raw_image[0:3]

    if depth == 1:
        raw_image = torch.cat([raw_image, raw_image, raw_image], dim=1)

    rgb = raw_image.permute(1, 2, 0).reshape(-1, 3).to(dtype=torch.float32, device=get_default_device())
    # slice to drop last two columns = shape(N,) and then unsqueeze to shape it (N,1)
    grayscale = rgb[:, 0].unsqueeze(1)
    grayscale = grayscale / 255.0
    return grayscale, width, height


# load (and unpack) XYZ normals from image and return it as a tensor with shape (w*h, 3)
# all normals are normalized and XYZ all values are in range -1..1
#
def load_as_normals(path):
    raw_image = torchvision.io.read_image(path)
    height = raw_image.shape[1]
    width = raw_image.shape[2]
    depth = raw_image.shape[0]

    # drop alpha if needed
    if depth == 4:
        raw_image = raw_image[0:3]

    normals = raw_image.permute(1, 2, 0).reshape(-1, 3).to(dtype=torch.float32, device=get_default_device())
    normals = (normals / 255.0) * 2.0 - 1.0
    normals = normalize_vec3(normals)
    return normals, width, height


def save_as_rgb(rgb, width, height, path):
    assert_is_tensor_shaped_nx3(rgb)
    # reshape
    raw_image = rgb.reshape(height, width, 3).permute(2, 0, 1)
    torchvision.utils.save_image(raw_image, path)


def save_as_grayscale(grayscale, width, height, path):
    assert_is_tensor_shaped_nx1(grayscale)
    rgb = torch.cat([grayscale, grayscale, grayscale], dim=1)
    assert_is_tensor_shaped_nx3(rgb)
    save_as_rgb(rgb, width, height, path)


def save_as_normals(normals, width, height, path):
    assert_is_tensor_shaped_nx3(normals)
    # normalize
    raw_image = normals / normals.norm(dim=1, keepdim=True)
    # reshape
    raw_image = raw_image.reshape(height, width, 3).permute(2, 0, 1)
    # convert to 0..1 range
    raw_image = (raw_image + 1.0) * 0.5
    torchvision.utils.save_image(raw_image, path)


def downsample_rgb_images_x2(images, w, h):
    assert_is_tensor_shaped_mxnx3(images)

    num_images = images.shape[0]

    # reshape to (num_images, H, W, 3)
    reshaped_tensor = images.view(num_images, h, w, 3)

    # initialize a list to hold downsampled images
    downsampled_images = []

    # downsample each image
    for img in reshaped_tensor:
        # add batch dimension and permute to (1, 3, H, W)
        img = img.permute(2, 0, 1).unsqueeze(0)

        # downsample
        downsampled_img = F.interpolate(img, scale_factor=0.5, mode='bilinear',
                                        align_corners=False)

        # remove batch dimension and permute back to (H/2, W/2, 3)
        downsampled_img = downsampled_img.squeeze(0).permute(1, 2, 0)

        # flatten and add to the list
        downsampled_images.append(downsampled_img.view(-1, 3))

    # concatenate all downsampled images into a single tensor
    output_tensor = torch.stack(downsampled_images)
    return output_tensor


def downsample_grayscale_image_x2(img, w, h):
    assert_is_tensor_shaped_nx1(img)

    img_tensor = img.view(h, w, 1)
    # reshape img_tensor to (1, 1, H, W)
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)

    # downsample by a factor of 2
    downsampled_tensor = F.interpolate(img_tensor, scale_factor=0.5, mode='bilinear',
                                       align_corners=False)

    # reshape back to (H/2, W/2, 1)
    downsampled_tensor = downsampled_tensor.squeeze(0).permute(1, 2, 0)

    # flatten the tensor to (W/2*H/2, 1)
    flattened_tensor = downsampled_tensor.view(-1, 1)

    return flattened_tensor


def downsample_rgb_image_x2(img, w, h):
    assert_is_tensor_shaped_nx3(img)
    img_tensor = img.view(h, w, 3)

    # reshape img_tensor to (1, 3, H, W) for interpolate
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)

    # downsample by a factor of 2
    downsampled_tensor = F.interpolate(img_tensor, scale_factor=0.5, mode='bilinear',
                                       align_corners=False)

    # reshape back to (H/2, W/2, 3)
    downsampled_tensor = downsampled_tensor.squeeze(0).permute(1, 2, 0)

    return downsampled_tensor.view(-1, 3)


# alternative approach
# def _downsample_rgb_image_x2(img, w, h):
#     assert_is_tensor_shaped_nx3(img)
#
#     # Reshape to (C, H, W) for torchvision transforms
#     img_tensor = img.view(h, w, 3).permute(2, 0, 1)
#
#     resize_transform = v2.Resize((h // 2, w // 2), antialias=True)
#     downsampled_img = resize_transform(img_tensor)
#
#     # Flatten back to linear shape (W/2 * H/2, 3)
#     flatten = downsampled_img.permute(1, 2, 0).view(-1, 3)
#     return flatten


# generate positions inside a "plane" with the following bounds
# X: -1..1
# Y: -1..1
# Z: 0
def generate_positions(width, height):
    x = torch.linspace(-1, 1, width)
    y = torch.linspace(-1, 1, height)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    zz = torch.zeros_like(xx)
    points = torch.stack((xx.t(), yy.t(), zz.t()), dim=-1)
    points = points.reshape(-1, 3)
    return points


#
# BRDF helpers
#
############################################################################################################

def gamma_to_linear(gamma):
    assert_is_tensor_shaped_nx3(gamma)
    return torch.pow(gamma, 2.2)


def linear_to_gamma(linear):
    assert_is_tensor(linear)
    linear_clamped = linear.clamp(0.0039, 100.0)
    res = torch.pow(linear_clamped, 0.45454545)
    # if torch.isnan(res).any():
    #     print("NAN!!!")
    return res


def perceptual_roughness_to_roughness(perceptual_roughness):
    assert_is_tensor_shaped_nx1(perceptual_roughness)
    return perceptual_roughness * perceptual_roughness


def fresnel_term(f0, cos_a):
    assert_is_tensor_shaped_nx3(f0)
    assert_is_tensor_shaped_nx1(cos_a)
    t = torch.pow(1.0 - cos_a, 5.0)
    return f0 + (1.0 - f0) * t


def fresnel_lerp(f0, f90, cos_a):
    assert_is_tensor_shaped_nx3(f0)
    assert_is_tensor_shaped_nx3(f90)
    assert_is_tensor_shaped_nx1(cos_a)
    t = torch.pow(1.0 - cos_a, 5.0)
    return torch.lerp(f0, f90, t)


# Note: Disney diffuse must multiply by diffuseAlbedo / PI. This is done outside this function.
def disney_diffuse(n_dot_v, n_dot_l, l_dot_h, perceptual_roughness):
    assert_is_tensor_shaped_nx1(n_dot_v)
    assert_is_tensor_shaped_nx1(n_dot_l)
    assert_is_tensor_shaped_nx1(l_dot_h)
    assert_is_tensor_shaped_nx1(perceptual_roughness)
    fd90 = 0.5 + 2.0 * l_dot_h * l_dot_h * perceptual_roughness
    # Two schlick fresnel term
    light_scatter = (1.0 + (fd90 - 1.0) * torch.pow((1.0 - n_dot_l), 5.0))
    view_scatter = (1.0 + (fd90 - 1.0) * torch.pow((1.0 - n_dot_v), 5.0))
    return light_scatter * view_scatter


def fresnel_schlick(f0, cos_theta):
    assert_is_tensor_shaped_mxnx3(f0)
    assert_is_tensor_shaped_mxnx1(cos_theta)
    p5 = torch.pow(1.0 - cos_theta, 5.0)
    return p5 + f0 * (1.0 - p5)


#  http://jcgt.org/published/0003/02/03/paper.pdf
def v_smith_ggx_correlated(n_dot_l, n_dot_v, roughness):
    assert_is_tensor_shaped_mxnx1(n_dot_l)
    assert_is_tensor_shaped_mxnx1(n_dot_v)
    assert_is_tensor_shaped_mxnx1(roughness)
    a2 = roughness * roughness
    # Note: 'n_dot_l *' and 'n_dot_v *' are explicitly reversed
    # "Moving Frostbite to Physically Based Rendering" (Lagarde)
    # https://seblagarde.files.wordpress.com/2015/07/course_notes_moving_frostbite_to_pbr_v32.pdf
    # Listing 2: BSDF evaluation code.
    vis_v = n_dot_l * torch.sqrt((n_dot_v - a2 * n_dot_v) * n_dot_v + a2)
    vis_l = n_dot_v * torch.sqrt((n_dot_l - a2 * n_dot_l) * n_dot_l + a2)
    return 0.5 / (vis_v + vis_l + 1e-5)


def d_ggx(n_dot_h, roughness):
    assert_is_tensor_shaped_mxnx1(n_dot_h)
    assert_is_tensor_shaped_mxnx1(roughness)
    a2 = roughness * roughness
    d = (n_dot_h * a2 - n_dot_h) * n_dot_h + 1.0
    return a2 / (d * d + 1e-7)   # note: no Pi


def brdf_ggx(roughness, n_dot_h, n_dot_v, n_dot_l):
    assert_is_tensor_shaped_mxnx1(roughness)
    assert_is_tensor_shaped_mxnx1(n_dot_h)
    assert_is_tensor_shaped_mxnx1(n_dot_v)
    assert_is_tensor_shaped_mxnx1(n_dot_l)
    d = d_ggx(n_dot_h, roughness)
    v = v_smith_ggx_correlated(roughness, n_dot_v, n_dot_l)
    return d * v


# def sqr(x):
#     assert_is_tensor(x)
#     return x*x
#
#
# def schlick_fresnel(u):
#     assert_is_tensor(u)
#     m = (1 - u).clamp(0, 1)
#     m2 = m * m
#     # pow(m, 5)
#     return m2 * m2 * m
#
#
# def gtr1(n_dot_h, a):
#     assert_is_tensor(n_dot_h)
#     assert_is_tensor(a)
#     condition = a >= 1
#     a2 = a * 2
#     t = 1 + (a2 - 1) * n_dot_h * n_dot_h
#     return torch.where(condition, 1 / math.pi, (a2 - 1) / (math.pi * torch.log(a2) * t))
#
#
# def gtr2(n_dot_h, a):
#     assert_is_tensor(n_dot_h)
#     assert_is_tensor(a)
#     a2 = a * 2
#     t = 1 + (a2 - 1) * n_dot_h * n_dot_h
#     a2 / (math.pi * t * t)

class SurfaceProperties:
    def __init__(self, base_color: torch.Tensor, normal: torch.Tensor, roughness: torch.Tensor, metallic: torch.Tensor,
                 positions: torch.Tensor):
        # base_color = base_color.detach()
        # normal = normal.detach()
        # roughness = roughness.detach()
        # metallic = metallic.detach()
        # positions = positions.detach()

        # base_color.requires_grad_(True)
        # normal.requires_grad_(True)
        # roughness.requires_grad_(True)
        # metallic.requires_grad_(True)
        # positions.requires_grad_(False)

        assert_is_tensor_shaped_nx3(base_color)
        assert_is_tensor_shaped_nx3(normal)
        assert_is_tensor_shaped_nx1(roughness)
        assert_is_tensor_shaped_nx1(metallic)
        assert_is_tensor_shaped_nx3(positions)

        n_clr = base_color.shape[0]
        n_nrm = normal.shape[0]
        n_rgh = roughness.shape[0]
        n_mtl = metallic.shape[0]
        n_pos = positions.shape[0]

        # check all tensors have the same resolution
        if n_clr != n_nrm or n_clr != n_rgh or n_clr != n_mtl or n_clr != n_pos:
            raise ValueError

        dielectric_f0 = torch.full_like(base_color, 0.04)

        self.albedo = base_color
        self.normal = normal
        self.perceptual_roughness = roughness
        self.metallic = metallic
        self.positions = positions
        self.dielectric_f0 = dielectric_f0


class SceneProperties:
    def __init__(self, light_dir, light_color, ambient_color):
        assert_is_tensor_shaped_nx3(light_dir)
        assert_is_tensor_shaped_nx3(light_color)
        assert_is_tensor_shaped_1x3(ambient_color)

        n_dir = light_dir.shape[0]
        n_clr = light_color.shape[0]
        if n_dir != n_clr:
            raise ValueError

        self.light_dir = normalize_vec3(light_dir)
        self.light_color = light_color
        self.ambient_color = ambient_color


def compute_loss(images_a, images_b):
    assert_is_tensor_shaped_mxnx3(images_a)
    assert_is_tensor_shaped_mxnx3(images_b)

    flat_a = images_a.view(-1)
    flat_b = images_b.view(-1)

    loss = (flat_a - flat_b).pow(2).mean()

    return loss


# diffuse only
def dbg_render_brdf(surface: SurfaceProperties, scene: SceneProperties, w, h):

    # shape(num_pixels, 3)
    albedo = gamma_to_linear(_saturate(surface.albedo))

    # shape(num_pixels, 3)
    normal = normalize_vec3(surface.normal)

    # direction from a pixel position to a light position (-light_dir in case of directional light)
    # shape(num_frames, 3)
    light_dir = scene.light_dir

    # shape(num_frames, 3)
    light_color = scene.light_color

    # shape(1,3)
    ambient_color = scene.ambient_color

    num_frames = light_dir.shape[0]
    num_pixels = albedo.shape[0]

    # promote to 3d tensors
    albedo = promote_2d_to_3d(albedo, num_frames)
    normal = promote_2d_to_3d(normal, num_frames)

    light_dir = light_dir.unsqueeze(1).repeat(1, num_pixels, 1)
    light_color = light_color.unsqueeze(1).repeat(1, num_pixels, 1)

    # --- brdf ----------

    # shape(num_frames, num_pixels, 3)
    n_dot_l = _saturate(_dot_as_vec3(light_dir, normal))

    diffuse_term = light_color * n_dot_l
    color = diffuse_term * albedo + ambient_color
    return color


def render_brdf(surface: SurfaceProperties, scene: SceneProperties, w, h):

    # shape(num_pixels, 3)
    albedo = gamma_to_linear(_saturate(surface.albedo))

    # shape(num_pixels, 1)
    min_roughness = 0.045
    perceptual_roughness = _clamp(surface.perceptual_roughness, min_roughness, 1.0)

    # shape(num_pixels, 3)
    normal = normalize_vec3(surface.normal)

    # shape(num_pixels, 3)
    f0 = gamma_to_linear(_saturate(torch.lerp(surface.dielectric_f0, surface.albedo, surface.metallic)))

    # shape(num_pixels, 3)
    one_minus_metalness = _saturate(1.0 - surface.metallic)

    # TODO: multiple camera positions?
    # virtual camera pos (2 units above the plane)
    # shape(1,3)
    cam_pos = vec3(0.0, 0.0, 1.2)

    # direction from a pixel to a camera position
    # shape(num_pixels, 3) (since we have a single camera)
    view_dir = normalize_vec3(cam_pos - surface.positions)

    # direction from a pixel position to a light position (-light_dir in case of directional light)
    # shape(num_frames, 3)
    light_dir = scene.light_dir

    # shape(num_frames, 3)
    light_color = scene.light_color

    # shape(1,3)
    ambient_color = scene.ambient_color

    # shape(num_pixels, 1)
    roughness = perceptual_roughness_to_roughness(perceptual_roughness)
    roughness = torch.max(roughness, scalar(0.045))

    num_frames = light_dir.shape[0]
    num_pixels = albedo.shape[0]

    # promote to 3d tensors

    albedo = promote_2d_to_3d(albedo, num_frames)
    normal = promote_2d_to_3d(normal, num_frames)
    f0 = promote_2d_to_3d(f0, num_frames)
    roughness = promote_2d_to_3d(roughness, num_frames)
    view_dir = promote_2d_to_3d(view_dir, num_frames)
    one_minus_metalness = promote_2d_to_3d(one_minus_metalness, num_frames)

    light_dir = light_dir.unsqueeze(1).repeat(1, num_pixels, 1)
    light_color = light_color.unsqueeze(1).repeat(1, num_pixels, 1)

    # --- debug ----------

    # debug_frame_num = 0
    # save_as_normals(surface.positions, w, h, "out/debug_pos.png")
    # save_as_normals(slice_3d(normal, debug_frame_num), w, h, "out/debug_normals.png")
    # save_as_normals(slice_3d(view_dir, debug_frame_num), w, h, "out/debug_viewdir.png")
    # save_as_rgb(linear_to_gamma(slice_3d(albedo, debug_frame_num)), w, h, "out/debug_albedo.png")
    # save_as_rgb(linear_to_gamma(slice_3d(f0, debug_frame_num)), w, h, "out/debug_f0.png")

    # --- brdf ----------

    # half-angle vector
    # shape(num_frames, num_pixels, 3)
    half_angle_vec = _normalize_as_vec3(light_dir + view_dir)

    # shape(num_pixels, 3)
    n_dot_v = _abs(_dot_as_vec3(normal, view_dir)) + 1e-5

    # shape(num_frames, num_pixels, 3)
    n_dot_l = _saturate(_dot_as_vec3(light_dir, normal))
    n_dot_h = _saturate(_dot_as_vec3(normal, half_angle_vec))

    # note: intentionally unclamped
    l_dot_h = _dot_as_vec3(half_angle_vec, light_dir)

    f = fresnel_schlick(f0, l_dot_h)

    specular = (brdf_ggx(roughness, n_dot_h, n_dot_v, n_dot_l) * n_dot_l) * f * light_color
    diffuse_energy = one_minus_metalness - f * one_minus_metalness
    diffuse_intensity = diffuse_energy * light_color * n_dot_l

    color = diffuse_intensity * albedo + specular + ambient_color
    return _saturate(linear_to_gamma(color))


def reinhard_tonemapper(color):
    assert_is_tensor_shaped_nx3(color)
    return _saturate(color / (1 + color))


def downsample_frames(hdr_color, w, h, num_mips):
    reference = [{'image': hdr_color, 'width': w, 'height': h}]
    _current = hdr_color
    _width = w
    _height = h
    for mip_num in range(num_mips - 1):
        _current = downsample_rgb_images_x2(_current, _width, _height)
        _width = int(_width / 2)
        _height = int(_height / 2)
        reference.append({'image': _current, 'width': _width, 'height': _height})

    return reference


def test(num_iterations=1500):

    # device = "cpu"
    torch.set_default_device(get_default_device())

    # torch.manual_seed(13)
    # N = 2
    # M = 4
    # A = torch.rand(N, 3)
    # B = torch.rand(M, 3)
    # res = add3(A,B)

    print("Load Textures")
    base_color, w, h = load_as_rgb("pbr/albedo.png")
    normals, _, _ = load_as_normals("pbr/normal.png")
    roughness, _, _ = load_as_grayscale("pbr/roughness.png")
    metallic, _, _ = load_as_grayscale("pbr/metallic.png")

    print("Save mip0 (original)")
    save_as_rgb(_saturate(base_color), w, h, "out/albedo_mip_0.png")
    save_as_normals(normalize_vec3(normals), w, h, "out/normal_mip_0.png")
    save_as_grayscale(_saturate(roughness), w, h, "out/roughness_mip_0.png")
    save_as_grayscale(_saturate(metallic), w, h, "out/metallic_mip_0.png")


    # save_as_rgb(base_color, w, h, "out/debug_albedo.png")
    # save_as_normals(normals, w, h, "out/debug_normal.png")
    # save_as_grayscale(roughness, w, h, "out/debug_roughness.png")
    # save_as_grayscale(metallic, w, h, "out/debug_metallic.png")

    positions = generate_positions(w, h)
    # metallic = torch.full_like(metallic, 0.0)
    # roughness = torch.full_like(roughness, 1.0)

    num_mips = int(math.log2(min(w, h)) + 1)
    print("Num mips {0}".format(num_mips))

    texture_mips = [
        {
            'base_color': base_color,
            'normals': normals,
            'roughness': roughness,
            'metallic': metallic,
            'positions': positions,
            'width': w,
            'height': h
         }
    ]

    print("Downsample source textures")
    for mip_num in range(num_mips - 1):
        print("Mip {0}".format(mip_num))
        _w = texture_mips[-1]['width']
        _h = texture_mips[-1]['height']
        _base_color = texture_mips[-1]['base_color']
        _normals = texture_mips[-1]['normals']
        _roughness = texture_mips[-1]['roughness']
        _metallic = texture_mips[-1]['metallic']

        _base_color2 = downsample_rgb_image_x2(_base_color, _w, _h)
        _normals2 = normalize_vec3(downsample_rgb_image_x2(_normals, _w, _h))
        _roughness2 = downsample_grayscale_image_x2(_roughness, _w, _h)
        _metallic2 = downsample_grayscale_image_x2(_metallic, _w, _h)
        _w2 = int(_w / 2)
        _h2 = int(_h / 2)
        _positions2 = generate_positions(_w2, _h2)

        texture_mips.append(
            {
                'base_color': _base_color2,
                'normals': _normals2,
                'roughness': _roughness2,
                'metallic': _metallic2,
                'positions': _positions2,
                'width': _w2,
                'height': _h2
             }
        )

    # coord system
    # x,y = screen x, y
    # +z = up (from surface to screen)
    surface = SurfaceProperties(base_color, normals, roughness, metallic, positions)

    num_frames = 16
    torch.manual_seed(13423423)
    # light_dir = torch.randn(num_frames, 3)
    # make sure Z is always positive (hemisphere)
    # light_dir[:, 2] = torch.abs(light_dir[:, 2])
    # light_dir = normalize_vec3(light_dir)
    # light_color = torch.randn(num_frames, 3)

    # light_dir = vec3(0.00, -0.87, 0.5).repeat(num_frames, 1)
    #light_color = vec3(1.0, 0.957, 0.839).repeat(num_frames, 1)

    light_color = torch.tensor([
        [1.0, 0.957, 0.839],
        [1.0, 0.957, 0.839],
        [1.0, 0.957, 0.839],
        [1.0, 0.957, 0.839],
        [1.0, 0.957, 0.839],
        [1.0, 0.0, 0.839],
        [1.0, 0.957, 0],
        [0, 0.957, 0.839],
        [1.0, 0.957, 0.839],
        [1.0, 0.957, 0.839],
        [1.0, 0.957, 0.839],
        [1.0, 1.0, 1.0],
        [0.5, 0.5, 0.5],
        [1.0, 0.957, 0.839],
        [1.0, 0.957, 0.839],
        [1.0, 0.957, 0.839],
    ])

    # light_dir = vec3(0.00, -0.87, 0.5)           # note: neg light_dir!

    light_dir = torch.tensor([
        [0.00, -0.87, 0.5],
        [0.00, 0.87, 0.5],
        [0.0, 0.77, 0.64],
        [0.0, -0.77, 0.64],
        [-0.58, 0.73, 0.34],
        [-0.58, -0.73, 0.34],
        [0.58, 0.73, 0.34],
        [0.58, -0.73, 0.34],
        [0.72, 0.59, 0.38],
        [-0.72, 0.59, 0.38],
        [0.72, -0.59, 0.38],
        [-0.72, -0.59, 0.38],
        [0.19, 0.12, 0.98],
        [-0.19, 0.12, 0.98],
        [0.19, -0.12, 0.98],
        [-0.19, -0.12, 0.98]
    ])

    light_dir = normalize_vec3(light_dir)

    # light_color = vec3(1.0, 0.957, 0.839)
    ambient_color = vec3(0.02, 0.02, 0.02)
    scene = SceneProperties(light_dir, light_color, ambient_color)

    print("Render {0} frames".format(num_frames))
    # shape (num_frames, num_pixels, 3)
    hdr_color = render_brdf(surface, scene, w, h)

    print("Downsample ground truth textures")

    ground_truth = downsample_frames(hdr_color, w, h, num_mips)

    if False:
        print("Save ground truth")
        for frame_num in range(num_frames):
            image1 = slice_3d(hdr_color, frame_num)
            img_name = "out/gt/gt_render_{0}.png".format(frame_num)
            print(img_name)
            # ldr_color1 = _saturate(linear_to_gamma(image1))
            save_as_rgb(image1, w, h, img_name)

    for mip_num in range(num_mips - 1):
        gt_hdr_color = ground_truth[mip_num]['image']
        gt_w = ground_truth[mip_num]['width']
        gt_h = ground_truth[mip_num]['height']
        #for frame_num in range(num_frames):
        if True:
            frame_num = 0
            image1 = slice_3d(gt_hdr_color, frame_num)
            img_name = "out/steps/gt_mip_render_{0}_{1}x{2}.png".format(frame_num, gt_w, gt_h)
            print(img_name)
            # ldr_color1 = _saturate(linear_to_gamma(image1))
            save_as_rgb(image1, gt_w, gt_h, img_name)


    # render using downsampled source textures

    print("Save mip #0")
    _w = texture_mips[0]['width']
    _h = texture_mips[0]['height']
    save_as_rgb(_saturate(texture_mips[0]['base_color'].detach()), _w, _h, "out/_albedo_mip_0.png")
    save_as_normals(normalize_vec3(texture_mips[0]['normals'].detach()), _w, _h, "out/_normal_mip_0.png")
    save_as_grayscale(_saturate(texture_mips[0]['roughness'].detach()), _w, _h, "out/_roughness_mip_0.png")
    save_as_grayscale(_saturate(texture_mips[0]['metallic'].detach()), _w, _h, "out/_metallic_mip_0.png")

    for current_mip in range(1, num_mips):
        print("Optimize mip #" + str(current_mip))
        _w = texture_mips[current_mip]['width']
        _h = texture_mips[current_mip]['height']

        _base_color = texture_mips[current_mip]['base_color'].detach()
        _base_color.requires_grad_(True)

        _normals = texture_mips[current_mip]['normals'].detach()
        _normals.requires_grad_(True)

        _roughness = texture_mips[current_mip]['roughness'].detach()
        _roughness.requires_grad_(True)

        _metallic = texture_mips[current_mip]['metallic'].detach()
        _metallic.requires_grad_(True)

        _positions = texture_mips[current_mip]['positions']

        # _base_color = torch.full_like(_base_color, 0.01, requires_grad=True)
        # _normals = torch.full_like(_normals, 0.01, requires_grad=True)
        # _roughness = torch.full_like(_roughness, 0.01, requires_grad=True)
        # _metallic = torch.full_like(_metallic, 0.01, requires_grad=True)

        _surface = SurfaceProperties(_base_color, _normals, _roughness, _metallic, _positions)

        ref = ground_truth[current_mip]['image'].detach()

        assert_is_tensor_with_autograd(_base_color)
        assert_is_tensor_with_autograd(_normals)
        assert_is_tensor_with_autograd(_roughness)
        assert_is_tensor_with_autograd(_metallic)
        learning_rate = 0.005
        optimizer = optim.Adam([_base_color, _normals, _roughness, _metallic], lr=learning_rate)

        # learning_rate = 1.0
        # optimizer = optim.SGD([_base_color, _normals, _roughness, _metallic], lr=learning_rate)

        #torch.autograd.set_detect_anomaly(True)
        _hdr_color = None

        print("Save original mip")
        save_as_rgb(_base_color, _w, _h, "out/_albedo_mip_{0}.png".format(current_mip))
        save_as_normals(_normals, _w, _h, "out/_normal_mip_{0}.png".format(current_mip))
        save_as_grayscale(_roughness, _w, _h, "out/_roughness_mip_{0}.png".format(current_mip))
        save_as_grayscale(_metallic, _w, _h, "out/_metallic_mip_{0}.png".format(current_mip))

        for t in range(num_iterations):
            _hdr_color = render_brdf(_surface, scene, _w, _h)

            loss = compute_loss(ref, _hdr_color)

            if t % 100 == 0:
                print(t, loss.item())

            if t % 1000 == 0:
                #for frame_num in range(num_frames):
                if True:
                    frame_num = 0
                    image2 = slice_3d(_hdr_color, frame_num)
                    img_name = "out/steps/step{3}_mip_render_{0}_{1}x{2}.png".format(frame_num, _w, _h, t)
                    print(img_name)
                    #ldr_color2 = _saturate(linear_to_gamma(image2))
                    save_as_rgb(image2, _w, _h, img_name)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print("Save resulting mip")
        save_as_rgb(_saturate(_base_color), _w, _h, "out/albedo_mip_{0}.png".format(current_mip))
        save_as_normals(normalize_vec3(_normals), _w, _h, "out/normal_mip_{0}.png".format(current_mip))
        save_as_grayscale(_saturate(_roughness), _w, _h, "out/roughness_mip_{0}.png".format(current_mip))
        save_as_grayscale(_saturate(_metallic), _w, _h, "out/metallic_mip_{0}.png".format(current_mip))

    #for frame_num in range(num_frames):
    if True:
        frame_num = 0
        image2 = slice_3d(_hdr_color, frame_num)
        img_name = "out/steps/mip_render_{0}_{1}x{2}.png".format(frame_num, _w, _h)
        print(img_name)
        #ldr_color2 = _saturate(linear_to_gamma(image2))
        save_as_rgb(image2, _w, _h, img_name)


test(600)
