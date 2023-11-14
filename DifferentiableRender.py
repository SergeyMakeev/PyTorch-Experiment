import math

import torch
import torchvision
import torch.nn.functional as F
from torchvision.transforms import v2


def get_default_device():
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    return "cuda"


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
    rgb = raw_image.permute(1, 2, 0).reshape(-1, 3).to(dtype=torch.float32, device=get_default_device())
    rgb = rgb / 255.0
    return rgb, width, height


# load RGB image from disk and return R channel  as a tensor with shape (w*h, 1)
# grayscale values are normalized in range 0..1
def load_as_grayscale(path):
    raw_image = torchvision.io.read_image(path)
    height = raw_image.shape[1]
    width = raw_image.shape[2]
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
    normals = raw_image.permute(1, 2, 0).reshape(-1, 3).to(dtype=torch.float32, device=get_default_device())
    normals = (normals / 255.0) * 2.0 - 1.0
    normals = normalize_vec3(normals)
    return normals, width, height


def save_as_rgb(rgb, width, height, path):
    # reshape
    raw_image = rgb.reshape(height, width, 3).permute(2, 0, 1)
    torchvision.utils.save_image(raw_image, path)


def save_as_normals(normals, width, height, path):
    # normalize
    raw_image = normals / normals.norm(dim=1, keepdim=True)
    # reshape
    raw_image = raw_image.reshape(height, width, 3).permute(2, 0, 1)
    # convert to 0..1 range
    raw_image = (raw_image + 1.0) * 0.5
    torchvision.utils.save_image(raw_image, path)


def downsample_image_x2(img, w, h):
    assert_is_tensor_shaped_nx3(img)
    img_tensor = img.view(h, w, 3)

    # Reshape img_tensor to (1, 3, H, W) for interpolate
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)

    # downsample by a factor of 2
    downsampled_tensor = F.interpolate(img_tensor, scale_factor=0.5, mode='bilinear',
                                       align_corners=False, antialias=True)

    # reshape back to (H/2, W/2, 3)
    downsampled_tensor = downsampled_tensor.squeeze(0).permute(1, 2, 0)

    return downsampled_tensor.view(-1, 3)


def _downsample_image_x2(img, w, h):
    assert_is_tensor_shaped_nx3(img)

    # Reshape to (C, H, W) for torchvision transforms
    img_tensor = img.view(h, w, 3).permute(2, 0, 1)

    resize_transform = v2.Resize((h // 2, w // 2), antialias=True)
    downsampled_img = resize_transform(img_tensor)

    # Flatten back to linear shape (W/2 * H/2, 3)
    flatten = downsampled_img.permute(1, 2, 0).view(-1, 3)
    return flatten


# generate positions inside a "plane" with the following bounds
# X: -1..1
# Y: -1..1
# Z: 0
def generate_positions(width, height):
    x = torch.linspace(-1, 1, width)
    y = torch.linspace(-1, 1, height)
    xx, yy = torch.meshgrid(x, y)
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
    return pow(linear, 0.45454545)


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
    # Listing 2
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
        assert_is_tensor_shaped_nx3(base_color)
        assert_is_tensor_shaped_nx3(normal)
        assert_is_tensor_shaped_nx1(roughness)
        assert_is_tensor_shaped_nx1(metallic)
        assert_is_tensor_shaped_nx3(positions)

        base_color.requires_grad_(True)
        normal.requires_grad_(True)
        roughness.requires_grad_(True)
        metallic.requires_grad_(True)
        positions.requires_grad_(False)

        assert_is_tensor_with_autograd(base_color)
        assert_is_tensor_with_autograd(normal)
        assert_is_tensor_with_autograd(roughness)
        assert_is_tensor_with_autograd(metallic)
        assert_is_tensor_without_autograd(positions)

        n_clr = base_color.shape[0]
        n_nrm = normal.shape[0]
        n_rgh = roughness.shape[0]
        n_mtl = metallic.shape[0]
        n_pos = positions.shape[0]

        # check all tensors have the same resolution
        if n_clr != n_nrm or n_clr != n_rgh or n_clr != n_mtl or n_clr != n_pos:
            raise ValueError

        dielectric_f0 = torch.full_like(base_color, 0.04)
        f0 = torch.lerp(dielectric_f0, base_color, metallic)

        one_minus_metalness = _saturate(1.0 - metallic)

        self.albedo = base_color
        self.normal = normal
        self.perceptual_roughness = roughness
        self.metallic = metallic
        self.positions = positions
        self.f0 = f0
        self.one_minus_metalness = one_minus_metalness

    def parameters(self):
        return [self.albedo, self.normal, self.perceptual_roughness, self.metallic]


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


def render_brdf(surface: SurfaceProperties, scene: SceneProperties, w, h):

    # shape(num_pixels, 3)
    albedo = gamma_to_linear(_saturate(surface.albedo))

    # shape(num_pixels, 1)
    perceptual_roughness = _saturate(surface.perceptual_roughness)

    # shape(num_pixels, 3)
    normal = normalize_vec3(surface.normal)

    # shape(num_pixels, 3)
    f0 = gamma_to_linear(_saturate(surface.f0))

    # shape(num_pixels, 3)
    one_minus_metalness = surface.one_minus_metalness

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

    debug_frame_num = 0
    save_as_normals(surface.positions, w, h, "out/debug_pos.png")
    save_as_normals(slice_3d(normal, debug_frame_num), w, h, "out/debug_normals.png")
    save_as_normals(slice_3d(view_dir, debug_frame_num), w, h, "out/debug_viewdir.png")
    save_as_rgb(linear_to_gamma(slice_3d(albedo, debug_frame_num)), w, h, "out/debug_albedo.png")
    save_as_rgb(linear_to_gamma(slice_3d(f0, debug_frame_num)), w, h, "out/debug_f0.png")

    # --- brdf ----------

    # half-angle vector
    # shape(num_frames, num_pixels, 3)
    half_angle_vec = _normalize_as_vec3(light_dir + view_dir)

    # shape(num_pixels, 3)
    n_dot_v = _abs(_dot_as_vec3(normal, view_dir))

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
    return color


def reinhard_tonemapper(color):
    assert_is_tensor_shaped_nx3(color)
    return _saturate(color / (1 + color))


def test():

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

    positions = generate_positions(w, h)
    # metallic = torch.full_like(metallic, 0.0)
    # roughness = torch.full_like(roughness, 1.0)

    surface = SurfaceProperties(base_color, normals, roughness, metallic, positions)

    # x,y = screen x, y
    # +z = up (from surface to screen)

    num_frames = 5
    torch.manual_seed(13)
    # light_dir = torch.randn(num_frames, 3)
    # make sure Z is always positive (hemisphere)
    # light_dir[:, 2] = torch.abs(light_dir[:, 2])
    # light_color = torch.randn(num_frames, 3)

    light_dir = vec3(0.00, -0.87, 0.5).repeat(num_frames, 1)
    light_color = vec3(1.0, 0.957, 0.839).repeat(num_frames, 1)

    # light_dir = vec3(0.00, -0.87, 0.5)           # note: neg light_dir!
    # light_color = vec3(1.0, 0.957, 0.839)
    ambient_color = vec3(0.0, 0.0, 0.0)
    scene = SceneProperties(light_dir, light_color, ambient_color)

    num_samples = 100
    # generate random light directions
    torch.manual_seed(13)
    random_light_directions = torch.randn(num_samples, 3)

    print("Render {0} frames".format(num_frames))
    # shape (num_frames, num_pixels, 3)
    hdr_color = render_brdf(surface, scene, w, h)

    # TODO what to do with fireflies? (random super bright pixels)
    mdr_color = hdr_color.clamp(0.0, 4.0)

    loss = compute_loss(hdr_color, mdr_color)
    loss.backward()

    num_mips = int(math.log2(min(w, h)) + 1)
    print("Num mips {0}".format(num_mips))

    print("Save results")

    for frame_num in range(num_frames):
        width = w
        height = h
        image = slice_3d(mdr_color, frame_num)
        for mip_num in range(num_mips-1):
            img_name = "out/render_{0}_{1}.png".format(frame_num, mip_num)
            print(img_name)
            ldr_color = _saturate(linear_to_gamma(image))
            save_as_rgb(ldr_color, width, height, img_name)
            image = downsample_image_x2(image, width, height)
            width = int(width / 2)
            height = int(height / 2)


test()
