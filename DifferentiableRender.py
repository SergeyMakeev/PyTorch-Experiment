import torch
import torchvision


def is_tensor(v) -> bool:
    if isinstance(v, torch.Tensor):
        return True
    return False


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


def normalize_vec3(v):
    assert_is_tensor_shaped_nx3(v)
    v = v / v.norm(dim=1, keepdim=True)
    return v


def vec3(x, y, z):
    return torch.tensor([[x, y, z]]).to(torch.float32)


def scalar(x):
    return torch.tensor(x)


def pi():
    return scalar(3.14159265359)


# take two arrays of vec3 as input
#   a = [N,3]
#   b = [1,3]
# compute dot product between all As and Bs
#  and return the result as tensor shaped [N, 1]
def dot3(a, b):
    assert_is_tensor_shaped_nx3(a)
    assert_is_tensor_shaped_1x3(b)
    dp = torch.matmul(a, b.t())
    num_a = a.shape[0]
    num_b = b.shape[0]
    if dp.shape[0] != num_a:
        raise ValueError
    if dp.shape[1] != num_b:
        raise ValueError
    return dp


# take two arrays of vec3 as input
#   a = [N,3]
#   b = [N,3]
# compute dot product between all As and Bs
#  and return the result as tensor shaped [N, 1]
def _dot3(a, b):
    assert_is_tensor_shaped_nx3(a)
    assert_is_tensor_shaped_nx3(b)
    if a.shape[0] != b.shape[0]:
        raise ValueError
    dp = (a * b).sum(dim=1, keepdim=True)
    return dp


def _abs(v):
    assert_is_tensor(v)
    return torch.abs(v)


def saturate(v):
    assert_is_tensor(v)
    return v.clamp(0.0, 1.0)


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
    rgb = raw_image.permute(1, 2, 0).reshape(-1, 3).to(torch.float32)
    rgb = rgb / 255.0
    return rgb, width, height


# load RGB image from disk and return R channel  as a tensor with shape (w*h, 1)
# grayscale values are normalized in range 0..1
def load_as_grayscale(path):
    raw_image = torchvision.io.read_image(path)
    height = raw_image.shape[1]
    width = raw_image.shape[2]
    rgb = raw_image.permute(1, 2, 0).reshape(-1, 3).to(torch.float32)
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
    normals = raw_image.permute(1, 2, 0).reshape(-1, 3).to(torch.float32)
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
    assert_is_tensor_shaped_nx3(linear)
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
    assert_is_tensor_shaped_nx3(f0)
    assert_is_tensor_shaped_nx1(cos_theta)
    p5 = torch.pow(1.0 - cos_theta, 5.0)
    return p5 + f0 * (1.0 - p5)


#  http://jcgt.org/published/0003/02/03/paper.pdf
def v_smith_ggx_correlated(n_dot_l, n_dot_v, roughness):
    assert_is_tensor_shaped_nx1(n_dot_l)
    assert_is_tensor_shaped_nx1(n_dot_v)
    assert_is_tensor_shaped_nx1(roughness)
    a2 = roughness * roughness
    # Note: 'n_dot_l *' and 'n_dot_v *' are explicitly reversed
    # "Moving Frostbite to Physically Based Rendering" (Lagarde)
    # https://seblagarde.files.wordpress.com/2015/07/course_notes_moving_frostbite_to_pbr_v32.pdf
    # Listing 2
    vis_v = n_dot_l * torch.sqrt((n_dot_v - a2 * n_dot_v) * n_dot_v + a2)
    vis_l = n_dot_v * torch.sqrt((n_dot_l - a2 * n_dot_l) * n_dot_l + a2)
    return 0.5 / (vis_v + vis_l + 1e-5)


def d_ggx(n_dot_h, roughness):
    assert_is_tensor_shaped_nx1(n_dot_h)
    assert_is_tensor_shaped_nx1(roughness)
    a2 = roughness * roughness
    d = (n_dot_h * a2 - n_dot_h) * n_dot_h + 1.0
    return a2 / (d * d + 1e-7)   # note: no Pi


def brdf_ggx(roughness, n_dot_h, n_dot_v, n_dot_l):
    assert_is_tensor_shaped_nx1(roughness)
    assert_is_tensor_shaped_nx1(n_dot_h)
    assert_is_tensor_shaped_nx1(n_dot_v)
    assert_is_tensor_shaped_nx1(n_dot_l)
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

        n_clr = base_color.shape[0]
        n_nrm = normal.shape[0]
        n_rgh = roughness.shape[0]
        n_mtl = metallic.shape[0]
        n_pos = positions.shape[0]

        # check all tensors have the same resolution
        if n_clr != n_nrm or n_clr != n_rgh or n_clr != n_mtl or n_clr != n_pos:
            raise ValueError

        self.base_color = base_color
        self.normal = normal
        self.perceptual_roughness = roughness
        self.metallic = metallic
        self.positions = positions


class SceneProperties:
    def __init__(self, light_dir, light_color, ambient_color):
        assert_is_tensor_shaped_1x3(light_dir)
        assert_is_tensor_shaped_1x3(light_color)
        assert_is_tensor_shaped_1x3(ambient_color)
        self.light_dir = normalize_vec3(light_dir)
        self.light_color = light_color
        self.ambient_color = ambient_color


def render_brdf(surface: SurfaceProperties, scene: SceneProperties, w, h):

    dielectric_f0 = torch.full_like(surface.base_color, 0.04)
    albedo = gamma_to_linear(surface.base_color)
    perceptual_roughness = surface.perceptual_roughness
    normal = surface.normal
    f0 = torch.lerp(dielectric_f0, albedo, surface.metallic)
    one_minus_metalness = 1 - surface.metallic

    # virtual camera pos (10 units above the plane)
    cam_pos = vec3(0.0, 0.0, 10.0)

    # direction from a pixel to a camera position
    _view_dir = cam_pos - surface.positions

    view_dir = normalize_vec3(_view_dir)

    save_as_normals(surface.positions, w, h, "out/debug_pos.png")
    save_as_normals(normal, w, h, "out/debug_normals.png")
    save_as_normals(view_dir, w, h, "out/debug_view_dir.png")
    save_as_rgb(linear_to_gamma(albedo), w, h, "out/debug_albedo.png")
    save_as_rgb(linear_to_gamma(f0), w, h, "out/debug_f0.png")

    # direction from a pixel position to a light position (-light_dir in case of directional light)
    light_dir = scene.light_dir
    light_color = scene.light_color
    ambient_color = scene.ambient_color

    ####################
    roughness = perceptual_roughness_to_roughness(perceptual_roughness)
    roughness = torch.max(roughness, scalar(0.045))

    # half-angle vector
    half_angle_vec = normalize_vec3(light_dir + view_dir)
    n_dot_v = _abs(_dot3(normal, view_dir))

    n_dot_l = saturate(dot3(normal, light_dir))
    n_dot_h = saturate(_dot3(normal, half_angle_vec))

    # note: intentionally unclamped
    l_dot_h = dot3(half_angle_vec, light_dir)

    f = fresnel_schlick(f0, l_dot_h)

    specular = (brdf_ggx(roughness, n_dot_h, n_dot_v, n_dot_l) * n_dot_l) * f * light_color
    diffuse_energy = one_minus_metalness - f * one_minus_metalness
    diffuse_intensity = diffuse_energy * light_color * n_dot_l

    color = diffuse_intensity * albedo + specular + ambient_color
    return color


def reinhard_tonemapper(color):
    assert_is_tensor_shaped_nx3(color)
    return saturate(color / (1 + color))


def test():

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

    light_dir = vec3(1.0, 1.0, 1.0)           # note: neg light_dir!
    light_color = vec3(1.0, 0.95, 0.83)
    ambient_color = vec3(0.02, 0.02, 0.02)
    scene = SceneProperties(light_dir, light_color, ambient_color)

    num_samples = 100
    # generate random light directions
    torch.manual_seed(13)
    random_light_directions = torch.randn(num_samples, 3)

    print("Render")
    # TODO: replace scene parameters with tensor (N, 6)
    #       random_light_directions, random_light_colors
    #       to render multiple images at once
    hdr_color = render_brdf(surface, scene, w, h)

    print("Save results")
    ldr_color = saturate(linear_to_gamma(hdr_color))
    save_as_rgb(ldr_color, w, h, "out/render.png")


test()
