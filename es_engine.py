# -*- coding: utf-8 -*-
"""Readable 3D engine over Ren'Py's Model/Mesh3.

Object3D stores geometry + material + animation state. Subclass it and
override update() to animate. ObjectView drives one object per frame:
update() -> uniforms -> nested Transform (model matrix) -> child render.

Proven rules this encodes (see project history):
- rotation lives in the model matrix (auto-centered pivot), never in verts,
- Mesh3 fields are cdef-only: uploads go through es_step3's poked offsets,
- depth test needs the direct-draw path: show WITHOUT dissolve,
- normals rotate analytically in-shader (u_transform collapses them),
- the bone palette composes in buffer (FIT) space: conjugation distributes
  over the joint chain, so FIT rides on bind constants, never per frame.
"""
import os
import sys

import renpy
import renpy.display.model
import renpy.display.render
import renpy.display.matrix
from renpy.display.core import Displayable
from renpy.display.transform import Transform

_mod_dir = os.path.dirname(
    renpy.loader.transfn("mods/es-3dmodels/es_engine.py"))
if _mod_dir not in sys.path:
    sys.path.insert(0, _mod_dir)

import es_step3
import es_shaders
import es_gltf
import es_obj
import es_morph
import es_caps
import es_matlib

print("ES_BUILD fit-space-palette")

_Matrix = renpy.display.matrix.Matrix


def es_gltf_bind(path, target=300.0):
    """Load a .gltf/.glb bind pose: (verts, normals, tris, uvs), fit to
    target. `uvs` is None when a primitive has no TEXCOORD_0."""
    return es_gltf.bind_pose_mesh(es_gltf.Gltf.load(path), target)


def define_material(name, fragment, params=None, requires=(), map_path=None,
                    applies=None, blend="opaque", cutoff=None, cull=None):
    """Author a shader recipe once; attach it anywhere with recipe="<name>".

    The fragment assigns `es_color`; the engine emits the discard, the
    premultiply and the write to gl_FragColor. It may read whatever the
    vertex stage provides: v_normal, v_diff, v_half, v_pos (the rotated
    position, which with u_eye_z gives a per-fragment view direction),
    u_base_color, u_light_dir, u_view_dir, u_spin -- plus v_uv when the mesh
    carries UVs.
    Everything tunable goes through `params` (a float, or a 2-4 element
    sequence); declarations are generated from those values.

    `blend` is "opaque", "alpha clip" (discards below `cutoff`) or "alpha"
    (blended, drawn in a second pass sorted farthest-first). `requires` names
    geometry inputs checked per object; `map_path` binds an image as tex1;
    `applies` is a glTF material name whose splits take this recipe; `cull`
    "back" throws away fragments whose normal faces away from the viewer,
    which is what stops a thin closed shell from shading itself twice.
    Tune an object with `obj.material.set(name, value)`.

    es-3dmodels/materials/*.mat is the preferred way to author: same backend,
    real GLSL files, and errors that carry a file and line number."""
    return es_shaders.define_recipe(name, fragment, params, requires,
                                    map_path, applies, blend, cutoff, cull)


def add_materials(prefix):
    """Register a content mod's materials directory. Call from `init 10`.

    `prefix` is the game-relative folder holding the .mat files, e.g.
    "mods/es-3dmodels-matitan/materials". Returns the material names, which
    objects then attach with recipe="<name>"; a recipe declaring `applies`
    takes over that glTF material's splits with no call site at all.
    """
    return es_matlib.load_materials(define_material, prefix)


# Global scene light. Materials with light_dir=None follow this;
# a per-material light_dir overrides it for that object only.
LIGHT_DIR = (0.5, -0.8, 0.7)


class Material(object):
    """Surface + lighting preset. Flags select a compiled shader variant.

    `recipe` names an authored fragment registered by define_material; its
    parameters start at the recipe defaults and are tuned per instance with
    set()."""

    def __init__(self, base_color=(0.9, 0.3, 0.3, 1.0),
                 light_dir=None,
                 shininess=16.0, ambient=0.3, diffuse=0.7,
                 spec_weight=0.5, specular=True, normal_viz=False,
                 skinned=False, max_bones=None, bind_pose=False,
                 recipe=None):
        self.base_color = base_color
        self.light_dir = light_dir
        self.shininess = shininess
        self.ambient = ambient
        self.diffuse = diffuse
        self.spec_weight = spec_weight
        self.specular = specular
        self.normal_viz = normal_viz
        self.skinned = skinned
        # None = decide from hardware (es_caps); lazy probe, label-time.
        self.max_bones = (max_bones if max_bones is not None
                          else es_caps.max_bones())
        self.bind_pose = bind_pose
        self.recipe = recipe
        # draw() raises on any uniform the program declares and never gets,
        # so every declared parameter carries a value from the start.
        self.params = (dict(es_shaders.recipe_defaults(recipe))
                       if recipe is not None else {})
        self.map_path = (es_shaders.recipe_map(recipe)
                         if recipe is not None else None)
        # Blend mode belongs to the recipe, not the caller: it is a property
        # of what the shader outputs.
        self.blend = (es_shaders.recipe_blend(recipe)
                      if recipe is not None else "opaque")
        self.alpha_clip = (es_shaders.recipe_cutoff(recipe)
                           if recipe is not None else 0.5)
        self.cull = (es_shaders.recipe_cull(recipe)
                     if recipe is not None else None)
        self._map_source = None

    def map_texture(self):
        """This material's own map as a texture source, or None.

        Built on first use so a recipe that is never attached neither
        decodes nor uploads; the source keeps it at native resolution."""
        if self.map_path is not None and self._map_source is None:
            self._map_source = es_morph.image(self.map_path)
        return self._map_source

    def set(self, name, value):
        """Tune one authored parameter: a float, or a 2-4 element sequence.

        Rejecting an unknown name or a type that disagrees with the recipe's
        declaration is deliberate -- Ren'Py's set_uniform ignores names the
        program does not declare, so a typo would otherwise change nothing
        and stay invisible."""
        if self.recipe is None:
            raise ValueError("Material.set: this material has no recipe")
        types = es_shaders.recipe_types(self.recipe)
        if name not in types:
            raise ValueError("%s has no parameter %r (has: %s)"
                             % (self.recipe, name, ", ".join(sorted(types))))
        ptype = es_shaders.param_type(value)
        if ptype != types[name]:
            raise ValueError("%s.%s wants %s, got %r"
                             % (self.recipe, name, types[name], value))
        self.params[name] = value
        return self

    def variant_key(self):
        return (bool(self.specular), bool(self.normal_viz),
                float(self.shininess), float(self.ambient),
                float(self.diffuse), float(self.spec_weight),
                bool(self.skinned), int(self.max_bones),
                bool(self.bind_pose), self.recipe, self.blend, self.cull)

    def shader_name(self, morph=0, uvs=False):
        # `morph` = active shape-key slot count, `uvs` = the mesh binds a_uv.
        # Both are geometry facts, so the caller (a view, which knows its
        # splits) passes them; only atlas-carrying splits pass a morph count.
        if self.recipe is not None and not uvs:
            if "uv" in es_shaders.recipe_requires(self.recipe):
                raise ValueError("%s requires uv but this mesh carries none"
                                 % self.recipe)
        try:
            return es_shaders.get_variant(
                specular=self.specular, normal_viz=self.normal_viz,
                shininess=self.shininess, ambient=self.ambient,
                diffuse=self.diffuse, spec_weight=self.spec_weight,
                skinned=self.skinned, max_bones=self.max_bones,
                bind_pose=self.bind_pose, morph=morph, uvs=uvs,
                recipe=self.recipe, blend=self.blend, cull=self.cull)
        except Exception:
            return es_shaders.ensure_fallback()


class Object3D(object):
    """Geometry + material + animation state. Subclass, override update()."""

    view_dir = (0.0, 0.0, 1.0)

    def __init__(self, size=400.0, speed=120.0, tilt=25.0, material=None):
        self.size = size
        self.speed = speed
        self.tilt = tilt
        self.angle = 0.0
        self.animated = True
        self.material = material if material is not None else Material()
        self.verts = []
        self.normals = []
        self.tris = []
        # One UV channel, or [] when the geometry carries no texture coords:
        # a_uv is a per-mesh attribute, so it is all-or-nothing.
        self.uvs = []
        self.mesh = None
        self._last_st = None

    def add_quad(self, corners, normal):
        base = len(self.verts)
        for v in corners:
            self.verts.append(v)
            self.normals.append(normal)
        self.tris.append((base, base + 1, base + 2))
        self.tris.append((base, base + 2, base + 3))

    def update(self, st, dt):
        """Advance animation state. Override in subclasses."""
        self.angle = (st * self.speed) % 360.0

    def centered(self):
        """Mesh-space verts with the object centered in its box."""
        half_size = self.size / 2.0
        verts = [(half_size + x, half_size + y, z) for (x, y, z) in self.verts]
        return verts, list(self.normals), list(self.tris)

    def build_mesh(self):
        v, n, t = self.centered()
        if self.uvs:
            # Layout order only has to agree with itself: the shader binds
            # attributes by name through mesh.layout.offset.
            self.mesh = es_step3.mesh3_from_layout(
                v, [n, self.uvs], t,
                [("a_normal", 3), ("a_uv", 2)])
        else:
            self.mesh = es_step3.mesh3_from_lists(v, n, t)
        return self.mesh


class CubeObject(Object3D):
    """Unit cube, outward normals, edge length 300 in a 400 box."""

    HALF = 150.0
    CORNERS = [(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
               (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]
    FACES = [([0, 1, 2, 3], (0, 0, -1)), ([4, 5, 6, 7], (0, 0, 1)),
             ([0, 1, 5, 4], (0, -1, 0)), ([2, 3, 7, 6], (0, 1, 0)),
             ([0, 3, 7, 4], (-1, 0, 0)), ([1, 2, 6, 5], (1, 0, 0))]

    def __init__(self, size=400.0, speed=120.0, tilt=25.0,
                 color=(0.9, 0.3, 0.3)):
        super(CubeObject, self).__init__(
            size, speed, tilt,
            Material(base_color=(color[0], color[1], color[2], 1.0)))
        h = CubeObject.HALF
        for quad, normal in CubeObject.FACES:
            self.add_quad(
                [(CubeObject.CORNERS[i][0] * h,
                  CubeObject.CORNERS[i][1] * h,
                  CubeObject.CORNERS[i][2] * h) for i in quad],
                normal)
        self.build_mesh()


class MeshObject(Object3D):
    """Loaded geometry (OBJ file or decoder lists). Animate via update()."""

    @classmethod
    def from_data(cls, verts, normals, tris, uvs=None, size=500.0,
                  speed=45.0, tilt=25.0, color=(0.85, 0.85, 0.9),
                  recipe=None):
        obj = cls(size=size, speed=speed, tilt=tilt,
                  material=Material(
                      base_color=(color[0], color[1], color[2], 1.0),
                      recipe=recipe))
        obj.verts = list(verts)
        obj.normals = list(normals)
        obj.tris = list(tris)
        obj.uvs = list(uvs) if uvs else []
        obj.build_mesh()
        return obj

    @classmethod
    def from_obj(cls, path, size=500.0, speed=60.0, tilt=25.0,
                 color=(0.85, 0.85, 0.9), target=300.0, recipe=None):
        obj = cls(size=size, speed=speed, tilt=tilt,
                  material=Material(
                      base_color=(color[0], color[1], color[2], 1.0),
                      recipe=recipe))
        verts, normals, tris, uvs = es_obj.load_obj(path, target)
        obj.verts = list(verts)
        obj.normals = list(normals)
        obj.tris = list(tris)
        obj.uvs = list(uvs) if uvs else []
        obj.build_mesh()
        return obj


class ObjectView(Displayable):
    """Drives one Object3D: update() -> uniforms -> matrix -> child render."""

    def __init__(self, obj, **kwargs):
        super(ObjectView, self).__init__(**kwargs)
        self.obj = obj
        self.model = None
        self._shader_key = None
        self._sync_material()

    def _sync_material(self):
        # a_uv is a geometry fact the material's flags cannot see; it rides
        # on the key so a UV-carrying mesh never reuses a non-UV program.
        has_uvs = bool(self.obj.uvs)
        key = self.obj.material.variant_key() + (has_uvs,)
        if key == self._shader_key and self.model is not None:
            return
        self._shader_key = key
        model = renpy.display.model.Model(size=(self.obj.size, self.obj.size))
        model.shader("renpy.geometry")
        model.shader(self.obj.material.shader_name(uvs=has_uvs))
        model.property("depth", True)
        material = self.obj.material
        if material.blend == "alpha clip":
            model.uniform("u_alpha_clip", material.alpha_clip)
        elif material.blend == "alpha":
            model.property("blend_func", renpy.config.gl_blend_func["normal"])
        model.uniform("u_view_dir", Object3D.view_dir)
        model.uniform("u_base_color", material.base_color)
        # Same slot discipline as the skinned path: a rigid draw has no
        # atlas, so without this a material map would land on tex0 here but
        # tex1 on a skinned split, and one authored fragment could not serve
        # both.
        model.texture(es_morph.placeholder())
        texture = self.obj.material.map_texture()
        if texture is not None:
            model.texture(texture)
        model._mesh = self.obj.mesh
        self.model = model

    def render(self, w, h, st, at):
        obj = self.obj
        last = obj._last_st
        dt = 0.0 if last is None else st - last
        obj._last_st = st
        obj.update(st, dt)
        self._sync_material()
        self.model.uniform("u_spin", obj.angle)
        # The vertex stage rotates normals analytically, so it needs the same
        # tilt the model matrix was built with -- a hardcoded one put the
        # shading 25 degrees off any object that did not tilt by 25.
        self.model.uniform("u_tilt", obj.tilt)
        # Eye distance in buffer pixels, from the same perspective the
        # transform is built with: v_pos + u_eye_z is what lets a fragment
        # derive a per-pixel view vector, the only thing that gives a flat
        # panel a gradient at all.
        self.model.uniform("u_eye_z", renpy.config.perspective[1])
        self.model.uniform(
            "u_light_dir",
            obj.material.light_dir if obj.material.light_dir is not None
            else LIGHT_DIR)
        for name, value in obj.material.params.items():
            self.model.uniform(name, value)
        if obj.material.blend == "alpha clip":
            self.model.uniform("u_alpha_clip", obj.material.alpha_clip)
        turntable = Transform(
            self.model,
            matrixtransform=_Matrix.rotate(obj.tilt, obj.angle, 0.0),
            perspective=True)
        if obj.animated:
            # redraw() должен получать И model: кеш render() ключуется
            # (id, w/h, frame_time-st), где frame_time-st константа, а id
            # модели стабилен — без redraw(model) Model.render не rerun-ится
            # и draw вечно видит uniform'ы первого кадра (u_spin=0).
            # (Меш-мутации видны всегда: draw читает живые C-буферы Mesh,
            # а uniform'ы — скопированные значения Render.)
            renpy.display.render.redraw(self, 0.016)
            renpy.display.render.redraw(self.model, 0.016)
        return renpy.display.render.render(turntable, w, h, st, at)


class Skeleton(object):
    """Joint hierarchy + bind pose + procedural overrides.

    Composes skinning matrices per frame on CPU (215 joints = milliseconds);
    per-vertex deformation runs on GPU from the resulting palette.

    update() answers in the mesh's buffer space -- the one _fit_transforms
    describes. Conjugation distributes over the hierarchy
    (FIT*(A*B)*FIT^-1 == FIT*A*FIT^-1 * FIT*B*FIT^-1), so use_fit_space()
    folds FIT into the node and inverse-bind constants once and no frame
    ever conjugates again.
    """

    def __init__(self, gltf, skin_idx=0):
        import es_gltf
        self.gltf = gltf
        skin = gltf.skin(skin_idx)
        self.joint_nodes = list(skin["joints"])
        self.joint_names = [gltf.json["nodes"][ni].get("name", "j%d" % k)
                            for k, ni in enumerate(self.joint_nodes)]
        node_count = len(gltf.json.get("nodes", []))
        self.node_parent = {}
        for parent_index, kids in enumerate(gltf.children()):
            for kid in kids:
                self.node_parent[kid] = parent_index
        self.node_order = []
        visited = set()

        def visit(node_index):
            if node_index in visited:
                return
            if node_index in self.node_parent:
                visit(self.node_parent[node_index])
            visited.add(node_index)
            self.node_order.append(node_index)

        for node_index in range(node_count):
            visit(node_index)
        # A node carries either a matrix or TRS in the file; keep both shapes.
        self.local_specs = {}
        for node_index in range(node_count):
            spec = gltf.node_local(node_index)
            if spec[0] == "matrix":
                self.local_specs[node_index] = ("matrix", spec[1])
            else:
                self.local_specs[node_index] = ("trs",
                                                 spec[1], spec[2], spec[3])
        # Procedural posing: node index -> (x, y, z, w) replacing its rotation.
        self.rotation_overrides = {}
        # Rebind: the file's inverseBindMatrices match a different node pose
        # than the file's node TRS (retopo asset). The rest pose the NODES
        # define is what the rigid demos render, so invert THOSE worlds.
        bind_world = {}
        for node_index in self.node_order:
            local = self._rest_local(node_index)
            parent = self.node_parent.get(node_index)
            if parent is None:
                bind_world[node_index] = local
            else:
                bind_world[node_index] = es_gltf.mat4_mul(
                    bind_world[parent], local)
        self.inverse_bind = [
            es_gltf.mat4_inverse(bind_world[ni]) for ni in self.joint_nodes]
        # Fit-space constants. SkinnedObject._build_splits supplies the fit
        # once it has measured the mesh; use_fit_space() must have run before
        # update().
        self.fit = None
        self.fit_inverse = None
        self.local_fit = {}
        self.inverse_bind_fit = []

    def use_fit_space(self, fit, fit_inverse):
        """Fold the mesh's fit transform into every constant matrix.

        Called once by SkinnedObject._build_splits, which owns the scale,
        center and pivot mapping rig space onto the vertex buffer. This pays
        for the conjugations the palette used to redo per frame and per slot.
        """
        self.fit = fit
        self.fit_inverse = fit_inverse
        self.local_fit = {}
        for node_index in self.local_specs:
            self.local_fit[node_index] = self._to_fit(
                self._rest_local(node_index))
        self.inverse_bind_fit = [self._to_fit(ibm)
                                 for ibm in self.inverse_bind]

    def _to_fit(self, matrix):
        """Rig-space matrix -> the same transform in buffer (fit) space."""
        import es_gltf
        return es_gltf.mat4_mul(
            es_gltf.mat4_mul(self.fit, matrix), self.fit_inverse)

    def _rest_local(self, node_index):
        """Node's local matrix exactly as the file defines it."""
        import es_gltf
        spec = self.local_specs[node_index]
        if spec[0] == "matrix":
            return spec[1]
        _, translation, rotation, scale = spec
        return es_gltf.mat4_trs(translation, rotation, scale)

    def _posed_local(self, node_index, rotation):
        """Local matrix with `rotation` replacing the file's own."""
        import es_gltf
        spec = self.local_specs[node_index]
        if spec[0] == "matrix":
            # A matrix node has no TRS rotation to replace: not posable.
            return spec[1]
        _, translation, _rotation, scale = spec
        return es_gltf.mat4_trs(translation, rotation, scale)

    def joint_index(self, name_or_index):
        if isinstance(name_or_index, int):
            return name_or_index
        return self.joint_names.index(name_or_index)

    def pose_quat(self, name_or_index, quat):
        """Override a joint's local rotation (procedural posing)."""
        joint = self.joint_index(name_or_index)
        self.rotation_overrides[self.joint_nodes[joint]] = tuple(quat)

    def pose_euler(self, name_or_index, rx=0.0, ry=0.0, rz=0.0):
        import math as _m
        ax, ay, az = _m.radians(rx), _m.radians(ry), _m.radians(rz)
        cx, sx = _m.cos(ax / 2.0), _m.sin(ax / 2.0)
        cy, sy = _m.cos(ay / 2.0), _m.sin(ay / 2.0)
        cz, sz = _m.cos(az / 2.0), _m.sin(az / 2.0)
        # q = qy * qx * qz applied to (x,y,z,w) components
        qx = (sx, 0.0, 0.0, cx)
        qy = (0.0, sy, 0.0, cy)
        qz = (0.0, 0.0, sz, cz)

        def quat_mul(a, b):
            ax_, ay_, az_, aw_ = a
            bx_, by_, bz_, bw_ = b
            return (aw_ * bx_ + ax_ * bw_ + ay_ * bz_ - az_ * by_,
                    aw_ * by_ - ax_ * bz_ + ay_ * bw_ + az_ * bx_,
                    aw_ * bz_ + ax_ * by_ - ay_ * bx_ + az_ * bw_,
                    aw_ * bw_ - ax_ * bx_ - ay_ * by_ - az_ * bz_)

        self.pose_quat(name_or_index, quat_mul(quat_mul(qy, qx), qz))

    def update(self):
        """Skinning matrix per joint, in buffer space: {joint index: mat4}.

        One multiply per hierarchy edge and one per joint -- FIT is already
        folded into both factors by use_fit_space().
        """
        import es_gltf
        local = self.local_fit
        if self.rotation_overrides:
            local = dict(local)
            for node_index, rotation in self.rotation_overrides.items():
                local[node_index] = self._to_fit(
                    self._posed_local(node_index, rotation))
        chain = {}
        for node_index in self.node_order:
            parent = self.node_parent.get(node_index)
            if parent is None:
                chain[node_index] = local[node_index]
            else:
                chain[node_index] = es_gltf.mat4_mul(chain[parent],
                                                     local[node_index])
        return dict(
            (joint, es_gltf.mat4_mul(chain[node],
                                     self.inverse_bind_fit[joint]))
            for joint, node in enumerate(self.joint_nodes))


def _fit_transforms(half_size, fit_scale, bounds_center):
    """(fit, fit_inverse): rig space <-> the space the vertex buffer lives in.

    fit carries a rig-space point onto the uploaded verts -- bring the bounds
    center to the origin, scale to the target size, flip Y (glTF is Y-up, the
    screen is Y-down), then offset by the half-size the Model draws at.
    fit_inverse undoes the same steps in reverse order.
    """
    import es_gltf
    no_rotation = (0.0, 0.0, 0.0, 1.0)
    unit_scale = (1.0, 1.0, 1.0)
    origin = (0.0, 0.0, 0.0)
    pivot = (half_size, half_size, 0.0)
    mul = es_gltf.mat4_mul
    trs = es_gltf.mat4_trs
    to_bounds_center = trs(tuple(-x for x in bounds_center),
                           no_rotation, unit_scale)
    away_from_bounds_center = trs(tuple(bounds_center),
                                  no_rotation, unit_scale)
    to_pivot = trs(pivot, no_rotation, unit_scale)
    away_from_pivot = trs(tuple(-x for x in pivot), no_rotation, unit_scale)
    enlarge = trs(origin, no_rotation, (fit_scale, fit_scale, fit_scale))
    shrink = trs(origin, no_rotation, (1.0 / fit_scale, 1.0 / fit_scale,
                                       1.0 / fit_scale))
    flip_y = trs(origin, no_rotation, (1.0, -1.0, 1.0))
    return (mul(mul(mul(to_pivot, flip_y), enlarge), to_bounds_center),
            mul(mul(mul(away_from_bounds_center, shrink), flip_y),
                away_from_pivot))


def _view_depth(centroid, size, tilt, angle):
    """Rotated z of a buffer-space point; larger is nearer the viewer.

    Mirrors the rotation es_shaders._VERTEX applies to normals and the model
    matrix the Transform builds, so the blended pass can be ordered
    farthest-first without asking GL anything. Buffer verts sit around the
    pivot, hence the half-size subtraction."""
    import math
    x = centroid[0] - size / 2.0
    y = centroid[1] - size / 2.0
    z = centroid[2]
    sa = math.sin(math.radians(angle))
    ca = math.cos(math.radians(angle))
    sx = math.sin(math.radians(tilt))
    cx = math.cos(math.radians(tilt))
    return -sa * x + sx * ca * y + cx * ca * z


class SkinnedObject(Object3D):
    """glTF mesh skinned on GPU. update() poses joints, view pushes palette."""

    def __init__(self, gltf, size=600.0, speed=30.0, tilt=25.0,
                 color=(0.85, 0.8, 0.75), max_bones=None, target=300.0,
                 recipe=None):
        import es_gltf
        if isinstance(gltf, basestring):
            gltf = es_gltf.Gltf.load(gltf)
        material = Material(
            base_color=(color[0], color[1], color[2], 1.0),
            skinned=True, max_bones=max_bones, recipe=recipe)
        super(SkinnedObject, self).__init__(size, speed, tilt, material)
        self.skel = Skeleton(gltf)
        self.splits = []
        # A recipe that names a glTF material takes over that group's splits
        # only; every other group keeps this object's material.
        self.part_materials = self._part_materials(gltf)
        # Shape keys: name -> weight (active set), union of pool names.
        self.shapes = {}
        self.shape_names = []
        self._shape_seen = set()
        self._build_splits(gltf, target)

    def _part_materials(self, gltf):
        """{glTF material index: Material} for recipes declaring `applies`."""
        by_name = {}
        for index, entry in enumerate(gltf.json.get("materials", [])):
            by_name[entry.get("name")] = index
        found = {}
        for recipe in es_shaders.RECIPES:
            target = es_shaders.recipe_applies(recipe)
            if target is None:
                continue
            index = by_name.get(target)
            if index is None:
                # A mistyped name would otherwise be ignored in silence.
                print("ES_MAT %s applies to %r: no such material in this "
                      "asset" % (recipe, target))
                continue
            found[index] = Material(
                base_color=self.material.base_color, skinned=True,
                max_bones=self.material.max_bones, recipe=recipe)
        return found

    def set_shape(self, name, weight):
        """Set a shape key weight (typically 0..1). Names unknown to
        every morph-capable pool are dropped silently (per-mesh name
        sets differ)."""
        if name in self._shape_seen:
            self.shapes[name] = float(weight)

    def _build_splits(self, gltf, target):
        import es_gltf
        node_world = es_gltf.node_world_matrices(gltf)
        half_size = self.size / 2.0
        # One fit for every split (the same numbers the bind-pose demos use),
        # measured over all primitives so the parts stay aligned.
        bounds_min = [1e30, 1e30, 1e30]
        bounds_max = [-1e30, -1e30, -1e30]
        prim_data = []
        for prim in gltf.primitives():
            positions = gltf.read_accessor(prim["position"])
            normals = (gltf.read_accessor(prim["normal"])
                       if prim["normal"] is not None else None)
            indices = [int(x) for x in gltf.read_accessor(prim["indices"])]
            joint_rows = gltf.read_accessor(prim["joints"])
            weight_rows = gltf.read_accessor(prim["weights"], normalize=True)
            uv_rows = (gltf.read_accessor(prim["uv"])
                       if prim["uv"] is not None else None)
            node_matrix = (node_world[prim["nodes"][0]] if prim["nodes"]
                           else es_gltf.mat4_identity())
            prim_data.append((positions, normals, uv_rows, indices,
                              joint_rows, weight_rows, node_matrix,
                              prim["material"], prim.get("targets"),
                              prim.get("target_names")))
            for point in positions:
                world_point = es_gltf.mat4_transform_point(node_matrix, point)
                for axis in range(3):
                    if world_point[axis] < bounds_min[axis]:
                        bounds_min[axis] = world_point[axis]
                    if world_point[axis] > bounds_max[axis]:
                        bounds_max[axis] = world_point[axis]
        extent = max(bounds_max[k] - bounds_min[k] for k in range(3)) or 1.0
        fit_scale = target / extent
        bounds_center = [(bounds_min[k] + bounds_max[k]) / 2.0
                         for k in range(3)]
        print("ES_FIT nprims=%d extent=%.3f scale=%.3f center=%r size=%.1f target=%.1f" % (
            len(prim_data), extent, fit_scale,
            tuple(round(x, 3) for x in bounds_center), self.size, target))
        self.fit, self.fit_inverse = _fit_transforms(
            half_size, fit_scale, bounds_center)
        bone_cap = self.material.max_bones
        # Condense by material first: merge primitives sharing a material
        # into one pool, then joint-split each pool. Fewer state changes
        # per material, and a home for future per-material maps.
        pools = {}
        for (positions, normals, uv_rows, indices, joint_rows, weight_rows,
             node_matrix, material_id,
             morph_targets, morph_names) in prim_data:
            verts = []
            for point in positions:
                world_point = es_gltf.mat4_transform_point(node_matrix, point)
                verts.append(((world_point[0] - bounds_center[0]) * fit_scale,
                              (world_point[1] - bounds_center[1]) * fit_scale,
                              (world_point[2] - bounds_center[2]) * fit_scale))
            unit_normals = []
            for normal in (normals or [(0.0, 0.0, 1.0)] * len(positions)):
                length = ((normal[0] * normal[0] + normal[1] * normal[1]
                           + normal[2] * normal[2]) ** 0.5) or 1.0
                unit_normals.append(es_gltf.mat4_transform_dir(
                    node_matrix, (normal[0] / length, normal[1] / length,
                                  normal[2] / length)))
            pool = pools.setdefault(material_id, {
                "verts": [], "normals": [], "uvs": [], "tris": [],
                "joints": [], "weights": []})
            if uv_rows is None:
                # a_uv is one attribute buffer for the whole pool: mixing
                # UV'd and bare primitives draws without UVs at all.
                pool["uvs"] = None
            elif pool["uvs"] is not None:
                pool["uvs"].extend(uv_rows)
            if "morph_source" not in pool:
                # Morph source for this pool: (node matrix, targets,
                # target names). Merged multi-primitive pools clear it
                # below -- shared atlas rows would be ambiguous.
                pool["morph_source"] = (node_matrix, morph_targets,
                                        morph_names)
            else:
                pool["morph_source"] = None
            first_vert = len(pool["verts"])
            pool["verts"].extend(verts)
            pool["normals"].extend(unit_normals)
            pool["joints"].extend([tuple(int(x) for x in row)
                                   for row in joint_rows])
            pool["weights"].extend([tuple(float(x) for x in row)
                                    for row in weight_rows])
            pool["tris"].extend(
                [tuple(first_vert + x for x in indices[k:k + 3])
                 for k in range(0, len(indices), 3)])
        print("ES_MATS groups=%d tris=%r" % (
            len(pools),
            sorted([(material_id, len(pool["tris"]))
                    for material_id, pool in pools.items()])))
        ordered_pools = sorted(pools.items(), key=lambda kv: str(kv[0]))
        for material_id, pool in ordered_pools:
            verts = pool["verts"]
            normals = pool["normals"]
            uvs = pool["uvs"]
            tris = pool["tris"]
            joint_rows = pool["joints"]
            weight_rows = pool["weights"]
            # One RGBA8 atlas per pool: deltas in buffer space (fit scale
            # + Y-flip applied here, same as the base vertex copy).
            morph = None
            morph_source = pool.get("morph_source")
            # morph_ok: vertex texture fetch / uniform budget must exist.
            if (morph_source and morph_source[1] and morph_source[2]
                    and es_caps.morph_ok()):
                morph = es_morph.build(gltf, morph_source[0], morph_source[1],
                                       morph_source[2], len(verts), fit_scale)
                if morph is not None:
                    es_morph.attach(morph)
            if morph is not None:
                for shape_name in morph["names"]:
                    if shape_name not in self._shape_seen:
                        self._shape_seen.add(shape_name)
                        self.shape_names.append(shape_name)
                es_caps.check_atlas(morph["width"], morph["height"])
                print("ES_MORPH shapes=%d verts=%d atlas=%dx%d "
                      "mscale=%.5f nscale=%.5f" % (
                          morph["nshapes"], morph["nverts"],
                          morph["width"], morph["height"],
                          morph["mscale"], morph["nscale"]))
            for part_tris, part_joints in es_gltf.split_by_joints(
                    tris, joint_rows, weight_rows, bone_cap):
                referenced = set()
                for tri in part_tris:
                    referenced.update(tri)
                kept = sorted(referenced)
                global_to_local = dict((global_index, local_index)
                                       for local_index, global_index
                                       in enumerate(kept))
                part_verts = [None] * len(kept)
                part_normals = [None] * len(kept)
                part_joint_ids = [None] * len(kept)
                part_weights = [None] * len(kept)
                part_atlas_ids = [None] * len(kept)
                part_joint_set = set(part_joints)
                for global_index, local_index in global_to_local.items():
                    # Y-flip (glTF Y-up -> screen Y-down) baked here; the
                    # skeleton's fit-space constants carry the same flip.
                    part_verts[local_index] = (
                        half_size + verts[global_index][0],
                        half_size - verts[global_index][1],
                        verts[global_index][2])
                    part_normals[local_index] = (
                        normals[global_index][0],
                        -normals[global_index][1],
                        normals[global_index][2])
                    # Pool-local vertex id: indexes the morph atlas rows
                    # exactly like the base vertex list above. 1-tuple:
                    # mesh3_from_layout iterates each group's frame value.
                    part_atlas_ids[local_index] = (float(global_index),)
                    # Нулевые веса ссылаются на кости вне палитры —
                    # им все равно какой индекс (вклад нулевой).
                    part_joint_ids[local_index] = tuple(
                        float(part_joints.index(int(joint)))
                        if (float(weight) > 0.0
                            and int(joint) in part_joint_set)
                        else 0.0
                        for joint, weight in zip(joint_rows[global_index],
                                                 weight_rows[global_index]))
                    part_weights[local_index] = weight_rows[global_index]
                part_indices = [tuple(global_to_local[x] for x in tri)
                                for tri in part_tris]
                frames = [part_joint_ids, part_weights, part_normals,
                          part_atlas_ids]
                layout = [("a_joints", 4), ("a_weights", 4),
                          ("a_norm3", 3), ("a_vid", 1)]
                if uvs:
                    # kept[local] == global, so the pool's UV list slices
                    # straight into split order.
                    frames.append([uvs[g] for g in kept])
                    layout.append(("a_uv", 2))
                mesh = es_step3.mesh3_from_layout(
                    part_verts, frames, part_indices, layout)
                count = float(len(part_verts))
                # Buffer-space centroid: the transparent pass orders splits
                # by how near this point ends up after the turn.
                centroid = (sum(v[0] for v in part_verts) / count,
                            sum(v[1] for v in part_verts) / count,
                            sum(v[2] for v in part_verts) / count)
                self.splits.append({"mesh": mesh, "joints": part_joints,
                                    "model": None, "morph": morph,
                                    "uvs": bool(uvs), "centroid": centroid,
                                    "material": self.part_materials.get(
                                        material_id, self.material)})
        # The palette composes in this buffer space, not rig space: handing
        # the fit to the skeleton once is what keeps FIT out of every frame.
        self.skel.use_fit_space(self.fit, self.fit_inverse)

    def update(self, st, dt):
        self.angle = (st * self.speed) % 360.0


class SkinnedView(Displayable):
    """Drives one SkinnedObject: pose -> palette -> N model draws."""

    def __init__(self, obj, **kwargs):
        super(SkinnedView, self).__init__(**kwargs)
        es_caps.ensure()
        self.obj = obj
        self.splits = obj.splits
        self.models = []
        self.bone_count = obj.material.max_bones
        self.shape_slot_count = es_caps.slots()
        self._identity = None
        logged_morph_shader = False
        for split in obj.splits:
            model = renpy.display.model.Model(size=(obj.size, obj.size))
            model.shader("renpy.geometry")
            material = split["material"]
            morph = split["morph"]
            # Only atlas-carrying splits get the morph program: it alone
            # declares tex0/u_sw* (draw() raises on unset uniforms).
            shader_name = material.shader_name(
                morph=(self.shape_slot_count if morph is not None else 0),
                uvs=bool(split["uvs"]))
            model.shader(shader_name)
            if morph is not None and not logged_morph_shader:
                logged_morph_shader = True
                print("ES_DBG_VIEW size=%.0f shader=%s mscale=%.4f "
                      "nverts=%d shapes=%d seen=%d" % (
                          obj.size, shader_name, morph["mscale"],
                          morph["nverts"], len(obj.shape_names),
                          len(obj._shape_seen)))
            # Без личного depth: общий клеар — на родителе (см. render),
            # иначе каждый сплит чистит буфер и рушит окклюзию соседей.
            model.uniform("u_view_dir", Object3D.view_dir)
            model.uniform("u_base_color", material.base_color)
            # Slot 0 is always filled: the morph atlas where this pool has
            # one, the shared 1x1 white texture otherwise. Ren'Py names
            # samplers by insertion order, so filling it here is what lets a
            # material's own map count on being tex1 on every split.
            model.texture(morph["displayable"] if morph is not None
                          else es_morph.placeholder())
            texture = material.map_texture()
            if texture is not None:
                model.texture(texture)
            if morph is not None:
                model.uniform("u_mscale", morph["mscale"])
                model.uniform("u_nscale", morph["nscale"])
                model.uniform("u_nv", float(morph["nverts"]))
                model.uniform("u_as", (float(morph["width"]),
                                       float(morph["height"])))
            model._mesh = split["mesh"]
            if material.blend == "alpha":
                # Ren'Py's "normal" blend is premultiplied (GL_ONE,
                # GL_ONE_MINUS_SRC_ALPHA) and the generated fragment tail
                # premultiplies es_color to match. Depth writes cannot be
                # turned off -- glDepthMask is not bound in this build -- so
                # these draws go last, farthest-first, in render().
                model.property("blend_func",
                               renpy.config.gl_blend_func["normal"])
            split["model"] = model
            self.models.append(model)
        # Two ordered groups: opaque and alpha-clip write depth first, then
        # the blended splits go over them farthest-first.
        self.opaque_draws = [i for i, sp in enumerate(obj.splits)
                             if sp["material"].blend != "alpha"]
        self.blend_draws = [i for i, sp in enumerate(obj.splits)
                            if sp["material"].blend == "alpha"]

    @staticmethod
    def _flatten(matrix):
        """Row-major 4x4 -> the 16-float list a renpy Matrix is built from."""
        return [float(matrix[i][j]) for i in range(4) for j in range(4)]

    def render(self, w, h, st, at):
        obj = self.obj
        last = obj._last_st
        dt = 0.0 if last is None else st - last
        obj._last_st = st
        obj.update(st, dt)
        light = (obj.material.light_dir
                 if obj.material.light_dir is not None else LIGHT_DIR)
        # skel.update() already answers in buffer space, so a joint matrix
        # needs no per-slot work here. Splits share joints, so the list +
        # Matrix construction runs once per joint, not once per slot.
        palette = dict((joint, _Matrix(self._flatten(bone_matrix)))
                       for joint, bone_matrix in obj.skel.update().items())
        if self._identity is None:
            self._identity = _Matrix([1.0, 0.0, 0.0, 0.0,
                                      0.0, 1.0, 0.0, 0.0,
                                      0.0, 0.0, 1.0, 0.0,
                                      0.0, 0.0, 0.0, 1.0])
        for split, model in zip(self.splits, self.models):
            split_joints = split["joints"]
            for local_index, joint in enumerate(split_joints):
                model.uniform("u_bone%d" % local_index, palette[joint])
            # draw() требует ВСЕ объявленные uniform'ы: добиваем identity.
            for local_index in range(len(split_joints), self.bone_count):
                model.uniform("u_bone%d" % local_index, self._identity)
            model.uniform("u_spin", obj.angle)
            model.uniform("u_tilt", obj.tilt)
            model.uniform("u_eye_z", renpy.config.perspective[1])
            model.uniform("u_light_dir", light)
            for name, value in split["material"].params.items():
                model.uniform(name, value)
            if split["material"].blend == "alpha clip":
                # Pushed per frame rather than at build so the threshold is
                # tunable live, like every other parameter.
                model.uniform("u_alpha_clip", split["material"].alpha_clip)
            morph = split["morph"]
            if morph is not None:
                # Active (row, weight) pairs for this pool's atlas,
                # padded to the declared slot count like u_bone* above.
                atlas_rows = morph["row"]
                active_shapes = []
                for shape_name, weight in obj.shapes.items():
                    row = atlas_rows.get(shape_name)
                    if row is not None and weight:
                        active_shapes.append((row, float(weight)))
                active_shapes.sort()
                active_shapes = active_shapes[:self.shape_slot_count]
                if not getattr(self, "_logged_slots", False) and active_shapes:
                    self._logged_slots = True
                    print("ES_DBG_SLOTS %.1f size=%.0f %r" % (
                        st, obj.size, active_shapes))
                for slot in range(self.shape_slot_count):
                    if slot < len(active_shapes):
                        model.uniform("u_sw%d" % slot, active_shapes[slot])
                    else:
                        model.uniform("u_sw%d" % slot, (0.0, 0.0))
        rv = renpy.display.render.Render(int(obj.size), int(obj.size))
        # Один depth-клеар на всех соседей: draw() чистит буфер в узле
        # с property depth и ставит has_depth наследникам — сплиты делят
        # один тест вместо чистки на каждый (иначе порядок неверный).
        # That sharing is measured, not assumed: two quads at different Z
        # blitted into one depth-enabled parent in either order produce
        # byte-identical frames, so the blended splits can safely go last.
        rv.add_property("depth", True)
        order = list(self.opaque_draws)
        if self.blend_draws:
            order.extend(sorted(self.blend_draws, key=lambda i: _view_depth(
                self.splits[i]["centroid"], obj.size, obj.tilt, obj.angle)))
            if not getattr(self, "_logged_order", False):
                self._logged_order = True
                print("ES_DBG_ORDER opaque=%r blended=%r (farthest first)" % (
                    self.opaque_draws, order[len(self.opaque_draws):]))
        for index in order:
            model = self.models[index]
            drawn = renpy.display.render.render(
                _TransformWrap(model, obj), obj.size, obj.size, st, at)
            rv.blit(drawn, (0, 0))
            renpy.display.render.redraw(model, 0.016)
        if obj.animated:
            renpy.display.render.redraw(self, 0.016)
        return rv


class _TransformWrap(Displayable):
    """One Model + its rotation matrix, for compositing inside a view."""

    def __init__(self, model, obj, **kwargs):
        super(_TransformWrap, self).__init__(**kwargs)
        self.model = model
        self.obj = obj

    def render(self, w, h, st, at):
        turntable = Transform(
            self.model,
            matrixtransform=_Matrix.rotate(self.obj.tilt, self.obj.angle, 0.0),
            perspective=True)
        return renpy.display.render.render(turntable, w, h, st, at)
