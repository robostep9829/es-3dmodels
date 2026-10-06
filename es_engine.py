# -*- coding: utf-8 -*-
"""Readable 3D engine over Ren'Py's Model/Mesh3.

Object3D stores geometry + material + animation state. Subclass it and
override update() to animate. ObjectView drives one object per frame:
update() -> uniforms -> nested Transform (model matrix) -> child render.

Proven rules this encodes (see project history):
- rotation lives in the model matrix (auto-centered pivot), never in verts,
- Mesh3 fields are cdef-only: uploads go through es_step3's poked offsets,
- depth test needs the direct-draw path: show WITHOUT dissolve,
- normals rotate analytically in-shader (u_transform collapses them).
"""
import os
import sys

import renpy
import renpy.display.model
import renpy.display.render
import renpy.display.matrix
from renpy.display.core import Displayable
from renpy.display.transform import Transform

_md = os.path.dirname(renpy.loader.transfn("mods/es-3dmodels/es_engine.py"))
if _md not in sys.path:
    sys.path.insert(0, _md)

import es_step3
import es_shaders
import es_gltf


def es_gltf_bind(path, target=300.0):
    """Load a .gltf/.glb bind pose: (verts, normals, tris), fit to target."""
    return es_gltf.bind_pose_mesh(es_gltf.Gltf.load(path), target)
import es_obj

_Matrix = renpy.display.matrix.Matrix

# Global scene light. Materials with light_dir=None follow this;
# a per-material light_dir overrides it for that object only.
LIGHT_DIR = (0.5, -0.8, 0.7)


class Material(object):
    """Surface + lighting preset. Flags select a compiled shader variant."""

    def __init__(self, base_color=(0.9, 0.3, 0.3, 1.0),
                 light_dir=None,
                 shininess=16.0, ambient=0.3, diffuse=0.7,
                 spec_weight=0.5, specular=True, normal_viz=False,
                 skinned=False, max_bones=24):
        self.base_color = base_color
        self.light_dir = light_dir
        self.shininess = shininess
        self.ambient = ambient
        self.diffuse = diffuse
        self.spec_weight = spec_weight
        self.specular = specular
        self.normal_viz = normal_viz
        self.skinned = skinned
        self.max_bones = max_bones

    def variant_key(self):
        return (bool(self.specular), bool(self.normal_viz),
                float(self.shininess), float(self.ambient),
                float(self.diffuse), float(self.spec_weight),
                bool(self.skinned), int(self.max_bones))

    def shader_name(self):
        try:
            return es_shaders.get_variant(
                specular=self.specular, normal_viz=self.normal_viz,
                shininess=self.shininess, ambient=self.ambient,
                diffuse=self.diffuse, spec_weight=self.spec_weight,
                skinned=self.skinned, max_bones=self.max_bones)
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
        s = self.size / 2.0
        return ([(s + x, s + y, z) for (x, y, z) in self.verts],
                list(self.normals), list(self.tris))

    def build_mesh(self):
        v, n, t = self.centered()
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
    """Geometry loaded from a Wavefront OBJ file. Animate via update()."""

    @classmethod
    def from_data(cls, verts, normals, tris, size=500.0, speed=45.0,
                  tilt=25.0, color=(0.85, 0.85, 0.9)):
        obj = cls(size=size, speed=speed, tilt=tilt,
                  material=Material(
                      base_color=(color[0], color[1], color[2], 1.0)))
        obj.verts = list(verts)
        obj.normals = list(normals)
        obj.tris = list(tris)
        obj.build_mesh()
        return obj

    @classmethod
    def from_obj(cls, path, size=500.0, speed=60.0, tilt=25.0,
                 color=(0.85, 0.85, 0.9), target=300.0):
        obj = cls(size=size, speed=speed, tilt=tilt,
                  material=Material(
                      base_color=(color[0], color[1], color[2], 1.0)))
        verts, normals, tris = es_obj.load_obj(path, target)
        for v, n in zip(verts, normals):
            obj.verts.append(v)
            obj.normals.append(n)
        obj.tris = list(tris)
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
        key = self.obj.material.variant_key()
        if key == self._shader_key and self.model is not None:
            return
        self._shader_key = key
        m = renpy.display.model.Model(size=(self.obj.size, self.obj.size))
        m.shader("renpy.geometry")
        m.shader(self.obj.material.shader_name())
        m.property("depth", True)
        m.uniform("u_view_dir", Object3D.view_dir)
        m.uniform("u_base_color", self.obj.material.base_color)
        m._mesh = self.obj.mesh
        self.model = m

    def render(self, w, h, st, at):
        obj = self.obj
        last = obj._last_st
        dt = 0.0 if last is None else st - last
        obj._last_st = st
        obj.update(st, dt)
        self._sync_material()
        self.model.uniform("u_spin", obj.angle)
        self.model.uniform(
            "u_light_dir",
            obj.material.light_dir if obj.material.light_dir is not None
            else LIGHT_DIR)
        inner = Transform(
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
        return renpy.display.render.render(inner, w, h, st, at)


class Skeleton(object):
    """Joint hierarchy + bind pose + procedural overrides.

    Composes world matrices per frame on CPU (215 joints = milliseconds);
    per-vertex deformation runs on GPU from the resulting palette.
    """

    def __init__(self, gltf, skin_idx=0):
        import es_gltf
        self.gltf = gltf
        sk = gltf.skin(skin_idx)
        self.joint_nodes = list(sk["joints"])
        self.inverse_bind = [tuple(tuple(r) for r in m)
                             for m in sk["inverse_bind"]]
        self.names = [gltf.json["nodes"][ni].get("name", "j%d" % k)
                      for k, ni in enumerate(self.joint_nodes)]
        children = gltf.children()
        parent = {}
        for pi, ch in enumerate(children):
            for c in ch:
                parent[c] = pi
        order = []
        seen = set()

        def visit(ni):
            if ni in seen:
                return
            if ni in parent:
                visit(parent[ni])
            seen.add(ni)
            order.append(ni)

        for ni in range(len(gltf.json.get("nodes", []))):
            visit(ni)
        self.order = order
        self.parent = parent
        self.base_local = {}
        for ni in range(len(gltf.json.get("nodes", []))):
            kind = gltf.node_local(ni)
            if kind[0] == "matrix":
                self.base_local[ni] = ("matrix", kind[1])
            else:
                self.base_local[ni] = ("trs", kind[1], kind[2], kind[3])
        # overrides: node idx -> ("quat", (x,y,z,w)) replacing rotation
        self.overrides = {}
        self.worlds = {}

    def joint_index(self, name_or_idx):
        if isinstance(name_or_idx, int):
            return name_or_idx
        return self.names.index(name_or_idx)

    def pose_quat(self, name_or_idx, quat):
        """Override a joint's local rotation (procedural posing)."""
        ji = self.joint_index(name_or_idx)
        self.overrides[self.joint_nodes[ji]] = ("quat", tuple(quat))

    def pose_euler(self, name_or_idx, rx=0.0, ry=0.0, rz=0.0):
        import math as _m
        ax, ay, az = _m.radians(rx), _m.radians(ry), _m.radians(rz)
        cx, sx = _m.cos(ax / 2.0), _m.sin(ax / 2.0)
        cy, sy = _m.cos(ay / 2.0), _m.sin(ay / 2.0)
        cz, sz = _m.cos(az / 2.0), _m.sin(az / 2.0)
        # q = qy * qx * qz applied to (x,y,z,w) components
        qx = (sx, 0.0, 0.0, cx)
        qy = (0.0, sy, 0.0, cy)
        qz = (0.0, 0.0, sz, cz)

        def mul(a, b):
            ax_, ay_, az_, aw_ = a
            bx_, by_, bz_, bw_ = b
            return (aw_ * bx_ + ax_ * bw_ + ay_ * bz_ - az_ * by_,
                    aw_ * by_ - ax_ * bz_ + ay_ * bw_ + az_ * bx_,
                    aw_ * bz_ + ax_ * by_ - ay_ * bx_ + az_ * bw_,
                    aw_ * bw_ - ax_ * bx_ - ay_ * by_ - az_ * bz_)

        self.pose_quat(name_or_idx, mul(mul(qy, qx), qz))

    def update(self):
        """Recompute world matrices. Returns {joint_idx: mat4}."""
        import es_gltf
        local = {}
        for ni, spec in self.base_local.items():
            if ni in self.overrides:
                _, q = self.overrides[ni]
                if spec[0] == "matrix":
                    local[ni] = spec[1]
                else:
                    _, t, _r, s = spec
                    local[ni] = es_gltf.mat4_trs(t, q, s)
            else:
                if spec[0] == "matrix":
                    local[ni] = spec[1]
                else:
                    _, t, r, s = spec
                    local[ni] = es_gltf.mat4_trs(t, r, s)
        worlds = {}
        for ni in self.order:
            if ni in self.parent:
                worlds[ni] = es_gltf.mat4_mul(worlds[self.parent[ni]],
                                              local[ni])
            else:
                worlds[ni] = local[ni]
        self.worlds = worlds
        out = {}
        for ji, ni in enumerate(self.joint_nodes):
            out[ji] = es_gltf.mat4_mul(worlds[ni], self.inverse_bind[ji])
        return out


class SkinnedObject(Object3D):
    """glTF mesh skinned on GPU. update() poses joints, view pushes palette."""

    def __init__(self, gltf, size=600.0, speed=30.0, tilt=25.0,
                 color=(0.85, 0.8, 0.75), max_bones=24, target=300.0):
        import es_gltf
        if isinstance(gltf, basestring):
            gltf = es_gltf.Gltf.load(gltf)
        material = Material(
            base_color=(color[0], color[1], color[2], 1.0),
            skinned=True, max_bones=max_bones)
        super(SkinnedObject, self).__init__(size, speed, tilt, material)
        self.skel = Skeleton(gltf)
        self.splits = []
        self._build_splits(gltf, target)

    def _build_splits(self, gltf, target):
        import es_gltf
        world = es_gltf.node_world_matrices(gltf)
        s = self.size / 2.0
        # fit factor shared across splits (same as bind pose)
        all_min = [1e30, 1e30, 1e30]
        all_max = [-1e30, -1e30, -1e30]
        prims = []
        for p in gltf.primitives():
            pos = gltf.read_accessor(p["position"])
            nor = (gltf.read_accessor(p["normal"])
                   if p["normal"] is not None else None)
            idx = [int(x) for x in gltf.read_accessor(p["indices"])]
            jts = gltf.read_accessor(p["joints"])
            wts = gltf.read_accessor(p["weights"], normalize=True)
            m = (world[p["nodes"][0]] if p["nodes"]
                 else es_gltf.mat4_identity())
            prims.append((pos, nor, idx, jts, wts, m))
            for v in pos:
                wv = es_gltf.mat4_transform_point(m, v)
                for k in range(3):
                    if wv[k] < all_min[k]:
                        all_min[k] = wv[k]
                    if wv[k] > all_max[k]:
                        all_max[k] = wv[k]
        extent = max(all_max[k] - all_min[k] for k in range(3)) or 1.0
        scale = target / extent
        center = [(all_min[k] + all_max[k]) / 2.0 for k in range(3)]
        cap = self.material.max_bones
        for pos, nor, idx, jts, wts, m in prims:
            v = [((es_gltf.mat4_transform_point(m, p)[0] - center[0]) * scale,
                  (es_gltf.mat4_transform_point(m, p)[1] - center[1]) * scale,
                  (es_gltf.mat4_transform_point(m, p)[2] - center[2]) * scale)
                 for p in pos]
            n = []
            for k, nn in enumerate(nor or [(0.0, 0.0, 1.0)] * len(pos)):
                l = (nn[0] * nn[0] + nn[1] * nn[1] + nn[2] * nn[2]) ** 0.5 or 1.0
                n.append(es_gltf.mat4_transform_dir(
                    m, (nn[0] / l, nn[1] / l, nn[2] / l)))
            tris = [tuple(idx[k:k + 3]) for k in range(0, len(idx), 3)]
            jj = [tuple(int(x) for x in row) for row in jts]
            ww = [tuple(float(x) for x in row) for row in wts]
            for sub_tris, local in es_gltf.split_by_joints(tris, jj, ww, cap):
                used = set()
                for tri in sub_tris:
                    used.update(tri)
                order = sorted(used)
                remap = {g: l for l, g in enumerate(order)}
                sv = [None] * len(order)
                sn = [None] * len(order)
                sj = [None] * len(order)
                sw = [None] * len(order)
                for g, l in remap.items():
                    # Y-flip (glTF Y-up -> screen Y-down) baked here; joint
                    # matrices get the conjugate flip at palette time.
                    sv[l] = (s + v[g][0], s - v[g][1], v[g][2])
                    sn[l] = (n[g][0], -n[g][1], n[g][2])
                    # Нулевые веса ссылаются на кости вне палитры —
                    # им все равно какой индекс (вклад нулевой).
                    _loc = set(local)
                    sj[l] = tuple(
                        float(local.index(int(x)))
                        if (float(w) > 0.0 and int(x) in _loc) else 0.0
                        for x, w in zip(jj[g], ww[g]))
                    sw[l] = ww[g]
                st = [tuple(remap[x] for x in (tri[0], tri[2], tri[1]))
                      for tri in sub_tris]
                mesh = es_step3.mesh3_from_layout(
                    sv, [sj, sw, sn], st,
                    [("a_joints", 4), ("a_weights", 4), ("a_norm3", 3)])
                self.splits.append({"mesh": mesh, "palette": local,
                                    "model": None})

    def update(self, st, dt):
        self.angle = (st * self.speed) % 360.0


class SkinnedView(Displayable):
    """Drives one SkinnedObject: pose -> palettes -> N model draws."""

    def __init__(self, obj, **kwargs):
        super(SkinnedView, self).__init__(**kwargs)
        self.obj = obj
        self.models = []
        self._identity = None
        self._splits = obj.splits
        for sp in obj.splits:
            m = renpy.display.model.Model(size=(obj.size, obj.size))
            m.shader("renpy.geometry")
            m.shader(obj.material.shader_name())
            m.property("depth", True)
            m.uniform("u_view_dir", Object3D.view_dir)
            m.uniform("u_base_color", obj.material.base_color)
            m._mesh = sp["mesh"]
            sp["model"] = m
            self.models.append(m)

    @staticmethod
    def _flip_y(mat):
        # Conjugate Y-flip F*P*F, F=diag(1,-1,1): negate row/col 1
        # except the [1][1] diagonal. Matches the baked vert flip.
        return tuple(
            tuple(-mat[i][j] if (i == 1) != (j == 1) else mat[i][j]
                  for j in range(4))
            for i in range(4))

    @staticmethod
    def _flat16(mat):
        return [float(mat[i][j]) for i in range(4) for j in range(4)]

    def render(self, w, h, st, at):
        obj = self.obj
        last = obj._last_st
        dt = 0.0 if last is None else st - last
        obj._last_st = st
        obj.update(st, dt)
        light = (obj.material.light_dir
                 if obj.material.light_dir is not None else LIGHT_DIR)
        worlds = obj.skel.update()
        nbones = obj.material.max_bones
        if self._identity is None:
            self._identity = _Matrix([1.0, 0.0, 0.0, 0.0,
                                      0.0, 1.0, 0.0, 0.0,
                                      0.0, 0.0, 1.0, 0.0,
                                      0.0, 0.0, 0.0, 1.0])
        for sp, m in zip(self._splits, self.models):
            for li, ji in enumerate(sp["palette"]):
                flat = SkinnedView._flat16(
                    SkinnedView._flip_y(worlds[ji]))
                m.uniform("u_bone%d" % li, _Matrix(flat))
            # draw() требует ВСЕ объявленные uniform'ы: добиваем identity.
            for li in range(len(sp["palette"]), nbones):
                m.uniform("u_bone%d" % li, self._identity)
            m.uniform("u_spin", obj.angle)
            m.uniform("u_light_dir", light)
        rv = renpy.display.render.Render(int(obj.size), int(obj.size))
        for sp, m in zip(self._splits, self.models):
            cr = renpy.display.render.render(
                _TransformWrap(m, obj), obj.size, obj.size, st, at)
            rv.blit(cr, (0, 0))
            renpy.display.render.redraw(m, 0.016)
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
        inner = Transform(
            self.model,
            matrixtransform=_Matrix.rotate(self.obj.tilt, self.obj.angle,
                                           0.0),
            perspective=True)
        return renpy.display.render.render(inner, w, h, st, at)
