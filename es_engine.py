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

print("ES_BUILD fit-space-palette")

_Matrix = renpy.display.matrix.Matrix


def es_gltf_bind(path, target=300.0):
    """Load a .gltf/.glb bind pose: (verts, normals, tris), fit to target."""
    return es_gltf.bind_pose_mesh(es_gltf.Gltf.load(path), target)


# Global scene light. Materials with light_dir=None follow this;
# a per-material light_dir overrides it for that object only.
LIGHT_DIR = (0.5, -0.8, 0.7)


class Material(object):
    """Surface + lighting preset. Flags select a compiled shader variant."""

    def __init__(self, base_color=(0.9, 0.3, 0.3, 1.0),
                 light_dir=None,
                 shininess=16.0, ambient=0.3, diffuse=0.7,
                 spec_weight=0.5, specular=True, normal_viz=False,
                 skinned=False, max_bones=None, bind_pose=False):
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

    def variant_key(self):
        return (bool(self.specular), bool(self.normal_viz),
                float(self.shininess), float(self.ambient),
                float(self.diffuse), float(self.spec_weight),
                bool(self.skinned), int(self.max_bones),
                bool(self.bind_pose))

    def shader_name(self, morph=0):
        # `morph` = active shape-key slot count; only split models with
        # an atlas pass it (their shaders declare tex0/u_sw*).
        try:
            return es_shaders.get_variant(
                specular=self.specular, normal_viz=self.normal_viz,
                shininess=self.shininess, ambient=self.ambient,
                diffuse=self.diffuse, spec_weight=self.spec_weight,
                skinned=self.skinned, max_bones=self.max_bones,
                bind_pose=self.bind_pose, morph=morph)
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
        half_size = self.size / 2.0
        verts = [(half_size + x, half_size + y, z) for (x, y, z) in self.verts]
        return verts, list(self.normals), list(self.tris)

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
        model = renpy.display.model.Model(size=(self.obj.size, self.obj.size))
        model.shader("renpy.geometry")
        model.shader(self.obj.material.shader_name())
        model.property("depth", True)
        model.uniform("u_view_dir", Object3D.view_dir)
        model.uniform("u_base_color", self.obj.material.base_color)
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
        self.model.uniform(
            "u_light_dir",
            obj.material.light_dir if obj.material.light_dir is not None
            else LIGHT_DIR)
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


class SkinnedObject(Object3D):
    """glTF mesh skinned on GPU. update() poses joints, view pushes palette."""

    def __init__(self, gltf, size=600.0, speed=30.0, tilt=25.0,
                 color=(0.85, 0.8, 0.75), max_bones=None, target=300.0):
        import es_gltf
        if isinstance(gltf, basestring):
            gltf = es_gltf.Gltf.load(gltf)
        material = Material(
            base_color=(color[0], color[1], color[2], 1.0),
            skinned=True, max_bones=max_bones)
        super(SkinnedObject, self).__init__(size, speed, tilt, material)
        self.skel = Skeleton(gltf)
        self.splits = []
        # Shape keys: name -> weight (active set), union of pool names.
        self.shapes = {}
        self.shape_names = []
        self._shape_seen = set()
        self._build_splits(gltf, target)

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
            node_matrix = (node_world[prim["nodes"][0]] if prim["nodes"]
                           else es_gltf.mat4_identity())
            prim_data.append((positions, normals, indices, joint_rows,
                              weight_rows, node_matrix, prim["material"],
                              prim.get("targets"), prim.get("target_names")))
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
        for (positions, normals, indices, joint_rows, weight_rows,
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
                "verts": [], "normals": [], "tris": [],
                "joints": [], "weights": []})
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
                mesh = es_step3.mesh3_from_layout(
                    part_verts,
                    [part_joint_ids, part_weights, part_normals,
                     part_atlas_ids], part_indices,
                    [("a_joints", 4), ("a_weights", 4), ("a_norm3", 3),
                     ("a_vid", 1)])
                self.splits.append({"mesh": mesh, "joints": part_joints,
                                    "model": None, "morph": morph})
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
            morph = split["morph"]
            # Only atlas-carrying splits get the morph program: it alone
            # declares tex0/u_sw* (draw() raises on unset uniforms).
            shader_name = obj.material.shader_name(
                morph=(self.shape_slot_count if morph is not None else 0))
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
            model.uniform("u_base_color", obj.material.base_color)
            if morph is not None:
                # Atlas must be texture 0: its sampler is named tex0.
                model.texture(morph["displayable"])
                model.uniform("u_mscale", morph["mscale"])
                model.uniform("u_nscale", morph["nscale"])
                model.uniform("u_nv", float(morph["nverts"]))
                model.uniform("u_as", (float(morph["width"]),
                                       float(morph["height"])))
            model._mesh = split["mesh"]
            split["model"] = model
            self.models.append(model)

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
            model.uniform("u_light_dir", light)
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
        rv.add_property("depth", True)
        for split, model in zip(self.splits, self.models):
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
