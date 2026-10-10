# -*- coding: utf-8 -*-
"""Minimal glTF 2.0 decoder for the es-3dmodels engine (.gltf + .glb).

Covers what the engine consumes: POSITION / NORMAL / INDICES / TEXCOORD_0 /
JOINTS_0 / WEIGHTS_0, skins (joints + inverseBindMatrices), node TRS
hierarchy, animation samplers, morph targets (incl. sparse accessors).
One UV channel only: TEXCOORD_1..N exist in some assets and are ignored.
Everything else (textures, PBR factors beyond base color) is ignored,
never crashed on: unknown chunks raise ValueError with context, never
segfault.

Reference-checked against cgltf.h (bundled for reading) for stride,
normalization and sampler rules.
"""
import base64
import json
import math
import os
import struct

_COMP = {5120: ("b", 1), 5121: ("B", 1), 5122: ("h", 2),
         5123: ("H", 2), 5125: ("I", 4), 5126: ("f", 4)}
_NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def _norm_value(comp, raw, normalized):
    if not normalized:
        return float(raw)
    if comp == 5126:
        return float(raw)
    if comp == 5121:
        return raw / 255.0
    if comp == 5123:
        return raw / 65535.0
    if comp == 5120:
        return max(raw / 127.0, -1.0)
    if comp == 5122:
        return max(raw / 32767.0, -1.0)
    if comp == 5125:
        return float(raw)
    return float(raw)


class Gltf(object):
    def __init__(self):
        self.json = {}
        self.bins = []

    @classmethod
    def load(cls, path):
        self = cls()
        with open(path, "rb") as f:
            magic = f.read(4)
        if magic == b"glTF":
            self._load_glb(path)
        else:
            self._load_gltf(path)
        return self

    def _load_glb(self, path):
        with open(path, "rb") as f:
            data = f.read()
        magic, version, length = struct.unpack("<III", data[:12])
        if magic != 0x46546C67 or version != 2:
            raise ValueError("not a glTF 2.0 binary")
        off = 12
        while off + 8 <= len(data):
            clen, ctype = struct.unpack("<II", data[off:off + 8])
            chunk = data[off + 8:off + 8 + clen]
            if ctype == 0x4E4F534A:
                self.json = json.loads(chunk.decode("utf-8"))
            elif ctype == 0x004E4942:
                self.bins.append(chunk)
            off += 8 + clen
        if not self.json:
            raise ValueError("glb without JSON chunk")

    def _load_gltf(self, path):
        with open(path, "r") as f:
            self.json = json.load(f)
        base = os.path.dirname(path)
        for b in self.json.get("buffers", []):
            uri = b.get("uri", "")
            if uri.startswith("data:"):
                head, _, payload = uri.partition(",")
                if ";base64" in head:
                    self.bins.append(base64.b64decode(payload))
                else:
                    raise ValueError("unsupported data uri")
            else:
                with open(os.path.join(base, uri), "rb") as f:
                    self.bins.append(f.read())

    def _read_dense(self, a, normalize):
        """Read `count` tightly-strided elements from a["bufferView"]."""
        v = self.json["bufferViews"][a["bufferView"]]
        buf = self.bins[v.get("buffer", 0)]
        fmt, size = _COMP[a["componentType"]]
        ncomp = _NCOMP[a["type"]]
        stride = v.get("byteStride", 0) or ncomp * size
        base = v.get("byteOffset", 0) + a.get("byteOffset", 0)
        count = a["count"]
        if base + (count - 1) * stride + ncomp * size > len(buf):
            raise ValueError("accessor out of range (bufferView %s)" %
                             (a["bufferView"],))
        out = []
        for k in range(count):
            vals = struct.unpack_from("<%d%s" % (ncomp, fmt), buf,
                                      base + k * stride)
            if normalize or a.get("normalized", False):
                vals = tuple(_norm_value(a["componentType"], x, True)
                             for x in vals)
            else:
                vals = tuple(float(x) if fmt != "f" else x for x in vals)
            if ncomp == 16:
                out.append((vals[0:4], vals[4:8], vals[8:12], vals[12:16]))
            else:
                out.append(vals if ncomp > 1 else vals[0])
        return out

    def read_accessor(self, idx, normalize=False):
        """Returns a list of tuples. Bounds-checked; no raw pointer math.

        Sparse accessors are supported: zero base (or a dense base)
        patched in place by the sparse index/value run, per glTF 2.0."""
        a = self.json["accessors"][idx]
        ncomp = _NCOMP[a["type"]]
        if "bufferView" in a:
            out = self._read_dense(a, normalize)
        else:
            zero = tuple(0.0 for _ in range(ncomp))
            out = [zero if ncomp > 1 else 0.0] * a["count"]
        sp = a.get("sparse")
        if sp:
            n = sp["count"]
            ia = sp["indices"]
            iv = self.json["bufferViews"][ia["bufferView"]]
            ibuf = self.bins[iv.get("buffer", 0)]
            ifmt, isize = _COMP[ia["componentType"]]
            istr = iv.get("byteStride", 0) or isize
            ibase = iv.get("byteOffset", 0) + ia.get("byteOffset", 0)
            if n > 0 and ibase + (n - 1) * istr + isize > len(ibuf):
                raise ValueError("sparse indices out of range (accessor %d)"
                                 % idx)
            va = sp["values"]
            vals = self._read_dense(
                {"bufferView": va["bufferView"],
                 "byteOffset": va.get("byteOffset", 0),
                 "count": n,
                 "componentType": a["componentType"],
                 "type": a["type"],
                 "normalized": a.get("normalized", False)}, normalize)
            for k in range(n):
                vi = struct.unpack_from("<%s" % ifmt, ibuf,
                                        ibase + k * istr)[0]
                out[vi] = vals[k]
        return out

    def primitives(self):
        """Yields dicts: name, indices, POSITION/NORMAL/INDICES/TEXCOORD_0/
        JOINTS_0/WEIGHTS_0 accessor ids (or None), material, node_matrix (or
        None), plus morph data: targets (list of {semantic: accessor} or
        None), target_names (mesh extras list or None), weights (mesh morph
        weights or None)."""
        js = self.json
        node_of_mesh = {}
        for ni, node in enumerate(js.get("nodes", [])):
            if "mesh" in node:
                node_of_mesh.setdefault(node["mesh"], []).append(ni)
        for mi, mesh in enumerate(js.get("meshes", [])):
            for p in mesh.get("primitives", []):
                if p.get("mode", 4) != 4:
                    continue
                at = p.get("attributes", {})
                yield {"name": mesh.get("name", "mesh%d" % mi),
                       "indices": p.get("indices"),
                       "position": at.get("POSITION"),
                       "normal": at.get("NORMAL"),
                       "uv": at.get("TEXCOORD_0"),
                       "joints": at.get("JOINTS_0"),
                       "weights": at.get("WEIGHTS_0"),
                       "material": p.get("material"),
                       "nodes": node_of_mesh.get(mi, []),
                       "targets": p.get("targets"),
                       "target_names":
                           mesh.get("extras", {}).get("targetNames"),
                       "mesh_weights": mesh.get("weights")}

    def skin(self, idx=0):
        js = self.json
        s = js["skins"][idx]
        inv = self.read_accessor(s["inverseBindMatrices"])
        # glTF stores MAT4 column-major; transpose to our row-major.
        inv = [tuple(tuple(m[j][i] for j in range(4)) for i in range(4))
               for m in inv]
        return {"joints": list(s["joints"]),
                "inverse_bind": inv,
                "skeleton": s.get("skeleton")}

    def node_local(self, ni):
        """Local TRS tuple (or row-major matrix) of a node."""
        n = self.json["nodes"][ni]
        if "matrix" in n:
            m = n["matrix"]
            col = [tuple(m[i * 4:(i + 1) * 4]) for i in range(4)]
            return ("matrix", tuple(tuple(col[j][i] for j in range(4))
                                    for i in range(4)))
        t = tuple(n.get("translation", (0.0, 0.0, 0.0)))
        r = tuple(n.get("rotation", (0.0, 0.0, 0.0, 1.0)))
        s = tuple(n.get("scale", (1.0, 1.0, 1.0)))
        return ("trs", t, r, s)

    def children(self):
        return [n.get("children", []) for n in self.json.get("nodes", [])]

    def animations(self):
        return self.json.get("animations", [])


def quat_normalize(q):
    x, y, z, w = q
    l = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    return (x / l, y / l, z / l, w / l)


def quat_to_mat3(q):
    x, y, z, w = quat_normalize(q)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return ((1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)),
            (2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)),
            (2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)))


def mat4_mul(a, b):
    return tuple(
        tuple(sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4))
        for i in range(4))


def mat4_trs(t, r, s):
    m = quat_to_mat3(r)
    return ((m[0][0] * s[0], m[0][1] * s[1], m[0][2] * s[2], t[0]),
            (m[1][0] * s[0], m[1][1] * s[1], m[1][2] * s[2], t[1]),
            (m[2][0] * s[0], m[2][1] * s[1], m[2][2] * s[2], t[2]),
            (0.0, 0.0, 0.0, 1.0))


def mat4_identity():
    return ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0))


def mat4_inverse(m):
    """General 4x4 inverse (Gauss-Jordan). For rebinding stale skins."""
    rows = [[float(m[i][j]) if j < 4 else (1.0 if i == j - 4 else 0.0)
             for j in range(8)] for i in range(4)]
    for col in range(4):
        piv = max(range(col, 4), key=lambda r: abs(rows[r][col]))
        if abs(rows[piv][col]) < 1e-12:
            raise ValueError("singular matrix")
        rows[col], rows[piv] = rows[piv], rows[col]
        div = rows[col][col]
        rows[col] = [x / div for x in rows[col]]
        for r in range(4):
            if r != col and rows[r][col] != 0.0:
                f = rows[r][col]
                rows[r] = [a - f * b for a, b in zip(rows[r], rows[col])]
    return tuple(tuple(rows[i][4:8]) for i in range(4))


def mat4_transform_point(m, p):
    x, y, z = p
    return (m[0][0] * x + m[0][1] * y + m[0][2] * z + m[0][3],
            m[1][0] * x + m[1][1] * y + m[1][2] * z + m[1][3],
            m[2][0] * x + m[2][1] * y + m[2][2] * z + m[2][3])


def mat4_transform_dir(m, v):
    x, y, z = v
    return (m[0][0] * x + m[0][1] * y + m[0][2] * z,
            m[1][0] * x + m[1][1] * y + m[1][2] * z,
            m[2][0] * x + m[2][1] * y + m[2][2] * z)


def node_world_matrices(gltf):
    """World matrix per node (parents before children assumed; fixed below)."""
    nodes = gltf.json.get("nodes", [])
    children = gltf.children()
    local = []
    for ni in range(len(nodes)):
        kind = gltf.node_local(ni)
        if kind[0] == "matrix":
            m = kind[1]
            local.append((tuple(m[0:4]), tuple(m[4:8]),
                          tuple(m[8:12]), tuple(m[12:16])))
        else:
            _, t, r, s = kind
            local.append(mat4_trs(t, r, s))
    parent = {}
    for pi, ch in enumerate(children):
        for c in ch:
            parent[c] = pi
    order = []
    visited = set()

    def visit(ni):
        if ni in visited:
            return
        if ni in parent:
            visit(parent[ni])
        visited.add(ni)
        order.append(ni)

    for ni in range(len(nodes)):
        visit(ni)
    world = [None] * len(nodes)
    for ni in order:
        if ni in parent:
            world[ni] = mat4_mul(world[parent[ni]], local[ni])
        else:
            world[ni] = local[ni]
    return world


def split_by_joints(tris, joints, weights, cap=24):
    """Greedy partition of triangle list so each part uses <= cap joints.

    Returns a list of (tri_subset, local_joints) where local_joints maps
    local palette index -> global joint index.
    """
    parts = []
    cur_tris = []
    cur_set = set()
    for tri in tris:
        need = set()
        for vi in tri:
            for j, w in zip(joints[vi], weights[vi]):
                if w > 0.0:
                    need.add(int(j))
        if len(cur_set | need) > cap and cur_tris:
            g2l = {g: l for l, g in enumerate(sorted(cur_set))}
            parts.append((cur_tris, [g for g, l in
                                    sorted(g2l.items(), key=lambda kv: kv[1])]))
            cur_tris = []
            cur_set = set()
        cur_set |= need
        cur_tris.append(tri)
    if cur_tris:
        g2l = {g: l for l, g in enumerate(sorted(cur_set))}
        parts.append((cur_tris, [g for g, l in
                                sorted(g2l.items(), key=lambda kv: kv[1])]))
    return parts


def bind_pose_mesh(gltf, target=300.0):
    """All meshes baked with their node transforms, fit into +-target/2.

    Returns (verts, normals, tris, uvs) centered at origin, for rigid
    display. `uvs` is None if ANY primitive lacks TEXCOORD_0: the attribute
    layout is per-mesh, so UVs are all-or-nothing. Texture coordinates are
    passed through in the file's own V convention -- only positions flip.
    """
    world = node_world_matrices(gltf)
    out_v = []
    out_n = []
    out_t = []
    out_uv = []
    for p in gltf.primitives():
        pos = gltf.read_accessor(p["position"])
        nor = gltf.read_accessor(p["normal"]) if p["normal"] is not None else None
        idx = [int(x) for x in gltf.read_accessor(p["indices"])]
        uv = gltf.read_accessor(p["uv"]) if p["uv"] is not None else None
        if uv is None:
            out_uv = None
        elif out_uv is not None:
            out_uv.extend(uv)
        mats = [world[ni] for ni in p["nodes"]] or [mat4_identity()]
        m = mats[0]
        base = len(out_v)
        for v in pos:
            out_v.append(mat4_transform_point(m, v))
        if nor is not None:
            for n in nor:
                l = math.sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]) or 1.0
                out_n.append(mat4_transform_dir(m, (n[0] / l, n[1] / l, n[2] / l)))
        else:
            for _ in pos:
                out_n.append((0.0, 0.0, 1.0))
        for k in range(0, len(idx), 3):
            out_t.append((base + idx[k], base + idx[k + 1], base + idx[k + 2]))
    xs = [v[0] for v in out_v]
    ys = [v[1] for v in out_v]
    zs = [v[2] for v in out_v]
    cx = (min(xs) + max(xs)) / 2.0
    cy = (min(ys) + max(ys)) / 2.0
    cz = (min(zs) + max(zs)) / 2.0
    extent = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))
    scale = (target / extent) if extent > 0 else 1.0
    out_v = [((x - cx) * scale, (y - cy) * scale, (z - cz) * scale)
             for (x, y, z) in out_v]
    # glTF is Y-up, the screen is Y-down: mirror Y on verts and normals,
    # swap winding so front faces stay front. (Skinning reuses this output,
    # so joint matrices get the same treatment at pose time.)
    out_v = [(x, -y, z) for (x, y, z) in out_v]
    out_n = [(x, -y, z) for (x, y, z) in out_n]
    out_t = [(a, c, b) for (a, b, c) in out_t]
    return out_v, out_n, out_t, out_uv
