# -*- coding: utf-8 -*-
"""Minimal Wavefront OBJ loader for the es-3dmodels engine.

Handles: v, vt, vn, f (tris/quads/n-gons, v/vt/vn, v//vn, v/vt or bare v).
Quads and n-gons are fan-triangulated. Vertices split on the (v, vt, vn)
triple, so smooth shading and per-face UVs from the exporter both survive.
Only one UV channel is read. No mtllib support yet (single material per
object), so `uvs` is None for files without any vt records.
"""


def load_obj(path, target=300.0):
    """Returns (verts, normals, tris, uvs) fit into a +-target/2 box.

    `uvs` is None when the file carries no texture coordinates at all.
    """
    verts = []
    texcoords = []
    normals = []
    faces = []
    with open(path, "r") as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "v" and len(parts) >= 4:
                verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif parts[0] == "vt" and len(parts) >= 3:
                texcoords.append((float(parts[1]), float(parts[2])))
            elif parts[0] == "vn" and len(parts) >= 4:
                normals.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif parts[0] == "f" and len(parts) >= 4:
                face = []
                for ref in parts[1:]:
                    idx = ref.split("/")
                    vi = int(idx[0]) - 1
                    ti = None
                    ni = None
                    if len(idx) >= 2 and idx[1]:
                        ti = int(idx[1]) - 1
                    if len(idx) >= 3 and idx[2]:
                        ni = int(idx[2]) - 1
                    face.append((vi, ti, ni))
                faces.append(face)
    out_v = []
    out_n = []
    out_uv = []
    out_t = []
    index_of = {}
    has_uv = bool(texcoords)

    def emit(vi, ti, ni):
        key = (vi, ti, ni)
        idx = index_of.get(key)
        if idx is None:
            idx = len(out_v)
            index_of[key] = idx
            out_v.append(verts[vi])
            if ni is not None and 0 <= ni < len(normals):
                out_n.append(normals[ni])
            else:
                out_n.append((0.0, 0.0, 1.0))
            if ti is not None and 0 <= ti < len(texcoords):
                out_uv.append(texcoords[ti])
            else:
                out_uv.append((0.0, 0.0))
        return idx

    for face in faces:
        v0 = face[0]
        for k in range(1, len(face) - 1):
            out_t.append((emit(*v0), emit(*face[k]), emit(*face[k + 1])))

    if not out_v:
        raise ValueError("no geometry in %r" % (path,))

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
    return out_v, out_n, out_t, (out_uv if has_uv else None)
