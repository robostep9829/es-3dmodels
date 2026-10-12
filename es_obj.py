# -*- coding: utf-8 -*-
"""Minimal Wavefront OBJ loader for the es-3dmodels engine.

Handles: v, vt, vn, f (tris/quads/n-gons, v/vt/vn, v//vn, v/vt or bare v).
Quads and n-gons are fan-triangulated. Vertices split on the (v, vt, vn)
triple, so smooth shading and per-face UVs from the exporter both survive.
Only one UV channel is read. No mtllib support yet (single material per
object), so `uvs` is None for files without any vt records.
"""
import es_fit


def load_obj(path, target=None, anchor="origin"):
    """Returns (verts, normals, tris, uvs) with es_fit's policy applied.

    `target` is the extent the longest axis is fitted to, or None to keep the
    file's own units; `anchor` is the point that lands at the centre of the
    buffer (see es_fit). Positions only -- a normal is a direction, so it is
    neither scaled nor moved. `uvs` is None when the file carries no texture
    coordinates at all.

    Unlike the glTF loaders this one mirrors nothing: an OBJ is drawn with
    whatever up direction the file was written with.
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

    # Positions only: a normal is a direction, so neither the scale nor the
    # anchor subtraction applies to it.
    out_v, _scale, _anchor = es_fit.fit_positions(out_v, target, anchor)
    return out_v, out_n, out_t, (out_uv if has_uv else None)
