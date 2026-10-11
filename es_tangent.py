# -*- coding: utf-8 -*-
"""Per-vertex tangents for normal mapping, baked from geometry at load time.

Why geometry and not the shader: a derivative tangent space (dFdx/dFdy of the
position and UV) needs an extension directive in GLSL ES 1.00, and there is no
legal place to put one in this pipeline -- renpy.gl2.gl2shadercache.source()
emits `precision mediump float;` before the mod's function slot, and
ShaderPart's `variables` parser rejects any line that is not exactly
"{storage} {type} {name}". So tangents arrive as an attribute, like a_uv does.

Convention is glTF 2.0's, not another engine's, because the assets come from
glTF: for a triangle (p0,p1,p2) with UVs (t0,t1,t2),

    e1 = p1 - p0,  e2 = p2 - p0
    s1 = t1 - t0,  s2 = t2 - t0        (s = (du, dv))
    r  = 1 / (s1[0]*s2[1] - s2[0]*s1[1])
    T  = r * ( s2[1]*e1 - s1[1]*e2 )
    B  = r * ( s1[0]*e2 - s2[0]*e1 )

which is the 2x2 solve of `e = T*du + B*dv` for the pair (T, B). T and B are
accumulated over every triangle a vertex belongs to (that is what
makes one tangent per vertex instead of one per face), T is then orthogonalised
against the vertex normal, and w records whether cross(N, T) points along the
accumulated B. A shader therefore rebuilds the basis as

    vec3 T = normalize(v_tangent.xyz);
    vec3 N = normalize(v_normal);
    vec3 B = cross(N, T) * v_tangent.w;
    vec3 world = T * n.x + B * n.y + N * n.z;

which is the same basis the map was baked against, so a map authored by any
glTF-conformant tool decodes correctly. The one thing this file cannot decide
is the map's *green* direction: glTF/OpenGL says +V is up in the map, DirectX
says down. That is a per-asset property, so it stays an author's parameter
(see MATERIALS.md).

Everything here is pure and imports nothing, so it is unit-testable under a
bare python3.
"""


def _length(v):
    return (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) ** 0.5


def _unit(v):
    length = _length(v)
    if length <= 0.0:
        return None
    return (v[0] / length, v[1] / length, v[2] / length)


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _perpendicular(normal):
    """Any unit vector at right angles to `normal`.

    For a vertex no triangle gave a tangent to -- a fully degenerate UV island
    -- the basis still has to exist: a zero-length tangent normalises to NaN in
    the shader and NaN propagates into every channel of the pixel. The axis
    least aligned with the normal gives the least rounded result."""
    axis = (1.0, 0.0, 0.0)
    if abs(normal[0]) >= abs(normal[1]) and abs(normal[0]) >= abs(normal[2]):
        axis = (0.0, 1.0, 0.0)
    elif abs(normal[1]) >= abs(normal[2]):
        axis = (0.0, 0.0, 1.0)
    return _unit(_cross(normal, axis)) or _unit(_cross(axis, normal))


def bake(verts, normals, uvs, tris):
    """Tangent per vertex as (tx, ty, tz, w), w in {-1.0, +1.0}.

    All three inputs must live in the space the mesh is uploaded in -- the
    caller cannot bake in rig space and convert afterwards, because the
    buffer-space Y flip is a mirror (det -1) and mirroring flips the sign of
    cross(N, T) while leaving the accumulated B direction transformed but not
    negated, so w would come out wrong for every vertex.
    """
    count = len(verts)
    tangents = [(0.0, 0.0, 0.0)] * count
    bitangents = [(0.0, 0.0, 0.0)] * count
    for tri in tris:
        i0, i1, i2 = tri[0], tri[1], tri[2]
        p0, p1, p2 = verts[i0], verts[i1], verts[i2]
        t0, t1, t2 = uvs[i0], uvs[i1], uvs[i2]
        e1 = (p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2])
        e2 = (p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2])
        s1 = (t1[0] - t0[0], t1[1] - t0[1])
        s2 = (t2[0] - t0[0], t2[1] - t0[1])
        # UV-space area: scale-free, so a fixed epsilon is meaningful here.
        denom = s1[0] * s2[1] - s2[0] * s1[1]
        if -1e-8 < denom < 1e-8:
            continue
        r = 1.0 / denom
        tangent = (r * (s2[1] * e1[0] - s1[1] * e2[0]),
                   r * (s2[1] * e1[1] - s1[1] * e2[1]),
                   r * (s2[1] * e1[2] - s1[1] * e2[2]))
        bitangent = (r * (s1[0] * e2[0] - s2[0] * e1[0]),
                     r * (s1[0] * e2[1] - s2[0] * e1[1]),
                     r * (s1[0] * e2[2] - s2[0] * e1[2]))
        for index in (i0, i1, i2):
            old = tangents[index]
            tangents[index] = (old[0] + tangent[0], old[1] + tangent[1],
                               old[2] + tangent[2])
            old = bitangents[index]
            bitangents[index] = (old[0] + bitangent[0], old[1] + bitangent[1],
                                 old[2] + bitangent[2])
    out = []
    for index in range(count):
        normal = _unit(normals[index]) or (0.0, 0.0, 1.0)
        tangent = _unit(tangents[index])
        if tangent is None:
            out.append(_perpendicular(normal) + (1.0,))
            continue
        # Gram-Schmidt: the accumulated direction is only roughly tangential,
        # and a normal component in T would tilt the rebuilt basis.
        tangent = _unit((tangent[0] - normal[0] * _dot(normal, tangent),
                         tangent[1] - normal[1] * _dot(normal, tangent),
                         tangent[2] - normal[2] * _dot(normal, tangent)))
        if tangent is None:
            out.append(_perpendicular(normal) + (1.0,))
            continue
        handed = _dot(_cross(normal, tangent), bitangents[index])
        out.append(tangent + (-1.0 if handed < 0.0 else 1.0,))
    return out
