# -*- coding: utf-8 -*-
# Geometry upload into renpy.gl2.gl2mesh3.Mesh3.
#
# Mesh3 has no Python-facing fill path: its __init__ docstring says the creator
# is responsible for point, point_data, attribute and triangle, and every one of
# Ren'Py's own writers is Cython code assigning them directly. So the only way
# in from Python is ctypes against the compiled struct -- which means the layout
# has to be known, not assumed.
#
# The five offsets used to be hand-measured constants valid for one build. They
# are now derived at runtime by _discover(), which finds each one by matching
# known data that Cython wrote for us: Mesh3.texture_rectangle() fills point
# coordinates, texture coordinates and triangle indices with caller-supplied
# values, so reading each header slot and comparing against those arrays locates
# the three buffers without writing anything. The counts are then the int32
# slots that make get_points()/get_triangles() -- which loop over the compiled
# fields -- start returning rows.
import ctypes, struct, array

HEADER = 160            # bytes of object header scanned

# Installed by _discover() via verify_layout(). None means unproven, and every
# entry point refuses to run before that has succeeded.
OFF_POINTS = None
OFF_TRIS = None
OFF_POINT = None        # point_data, the array glVertexAttribPointer reads
OFF_ATTR = None
OFF_TRI = None
POINT_FMT = None        # "f" or "d": the width of one Point3 coordinate

_LAYOUT = [None]

# cdef-классам нельзя вешать python-атрибуты — адреса храним здесь по id меша.
_ADDRS = {}

# Distinct values everywhere, so a match cannot be coincidence: the rectangle's
# four corners, its eight UVs, and a mesh sized so the allocated counts (5, 3)
# differ from every coordinate and index in play.
RECT = (1.25, -2.5, 3.75, 5.0)
RECT_UVS = (0.0625, 0.9375, 0.5625, 0.4375)
RECT_POINTS = [(1.25, -2.5, 0.0), (3.75, -2.5, 0.0),
               (3.75, 5.0, 0.0), (1.25, 5.0, 0.0)]
RECT_ATTRS = [0.0625, 0.9375, 0.5625, 0.9375, 0.5625, 0.4375, 0.0625, 0.4375]
RECT_TRIS = [0, 1, 2, 0, 2, 3]
PROBE_N, PROBE_T = 5, 3


def _i32(raw, off):
    return struct.unpack_from("<i", raw, off)[0]


def _i64(raw, off):
    return struct.unpack_from("<Q", raw, off)[0]


def _candidate(value):
    """Guard before dereferencing a header slot as a pointer.

    Deliberately weak: the buffers come from libc's heap while the Python object
    comes from CPython's, so they can sit in completely different ranges and no
    address band would hold. Requiring an aligned value past the null page is
    enough to skip the small ints in the header, and every dereference is a read
    of at most 32 bytes -- wrong-but-mapped candidates simply fail the content
    comparison."""
    return value >= 4096 and value % 8 == 0


def _slots_with_floats(raw, base, want, fmt):
    size = struct.calcsize(fmt)
    target = len(want) * size
    found = []
    for off in range(0, HEADER - 7, 8):
        value = _i64(raw, off)
        if not _candidate(value):
            continue
        try:
            blob = ctypes.string_at(value, target)
        except Exception:
            continue
        if list(struct.unpack("<%d%s" % (len(want), fmt), blob)) == want:
            found.append(off)
    return found


def _slots_with_ushorts(raw, want):
    blob_want = struct.pack("<%dH" % len(want), *want)
    found = []
    for off in range(0, HEADER - 7, 8):
        value = _i64(raw, off)
        if not _candidate(value):
            continue
        try:
            blob = ctypes.string_at(value, len(blob_want))
        except Exception:
            continue
        if blob == blob_want:
            found.append(off)
    return found


def _mesh(n, t, textured=False):
    from renpy.gl2.gl2mesh3 import Mesh3
    from renpy.gl2.gl2mesh import AttributeLayout as AL
    if textured:
        return Mesh3.texture_rectangle(RECT[0], RECT[1], RECT[2], RECT[3],
                                       RECT_UVS[0], RECT_UVS[1], RECT_UVS[2],
                                       RECT_UVS[3])
    layout = AL()
    layout.add_attribute("a_normal", 3)
    return Mesh3(layout, n, t)


def _count_offset(mesh, off, want, rows):
    """Does writing `want` at header offset `off` make the compiled reader
    return `rows` entries? Safe for every wrong answer: if `off` is not the
    count field the count stays 0, so the reader loops zero times and never
    dereferences a pointer we may have just corrupted."""
    base = id(mesh)
    snapshot = ctypes.string_at(base, HEADER)
    try:
        _move(base + off, struct.pack("i", want))
        if _i32(ctypes.string_at(base, HEADER), off) != want:
            return False
        return len(rows(mesh)) == want
    except Exception:
        return False
    finally:
        _move(base, snapshot)


def _find_count_offset(n, t):
    """Locate `points` and `triangles`, the two int32 fields a fresh Mesh3
    leaves at 0. Returns (off_points, off_tris, reason)."""
    points = tris = None
    for off in range(0, HEADER - 3, 4):
        m = _mesh(PROBE_N, PROBE_T)
        if _i32(ctypes.string_at(id(m), HEADER), off) != 0:
            continue
        if points is None and _count_offset(m, off, PROBE_N,
                                            lambda x: x.get_points()):
            points = off
            continue
        m2 = _mesh(PROBE_N, PROBE_T)
        if tris is None and _count_offset(m2, off, PROBE_T,
                                          lambda x: x.get_triangles()):
            tris = off
    if points is None or tris is None:
        return None, None, "could not locate the points/triangles counts"
    if points == tris:
        return None, None, "one slot answered to both counts"
    return points, tris, None


def _discover():
    """Returns (offsets, reason); offsets is None when anything is unclear."""
    from renpy.gl2.gl2mesh import AttributeLayout as AL
    probe = AL()
    probe.add_attribute("a_normal", 3)
    stride = probe.__reduce__()[1][1]
    if stride != 3:
        return None, "one vec3 attribute reports stride %r" % (stride,)

    ref = _mesh(0, 0, textured=True)
    if [tuple(p[:3]) for p in ref.get_points()] != RECT_POINTS:
        return None, "texture_rectangle points did not read back"
    if ref.get_triangles() != [tuple(RECT_TRIS[k:k + 3]) for k in (0, 3)]:
        return None, "texture_rectangle indices did not read back"
    base = id(ref)
    raw = ctypes.string_at(base, HEADER)

    fmt = None
    point_hits = []
    want = [v for point in RECT_POINTS for v in point]
    for fmt_try in ("f", "d"):
        hits = _slots_with_floats(raw, base, want, fmt_try)
        if hits:
            fmt, point_hits = fmt_try, hits
            break
    if not point_hits:
        return None, "no slot holds the rectangle's points"
    # Mesh3.__init__ assigns self.point_data = self.point, so the same address
    # sits in two fields and both dereference to the same data. That is the
    # alias relation being confirmed, not ambiguity -- but every hit must agree
    # on one address, and the draw path can use either, so the last is taken.
    addresses = set(_i64(raw, off) for off in point_hits)
    if len(addresses) != 1:
        return None, "the point slots disagree on an address: %r" % (
            sorted(addresses),)
    point_off = point_hits[-1]

    attr_offs = _slots_with_floats(raw, base, RECT_ATTRS, "f")
    if len(attr_offs) != 1:
        return None, "attribute slot is ambiguous: %r" % (attr_offs,)
    tri_offs = _slots_with_ushorts(raw, RECT_TRIS)
    if len(tri_offs) != 1:
        return None, "triangle slot is ambiguous: %r" % (tri_offs,)

    points_off, tris_off, reason = _find_count_offset(PROBE_N, PROBE_T)
    if reason:
        return None, reason

    return {"OFF_POINTS": points_off, "OFF_TRIS": tris_off,
            "OFF_POINT": point_off, "OFF_ATTR": attr_offs[0],
            "OFF_TRI": tri_offs[0], "POINT_FMT": fmt}, None


def _round_trip(offsets):
    """End-to-end proof: build a mesh through the discovered offsets and read
    every field back through Cython's own accessors."""
    from renpy.gl2.gl2mesh3 import Mesh3
    from renpy.gl2.gl2mesh import AttributeLayout as AL
    layout = AL()
    layout.add_attribute("a_normal", 3)
    verts = [(1.5, 2.5, 3.5), (4.5, 5.5, 6.5), (7.5, 8.5, 9.5)]
    normals = [(0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)]
    tris = [(0, 1, 2)]
    m = Mesh3(layout, len(verts), len(tris))
    base = id(m)
    _move(base + offsets["OFF_POINTS"], struct.pack("i", len(verts)))
    _move(base + offsets["OFF_TRIS"], struct.pack("i", len(tris)))
    _write_buffers(base, verts, normals, tris, offsets)
    got_points = [tuple(p[:3]) for p in m.get_points()]
    got_tris = m.get_triangles()
    if got_points != [tuple(v) for v in verts]:
        return False
    if got_tris != tris:
        return False
    text = repr(m)
    return all("{: >8.3f} ".format(v) in text for row in normals for v in row)


def verify_layout():
    """True once this build's Mesh3 layout has been derived and installed."""
    if _LAYOUT[0] is not None:
        return _LAYOUT[0]
    ok = False
    try:
        offsets, reason = _discover()
        if offsets is None:
            print("ES_LAYOUT rejected: %s" % reason)
        elif not _round_trip(offsets):
            print("ES_LAYOUT rejected: a mesh built through the discovered "
                  "offsets did not read back")
        else:
            globals().update(offsets)
            ok = True
            print("ES_LAYOUT points=%d tris=%d point=%d attr=%d tri=%d "
                  "coord=%s" % (offsets["OFF_POINTS"], offsets["OFF_TRIS"],
                                offsets["OFF_POINT"], offsets["OFF_ATTR"],
                                offsets["OFF_TRI"], offsets["POINT_FMT"]))
    except Exception as error:
        print("ES_LAYOUT rejected: discovery raised %r" % (error,))
    _LAYOUT[0] = ok
    return ok


def _check_layout():
    if not verify_layout():
        raise RuntimeError(
            "es-3dmodels: could not derive the layout of "
            "renpy.gl2.gl2mesh3.Mesh3 on this build, so vertex data cannot be "
            "uploaded safely. See _discover() in es_step3.py.")


def _ptr(base, off):
    return struct.unpack("P", ctypes.string_at(base + off, ctypes.sizeof(ctypes.c_void_p)))[0]


def _raw(ctype_code, values):
    out = array.array(ctype_code)
    out.extend(values)
    return out.tobytes() if hasattr(out, "tobytes") else out.tostring()


def _move(dst, data):
    """memmove that keeps the source referenced for the whole call.

    Every buffer written here is produced by a function call, so passing it
    straight into ctypes.memmove hands the interpreter a temporary whose only
    owner is the argument tuple -- and ctypes takes the buffer *address*, not
    the object. Binding the data to a name first is what makes the source
    provably alive when the copy runs. Deriving the count from the data also
    removes the second way a write can end up the wrong size: recomputing it
    from the vertex list.
    """
    ctypes.memmove(dst, data, len(data))


def _pack_points(verts, offsets=None):
    fmt = (offsets or {}).get("POINT_FMT") or POINT_FMT
    flat = [float(v[i]) for v in verts for i in (0, 1, 2)]
    return struct.pack("<%d%s" % (len(flat), fmt), *flat)


def _write_buffers(base_or_addrs, verts, normals, tris, offsets=None):
    offsets = offsets or globals()
    if isinstance(base_or_addrs, tuple):
        p_point, p_attr, p_tri = base_or_addrs
    else:
        base = base_or_addrs
        p_point = _ptr(base, offsets["OFF_POINT"])
        p_attr = _ptr(base, offsets["OFF_ATTR"])
        p_tri = _ptr(base, offsets["OFF_TRI"])
    _move(p_point, _pack_points(verts, offsets))
    flat_n = [float(v[i]) for v in normals for i in (0, 1, 2)]
    flat_t = [int(i) for tri in tris for i in tri]
    _move(p_attr, _raw("f", flat_n))
    _move(p_tri, _raw("H", flat_t))


def mesh3_from_lists(verts, normals, tris):
    _check_layout()
    from renpy.gl2.gl2mesh3 import Mesh3
    from renpy.gl2.gl2mesh import AttributeLayout as AL
    l = AL()
    l.add_attribute("a_normal", 3)
    n, t = len(verts), len(tris)
    m = Mesh3(l, n, t)
    base = id(m)
    _move(base + OFF_POINTS, struct.pack("i", n))
    _move(base + OFF_TRIS, struct.pack("i", t))
    _write_buffers(base, verts, normals, tris)
    _ADDRS[id(m)] = (_ptr(base, OFF_POINT), _ptr(base, OFF_ATTR),
                     _ptr(base, OFF_TRI))
    return m


def mesh3_from_layout(verts, attrib_frames, tris, layout_spec):
    """Generic builder. layout_spec: [(name, floats)], attrib_frames: list of
    per-vert float lists in the same order. Attributes packed interleaved."""
    _check_layout()
    from renpy.gl2.gl2mesh3 import Mesh3
    from renpy.gl2.gl2mesh import AttributeLayout as AL
    l = AL()
    for name, size in layout_spec:
        l.add_attribute(name, size)
    n, t = len(verts), len(tris)
    m = Mesh3(l, n, t)
    base = id(m)
    _move(base + OFF_POINTS, struct.pack("i", n))
    _move(base + OFF_TRIS, struct.pack("i", t))
    frames = []
    for k in range(n):
        for frame in attrib_frames:
            for value in frame[k]:
                frames.append(float(value))
    p_point = _ptr(base, OFF_POINT)
    p_attr = _ptr(base, OFF_ATTR)
    p_tri = _ptr(base, OFF_TRI)
    _move(p_point, _pack_points(verts))
    _move(p_attr, _raw("f", frames))
    flat_t = [int(i) for tri in tris for i in tri]
    _move(p_tri, _raw("H", flat_t))
    _ADDRS[id(m)] = (p_point, p_attr, p_tri)
    return m


def mesh3_update(m, verts, normals):
    # Только вершины+нормали (топология фиксирована): без malloc/free каждый кадр.
    _check_layout()
    p_point, p_attr, p_tri = _ADDRS[id(m)]
    _move(p_point, _pack_points(verts))
    flat_n = [float(v[i]) for v in normals for i in (0, 1, 2)]
    _move(p_attr, _raw("f", flat_n))
    return m


def mesh3_read_attrib(m, count=11):
    """Read back the first `count` floats of the attribute buffer."""
    p_attr = _ADDRS[id(m)][1]
    raw = ctypes.string_at(p_attr, count * 4)
    return list(struct.unpack("<%df" % count, raw))


def mesh3_update_tris(m, tris):
    # Перезапись индексного буфера + счетчика (для CPU-culling без depth-буфера).
    _check_layout()
    p_point, p_attr, p_tri = _ADDRS[id(m)]
    flat_t = [int(i) for tri in tris for i in tri]
    _move(p_tri, _raw("H", flat_t))
    _move(id(m) + OFF_TRIS, struct.pack("i", len(tris)))
    return m
