# -*- coding: utf-8 -*-
# Шаг 3: полная запись меша по доказанным оффсетам (20/60 counts, 80 point, 48 attr, 64 tri).
import ctypes, struct, array

OFF_POINTS = 20
OFF_TRIS = 60
OFF_POINT = 80
OFF_ATTR = 48
OFF_TRI = 64

# cdef-классам нельзя вешать python-атрибуты — адреса храним здесь по id меша.
_ADDRS = {}

def _ptr(base, off):
    return struct.unpack("P", ctypes.string_at(base + off, ctypes.sizeof(ctypes.c_void_p)))[0]

def mesh3_from_lists(verts, normals, tris):
    from renpy.gl2.gl2mesh3 import Mesh3
    from renpy.gl2.gl2mesh import AttributeLayout as AL
    l = AL()
    l.add_attribute("a_normal", 3)
    n, t = len(verts), len(tris)
    m = Mesh3(l, n, t)
    base = id(m)
    ctypes.memmove(base + OFF_POINTS, struct.pack("i", n), 4)
    ctypes.memmove(base + OFF_TRIS, struct.pack("i", t), 4)
    _write_buffers(base, verts, normals, tris)
    _ADDRS[id(m)] = (_ptr(base, OFF_POINT), _ptr(base, OFF_ATTR), _ptr(base, OFF_TRI))
    return m

def _write_buffers(base_or_addrs, verts, normals, tris):
    if isinstance(base_or_addrs, tuple):
        p_point, p_attr, p_tri = base_or_addrs
    else:
        base = base_or_addrs
        p_point = _ptr(base, OFF_POINT)
        p_attr = _ptr(base, OFF_ATTR)
        p_tri = _ptr(base, OFF_TRI)
    parr = array.array("f")
    for v in verts:
        parr.extend([float(v[0]), float(v[1]), float(v[2])])
    narr = array.array("f")
    for v in normals:
        narr.extend([float(v[0]), float(v[1]), float(v[2])])
    tarr = array.array("H")
    for a, b, c_ in tris:
        tarr.extend([int(a), int(b), int(c_)])
    ctypes.memmove(p_point, parr.tostring(), len(parr.tostring()))
    ctypes.memmove(p_attr, narr.tostring(), len(narr.tostring()))
    ctypes.memmove(p_tri, tarr.tostring(), len(tarr.tostring()))

def mesh3_update(m, verts, normals):
    # Только вершины+нормали (топология фиксирована): без malloc/free каждый кадр.
    p_point, p_attr, p_tri = _ADDRS[id(m)]
    parr = array.array("f")
    for v in verts:
        parr.extend([float(v[0]), float(v[1]), float(v[2])])
    narr = array.array("f")
    for v in normals:
        narr.extend([float(v[0]), float(v[1]), float(v[2])])
    ctypes.memmove(p_point, parr.tostring(), len(parr.tostring()))
    ctypes.memmove(p_attr, narr.tostring(), len(narr.tostring()))
    return m

def mesh3_update_tris(m, tris):
    # Перезапись индексного буфера + счетчика (для CPU-culling без depth-буфера).
    import ctypes as _ct
    base = None
    p_point, p_attr, p_tri = _ADDRS[id(m)]
    tarr = array.array("H")
    for a, b, c_ in tris:
        tarr.extend([int(a), int(b), int(c_)])
    _ct.memmove(p_tri, tarr.tostring(), len(tarr.tostring()))
    # triangles count по оффсету 60 — ищем по id заново (адрес объекта стабилен).
    import ctypes as _ct2, struct as _st
    _base = id(m)
    _ct2.memmove(_base + OFF_TRIS, _st.pack("i", len(tris)), 4)
    return m
