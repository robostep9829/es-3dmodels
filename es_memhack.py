# -*- coding: utf-8 -*-
# MEMHACK v2: все поля Mesh/Mesh3 — cdef, из Python недоступны.
# Поэтому points/triangles/point/attribute/triangle правим через память.
import ctypes, struct, array

def _read_i32(addr):
    return struct.unpack("i", ctypes.string_at(addr, 4))[0]

def _write_i32(addr, v):
    ctypes.memmove(addr, struct.pack("i", int(v)), 4)

def _read_ptr(addr):
    return struct.unpack("P", ctypes.string_at(addr, ctypes.sizeof(ctypes.c_void_p)))[0]

def mesh3_from_lists(verts, normals, tris):
    from renpy.gl2.gl2mesh3 import Mesh3
    from renpy.gl2.gl2mesh import AttributeLayout as AL
    layout = AL()
    layout.add_attribute("a_normal", 3)
    n, t = len(verts), len(tris)
    m = Mesh3(layout, n, t)
    base = id(m)
    # 1) Найти allocated_points/allocated_triangles (установлены __init__),
    # рядом с ними лежат points/triangles. Сканируем int32.
    off_points, off_tris = None, None
    seen_n, seen_t = [], []
    for off in range(0, 256, 4):
        try:
            v = _read_i32(base + off)
        except Exception:
            continue
        if v == n:
            seen_n.append(off)
        if v == t:
            seen_t.append(off)
    # Эвристика по порядку полей из gl2mesh.pxd:
    # allocated_points, points, ..., allocated_triangles, triangles
    # points идёт сразу за allocated_points.
    if seen_n:
        # allocated_points — первое вхождение n, points — +4
        off_points = seen_n[0] + 4
        _write_i32(base + off_points, n)
    if seen_t:
        # последний кандидат t ближе к концу структуры
        off_tris = seen_t[-1] + 4
        # если там уже t (allocated==triangles==0? нет) — всё равно пишем
        try:
            _write_i32(base + off_tris, t)
        except Exception:
            pass
    # 2) Буферы: ищем указатели на читаемые регионы нужного размера.
    P = ctypes.sizeof(ctypes.c_void_p)
    cands = []
    for off in range(0, 64 * P, P):
        try:
            p = _read_ptr(base + off)
        except Exception:
            continue
        if not p:
            continue
        try:
            ctypes.string_at(p, max(n * 12, t * 6))
            cands.append((off, p))
        except Exception:
            continue
    if len(cands) < 2:
        raise RuntimeError("memhack: buffers not found")
    parr = array.array("f")
    for v in verts:
        parr.extend([float(v[0]), float(v[1]), float(v[2])])
    narr = array.array("f")
    for v in normals:
        narr.extend([float(v[0]), float(v[1]), float(v[2])])
    tarr = array.array("H")
    for a, b, c_ in tris:
        tarr.extend([int(a), int(b), int(c_)])
    # point, attribute, triangle — первые три крупных кандидата
    ctypes.memmove(cands[0][1], parr.tostring(), min(len(parr.tostring()), n * 12))
    if len(cands) > 1:
        ctypes.memmove(cands[1][1], narr.tostring(), min(len(narr.tostring()), n * 12))
    if len(cands) > 2:
        ctypes.memmove(cands[2][1], tarr.tostring(), min(len(tarr.tostring()), t * 6))
    return m
