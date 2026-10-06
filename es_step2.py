# -*- coding: utf-8 -*-
# Шаг 2: точечная запись по доказанным оффсетам из дампа:
#  P=8: allocated_points=16, points=20, allocated_triangles=56, triangles=60,
#  point(Mesh3)=80, attribute=48, triangle=64. Остальное не трогаем.
import ctypes, struct, array

OFF_POINTS = 20
OFF_TRIS = 60
OFF_POINT = 80
OFF_ATTR = 48
OFF_TRI = 64

def _ptr(base, off):
    return struct.unpack("P", ctypes.string_at(base + off, ctypes.sizeof(ctypes.c_void_p)))[0]

def step2_single_float():
    from renpy.gl2.gl2mesh3 import Mesh3
    from renpy.gl2.gl2mesh import AttributeLayout as AL
    l = AL()
    l.add_attribute("a_normal", 3)
    m = Mesh3(l, 4, 2)
    base = id(m)
    p_point = _ptr(base, OFF_POINT)
    p_attr = _ptr(base, OFF_ATTR)
    p_tri = _ptr(base, OFF_TRI)
    # шаг 2а: только counts (безопасно — свои int-поля)
    ctypes.memmove(base + OFF_POINTS, struct.pack("i", 4), 4)
    ctypes.memmove(base + OFF_TRIS, struct.pack("i", 2), 4)
    # шаг 2б: один float x=1.5 в point[0].x, проверяем через get_points()
    ctypes.memmove(p_point, struct.pack("f", 1.5), 4)
    pts = None
    try:
        pts = m.get_points()
    except Exception as e:
        pts = "get_points err %r" % (e,)
    info = "p_point=%x p_attr=%x p_tri=%x pts0=%r" % (p_point, p_attr, p_tri, (pts[0] if isinstance(pts, list) and pts else pts))
    return info, m
