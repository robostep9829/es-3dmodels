# -*- coding: utf-8 -*-
# Шаг 2: зная off allocated_points=16, allocated_triangles=56 (P=8, 64-bit),
# points=20, triangles=60. Здесь только проверяем zero-поля + дампим указатели.
import ctypes, struct

def dump_mesh_info(n1=4, t1=2, n2=8, t2=5):
    from renpy.gl2.gl2mesh3 import Mesh3
    from renpy.gl2.gl2mesh import AttributeLayout as AL
    def make(n, t):
        l = AL()
        l.add_attribute("a_normal", 3)
        return Mesh3(l, n, t)
    m1, m2 = make(n1, t1), make(n2, t2)
    b1, b2 = id(m1), id(m2)
    P = ctypes.sizeof(ctypes.c_void_p)
    out = []
    out.append("P=%d" % P)
    for off in (16, 20, 56, 60):
        try:
            v1 = struct.unpack("i", ctypes.string_at(b1 + off, 4))[0]
            v2 = struct.unpack("i", ctypes.string_at(b2 + off, 4))[0]
            out.append("i32 off=%d m1=%d m2=%d" % (off, v1, v2))
        except Exception as e:
            out.append("i32 off=%d err" % off)
    # указатели первые 96 байт — коротко, чтобы влезло в textbox
    for off in range(0, 96, P):
        try:
            p1 = struct.unpack("P", ctypes.string_at(b1 + off, P))[0]
            p2 = struct.unpack("P", ctypes.string_at(b2 + off, P))[0]
        except Exception:
            continue
        # показываем только ненулевые и различающиеся (malloc-буферы)
        if p1 and p1 != p2:
            out.append("ptr off=%d %x/%x" % (off, p1 & 0xfffff, p2 & 0xfffff))
    return "\n".join(out), (m1, m2)

def step2_set_counts(m, n, t, off_points=20, off_triangles=60):
    base = id(m)
    ctypes.memmove(base + off_points, struct.pack("i", int(n)), 4)
    ctypes.memmove(base + off_triangles, struct.pack("i", int(t)), 4)
    return True
