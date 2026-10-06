# -*- coding: utf-8 -*-
"""Software fallback: Blinn-Phong cube on a 2D canvas. No GL required.

Used when the model renderer is unavailable. Lives here (not in the engine)
so the GPU path never imports canvas code.
"""
import math

from renpy.display.core import Displayable
import renpy.display.render


class Cube(Displayable):
    # Залитый куб с Blinn-Phong: ambient + diffuse + specular, сортировка по глубине.
    def __init__(self, size=300.0, **kwargs):
        super(Cube, self).__init__(**kwargs)
        self.size = size
        self.verts = [(-1,-1,-1),(1,-1,-1),(1,1,-1),(-1,1,-1),
                      (-1,-1,1),(1,-1,1),(1,1,1),(-1,1,1)]
        # 6 граней: индексы, базовый цвет, нормаль в object-space
        self.faces = [
            ([0,1,2,3], (200,70,70), (0,0,-1)),
            ([4,5,6,7], (70,200,70), (0,0,1)),
            ([0,1,5,4], (70,70,200), (0,-1,0)),
            ([2,3,7,6], (200,200,70), (0,1,0)),
            ([0,3,7,4], (70,200,200), (-1,0,0)),
            ([1,2,6,5], (200,70,200), (1,0,0)),
        ]
        L = (0.5, -0.8, 0.7)
        ll = math.sqrt(sum(v*v for v in L))
        self.L = tuple(v/ll for v in L)
        self.ambient = 0.25
        self.shininess = 32.0
    def render(self, w, h, st, at):
        import __builtin__ as _b
        _min, _max = _b.min, _b.max
        s = self.size
        rv = renpy.display.render.Render(int(s), int(s))
        c = rv.canvas()
        ay = math.radians((st * 60.0) % 360.0)
        ax = math.radians(25.0)
        cy, sy = math.cos(ay), math.sin(ay)
        cx, sx = math.cos(ax), math.sin(ax)
        def rot(p):
            x, y, z = p
            x1 = x * cy + z * sy
            z1 = -x * sy + z * cy
            y1 = y * cx - z1 * sx
            z2 = y * sx + z1 * cx
            return (x1, y1, z2)
        rp = [rot(v) for v in self.verts]
        # проекция
        pts = []
        for (x1, y1, z2) in rp:
            f = 1.0 / (3.2 - z2 * 0.6)
            px = s/2 + x1 * s*0.32 * f * 2.2
            py = s/2 + y1 * s*0.32 * f * 2.2
            pts.append((px, py))
        V = (0.0, 0.0, 1.0)
        H = (self.L[0]+V[0], self.L[1]+V[1], self.L[2]+V[2])
        hl = math.sqrt(sum(v*v for v in H))
        H = tuple(v/hl for v in H)
        draw = []
        for (idx, base, n) in self.faces:
            nx, ny, nz = rot(n)
            nl = math.sqrt(nx*nx+ny*ny+nz*nz) or 1.0
            nx, ny, nz = nx/nl, ny/nl, nz/nl
            diff = _max(0.0, nx*self.L[0]+ny*self.L[1]+nz*self.L[2])
            spec = _max(0.0, nx*H[0]+ny*H[1]+nz*H[2]) ** self.shininess
            k = _min(1.0, self.ambient + 0.75*diff)  # diffuse
            col = tuple(_min(255, int(b*k + 255*spec*0.9)) for b in base)
            depth = sum(rp[i][2] for i in idx) / 4.0
            draw.append((depth, idx, col))
        draw.sort(key=lambda d: d[0])  # дальние первыми
        for (_, idx, col) in draw:
            c.polygon(col, [pts[i] for i in idx])
        renpy.display.render.redraw(self, 0.016)
        return rv
