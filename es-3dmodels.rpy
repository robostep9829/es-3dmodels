init python:
    # Встроенный GL-контекст Ren'Py, без внешнего PyOpenGL.
    try:
        import renpy.gl.glfunctions as glf
        HAS_BUILTIN_GL = True
        _gl_info = "glfunctions: %d funcs available" % len(glf.required_functions)
    except Exception as e:
        HAS_BUILTIN_GL = False
        _gl_info = "no builtin GL: %r" % (e,)
    import math

    def use_model_renderer():
        import renpy.display.render as R
        return bool(getattr(R, "models", False))

    class Cube(renpy.Displayable):
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
            rv = renpy.Render(int(s), int(s))
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
            renpy.redraw(self, 0.016)
            return rv

    import sys as _sys2
    import os as _os2
    _md2 = _os2.path.dirname(renpy.loader.transfn("mods/es-3dmodels/es-3dmodels.rpy"))
    if _md2 not in _sys2.path:
        _sys2.path.insert(0, _md2)

    import es_step3 as _s3
    import math as _math2

    class SpinCube(renpy.Displayable):
        # Меш выделяется ОДИН раз; каждый кадр вершины+нормали переписываются
        # на месте (без malloc/free). Пивот/позиция — baked-математикой.
        def __init__(self, size=400.0, speed=120.0, frozen=None, **kwargs):
            super(SpinCube, self).__init__(**kwargs)
            self.size = size
            self.speed = speed
            self.frozen = frozen
            bv = [(-1,-1,-1),(1,-1,-1),(1,1,-1),(-1,1,-1),
                  (-1,-1,1),(1,-1,1),(1,1,1),(-1,1,1)]
            self.base_v, self.base_n, self.tris = [], [], []
            self.quads = [([0,1,2,3],(0,0,1)),([4,5,6,7],(0,0,-1)),
                          ([0,1,5,4],(0,1,0)),([2,3,7,6],(0,-1,0)),
                          ([0,3,7,4],(1,0,0)),([1,2,6,5],(-1,0,0))]
            for q, n_ in self.quads:
                b = len(self.base_v)
                for i in q:
                    self.base_v.append((bv[i][0]*150.0, bv[i][1]*150.0, bv[i][2]*150.0))
                    self.base_n.append(n_)
                self.tris.append((b+0,b+1,b+2)); self.tris.append((b+0,b+2,b+3))
            self.mesh = _s3.mesh3_from_lists(self.base_v, self.base_n, self.tris)
            self.model = renpy.display.model.Model(size=(size, size))
            self.model.shader("renpy.geometry")
            self.model.shader("es_blinn_phong")
            # БЕЗ depth: в offscreen-FBO нет depth-буфера (config.depth_size=0),
            # включенный DEPTH_TEST режет все фрагменты. Вместо этого CPU-culling.
            self.model.uniform("u_light_dir", (0.5,-0.8,0.7))
            self.model.uniform("u_view_dir", (0.0,0.0,1.0))
            self.model.uniform("u_base_color", (0.9,0.3,0.3,1.0))
            self.model.uniform("u_shininess", 16.0)
            self.model._mesh = self.mesh
        def render(self, w, h, st, at):
            ay = _math2.radians(self.frozen if self.frozen is not None else (st * self.speed) % 360.0)
            ax = _math2.radians(25.0)
            cy, sy = _math2.cos(ay), _math2.sin(ay)
            cx, sx = _math2.cos(ax), _math2.sin(ax)
            s = self.size / 2.0
            vv, nn = [], []
            for (x, y, z) in self.base_v:
                x1 = x * cy + z * sy
                z1 = -x * sy + z * cy
                y1 = y * cx - z1 * sx
                z2 = y * sx + z1 * cx
                vv.append((s + x1, s + y1, z2))
            for (x, y, z) in self.base_n:
                x1 = x * cy + z * sy
                z1 = -x * sy + z * cy
                y1 = y * cx - z1 * sx
                z2 = y * sx + z1 * cx
                nn.append((x1, y1, z2))
            _s3.mesh3_update(self.mesh, vv, nn)
            # CPU backface-culling: грани с нормалью от камеры не сабмитим.
            # Квады лежат блоками по 4 вершины / 2 треугольника (см. __init__).
            _tris, _qi = [], 0
            for _qn in self.quads:
                _nx, _ny, _nz = nn[_qi * 4]
                if _nz > 0.0:
                    _b = _qi * 4
                    _tris.append((_b+0,_b+1,_b+2)); _tris.append((_b+0,_b+2,_b+3))
                _qi += 1
            _s3.mesh3_update_tris(self.mesh, _tris)
            if self.frozen is None:
                renpy.redraw(self, 0.016)
            return renpy.display.render.render(self.model, w, h, st, at)

    def make_gpu_quad():
        # Отладка: плоский квад z=0 с центром (0,0,0).
        import es_step3 as _s3q
        v = [(-150.0,-150.0,0.0),(150.0,-150.0,0.0),(150.0,150.0,0.0),(-150.0,150.0,0.0)]
        n = [(0.0,0.0,1.0)]*4
        t = [(0,1,2),(0,2,3)]
        m = renpy.display.model.Model(size=(400, 400))
        m.shader("renpy.geometry")
        m.shader("es_blinn_phong")
        m.uniform("u_light_dir", (0.0,0.0,1.0))
        m.uniform("u_view_dir", (0.0,0.0,1.0))
        m.uniform("u_base_color", (0.9,0.3,0.3,1.0))
        m.uniform("u_shininess", 32.0)
        m._mesh = _s3q.mesh3_from_lists(v, n, t)
        return m

init:
    $ mods["es_3dmodels"]=u"Просто тестовый мод"
label es_3dmodels:
    "Я говорю..."
    "[_gl_info]"
    $ _use_models = use_model_renderer()
    $ _GPU_READY = _use_models
    $ _GPU_ERR = "" if _use_models else "models renderer off"
    $ _RENDER_PATH = "GPU: Model/es_blinn_phong/SpinCube" if _GPU_READY else "CPU: canvas.polygon Blinn-Phong"
    "Рендер: [_RENDER_PATH]"
    if _GPU_READY:
        show expression SpinCube(size=400.0) as gltest at center with dissolve
        "GPU-куб: Blinn-Phong на видеокарте, вращение запечено в вершины."
        hide gltest with dissolve
    else:
        "ВНИМАНИЕ: GPU-пайплайн недоступен ([_GPU_ERR]) — запасной CPU-рендер."
        show expression Cube(size=400.0) as gltest at center with dissolve
        "CPU-пасс: canvas.polygon Blinn-Phong (плоские грани, пересчёт на CPU)."
        hide gltest with dissolve
    "Конец теста."
    return