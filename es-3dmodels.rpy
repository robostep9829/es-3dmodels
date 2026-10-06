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

    class SpinCube(renpy.Displayable):
        # Вращение — матрицей модели (matrixtransform, пивот авто-центр бокса).
        # Меш СТАТИЧНЫЙ (один аплоад); в кадре меняется только uniform u_spin.
        # Формула поворота нормалей в шейдере — строки rotate_matrix из matrix_functions.pxi.
        def __init__(self, size=400.0, speed=120.0, **kwargs):
            super(SpinCube, self).__init__(**kwargs)
            self.size = size
            self.speed = speed
            s = size / 2.0
            bv = [(-1,-1,-1),(1,-1,-1),(1,1,-1),(-1,1,-1),
                  (-1,-1,1),(1,-1,1),(1,1,1),(-1,1,1)]
            self.base_n = []
            self.quads = [([0,1,2,3],(0,0,-1)),([4,5,6,7],(0,0,1)),
                          ([0,1,5,4],(0,-1,0)),([2,3,7,6],(0,1,0)),
                          ([0,3,7,4],(-1,0,0)),([1,2,6,5],(1,0,0))]
            base_v, tris = [], []
            for q, n_ in self.quads:
                b = len(base_v)
                for i in q:
                    base_v.append((s + bv[i][0]*150.0, s + bv[i][1]*150.0, bv[i][2]*150.0))
                    self.base_n.append(n_)
                tris.append((b+0,b+1,b+2)); tris.append((b+0,b+2,b+3))
            self.mesh = _s3.mesh3_from_lists(base_v, self.base_n, tris)
            self.model = renpy.display.model.Model(size=(size, size))
            self.model.shader("renpy.geometry")
            self.model.shader("es_blinn_phong")
            # Эксперимент: DEPTH_TEST против оконного depth-буфера (в FB tab он есть,
            # пуст). Показ — БЕЗ dissolve, иначе render_to_texture в FBO без depth.
            self.model.property("depth", True)
            self.model.uniform("u_light_dir", (0.5,-0.8,0.7))
            self.model.uniform("u_view_dir", (0.0,0.0,1.0))
            self.model.uniform("u_base_color", (0.9,0.3,0.3,1.0))
            self.model.uniform("u_shininess", 16.0)
            self.model.uniform("u_spin", 0.0)
            self.model._mesh = self.mesh
        def render(self, w, h, st, at):
            # Окклюзия — DEPTH_TEST против оконного depth-буфера (прямой draw,
            # без dissolve). Здесь только угол для шейдера; сабмит всегда полный.
            a = (st * self.speed) % 360.0
            self.model.uniform("u_spin", a)
            renpy.redraw(self, 0.016)
            return renpy.display.render.render(self.model, w, h, st, at)

    def spin_mtx(trans, st, at):
        # Пивот — авто-центр бокса (matrixanchor по умолчанию w/2,h/2).
        trans.matrixtransform = Matrix.rotate(25.0, (st * 120.0) % 360.0, 0.0)
        return 0.016

init:
    $ mods["es_3dmodels"]=u"Просто тестовый мод"
    transform gpu_spin:
        align (0.5, 0.5)
        perspective True
        function spin_mtx
label es_3dmodels:
    "Я говорю..."
    "[_gl_info]"
    $ _use_models = use_model_renderer()
    $ _GPU_READY = _use_models
    $ _GPU_ERR = "" if _use_models else "models renderer off"
    $ _RENDER_PATH = "GPU: Model/es_blinn_phong/SpinCube" if _GPU_READY else "CPU: canvas.polygon Blinn-Phong"
    "Рендер: [_RENDER_PATH]"
    if _GPU_READY:
        show expression SpinCube(size=400.0) as gltest at gpu_spin
        "GPU-куб: вращение матрицей, свет — шейдер, окклюзия — depth-тест."
        hide gltest
    else:
        "ВНИМАНИЕ: GPU-пайплайн недоступен ([_GPU_ERR]) — запасной CPU-рендер."
        show expression Cube(size=400.0) as gltest at center with dissolve
        "CPU-пасс: canvas.polygon Blinn-Phong (плоские грани, пересчёт на CPU)."
        hide gltest with dissolve
    "Конец теста."
    return