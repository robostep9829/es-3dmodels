init python:
    # Встроенный GL-контекст Ren'Py, без внешнего PyOpenGL.
    try:
        import renpy.gl.glfunctions as glf
        HAS_BUILTIN_GL = True
        _gl_info = "glfunctions: %d funcs available" % len(glf.required_functions)
    except Exception as e:
        HAS_BUILTIN_GL = False
        _gl_info = "no builtin GL: %r" % (e,)
    import sys as _sys2
    import os as _os2
    _md2 = _os2.path.dirname(renpy.loader.transfn("mods/es-3dmodels/es-3dmodels.rpy"))
    if _md2 not in _sys2.path:
        _sys2.path.insert(0, _md2)

    import es_cpu as _cpu
    import es_engine as _eng

    def use_model_renderer():
        import renpy.display.render as R
        return bool(getattr(R, "models", False))

    class SlowCube(_eng.CubeObject):
        # Subclass demo: same cube, own animation (half speed, st-driven).
        def update(self, st, dt):
            self.angle = (st * 45.0) % 360.0

    class Nodding(_eng.SkinnedObject):
        # Turntable + two-joint nod (bend distributed Neck->Head).
        def update(self, st, dt):
            super(Nodding, self).update(st, dt)
            import math as _m
            ph = _m.sin(st * 1.5)
            self.skel.pose_euler("Neck", rx=4.0 * ph)
            self.skel.pose_euler("Head", rx=6.0 * ph,
                                 ry=3.0 * _m.sin(st * 0.9))

init:
    $ mods["es_3dmodels"]=u"Просто тестовый мод"
label es_3dmodels:
    "Я говорю..."
    "[_gl_info]"
    $ _use_models = use_model_renderer()
    $ _GPU_READY = _use_models
    $ _GPU_ERR = "" if _use_models else "models renderer off"
    $ _RENDER_PATH = "GPU: engine/ObjectView" if _GPU_READY else "CPU: canvas.polygon Blinn-Phong"
    "Рендер: [_RENDER_PATH]"
    if _GPU_READY:
        $ _red = _eng.CubeObject(color=(0.9, 0.3, 0.3))
        $ _teal = SlowCube(color=(0.2, 0.8, 0.75))
        $ _view1 = _eng.ObjectView(_red)
        $ _view2 = _eng.ObjectView(_teal)
        show expression _view1 as cube_red at left
        show expression _view2 as cube_teal at right
        "Движок: два куба, второй — сабкласс со своей анимацией."
        hide cube_red
        hide cube_teal
        $ _objpath = renpy.loader.transfn("mods/es-3dmodels/game-ready.obj")
        $ _obj = _eng.MeshObject.from_obj(_objpath, size=500.0, speed=45.0)
        $ _view3 = _eng.ObjectView(_obj)
        show expression _view3 as obj_view at center
        "Модель game-ready.obj: движок, OBJ-лоадер, тот же Blinn-Phong."
        hide obj_view
        $ _glbpath = renpy.loader.transfn("mods/es-3dmodels/matitan-retopo-names.glb")
        $ _glbv, _glbn, _glbt = _eng.es_gltf_bind(_glbpath)
        $ _glb = _eng.MeshObject.from_data(_glbv, _glbn, _glbt, size=600.0, speed=30.0)
        $ _view4 = _eng.ObjectView(_glb)
        show expression _view4 as glb_view at center
        "glTF bind-pose: 5 мешей, 18.5k tris, жесткий проворот."
        hide glb_view
        $ _nod = Nodding(_glbpath, size=600.0, speed=20.0)
        $ _view5 = _eng.SkinnedView(_nod)
        $ _vname = _nod.material.shader_name()
        "Вариант шейдера: [_vname]"
        show expression _view5 as skin_view at center
        "Скиннинг: кивок головой, палитра костей на GPU."
        hide skin_view
    else:
        "ВНИМАНИЕ: GPU-пайплайн недоступен ([_GPU_ERR]) — запасной CPU-рендер."
        show expression _cpu.Cube(size=400.0) as gltest at center with dissolve
        "CPU-пасс: canvas.polygon Blinn-Phong (плоские грани, пересчёт на CPU)."
        hide gltest with dissolve
    "Конец теста."
    return