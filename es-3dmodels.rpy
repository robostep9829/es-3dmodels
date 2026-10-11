# -*- coding: utf-8 -*-
# es-3dmodels: a 3D engine for Ren'Py's GL2 renderer. This file is a library
# bootstrap, not a playable mod: it registers no label and no entry in `mods`.
# Content lives in sibling mods such as es-3dmodels-matitan, which import
# these modules from their `init 10` block -- this one runs first, at the
# default priority 0, and it is the step that puts this folder on sys.path.
init python:
    # Встроенный GL-контекст Ren'Py, без внешнего PyOpenGL.
    try:
        import renpy.gl.glfunctions as glf
        HAS_BUILTIN_GL = True
        _gl_info = "glfunctions: %d funcs available" % len(glf.required_functions)
    except Exception as e:
        HAS_BUILTIN_GL = False
        _gl_info = "no builtin GL: %r" % (e,)
    import sys as _sys
    import os as _os
    _md = _os.path.dirname(renpy.loader.transfn("mods/es-3dmodels/es-3dmodels.rpy"))
    if _md not in _sys.path:
        _sys.path.insert(0, _md)

    import es_engine as _eng

    # Shift+R purges these from sys.modules, so their module-level state --
    # the recipe table and the variant cache -- is rebuilt by re-reading the
    # .mat/.glsl files instead of serving stale programs inside one process.
    config.reload_modules += ["es_matlib", "es_engine", "es_shaders",
                              "es_gltf", "es_obj", "es_morph", "es_step3",
                              "es_caps", "es_tangent"]


init python:
    def use_model_renderer():
        """True when Ren'Py's Model/Mesh3 path is live. There is no second
        renderer: if this is False, the engine cannot draw and a content mod
        should say so rather than try."""
        import renpy.display.render as R
        return bool(getattr(R, "models", False))
