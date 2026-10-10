# -*- coding: utf-8 -*-
"""The material DSL: .mat files compiled into es_shaders recipes.

    material matitan_toon {
        fragment "toon.glsl"                  # body, read next to this file
        float u_bands = 4.0                   # typed parameters, with defaults
        vec3  u_rim_color = 0.85 0.45 0.40
        requires uv                           # this fragment reads v_uv
    }

    material body_diffuse {
        fragment "body.glsl"
        map "textures/body.png"               # bound as tex1
        applies "mtl_bdy1062_00"              # only this glTF material
        blend "alpha clip"                    # opaque | alpha clip | alpha
        cutoff 0.5
        cull "back"                           # drop fragments facing away
    }

parse() imports no Ren'Py and takes a `read` callback, so the grammar and its
diagnostics are unit-testable under a bare python3; load_materials() is the
only function that touches renpy.loader. Checks that need a position live here
(an author sees toon.glsl:7, not "the recipe was rejected");
es_shaders.define_recipe re-validates semantics for every entry point,
including the Python API, so neither path can be bypassed.
"""
import re

TYPES = {"float": 1, "vec2": 2, "vec3": 3, "vec4": 4}
FLAGS = ("uv",)
# Mirrors es_shaders.BLEND_MODES: this copy exists so a bad blend name is
# reported with its .mat line number, and the parser stays importable without
# Ren'Py (es_shaders cannot be imported outside a running game).
BLEND_MODES = ("opaque", "alpha clip", "alpha")
# Mirrors es_shaders.CULL_MODES, same reason.
CULL_MODES = ("back",)

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_NUMBER = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")
_TOKEN = re.compile(r"\"[^\"]*\"|'[^']*'|[^\s]+")
_DECL = re.compile(r"^(uniform|attribute|varying)\b")


class MaterialError(Exception):
    """A material file problem. The message already carries file:line."""


def _at(path, line, message):
    return MaterialError("%s:%d: %s" % (path, line, message))


def _strip_comment(line):
    """Drop # and // comments; a quote hides the rest of the line."""
    out = []
    quote = None
    index = 0
    length = len(line)
    while index < length:
        char = line[index]
        if quote is not None:
            out.append(char)
            if char == quote:
                quote = None
        elif char in "\"'":
            quote = char
            out.append(char)
        elif char == "#" or (char == "/" and line[index + 1:index + 2] == "/"):
            break
        else:
            out.append(char)
        index += 1
    return "".join(out)


def _tokens(line):
    return _TOKEN.findall(line)


def _is_ascii(text):
    # UnicodeError is the common base of the encode and decode failures that
    # the two interpreters raise for this on str versus unicode.
    try:
        text.encode("ascii")
        return True
    except UnicodeError:
        return False


def _block(lines, index, total, path, name):
    """Collect one material's statements, stopping at the closing brace."""
    stmts = []
    while True:
        if index >= total:
            raise _at(path, total, "material %s: block never closes" % name)
        line = _strip_comment(lines[index]).strip()
        index += 1
        if not line:
            continue
        if line == "}":
            return stmts, index
        stmts.append((index, _tokens(line)))


def _param(name, lineno, key, toks, path, params):
    """Parse and store one typed parameter statement."""
    want = TYPES[key]
    if len(toks) < 4 or toks[2] != "=":
        raise _at(path, lineno, "material %s: expected "
                                "'%s <name> = <value...>'" % (name, key))
    pname = toks[1]
    if not _IDENT.match(pname):
        raise _at(path, lineno, "material %s: bad parameter name %r"
                  % (name, pname))
    if pname in params:
        raise _at(path, lineno, "material %s: parameter %s given twice"
                  % (name, pname))
    values = toks[3:]
    if len(values) != want:
        raise _at(path, lineno, "material %s.%s is a %s: wants %d value%s, "
                                "got %d" % (name, pname, key, want,
                                            "" if want == 1 else "s",
                                            len(values)))
    numbers = []
    for token in values:
        if not _NUMBER.match(token):
            raise _at(path, lineno, "material %s.%s: %r is not a number"
                      % (name, pname, token))
        numbers.append(float(token))
    params[pname] = numbers[0] if want == 1 else tuple(numbers)


def _first_line(text, needle):
    """1-based number of the first line containing `needle`."""
    for number, line in enumerate(text.split("\n"), 1):
        if needle in line:
            return number
    return 1


def _check_fragment(name, frag_path, text, requires, has_map):
    """Source checks that pay for themselves by naming a line number."""
    for number, line in enumerate(text.split("\n"), 1):
        where = "%s:%d" % (frag_path, number)
        # The raw line: a comment is still part of the source that reaches
        # the compiler, so non-ASCII has to fail here and not later in
        # _validate, which cannot name a line.
        if not _is_ascii(line):
            raise MaterialError("%s: non-ASCII character -- GLSL sources must "
                                "be pure ASCII" % where)
        code = _strip_comment(line).strip()
        if "gl_FragColor" in code:
            raise MaterialError("%s: writes gl_FragColor; assign es_color "
                                "instead -- the engine emits the output line, "
                                "the discard and the premultiply" % where)
        if _DECL.match(code):
            raise MaterialError("%s: declares %r; parameters belong in the "
                                ".mat, which generates this block"
                                % (where, code[:48]))
        if "tex0" in code:
            raise MaterialError("%s: tex0 is the engine's morph atlas or "
                                "placeholder, not an authoring surface; a "
                                "material's own map is tex1" % where)
        if ("texture2D" in code or "tex1" in code) and not has_map:
            raise MaterialError("%s: samples a texture but %s declares no map "
                                "(nothing is bound at tex1)"
                                % (where, name))
    if "es_color" not in text:
        raise MaterialError("%s: material %s never assigns es_color, so "
                            "nothing it computes reaches the screen"
                            % (frag_path, name))
    reads_uv = "v_uv" in text
    if reads_uv and "uv" not in requires:
        raise MaterialError("%s:%d: material %s reads v_uv but does not "
                            "declare 'requires uv'"
                            % (frag_path, _first_line(text, "v_uv"), name))
    if "uv" in requires and not reads_uv:
        raise MaterialError("%s: material %s declares 'requires uv' but never "
                            "reads v_uv" % (frag_path, name))


def _string_arg(name, lineno, toks, path, keyword):
    """Validate a `keyword "value"` statement and return the value."""
    if len(toks) != 2 or toks[1][:1] not in "\"'":
        raise _at(path, lineno, "material %s: expected %s \"value\""
                  % (name, keyword))
    return toks[1][1:-1]


def _material(name, decl_line, stmts, path, base, read):
    relpath = None
    map_rel = None
    applies = None
    blend = None
    blend_line = 0
    cutoff = None
    cutoff_line = 0
    cull = None
    params = {}
    requires = set()
    for lineno, toks in stmts:
        key = toks[0]
        if key == "fragment":
            if relpath is not None:
                raise _at(path, lineno, "material %s: fragment given twice"
                          % name)
            relpath = _string_arg(name, lineno, toks, path, "fragment")
        elif key == "map":
            if map_rel is not None:
                raise _at(path, lineno, "material %s: map given twice" % name)
            map_rel = _string_arg(name, lineno, toks, path, "map")
        elif key == "applies":
            if applies is not None:
                raise _at(path, lineno, "material %s: applies given twice"
                          % name)
            applies = _string_arg(name, lineno, toks, path, "applies")
        elif key == "blend":
            if blend is not None:
                raise _at(path, lineno, "material %s: blend given twice"
                          % name)
            blend = _string_arg(name, lineno, toks, path, "blend")
            blend_line = lineno
            if blend not in BLEND_MODES:
                raise _at(path, lineno, "material %s: unknown blend %r "
                                        "(known: %s)"
                          % (name, blend, ", ".join(BLEND_MODES)))
        elif key == "cutoff":
            if cutoff is not None:
                raise _at(path, lineno, "material %s: cutoff given twice"
                          % name)
            if len(toks) != 2 or not _NUMBER.match(toks[1]):
                raise _at(path, lineno, "material %s: expected "
                                        "cutoff <number in 0..1>" % name)
            cutoff = float(toks[1])
            cutoff_line = lineno
        elif key == "cull":
            if cull is not None:
                raise _at(path, lineno, "material %s: cull given twice"
                          % name)
            cull = _string_arg(name, lineno, toks, path, "cull")
            if cull not in CULL_MODES:
                raise _at(path, lineno, "material %s: unknown cull %r "
                                        "(known: %s)"
                          % (name, cull, ", ".join(CULL_MODES)))
        elif key == "requires":
            if len(toks) < 2:
                raise _at(path, lineno, "material %s: requires names nothing"
                          % name)
            for flag in toks[1:]:
                if flag not in FLAGS:
                    raise _at(path, lineno, "material %s: unknown requirement "
                                            "%r (known: %s)"
                              % (name, flag, ", ".join(FLAGS)))
                requires.add(flag)
        elif key in TYPES:
            _param(name, lineno, key, toks, path, params)
        else:
            raise _at(path, lineno, "material %s: unknown statement %r"
                      % (name, key))
    if relpath is None:
        raise _at(path, decl_line, "material %s: no fragment given" % name)
    frag_path = base + relpath
    try:
        text = read(frag_path)
    except Exception as error:
        raise _at(path, decl_line, "material %s: cannot read %s (%s)"
                  % (name, relpath, error))
    # The map is not read here: it is binary and only becomes a texture when
    # a view is built. load_materials checks it resolves, es_morph.image()
    # decodes it on first render.
    map_path = (base + map_rel) if map_rel is not None else None
    if cutoff is not None and blend not in (None, "alpha clip"):
        raise _at(path, cutoff_line, "material %s: cutoff needs "
                                    "blend \"alpha clip\", not %r"
                  % (name, blend))
    _check_fragment(name, frag_path, text, requires, map_path is not None)
    return {"name": name, "fragment": text, "params": params,
            "requires": requires, "map": map_path, "applies": applies,
            "blend": blend or "opaque", "cutoff": cutoff, "cull": cull,
            "source": path}


def parse(text, path, read):
    """Parse .mat source into recipe dicts.

    `read(relpath)` returns a fragment file's contents for a path resolved
    against `path`'s directory. Returns a list of
    {name, fragment, params, requires, source}.
    """
    lines = text.split("\n")
    total = len(lines)
    base = path[:path.rfind("/") + 1] if "/" in path else ""
    found = []
    index = 0
    while index < total:
        line = _strip_comment(lines[index]).strip()
        index += 1
        if not line:
            continue
        toks = _tokens(line)
        if len(toks) != 3 or toks[0] != "material" or toks[2] != "{":
            raise _at(path, index, "expected 'material <name> {', got %r"
                      % line)
        name = toks[1]
        if not _IDENT.match(name):
            raise _at(path, index, "bad material name %r" % name)
        decl_line = index
        stmts, index = _block(lines, index, total, path, name)
        found.append(_material(name, decl_line, stmts, path, base, read))
    if not found:
        raise _at(path, 1, "no materials defined")
    return found


def _read_file(path):
    import renpy
    handle = renpy.loader.load(path)
    try:
        return handle.read().decode("utf-8")
    finally:
        handle.close()


def discover(prefix):
    """Every .mat under `prefix`, newest files included.

    The scan is re-run because a .mat created since boot is not in the cached
    file list, and reload is exactly when someone adds one.
    """
    import renpy
    renpy.loader.scandirfiles()
    found = set()
    for _dir, name in renpy.loader.listdirfiles(common=False):
        if name.startswith(prefix + "/") and name.endswith(".mat"):
            found.add(name)
    return sorted(found)


def load_materials(define, prefix):
    """Parse every .mat under `prefix` and hand each to `define`.

    `prefix` is a game-relative folder holding .mat files, e.g.
    "mods/es-3dmodels-matitan/materials" -- materials belong to the content
    mod that ships them, so there is no engine default.
    `define(name, fragment, params, requires, map_path, applies, blend,
    cutoff, cull)` is the recipe backend: es_engine.define_material in the
    game, a test double in unit tests. Returns the registered material names.
    """
    import renpy
    loaded = []
    for path in discover(prefix):
        recipes = parse(_read_file(path), path, _read_file)
        for recipe in recipes:
            if recipe["map"] is not None and not renpy.loader.loadable(
                    recipe["map"]):
                raise _at(path, 0, "material %s: map %s is not a loadable "
                                   "file" % (recipe["name"], recipe["map"]))
            try:
                define(recipe["name"], recipe["fragment"], recipe["params"],
                       recipe["requires"], recipe["map"], recipe["applies"],
                       recipe["blend"], recipe["cutoff"], recipe["cull"])
            except Exception as error:
                raise _at(path, 0, "material %s rejected by the shader "
                                   "backend: %s" % (recipe["name"], error))
            loaded.append(recipe["name"])
    return sorted(loaded)
