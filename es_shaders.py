# -*- coding: utf-8 -*-
"""Shader preprocessor for the es-3dmodels engine.

Composes Ren'Py shader parts from material feature flags and bakes numeric
constants into the source (ifdef-style: each combination is its own program).
Results are cached, so identical materials share GL programs.

Hard lessons encoded here:
- sources must be pure ASCII (a non-ASCII GLSL comment once broke compile),
- never redeclare renpy.geometry's u_transform / a_position / gl_Position.
"""
import hashlib

from renpy.gl2.gl2shadercache import register_shader

VARIANTS = {}
RECIPES = {}
FALLBACK_SHADER = "es_fallback"
_fallback_done = [False]

# Anything register_shader's `variables` block or the engine's own parts
# already declare. A parameter reusing one of these names would emit a second
# declaration of it and the program would fail to compile.
_RESERVED = ("u_transform", "u_base_color", "u_light_dir", "u_view_dir",
             "u_spin", "u_tilt", "u_eye_z", "u_mscale", "u_nscale", "u_nv",
             "u_as",
             "v_normal", "v_half", "v_diff", "v_pos", "a_position",
             "a_normal",
             "a_norm3", "a_joints", "a_weights", "a_vid", "a_uv",
             "tex0", "tex1", "res0", "res1")

_VEC = {2: "vec2", 3: "vec3", 4: "vec4"}

# Geometry inputs an authored fragment may declare it needs. Mirrors
# es_matlib.FLAGS: both entry points guard themselves, because the .mat path
# wants line numbers and the Python API must not be able to skip the check.
RECIPE_FLAGS = ("uv",)

# Traditional material blend modes, named as most engines name them.
#   opaque      -- writes colour, no discard, first pass
#   alpha clip  -- discards below a threshold, still first pass, depth-correct
#   alpha       -- blended, drawn in a second pass sorted farthest-first
BLEND_MODES = ("opaque", "alpha clip", "alpha")

# Which faces the engine throws away before shading. "back" matters for the
# blended pass: a thin closed shell stacks its near and far face on the same
# pixel, and with per-layer alpha that stack reads as a patchwork of layer
# counts rather than as shading. gl_FrontFacing does not compile on this GLSL
# ES build, so facing comes from the sign of dot(N, V) -- which is why the
# normal rotation has to agree with the geometry (see u_tilt).
CULL_MODES = ("back",)


def normalize_cull(value):
    """Canonical cull mode, or ValueError. None and "" mean cull nothing."""
    if value is None or value == "" or value is False:
        return None
    words = str(value).replace("_", " ").replace("-", " ").strip().lower()
    words = " ".join(words.split())
    if words in CULL_MODES:
        return words
    raise ValueError("unknown cull mode %r (known: %s)"
                     % (value, ", ".join(CULL_MODES)))


def normalize_blend(value):
    """Canonical blend mode name, or ValueError. Accepts the obvious
    spellings so an author writing "Alpha_Clip" does not get a puzzle."""
    words = str(value).replace("_", " ").replace("-", " ").strip().lower()
    words = " ".join(words.split())
    if words == "alphaclip":
        words = "alpha clip"
    for mode in BLEND_MODES:
        if words == mode.lower():
            return mode
    raise ValueError("unknown blend mode %r (known: %s)"
                     % (value, ", ".join(BLEND_MODES)))


def param_type(value):
    """GLSL declaration type for a parameter value, None if unsupported."""
    if isinstance(value, (tuple, list)):
        return _VEC.get(len(value))
    if isinstance(value, bool):
        return None
    if isinstance(value, (float, int)):
        return "float"
    return None


def define_recipe(name, fragment, params=None, requires=(), map_path=None,
                  applies=None, blend="opaque", cutoff=None, cull=None):
    """Register an author-written fragment and the parameters it exposes.

    Authored text supplies the body of the fragment function only: it assigns
    `es_color`, and the engine emits the discard, the premultiply and the
    write to gl_FragColor. The vertex stage, its varyings and the sampler
    slots stay engine-owned. Every tunable goes through `params` -- a
    hand-written declaration would collide with the block this builds, and
    ShaderPart drops declarations whose name appears nowhere.

    `requires` names geometry inputs ("uv"), checked against the mesh by
    Material.shader_name. `map_path` binds an image as tex1 and legalises
    texture2D. `applies` is a glTF material name whose splits take this
    recipe. `blend` is one of BLEND_MODES; `cutoff` needs "alpha clip";
    `cull` is one of CULL_MODES or None.
    """
    params = params or {}
    blend = normalize_blend(blend)
    cull = normalize_cull(cull)
    if cutoff is None:
        cutoff = 0.5
    else:
        cutoff = float(cutoff)
        if blend != "alpha clip":
            raise ValueError("%s: cutoff %r needs blend \"alpha clip\", not "
                             "%r" % (name, cutoff, blend))
    if not 0.0 < cutoff < 1.0:
        raise ValueError("%s: cutoff must be strictly between 0 and 1, got %r"
                         % (name, cutoff))
    _validate(name + ".fragment", fragment)
    if "gl_FragColor" in fragment:
        raise ValueError("%s: fragment writes gl_FragColor -- the engine owns "
                         "that line; assign es_color instead" % name)
    if "es_color" not in fragment:
        raise ValueError("%s: fragment never assigns es_color, so nothing it "
                         "computes can reach the screen" % name)
    for line in fragment.split("\n"):
        head = line.strip()
        if head.startswith(("uniform ", "attribute ", "varying ")):
            raise ValueError("%s: fragment declares %r -- declare it through "
                             "params= instead" % (name, head[:48]))
        if "tex0" in head:
            raise ValueError("%s: tex0 is the engine's morph atlas or "
                             "placeholder; a material's own map is tex1"
                             % name)
        if ("texture2D" in head or "tex1" in head) and map_path is None:
            raise ValueError("%s: fragment samples a texture but declares no "
                             "map, so nothing is bound at tex1" % name)
    types = {}
    for pname, value in params.items():
        ptype = param_type(value)
        if ptype is None:
            raise ValueError("%s: parameter %r has unsupported value %r"
                             % (name, pname, value))
        if pname in _RESERVED or pname.startswith(("u_bone", "u_sw")):
            raise ValueError("%s: parameter name %r is engine-reserved"
                             % (name, pname))
        types[pname] = ptype
    for flag in requires:
        if flag not in RECIPE_FLAGS:
            raise ValueError("%s: unknown requirement %r (known: %s)"
                             % (name, flag, ", ".join(sorted(RECIPE_FLAGS))))
    RECIPES[name] = {"fragment": fragment, "params": dict(params),
                     "types": types, "requires": frozenset(requires),
                     "map": map_path, "applies": applies, "blend": blend,
                     "cutoff": cutoff, "cull": cull}
    return name


def recipe_types(name):
    return RECIPES[name]["types"]


def recipe_requires(name):
    return RECIPES[name]["requires"]


def recipe_map(name):
    return RECIPES[name]["map"]


def recipe_applies(name):
    return RECIPES[name]["applies"]


def recipe_blend(name):
    return RECIPES[name]["blend"]


def recipe_cutoff(name):
    return RECIPES[name]["cutoff"]


def recipe_cull(name):
    return RECIPES[name]["cull"]


def _recipe_fragment(name, blend, cull=None):
    """Wrap an authored body in the lines the engine owns.

    es_color is a plain local, so it never touches the `variables` block that
    ShaderPart parses line by line.

    Ren'Py composites every draw with a premultiplied blend (GL_ONE,
    GL_ONE_MINUS_SRC_ALPHA -- config.gl_blend_func["normal"], and the same
    pair draw_screen and the render-to-texture path leave set), so the alpha
    this tail writes is read as a weight on the destination, not as coverage
    of our own colour. Measured against a #e08c14 backdrop, a fragment of
    vec4(0.5, 0.2, 0.2, 0.6) lands at (214, 94, 75) -- destination bleeding
    through -- where straight alpha would give (166, 87, 39) and an opaque
    one (201, 80, 80). So each mode finishes the arithmetic here:
      alpha      -- premultiply, so the destination keeps its share.
      alpha clip -- the discard already decided coverage; what survives is
                    opaque, so alpha goes to 1 and a cutout can never tint
                    what it did not cut.
      opaque     -- alpha goes to 1: that is what the flag promises.

    The cull goes ahead of the body so a thrown-away face costs nothing, and
    uses dot(N, V) because gl_FrontFacing will not compile here.
    """
    head = "vec4 es_color = vec4(1.0);\n"
    if cull == "back":
        head += ("if (dot(normalize(v_normal), normalize(u_view_dir)) < 0.0)"
                 " discard;\n")
    if blend == "alpha":
        out = "gl_FragColor = vec4(es_color.rgb * es_color.a, es_color.a);"
    else:
        out = "gl_FragColor = vec4(es_color.rgb, 1.0);"
    clip = ("if (es_color.a < u_alpha_clip) discard;\n"
            if blend == "alpha clip" else "")
    return (head + RECIPES[name]["fragment"] + "\n" + clip + out + "\n")


def recipe_defaults(name):
    return RECIPES[name]["params"]


def _recipe_declares(name):
    """The `variables` block adding this recipe's parameter declarations.

    Sorted so the assembled source, and so the sampler/declaration order, is
    reproducible: ShaderPart parses these line by line and needs exactly
    '{uniform,attribute,varying} {type} {name}'. The leading newline is load
    bearing -- the block appended before this one ends in ';'.
    """
    lines = ["uniform %s %s;" % (t, n)
             for n, t in sorted(RECIPES[name]["types"].items())]
    if RECIPES[name]["map"] is not None:
        # tex1 is the material's own map -- slot 0 is always the atlas or the
        # placeholder, so this is where a bound image lands. res1 arrives
        # free: UniformSampler2D.assign sets the uniform named
        # "res<sampler>" to the texture's pixel size, and an author cannot
        # declare it by hand because fragment declarations are rejected.
        # ShaderPart drops either one if the fragment never mentions it.
        lines = lines + ["uniform sampler2D tex1;", "uniform vec2 res1;"]
    return "\n" + "\n".join(lines) + "\n"


def ensure_fallback():
    """Solid magenta: unmistakable when a material fails to compile."""
    if _fallback_done[0]:
        return FALLBACK_SHADER
    register_shader(
        FALLBACK_SHADER,
        variables="""
        uniform vec4 u_base_color;
        """,
        vertex_200="""""",
        fragment_200="""
        gl_FragColor = vec4(1.0, 0.0, 1.0, 1.0);
        """)
    _fallback_done[0] = True
    return FALLBACK_SHADER


_VARIABLES = """
uniform vec3 u_light_dir;
uniform vec3 u_view_dir;
uniform vec4 u_base_color;
uniform float u_spin;
uniform float u_tilt;
uniform float u_eye_z;
varying vec3 v_normal;
varying vec3 v_half;
varying float v_diff;
varying vec3 v_pos;
"""

# Normal attribute name differs per path (rigid meshes carry a_normal,
# skinned meshes a_norm3) so a program can never bind the wrong layout.
_NORMAL_DECL = "attribute vec3 %s;\n"

# UVs: one channel, mesh-space a_uv passed straight through. ShaderPart only
# keeps a declaration whose name appears in that stage's source, so the
# assignment below is what puts both declarations in the vertex shader -- and
# what gives a fragment that mentions v_uv its declaration for free.
_UV_DECL = "\nattribute vec2 a_uv;\nvarying vec2 v_uv;\n"
_UV_PASS = """
v_uv = a_uv;
"""

_VERTEX_RIGID = """
vec3 skP = a_position.xyz;
vec3 skN = a_normal;
"""

_SKIN_BLEND = """
mat4 es_sm = es_pick_%(n)d(a_joints.x) * a_weights.x;
es_sm += es_pick_%(n)d(a_joints.y) * a_weights.y;
es_sm += es_pick_%(n)d(a_joints.z) * a_weights.z;
es_sm += es_pick_%(n)d(a_joints.w) * a_weights.w;
vec3 skP = (es_sm * vec4(a_position.xyz, 1.0)).xyz;
vec3 skN = mat3(es_sm) * a_norm3;
"""

# Диагностика: обход скиннинга (позиции/нормали как есть).
_SKIN_IDENTITY = """
vec3 skP = a_position.xyz;
vec3 skN = a_norm3;
"""

# Analytic normal rotation. Row order copies rotate_matrix(x, y, 0)
# from matrix_functions.pxi, so it matches Matrix.rotate(tilt, angle, 0).
_VERTEX = """
vec3 wnormal = normalize(vec3(
    es_ca * skN.x + es_sx * es_sa * skN.y + es_cx * es_sa * skN.z,
    es_cx * skN.y - es_sx * skN.z,
    -es_sa * skN.x + es_sx * es_ca * skN.y + es_cx * es_ca * skN.z));
v_normal = wnormal;
v_pos = vec3(
    es_ca * skP.x + es_sx * es_sa * skP.y + es_cx * es_sa * skP.z,
    es_cx * skP.y - es_sx * skP.z,
    -es_sa * skP.x + es_sx * es_ca * skP.y + es_cx * es_ca * skP.z);
vec3 L = normalize(u_light_dir);
vec3 V = normalize(u_view_dir);
v_half = normalize(L + V);
v_diff = max(dot(wnormal, L), 0.0);
vec4 es_p = u_transform * vec4(skP.xyz, 1.0);
es_p.z = clamp(es_p.z, 0.0, 0.99 * es_p.w);
gl_Position = es_p;
"""

_VERTEX_SPIN = """
float es_sa = sin(radians(u_spin));
float es_ca = cos(radians(u_spin));
float es_sx = sin(radians(u_tilt));
float es_cx = cos(radians(u_tilt));
"""

# --- Shape-key (morph target) variant -----------------------------------
# Skin AFTER morphing (Godot's order): deltas land in bind space, the
# palette then rotates them with the base. Names must appear verbatim in
# vertex_200 or ShaderPart drops the declaration (gl2shadercache.py).

_SKIN_BLEND_MORPH = """
mat4 es_sm = es_pick_%(n)d(a_joints.x) * a_weights.x;
es_sm += es_pick_%(n)d(a_joints.y) * a_weights.y;
es_sm += es_pick_%(n)d(a_joints.z) * a_weights.z;
es_sm += es_pick_%(n)d(a_joints.w) * a_weights.w;
vec3 skP = (es_sm * vec4(a_position.xyz + es_dp, 1.0)).xyz;
vec3 skN = mat3(es_sm) * (a_norm3 + es_dn);
"""

_SKIN_IDENTITY_MORPH = """
vec3 skP = a_position.xyz + es_dp;
vec3 skN = a_norm3 + es_dn;
"""


def _morph_vars(k):
    # Declared via `variables` (not vertex_functions): these names occur
    # inline in the vertex_200 body, which is exactly what ShaderPart
    # requires to keep a declaration. Leading newline: ShaderPart parses
    # `variables` line-by-line and raises on any line that is not exactly
    # "{uniform,attribute,vertex} {type} {name}" -- the preceding
    # _skin_decls block ends with ';' and no newline.
    lines = ["uniform sampler2D tex0;",
             "uniform float u_mscale;",
             "uniform float u_nscale;",
             "uniform float u_nv;",
             "uniform vec2 u_as;",
             "attribute float a_vid;"]
    for i in range(k):
        lines.append("uniform vec2 u_sw%d;" % i)
    return "\n" + "\n".join(lines) + "\n"


def _morph_funcs():
    # Pre-declaration section: must not reference uniforms (functions
    # are emitted before `variables`; sizes come in as arguments).
    return """
float es_b(float x) {
  return floor(x * 255.0 + 0.5);
}
float es_d(float hi, float lo) {
  return (es_b(hi) * 255.0 + es_b(lo)) / 65535.0 * 2.0 - 1.0;
}
vec2 es_uv(float es_i, vec2 es_sz) {
  return vec2(mod(es_i, es_sz.x) + 0.5, floor(es_i / es_sz.x) + 0.5) / es_sz;
}
"""


def _morph_body(k):
    # Unrolled active-shape slots (Ren'Py cannot bind uniform arrays,
    # gl2shader.pyx find_variables rejects them). Zero weight exits
    # before any texture fetch. u_sw.x = local row, u_sw.y = weight.
    out = ["vec3 es_dp = vec3(0.0);",
           "vec3 es_dn = vec3(0.0);"]
    slot = """
if (u_sw%(i)d.y != 0.0) {
  float es_f%(i)d = (floor(u_sw%(i)d.x) * u_nv + a_vid) * 4.0;
  vec4 es_a%(i)d = texture2D(tex0, es_uv(es_f%(i)d, u_as));
  vec4 es_b%(i)d = texture2D(tex0, es_uv(es_f%(i)d + 1.0, u_as));
  vec4 es_c%(i)d = texture2D(tex0, es_uv(es_f%(i)d + 2.0, u_as));
  vec4 es_e%(i)d = texture2D(tex0, es_uv(es_f%(i)d + 3.0, u_as));
  es_dp += vec3(es_d(es_a%(i)d.r, es_a%(i)d.g),
                es_d(es_a%(i)d.b, es_b%(i)d.r),
                es_d(es_b%(i)d.g, es_b%(i)d.b)) * (u_mscale * u_sw%(i)d.y);
  es_dn += vec3(es_d(es_c%(i)d.r, es_c%(i)d.g),
                es_d(es_c%(i)d.b, es_e%(i)d.r),
                es_d(es_e%(i)d.g, es_e%(i)d.b)) * (u_nscale * u_sw%(i)d.y);
}
"""
    for i in range(k):
        out.append(slot % {"i": i})
    return "\n".join(out) + "\n"

_FRAG_BLINN = """
vec3 N = normalize(v_normal);
vec3 H = normalize(v_half);
float diff = v_diff;
float spec = pow(max(dot(N, H), 0.0), %(shininess)r);
vec3 col = u_base_color.rgb * (%(ambient)r + %(diffuse)r * diff) + vec3(1.0) * spec * %(spec_weight)r;
gl_FragColor = vec4(col, u_base_color.a);
"""

_FRAG_LAMBERT = """
vec3 N = normalize(v_normal);
float diff = v_diff;
vec3 col = u_base_color.rgb * (%(ambient)r + %(diffuse)r * diff);
gl_FragColor = vec4(col, u_base_color.a);
"""

_FRAG_NORMAL_VIZ = """
gl_FragColor = vec4(normalize(v_normal) * 0.5 + vec3(0.5), 1.0);
"""


def _validate(name, source):
    try:
        source.encode("ascii")
    except UnicodeEncodeError:
        raise ValueError("%s: shader source is not pure ASCII" % name)
    if source.count("{") != source.count("}"):
        raise ValueError("%s: unbalanced braces" % name)
    if source.count("(") != source.count(")"):
        raise ValueError("%s: unbalanced parens" % name)
    import re
    if re.search(r"vec3\s+\w+\s*=\s*a_position\s*;", source):
        raise ValueError("%s: vec4 a_position assigned to vec3 "
                         "(use a_position.xyz)" % name)
    for _ln in source.split("\n"):
        _s = _ln.strip()
        if _s.startswith("}") and ("attribute " in _s or "uniform " in _s):
            raise ValueError("%s: declaration fused with brace %r "
                             "(needs newline)" % (name, _ln.strip()[:60]))


def _skin_decls(n):
    # NOTE: bone uniforms must NOT go through `variables`: ShaderPart keeps
    # a declaration only if its name appears verbatim in a part, and emits
    # functions before declarations. Both drop/misplace u_boneN, so they are
    # declared at the top of vertex_functions instead (see _skin_pick).
    return "\n".join(["attribute vec4 a_joints;",
                      "attribute vec4 a_weights;"])


def _skin_pick(n):
    decls = ["uniform mat4 u_bone%d;" % i for i in range(n)]
    cases = []
    for i in range(n - 1):
        cases.append("  if (j < %(f)0.1f) return u_bone%(i)d;" %
                     {"f": i + 0.5, "i": i})
    return ("\n".join(decls) +
            "\nmat4 es_pick_%d(float j) {\n" % n + "\n".join(cases) +
            "\n  return u_bone%d;\n}\n" % (n - 1))


def get_variant(specular=True, normal_viz=False, shininess=16.0,
                ambient=0.3, diffuse=0.7, spec_weight=0.5,
                skinned=False, max_bones=24, bind_pose=False,
                morph=0, uvs=False, recipe=None, blend="opaque", cull=None):
    """Return a registered shader name for this feature combination.

    `morph` is the number of active shape-key slots (0 = no morphs).
    Morphs need the skinned path (a_vid lives in the split layout);
    ignored for rigid variants. `uvs` declares and forwards a_uv.
    `recipe` replaces the fragment with an authored one and adds its
    parameter declarations; the lighting flags then decide nothing, and
    `blend` and `cull` give that recipe its discard, premultiply and
    gl_FragColor."""
    if not skinned:
        morph = 0
    morph = max(0, int(morph))
    blend = normalize_blend(blend) if recipe is not None else "opaque"
    cull = normalize_cull(cull) if recipe is not None else None
    # The digest of the authored source belongs in the key, not just the
    # recipe name: after a reload edits the .glsl the name is unchanged and a
    # name-keyed cache would serve the stale program forever. ASCII was
    # enforced by define_recipe, so encode cannot raise here.
    digest = None
    if recipe is not None:
        digest = hashlib.md5(RECIPES[recipe]["fragment"]
                             .encode("ascii")).hexdigest()[:8]
    key = (bool(specular), bool(normal_viz), float(shininess),
           float(ambient), float(diffuse), float(spec_weight),
           bool(skinned), int(max_bones), bool(bind_pose), morph,
           bool(uvs), recipe, digest, blend, cull)
    if key in VARIANTS:
        return VARIANTS[key]
    if recipe is not None:
        fragment = _recipe_fragment(recipe, blend, cull)
    elif normal_viz:
        fragment = _FRAG_NORMAL_VIZ
    elif specular:
        fragment = _FRAG_BLINN % {"shininess": float(shininess),
                                  "ambient": float(ambient),
                                  "diffuse": float(diffuse),
                                  "spec_weight": float(spec_weight)}
    else:
        fragment = _FRAG_LAMBERT % {"ambient": float(ambient),
                                    "diffuse": float(diffuse)}
    variables = _VARIABLES
    if recipe is not None:
        variables = variables + _recipe_declares(recipe)
    if blend == "alpha clip":
        # Declared here rather than as a parameter: the view always pushes it,
        # so it cannot be left unset, and authors should not have to.
        variables = variables + "\nuniform float u_alpha_clip;\n"
    vfuncs = ""
    prelude = ""
    if skinned:
        n = max(1, int(max_bones))
        variables = variables + (_NORMAL_DECL % "a_norm3")
        variables = variables + _skin_decls(n)
        vfuncs = _skin_pick(n)
        if morph:
            variables = variables + _morph_vars(morph)
            vfuncs = vfuncs + _morph_funcs()
            prelude = _morph_body(morph)
        if bind_pose:
            skin = (_SKIN_IDENTITY_MORPH if morph else _SKIN_IDENTITY)
        else:
            skin = ((_SKIN_BLEND_MORPH if morph else _SKIN_BLEND)
                    % {"n": n})
    else:
        variables = variables + (_NORMAL_DECL % "a_normal")
        skin = _VERTEX_RIGID
    if uvs:
        variables = variables + _UV_DECL
    vertex = (prelude + skin + (_UV_PASS if uvs else "") + "\n"
              + _VERTEX_SPIN + _VERTEX)
    label = recipe or "es_mat3"
    _validate(label + ".vertex", vertex)
    _validate(label + ".fragment", fragment)
    # The program name hashes the assembled source, not the request that made
    # it: editing an authored fragment changes the name, so no stale compiled
    # program survives a reload, while unchanged content still lands on
    # Ren'Py's part-name cache instead of recompiling. Validate first, since
    # _validate reports non-ASCII sources far better than encode() does.
    source = variables + vfuncs + prelude + vertex + fragment
    name = "es_mat3_" + hashlib.md5(source.encode("ascii")).hexdigest()[:10]
    register_shader(name, variables=variables,
                    vertex_functions=vfuncs,
                    vertex_200=vertex, fragment_200=fragment)
    VARIANTS[key] = name
    return name
