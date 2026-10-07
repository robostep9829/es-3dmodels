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
import os
import time

from renpy.gl2.gl2shadercache import register_shader

VARIANTS = {}
FALLBACK_SHADER = "es_fallback"
_fallback_done = [False]

# Shift+R reload re-runs init python (re-registering parts with new source)
# but keeps compiled Programs cached by partname tuple. A per-process salt
# in every program name makes stale hits impossible by construction.
_SALT = "%x%x" % (os.getpid() & 0xfffff, int(time.time()) & 0xfffff)


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
varying vec3 v_normal;
varying vec3 v_half;
varying float v_diff;
"""

# Normal attribute name differs per path (rigid meshes carry a_normal,
# skinned meshes a_norm3) so a program can never bind the wrong layout.
_NORMAL_DECL = "attribute vec3 %s;\n"

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
float es_sx = sin(radians(25.0));
float es_cx = cos(radians(25.0));
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
                morph=0):
    """Return a registered shader name for this feature combination.

    `morph` is the number of active shape-key slots (0 = no morphs).
    Morphs need the skinned path (a_vid lives in the split layout);
    ignored for rigid variants."""
    if not skinned:
        morph = 0
    morph = max(0, int(morph))
    key = (bool(specular), bool(normal_viz), float(shininess),
           float(ambient), float(diffuse), float(spec_weight),
           bool(skinned), int(max_bones), bool(bind_pose), morph)
    if key in VARIANTS:
        return VARIANTS[key]
    if normal_viz:
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
    vertex = prelude + skin + "\n" + _VERTEX_SPIN + _VERTEX
    # v3 namespace + process salt: a name is never reused across a source
    # change OR a process boundary, so no stale compile can ever be served.
    name = "es_mat3_%s_%s" % (
        _SALT, hashlib.md5(repr(key).encode("ascii")).hexdigest()[:8])
    _validate(name + ".vertex", vertex)
    _validate(name + ".fragment", fragment)
    register_shader(name, variables=variables,
                    vertex_functions=vfuncs,
                    vertex_200=vertex, fragment_200=fragment)
    VARIANTS[key] = name
    return name
