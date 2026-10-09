# -*- coding: utf-8 -*-
"""Hardware capability probe: decides uniform-driven engine limits.

Queries GL through renpy.uguu (Python-callable; loaded by the GLDraw
constructor) and picks how many bone-palette uniforms (max_bones) and
morph slots a skinned draw may declare.

Proven rules this encodes:
- Ren'Py never validates uniform COUNTS (only unset declarations raise
  in draw), so the limits must be enforced here, before get_variant
  bakes them into a shader key,
- never call GL before renpy.uguu.gl.load() ran: found_functions gates
  every query (NULL function pointers are uncatchable),
- probe failure keeps today's defaults (24 bones, 32 slots); raise and
  shrink only happen on a successful probe, ES_MAX_BONES /
  ES_MORPH_SLOTS env vars pin the wants either way (budget still
  clamps them),
- decide() stays pure (no renpy imports) so it is table-testable
  under a headless python3.
"""
import os

DEFAULT_BONES = 24       # shipped cap; shrink reference
DEFAULT_SLOTS = 32       # shipped slots; shrink reference
RAISE_MAX_BONES = 64     # un-pinned want on a successful probe
RAISE_MAX_SLOTS = 64
MIN_BONES = 8            # es_pick chain floor under tiny budgets
SLOTS_MIN = 1            # >= 1 slot keeps morph meaningful
MARGIN = 0.9             # slack on the vertex-uniform budget

# Uniform components per skinned draw (GLSL counts: mat4 = 16):
# u_transform 16 + u_base_color 4 + u_light_dir 3 + u_view_dir 3 +
# u_spin 1 = 27; the morph variant adds u_mscale/u_nscale/u_nv 3 +
# u_as 2 + tex0 sampler 4 (counted fat) -> 40.
FIXED_PLAIN = 27
FIXED_MORPH = 40

_state = {"done": False, "decision": None, "caps": None, "fail": False}


def _env_int(name):
    """Positive int from env, or None (0/negative/garbage = unset)."""
    try:
        v = int(os.environ.get(name, ""))
    except (TypeError, ValueError):
        return None
    return v if v >= 1 else None


def decide(caps, bones_want=RAISE_MAX_BONES, slots_want=RAISE_MAX_SLOTS):
    """Map a capability dict to engine limits (pure; no renpy).

    caps=None -> no probe: wants are returned untouched (the caller
    passes the shipped defaults in that case). Otherwise wants are
    raised/shrunk to fit margin*vuc; morph drops when the hardware has
    no vertex texture units or the budget cannot hold one slot.
    """
    if caps is None or caps.get("vuc") is None:
        return {"bones": int(bones_want), "slots": int(slots_want),
                "morph": True, "reason": "noprobe"}
    vuc = float(caps["vuc"])
    vtf = caps.get("vtf")
    vtf = 1 if vtf is None else int(vtf)
    budget = MARGIN * vuc
    tags = []
    morph = vtf >= 1
    if morph:
        # Reserve at least one slot before sizing the palette.
        bmax = int((budget - FIXED_MORPH - 2 * SLOTS_MIN) // 16)
        if bmax < MIN_BONES:
            morph = False
            tags.append("nomorphbudget")
    if morph:
        bones = max(MIN_BONES, min(int(bones_want), bmax))
        smax = int((budget - FIXED_MORPH - 16.0 * bones) // 2)
        slots = max(SLOTS_MIN, min(int(slots_want), smax))
    else:
        if vtf < 1:
            tags.append("vtf0")
        bmax = int((budget - FIXED_PLAIN) // 16)
        bones = max(MIN_BONES, min(int(bones_want), bmax))
        slots = 0
    if bones > DEFAULT_BONES or slots > DEFAULT_SLOTS:
        tags.append("raised")
    if bones < DEFAULT_BONES or slots < DEFAULT_SLOTS:
        tags.append("shrunk")
    used = (16 * bones + (FIXED_MORPH if morph else FIXED_PLAIN) +
            (2 * slots if morph else 0))
    if used > budget:
        tags.append("underbudget")
    if not tags:
        tags.append("ok")
    return {"bones": bones, "slots": slots, "morph": morph,
            "reason": "/".join(tags)}


def _sane(v):
    """ASCII-only log text (GL strings come back arbitrary)."""
    if v is None:
        return "?"
    v = str(v)
    return "".join(c if 32 <= ord(c) < 127 else "?" for c in v)


def _query():
    """(caps dict, how) -- how is 'ok', 'noload' (not yet, retry
    later) or a failure tag. Never calls GL before load()."""
    try:
        import renpy.uguu.gl as ugl
        found = getattr(ugl, "found_functions", None) or set()
        for fn in ("glGetIntegerv", "glGetString", "glGetError"):
            if fn not in found:
                return None, "noload"
        import renpy.uguu as uguu

        def qi(pname):
            # py2 array.array has no new-style buffer interface here;
            # uguu.IntBuffer exports __getbuffer__ + __getitem__.
            b = uguu.IntBuffer([-1])
            uguu.glGetIntegerv(pname, b)
            return b[0]

        caps = {
            "vuc": qi(uguu.GL_MAX_VERTEX_UNIFORM_COMPONENTS),
            "fuc": qi(uguu.GL_MAX_FRAGMENT_UNIFORM_COMPONENTS),
            "vtf": qi(uguu.GL_MAX_VERTEX_TEXTURE_IMAGE_UNITS),
            "attribs": qi(uguu.GL_MAX_VERTEX_ATTRIBS),
            "tex_units": qi(uguu.GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS),
            "maxtex": qi(uguu.GL_MAX_TEXTURE_SIZE),
        }
        for _ in range(8):
            if uguu.glGetError() == 0:
                break
        caps["vendor"] = _sane(uguu.glGetString(uguu.GL_VENDOR))
        caps["renderer"] = _sane(uguu.glGetString(uguu.GL_RENDERER))
        caps["version"] = _sane(uguu.glGetString(uguu.GL_VERSION))
        if caps["vuc"] <= 0 or caps["vtf"] < 0 or caps["maxtex"] <= 0:
            return None, "invalid"
        return caps, "ok"
    except Exception as e:
        return None, "error:%s" % (e,)


def ensure():
    """Probe once, decide once, print ES_CAPS once. Idempotent.

    Returns the decision dict, or None while probing is impossible
    (callers then fall back to the defaults)."""
    if _state["done"]:
        return _state["decision"]
    caps, how = _query()
    if caps is None:
        if how != "noload" and not _state["fail"]:
            _state["fail"] = True
            print("ES_CAPS probe=fail reason=%s bones=%d slots=%d" % (
                how, _env_int("ES_MAX_BONES") or DEFAULT_BONES,
                _env_int("ES_MORPH_SLOTS") or DEFAULT_SLOTS))
        return None
    pin_b = _env_int("ES_MAX_BONES")
    pin_s = _env_int("ES_MORPH_SLOTS")
    bones_want = pin_b if pin_b is not None else RAISE_MAX_BONES
    slots_want = pin_s if pin_s is not None else RAISE_MAX_SLOTS
    d = decide(caps, bones_want, slots_want)
    _state["caps"] = caps
    _state["decision"] = d
    _state["done"] = True
    print("ES_CAPS probe=ok vuc=%d fuc=%d vtf=%d attribs=%d maxtex=%d "
          "bones=%d slots=%d morph=%d reason=%s vendor=%s renderer=%s "
          "version=%s" % (
              caps["vuc"], caps["fuc"], caps["vtf"], caps["attribs"],
              caps["maxtex"], d["bones"], d["slots"],
              1 if d["morph"] else 0, d["reason"], caps["vendor"],
              caps["renderer"], caps["version"]))
    return d


def max_bones():
    """Decided palette capacity (uniforms = mat4 x this per draw)."""
    d = ensure()
    if d is not None:
        return d["bones"]
    return _env_int("ES_MAX_BONES") or DEFAULT_BONES


def slots():
    """Decided morph slot count (u_sw uniforms per morph draw)."""
    d = ensure()
    if d is not None:
        return d["slots"]
    return _env_int("ES_MORPH_SLOTS") or DEFAULT_SLOTS


def morph_ok():
    """False when the hardware/budget cannot carry the atlas path
    (vertex texture fetch or uniform budget)."""
    d = ensure()
    if d is not None:
        return bool(d["morph"])
    return True


def check_atlas(width, height):
    """Warn when a packed atlas exceeds GL_MAX_TEXTURE_SIZE."""
    caps = _state["caps"]
    if not caps:
        return
    m = int(caps.get("maxtex") or 0)
    if m and (int(width) > m or int(height) > m):
        print("ES_ATLAS_WARN %dx%d exceeds GL_MAX_TEXTURE_SIZE %d" % (
            width, height, m))
