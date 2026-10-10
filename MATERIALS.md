# The `.mat` material format

Materials are authored as a pair of plain files: a `<name>.mat` that declares
the surface and its tunable parameters, and a `<name>.glsl` next to it holding
the fragment body. `es_matlib.py` parses them into `es_shaders` recipes.

They live with the **content mod that ships them** — there is no engine default
material directory. Load them from an `init 10` block:

    init 10 python:
        import es_engine as _eng
        _eng.add_materials("mods/es-3dmodels-matitan/materials")

`add_materials` returns the registered names. It must run at `init 10` or
later: the engine's own `init` (priority 0) is what puts its folder on
`sys.path`, so an import failure at priority 10 means the engine really is
absent rather than not-yet-loaded.

Attaching a material to a model:

    $ obj = _eng.SkinnedObject(path, recipe="matitan_toon")   # whole object
    $ obj.splits[8]["material"].set("u_opacity", 0.4)         # one group

A recipe that declares `applies` attaches itself instead: it takes over the
splits whose glTF material name matches, with no call site at all. **It wins
over an explicit `recipe=`** on the object: build a `SkinnedObject` with
`recipe="mine"` and any group another mod routed through `applies` still comes
out with that other recipe, so check what actually landed with

    print([(i, s["material"].recipe) for i, s in enumerate(obj.splits)])

Because that routing table is global and is consulted when an object is
**built**, a material registered late affects only objects built after that
point — which is how you scope an override to one step of a demo (load it from
a label, not from `init`).

`define_material()` in `es_engine` is the same backend through Python. Both
paths run every check below, so neither can skip them.


## Grammar

    material <name> {
        fragment "<file.glsl>"          # required, read next to the .mat
        map "<image.png>"               # optional, bound as tex1
        applies "<glTF material name>"  # optional, auto-attach target
        blend "opaque|alpha clip|alpha" # optional, default "opaque"
        cutoff <float strictly 0..1>    # optional, needs blend "alpha clip"
        cull "back"                     # optional, default: cull nothing
        requires uv                     # optional, geometry inputs needed
        float u_x = 1.0                 # typed parameters, with defaults
        vec2  u_p = 0.5 0.5
        vec3  u_tint = 1.0 1.0 1.0
        vec4  u_rect = 0.0 0.0 1.0 1.0
    }

Everything is one statement per line. `#` starts a comment; the fragment file
uses `//` and `/* */`. Repeating a keyword is an error (`blend given twice`),
and so is an unknown one. Parameter names must match `[A-Za-z_][A-Za-z0-9_]*`
and may not collide with engine-reserved names (`u_transform`, `u_n0`..`u_n2`,
`u_off`, `u_eye_z`, `v_normal`, `v_pos`, `tex0`, `a_position`, ...), which is
checked in `es_shaders.define_recipe` so both entry points are covered.

`blend`, `cutoff` and `cull` are **not** parameters. They select a compiled
program, so each distinct combination is its own shader variant; changing one
means rebuilding the view, not pushing a uniform. Parameters are uniforms:
they can be tuned live, every frame, at any value.


## What a fragment may read

Your `.glsl` is the body of the fragment function, nothing else. It assigns
`es_color`; the engine emits the output line, the discard and the premultiply.
Available to it:

| name | meaning |
|---|---|
| `v_normal` | the surface normal, rotated by the object's matrix |
| `v_diff` | `max(dot(N, L), 0.0)` — the wrap diffuse term, per-vertex |
| `v_half` | `normalize(L + V)` — per-vertex, see the gotchas |
| `v_pos` | position rotated by the object's matrix plus its `pos` offset |
| `u_eye_z` | eye distance in the same units as `v_pos` |
| `u_light_dir` | directional light, defaults to `LIGHT_DIR` |
| `u_view_dir` | the fixed view direction, `(0, 0, 1)` |
| `u_base_color` | the object's base colour, straight alpha |
| `v_uv` | `TEXCOORD_0`, only with `requires uv` |
| `tex1`, `res1` | the material's own `map`, and its size in pixels |

`v_pos` and `u_eye_z` together are what give a **per-fragment** view direction:

    vec3 V = normalize(vec3(0.0, 0.0, u_eye_z) - v_pos);

Take it from `u_view_dir` instead and every input is constant across a flat
face, so a panel cannot shade at all — no gradient, no moving highlight, just
one value per facet.

A worked example, a glossy translucent white panel:

    vec3 N = normalize(v_normal);
    vec3 V = normalize(vec3(0.0, 0.0, u_eye_z) - v_pos);
    vec3 H = normalize(normalize(u_light_dir) + V);
    float ndv = max(dot(N, V), 0.0);
    float fres = pow(1.0 - ndv, 4.0);
    float spec = pow(max(dot(N, H), 0.0), u_gloss);
    vec3 base = u_tint * (u_ambient + (1.0 - u_ambient) * v_diff);
    float hi = clamp(u_spec * spec + u_rim * fres, 0.0, 1.0);
    es_color = vec4(mix(base, vec3(1.0), hi), u_opacity);


## Blend modes

Ren'Py composites **every** draw with a premultiplied blend —
`glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA)` for the screen and for each
render-to-texture pass. So the alpha you write is a weight on the destination,
not coverage of your own colour, and the engine finishes that arithmetic per
mode rather than leaving it to you:

| mode | what the engine adds | pass |
|---|---|---|
| `opaque` | `gl_FragColor = vec4(rgb, 1.0)` | first |
| `alpha clip` | a `discard` below `cutoff`, then alpha 1 | first, depth-correct |
| `alpha` | `gl_FragColor = vec4(rgb * a, a)` | second, sorted farthest-first |

`alpha clip` is the one to reach for when you want holes rather than glass: the
surviving fragments are fully opaque, so a cutout can never tint what it did
not cut, and it keeps correct depth. Note that in this asset set the mask's
transparent texels are **also black in RGB**, so the sampled alpha has to reach
`es_color.a` or you paint the mask as dark patches:

    vec4 texel = texture2D(tex1, v_uv);
    es_color = vec4(texel.rgb * u_tint * light, texel.a * u_base_color.a);

`cull "back"` makes the engine emit, ahead of your body:

    if (dot(normalize(v_normal), normalize(u_view_dir)) < 0.0) discard;

This matters for blended geometry. A thin closed shell stacks its near and far
face on the same pixel, and with per-layer alpha that stack reads as a visible
map of layer counts rather than as shading — measured on the matitan tail, the
two dominant colours were exactly the predicted 2- and 3-layer results.
Culling leaves one layer. `gl_FrontFacing` is not an option: it does not
compile on this GLSL ES build.


## Rules the parser enforces

Each of these names the `.glsl` file and line, which is the point of checking
in the parser rather than at compile time.

- **Pure ASCII.** A non-ASCII comment once broke the compile with no useful
  message, so it fails here instead.
- **No `gl_FragColor`.** Assign `es_color`; the engine owns the output line.
- **No declarations.** No `uniform`/`attribute`/`varying` lines: parameters
  belong in the `.mat`, which generates that block. (Ren'Py's `ShaderPart`
  drops a declaration whose name appears nowhere in the stage that gets it, so
  a hand-written one is also how you silently lose a uniform.)
- **No `tex0`.** Slot 0 is the engine's morph atlas or 1x1 placeholder; a
  material's own map is always `tex1`.
- **Sampling needs `map`.** `texture2D` or `tex1` without a declared `map`
  means nothing is bound.
- **`es_color` must be assigned.** Otherwise nothing you compute reaches the
  screen.
- **`v_uv` needs `requires uv`, and `requires uv` needs a `v_uv` read.** Both
  directions are checked, so a stale requirement cannot linger.
- **`cutoff` needs `blend "alpha clip"`.**

`es_matlib.parse()` imports no Ren'Py and takes a `read` callback, so the
grammar and all of these diagnostics run under a bare interpreter — see
`HARNESS.md` for the unit test that exercises them on both bundled pythons.


## Gotchas worth knowing

- **The framebuffer is RGBA8 and clamps.** Additive highlights
  (`col + spec * k`, `k > 1`) clip across a wide area into a featureless blob.
  Mix toward white with a clamped weight instead; that cannot overshoot.
- **Don't drive `es_color.a` from a fresnel term on flat geometry.** A near-flat
  panel is mostly grazing angle, so `fres ~ 1` everywhere and
  `opacity + (1-opacity)*fres` turns the whole sheet opaque — 86-91% of its
  pixels showed no backdrop at all. Put gloss in the colour, keep alpha calm.
- **`v_half` and `v_diff` are per-vertex.** Interpolated across a coarse mesh
  they turn highlights into polygonal lumps. Rebuild `H` in the fragment from
  `u_light_dir` and the per-fragment `V` above.
- **A view on screen keeps the program it was built with.** Models are
  constructed in `SkinnedView.__init__`, so editing a `.glsl` and hitting
  Shift+R shows up only in newly built views.
- **Program names hash the assembled source**, and the variant cache carries a
  digest of your fragment, so editing it can never serve a stale program — but
  it also means every distinct `blend`/`cull`/digest is its own compile.
- **Transparency ordering is coarse.** Splits are sorted by centroid distance,
  farthest-first, and nothing is sorted inside a split. Depth writes cannot be
  turned off for a single draw (the only per-draw GL state is `blend_func`,
  `color_mask`, `texture_scaling`), so the first transparent fragment at a
  pixel claims its depth and a back face can occlude the front face of the same
  sheet.
