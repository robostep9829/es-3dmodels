# -*- coding: utf-8 -*-
"""The one fit policy every loader shares.

Three builders turn an authored mesh into vertex-buffer coordinates -- the glTF
skinning path, the glTF bind-pose path and the OBJ path -- and each used to
carry its own copy of "how big, and where". They had already drifted: the
skinned path anchored on the rig origin, the bind-pose path anchored on the
bounding box and rewound winding, and the OBJ path did neither. This is that
policy in one place, so a fourth loader cannot disagree with the other three.

`target` is the extent the *longest* axis is fitted to, in buffer units, or None
to leave the model in the units it was authored in. It is uniform, so aspect is
preserved and the two shorter axes land proportionally inside it.

`anchor` is the point that ends up at the centre of the buffer, which is also
the point the turntable rotates about because Ren'Py rotates a render about its
own centre. "origin" makes `target` behave as a pure scale about the rig's
(0, 0, 0); "center" frames the bounding box instead, which is what a character
showcase usually wants. Nothing here mirrors Y or rewinds winding: those are
per-format conventions, and they belong to the loader that owns the format.
"""

ANCHORS = ("origin", "center")


def normalize_anchor(value):
    """Canonical anchor name, or ValueError naming the legal ones."""
    if value is None:
        return "origin"
    words = str(value).replace("_", " ").replace("-", " ").strip().lower()
    words = " ".join(words.split())
    if words in ANCHORS:
        return words
    raise ValueError("fit anchor must be one of %r, got %r"
                     % (list(ANCHORS), value))


def fit_for_bounds(bounds_min, bounds_max, target=None, anchor="origin"):
    """(fit_scale, anchor_point) for an accumulated axis-aligned bounds.

    Split out from fit_positions because the skinned path accumulates its
    bounds over every primitive before it transforms any of them -- one fit for
    the whole object is what keeps material groups registered to each other --
    and it needs the scale and the anchor separately for the vertex copy, the
    morph atlas and the skeleton's conjugation.
    """
    anchor = normalize_anchor(anchor)
    extent = max(bounds_max[k] - bounds_min[k] for k in range(3))
    # A mesh with no extent cannot be fitted to a size: the ratio is undefined,
    # and multiplying by target anyway would fling a point mesh away from the
    # anchor rather than size it. So it keeps its authored units.
    scale = 1.0 if target is None or extent <= 0.0 else float(target) / extent
    if anchor == "center":
        point = [(bounds_min[k] + bounds_max[k]) / 2.0 for k in range(3)]
    else:
        point = [0.0, 0.0, 0.0]
    return scale, point


def fit_positions(verts, target=None, anchor="origin"):
    """(out_verts, fit_scale, anchor_point): the fit applied to positions.

    Directions are not touched -- a normal is not scaled, and an anchor
    subtraction would turn it into a point.
    """
    xs = [v[0] for v in verts]
    ys = [v[1] for v in verts]
    zs = [v[2] for v in verts]
    scale, point = fit_for_bounds((min(xs), min(ys), min(zs)),
                                  (max(xs), max(ys), max(zs)), target, anchor)
    out = [((v[0] - point[0]) * scale,
            (v[1] - point[1]) * scale,
            (v[2] - point[2]) * scale) for v in verts]
    return out, scale, point
