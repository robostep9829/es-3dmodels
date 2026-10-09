# -*- coding: utf-8 -*-
"""Shape-key (morph target) atlas for the es-3dmodels engine.

Packs glTF POSITION/NORMAL morph deltas into one RGBA8 texture sampled
in the vertex stage (GL 2.0-era desktop floor: no float textures, no
bindable uniform arrays, no gl_VertexID -- an a_vid attribute carries
the vertex index).

Encoding, 4 texels per (shape, vertex) -- 6 channels x 16 bits = 96 bits
= exactly 12 usable bytes, because alpha MUST stay 255: Ren'Py's texture
upload premultiplies, which would corrupt packed bytes below 255.

    t0 = (dx_hi, dx_lo, dy_hi, 255)
    t1 = (dy_lo, dz_hi, dz_lo, 255)
    t2 = (nx_hi, nx_lo, ny_hi, 255)
    t3 = (ny_lo, nz_hi, nz_lo, 255)

Fixed point: value in [-1,1] -> round((v+1)/2 * 65535) -> (hi, lo).
The shader decodes with floor(x*255+0.5) (GLSL 1.20 has no round()).

Layout is linear: texel t = (shape * nverts + vid) * 4 + k, row-major in
an ATLAS_WIDTH-wide RGBA image (W <= 2048 = ES 3.0 guaranteed
MAX_TEXTURE_SIZE; exact float32 index math, max < 2**24).

Delta transforms mirror the base-position pipeline in
SkinnedObject._build_splits: node world matrix direction transform, fit
scale for positions only (normals are directions), Y-flip (1,-1,1)
exactly as the split vertex copy does.
"""
import es_gltf

# Simultaneous active shapes per draw ("slots", u_sw0..u_sw{N-1} +
# the same number of unrolled fetches) are decided by es_caps.

ATLAS_WIDTH = 2048
TEXELS = 4
_ALPHA = 255


def _png_rgba(width, height, rgba):
    """Minimal PNG encoder (RGBA8, no interlace) -- pygame_sdl2 has no
    fromstring/frombytes, and im.Data-style loaders only take encoded
    files, so the atlas ships to SDL_image as a PNG stream."""
    import zlib
    import struct

    def chunk(tag, data):
        return (struct.pack(">I", len(data)) + tag + data +
                struct.pack(">I", zlib.crc32(tag + data) & 0xffffffff))

    stride = width * 4
    rows = []
    for y in range(height):
        rows.append("\x00" + rgba[y * stride:(y + 1) * stride])
    return ("\x89PNG\r\n\x1a\n" +
            chunk("IHDR", struct.pack(">IIBBBBB",
                                      width, height, 8, 6, 0, 0, 0)) +
            chunk("IDAT", zlib.compress("".join(rows), 6)) +
            chunk("IEND", ""))


def make_displayable(width, height, rgba):
    """Wrap atlas bytes as a size-stable texture source.

    render() ignores the requested size and always returns ONE cached
    Render at native atlas size: Model.texture() then sees identical
    texels whatever size the Model asks for, and Ren'Py's
    Render->texture cache (what.cached_texture, gl2draw.pyx) uploads
    the atlas to the GPU exactly once. The PNG is decoded on first
    render, then dropped. Ren'Py imports stay in here so the packer
    above remains usable outside a running game.
    """
    import renpy.display.core
    import renpy.display.render

    class AtlasDisplayable(renpy.display.core.Displayable):

        def __init__(self, width, height, png, **kwargs):
            super(AtlasDisplayable, self).__init__(**kwargs)
            self._duplicatable = False
            self.atlas_width = width
            self.atlas_height = height
            self.png = png
            self._cached = None
            # False = upload not attempted yet, None = fallback,
            # otherwise the pre-uploaded full-res GLTexture.
            self._gt = False
            # drawable_size at gt creation; on_resize wipes every GL
            # texture number (texture_loader.init -> quit), so a size
            # change means this gt is now a dangling name.
            self._draw_size = None

        def _decode(self):
            import io
            import pygame
            self._surf = pygame.image.load(io.BytesIO(self.png),
                                           "atlas.png")

        def _upload(self):
            # Hand Ren'Py a pre-uploaded texture instead of a pixel
            # blit: the Render->texture RTT (gl2texture.pyx from_render)
            # allocates its FBO at virt_to_draw scale, so whenever the
            # window is smaller than the virtual screen (e.g. 944x531 vs
            # 1920x1080) the packed atlas gets downsampled, neighbouring
            # texels bleed into each other and the decoded deltas turn
            # to garbage -- the face implodes. cached_texture
            # short-circuits that RTT (gl2draw.pyx render_to_texture)
            # and keeps the atlas at native resolution for any drawable
            # size.
            if self._surf is None:
                self._decode()
            gt = None
            err = None
            draw = getattr(renpy.display, "draw", None)
            if draw is not None:
                try:
                    gt = draw.load_texture(self._surf, False, {})
                except Exception as e:
                    err = e
            print("ES_ATLAS draw=%r gt=%r err=%r" % (
                type(draw).__name__ if draw is not None else None,
                type(gt).__name__ if gt is not None else None, err))
            self._gt = gt
            if gt is None:
                self._cached.blit(self._surf, (0, 0))
            self._surf = None

        def render(self, w, h, st, at):
            if self._cached is None:
                self._decode()
                self._cached = renpy.display.render.Render(
                    self.atlas_width, self.atlas_height)

            # A window resize deletes all GL texture names but keeps
            # loaded flags on orphaned GLTexture objects (Ren'Py only
            # rebuilds what its im.cache/text/video own). Our gt is
            # cached outside that machinery, so drop it and upload a
            # fresh one whenever the drawable size changes.
            draw = getattr(renpy.display, "draw", None)
            dsize = getattr(draw, "drawable_size", None)
            if (dsize is not None and self._gt is not None
                    and self._draw_size is not None
                    and dsize != self._draw_size):
                print("ES_ATLAS resize %r -> %r, reupload" % (
                    self._draw_size, dsize))
                self._gt = False
                if self._cached.cached_texture is not None:
                    self._cached.cached_texture = None
            if dsize is not None:
                self._draw_size = dsize

            if self._gt is False:
                self._upload()

            if self._gt is not None:
                # kill() (hide/re-show, resize sweep) clears
                # Render.cached_texture; restore it on every render so
                # the corrupting RTT never comes back.
                self._cached.cached_texture = self._gt

            return self._cached

        def visit(self):
            return []

    return AtlasDisplayable(width, height, _png_rgba(width, height, rgba))


def pack16(v):
    """[-1,1] float -> (hi, lo) bytes of the 16-bit fixed-point value."""
    x = int(round((v + 1.0) * 0.5 * 65535.0))
    if x < 0:
        x = 0
    elif x > 65535:
        x = 65535
    return x >> 8, x & 255


def unpack16(hi, lo):
    """Inverse of pack16 -- for tests / debugging."""
    return ((hi << 8) | lo) / 65535.0 * 2.0 - 1.0


def build(gltf, node_matrix, targets, names, nverts, scale):
    """Pack one pool's morph targets. Returns a morph dict or None.

    `targets` is the primitive's list of {semantic: accessor index};
    `names` the mesh extras.targetNames (defines local row order);
    `scale` the fit factor applied to positions at pool build. None
    means this pool has no usable morphs (rendering unchanged).
    """
    if not targets or not names or nverts <= 0:
        return None
    if len(targets) != len(names):
        print("ES_MORPH_SKIP targets=%d names=%d (order mismatch)" %
              (len(targets), len(names)))
        return None

    m = node_matrix
    nshapes = len(names)

    # Decode every shape into buffer space, track per-pool maxima.
    pos_d = []
    nrm_d = []
    maxp = 0.0
    maxn = 0.0
    for s in range(nshapes):
        t = targets[s] or {}
        pacc = t.get("POSITION")
        nacc = t.get("NORMAL")
        pd = (gltf.read_accessor(pacc) if pacc is not None
              else [(0.0, 0.0, 0.0)] * nverts)
        nd = (gltf.read_accessor(nacc) if nacc is not None
              else [(0.0, 0.0, 0.0)] * nverts)
        if len(pd) != nverts or len(nd) != nverts:
            print("ES_MORPH_SKIP shape %d count %d/%d != %d" %
                  (s, len(pd), len(nd), nverts))
            return None
        pb = []
        for d in pd:
            w = es_gltf.mat4_transform_dir(m, d)
            b = (w[0] * scale, -w[1] * scale, w[2] * scale)
            pb.append(b)
            a = max(abs(b[0]), abs(b[1]), abs(b[2]))
            if a > maxp:
                maxp = a
        nb = []
        for d in nd:
            w = es_gltf.mat4_transform_dir(m, d)
            b = (w[0], -w[1], w[2])
            nb.append(b)
            a = max(abs(b[0]), abs(b[1]), abs(b[2]))
            if a > maxn:
                maxn = a
        pos_d.append(pb)
        nrm_d.append(nb)

    if maxp <= 0.0:
        maxp = 1.0
    if maxn <= 0.0:
        maxn = 1.0

    total = nshapes * nverts * TEXELS
    height = (total + ATLAS_WIDTH - 1) // ATLAS_WIDTH
    if height > ATLAS_WIDTH:
        print("ES_MORPH_SKIP atlas %dx%d exceeds %d" %
              (ATLAS_WIDTH, height, ATLAS_WIDTH))
        return None

    out = bytearray()
    invp = 1.0 / maxp
    invn = 1.0 / maxn
    for s in range(nshapes):
        pb = pos_d[s]
        nb = nrm_d[s]
        for g in range(nverts):
            px, py, pz = pb[g]
            nx, ny, nz = nb[g]
            dxh, dxl = pack16(px * invp)
            dyh, dyl = pack16(py * invp)
            dzh, dzl = pack16(pz * invp)
            nxh, nxl = pack16(nx * invn)
            nyh, nyl = pack16(ny * invn)
            nzh, nzl = pack16(nz * invn)
            out.extend((dxh, dxl, dyh, _ALPHA))
            out.extend((dyl, dzh, dzl, _ALPHA))
            out.extend((nxh, nxl, nyh, _ALPHA))
            out.extend((nyl, nzh, nzl, _ALPHA))

    # Pad to full image rows (alpha bytes; last block untouched: row
    # padding is beyond any (shape, vid) block, so reads never see it).
    padded = ATLAS_WIDTH * height * 4
    if len(out) < padded:
        out.extend(bytearray(b"\xff" * (padded - len(out))))

    return {"width": ATLAS_WIDTH,
            "height": height,
            "nverts": nverts,
            "nshapes": nshapes,
            "names": list(names),
            "row": dict((n, i) for i, n in enumerate(names)),
            "mscale": maxp,
            "nscale": maxn,
            "rgba": bytes(out)}


def attach(morph):
    """Create the texture source for a built atlas (needs a running game)."""
    morph["displayable"] = make_displayable(morph["width"],
                                             morph["height"],
                                             morph["rgba"])
    return morph
