# HARNESS — autonomous verification process

How the shape-key system is verified without touching your desktop.
One command = one approval = a complete test cycle.

## How the environment was built ("SDK extraction")

No SDK was downloaded — a shipped Ren'Py game already contains the whole
engine; the installed game doubles as the SDK:

| Piece | Role |
|---|---|
| `Everlasting Summer.py` | stock Ren'Py 7.4.11 bootstrap; `main()` → `renpy.bootstrap.bootstrap(renpy_base)` |
| `renpy/arguments.py` | first **positional CLI arg** = `basedir` ("base directory of the project to run") — any project dir can be booted |
| `lib/linux-x86_64/python` | bundled CPython 2.7 interpreter that runs the bootstrap |
| `lib/linux-x86_64/librenpython.so` | compiled engine incl. `renpy.gl2` (verified via `strings`: `initrenpy_gl2_gl2draw`, source `gen-static/renpy.gl2.gl2draw.c`) |
| `renpy/*.py`, `renpy/gl2/*.pyx` | readable engine sources — the `.pyx` are compiled into the `.so` (edits are no-ops), used for reasoning only |
| `game/*.rpyc` | shipped game scripts (RPYC2); sources were stripped |

Construction steps:

1. **Boot a scratch project dir with the game's engine** (exactly what
   the runner does):
   ```bash
   cd "<game root>"
   lib/linux-x86_64/python "Everlasting Summer.py" <scratch_project> \
       > <run_log> 2>&1 &
   ```
   Same GL2 renderer and `librenpython.so` as the real game — nothing
   installed, nothing written back to the game dir.

2. **Minimal skeleton**: the scratch project holds only
   `game/script.rpy`. The engine creates `cache/`, `saves/`, `log.txt`,
   `traceback.txt`, `game/script.rpyc` beside it. The bare project has
   no screens (`Layout.yesno_prompt` missing) → hence
   `config.quit_action = None`.

3. **Symlink the mod in, never copy it**:
   `<scratch>/game/mods/es-3dmodels → <real>/game/mods/es-3dmodels`.
   `renpy.script.scan_script_files()` walks the whole basedir
   (`renpy.loader.listdirfiles`), so the mod's real `es-3dmodels.rpy`
   parses and runs inside the harness (proven by `ES_NOD start`), and
   `renpy.loader.transfn("mods/es-3dmodels/...")` resolves the GLB/OBJ
   assets. Tests therefore always exercise the exact shipped files —
   no copies that can drift. Side effect: Ren'Py recompiles
   `es-3dmodels.rpyc` next to the real source (the same cache file the
   real game maintains); every other artifact stays in `/tmp`.

4. **Init ordering**: harness `init -100 python: mods = {}` creates the
   mod registry before the mod's `init:` (priority 0) writes
   `mods["es_3dmodels"] = ...`.

5. **Self-driving script**: the scratch script puts the mod dir on `sys.path`,
   imports `es_engine`, screenshots to an absolute `OUTDIR`, and
   sequences phases with `pause` + file-driven runner waits. The only
   input is the runner's single Return at the menu.

6. **Offline metrics use a different interpreter**: pixel checks run
   under system `python3` + PIL; the bundled CPython
   has no PIL and no `runpy`.

