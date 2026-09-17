"""First Light PHerc0211 -- run the published First Letters workflow on Modal.

One Modal App, one persistent Volume (/data), one container image built from
the Vesuvius team's own VC3D CI image (so vc_render_tifxyz etc. are present)
plus villa's spiral-fitting environment pinned to a commit.

Functions (each is one runbook step):
  probe          -- prove the image works: python, torch+CUDA, vc binaries
  fetch_dataset  -- tracks DBM + crossings + umbilicus + lasagna normals -> /data
  render_slices  -- axial slices with the umbilicus marked, for the CW/ACW read
  pack_pools     -- pack_resident_pools.py (required on current villa main)
  spiral_fit     -- fit_spiral.py over a z window, CW or ACW, N steps

Usage:
  modal run modal_app.py::probe
  modal run modal_app.py::fetch_dataset
  modal run modal_app.py::render_slices
  modal run modal_app.py::pack_pools
  modal run modal_app.py::spiral_fit --sense CW --steps 1500 --z-begin 10000 --z-end 11000
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Pins. Everything the writeup claims traces back to these.
# ---------------------------------------------------------------------------
VILLA_COMMIT = "2dcfaf6a08c3bc796fde726c4fa32050c8fc90e7"  # villa main, 2026-09-17
VC3D_IMAGE = "ghcr.io/scrollprize/villa/volume-cartographer:main"

SCROLL = "PHerc0211"
VOLUME_ID = "20250821151803"
VOXEL_SIZE_UM = 9.362
VOLUME_ZARR = (
    f"s3://vesuvius-challenge-open-data/{SCROLL}/volumes/"
    f"{VOLUME_ID}-9.362um-1.2m-113keV-masked.zarr"
)
VOLUME_HTTP = VOLUME_ZARR.replace("s3://", "https://").replace(
    "vesuvius-challenge-open-data/", "vesuvius-challenge-open-data.s3.amazonaws.com/"
)
UMBILICUS_S3 = (
    f"s3://vesuvius-challenge-open-data/{SCROLL}/representations/umbilicus/"
    f"{VOLUME_ID}-umbilicus-20260808112626.json"
)
LASAGNA_S3 = (
    f"s3://vesuvius-challenge-open-data/{SCROLL}/representations/predictions/lasagna/"
    f"{VOLUME_ID}-lasagna-20260419180421"
)
TRACKS_HTTP = f"https://dl.ash2txt.org/datasets/spiral_datasets/{SCROLL}/{VOLUME_ID}/tracks"
TRACKS_BASE = f"{SCROLL}_{VOLUME_ID}_surface_m7_L0_th0.2"
VCTRACKS_FILES = [
    "arclengths.f64", "coordinates.i32", "family_codes.i8", "header.bin",
    "metadata.json", "offsets.i64", "source_ids.u64", "tortuosities.f64", "z_bounds.i32",
]

DATA = Path("/data")
DS = DATA / "spiral_datasets" / SCROLL / VOLUME_ID   # fit_spiral --dataset root
VILLA = Path("/opt/villa")

app = modal.App("first-light-pherc0211")
vol = modal.Volume.from_name("vesuvius-pherc0211", create_if_missing=True)

# ---------------------------------------------------------------------------
# Image. The VC3D CI image is Ubuntu 26.04 with the full C++ toolchain and the
# vc_* apps installed in /usr/local/bin. We add Python 3.14 (villa requires it),
# uv, and villa's spiral-fitting venv.
# ---------------------------------------------------------------------------
image = (
    modal.Image.from_registry(VC3D_IMAGE, add_python="3.14")
    .apt_install("aria2", "rclone", "git", "curl")
    .run_commands(
        "curl -LsSf https://astral.sh/uv/install.sh | sh",
        f"git clone https://github.com/ScrollPrize/villa.git {VILLA} && cd {VILLA} && git checkout {VILLA_COMMIT}",
    )
    .env({"PATH": "/root/.local/bin:/usr/local/bin:/usr/bin:/bin", "UV_LINK_MODE": "copy"})
    .run_commands(
        # spiral-fitting venv: torch cu128 from the lockfile, plus the vc_spiral C++ ext.
        f"cd {VILLA}/spiral-fitting && uv python install 3.14 && uv sync --frozen",
        # lasagna's flattener is invoked with the spiral-fitting interpreter
        # (render_ink.py runs fit.py with sys.executable). Its fit-path deps
        # not already in the spiral lockfile:
        f"cd {VILLA}/spiral-fitting && uv pip install 'scikit-image>=0.21'",
    )
)

# Second layer: villa/vesuvius with the `models` extra, for ink_9um inference.
# (cucim-cu13 has Linux-only wheels, which is fine here.) Kept separate so a
# failure in this heavy install cannot block the spiral-fit steps.
ink_image = image.run_commands(
    f"cd {VILLA}/vesuvius && uv sync --frozen --extra models",
)

GPU = "A10"          # 24 GB, same class as the RTX 3090 the survey tool measured on
CHECKPOINT_REPO = "scrollprize/ink_9um"
CHECKPOINT_FILE = "hybrid_3d2d-seed42/step-075000.pth"
HOURS = 3600


def sh(cmd: str, **kw) -> None:
    print(f"\n$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True, **kw)


# ---------------------------------------------------------------------------
@app.function(image=image, volumes={str(DATA): vol}, timeout=20 * 60)
def probe() -> None:
    sh("python --version; which python")
    sh(f"cd {VILLA} && git rev-parse HEAD")
    sh(f"cd {VILLA}/spiral-fitting && uv run python -c \"import torch, zarr, vc_spiral; print('torch', torch.__version__, 'cuda build', torch.version.cuda, 'zarr', zarr.__version__)\"")
    for b in ["vc_render_tifxyz", "vc_tifxyz2obj", "vc_obj2tifxyz", "vc_obj_uv_lift", "vc_tifxyz_trim", "flatboi", "aws", "aria2c"]:
        subprocess.run(f"which {b} || echo 'MISSING: {b}'", shell=True)
    sh(f"df -h {DATA} && ls -la {DATA}")


@app.function(image=image, volumes={str(DATA): vol}, timeout=3 * HOURS, cpu=4, memory=8192)
def fetch_dataset() -> None:
    """Mirror the published inputs into the fit_spiral dataset layout."""
    t0 = time.time()
    tracks = DS / "tracks"
    (tracks / f"{TRACKS_BASE}.dbm.vctracks").mkdir(parents=True, exist_ok=True)
    (DS / "lasagna").mkdir(exist_ok=True)

    def aria(url: str, dest: Path) -> None:
        if dest.exists():
            print(f"exists, skipping: {dest}")
            return
        sh(f"aria2c -x 8 -s 8 --file-allocation=none --console-log-level=warn --summary-interval=30 "
           f"-d {dest.parent} -o {dest.name} '{url}'")

    # 1. tracks: DBM, crossings sidecar, extract provenance, hidden .vctracks companion dir
    aria(f"{TRACKS_HTTP}/{TRACKS_BASE}.dbm", tracks / f"{TRACKS_BASE}.dbm")
    aria(f"{TRACKS_HTTP}/{TRACKS_BASE}.dbm.crossings.npz", tracks / f"{TRACKS_BASE}.dbm.crossings.npz")
    aria(f"{TRACKS_HTTP}/{TRACKS_BASE}.extract.json", tracks / f"{TRACKS_BASE}.extract.json")
    for f in VCTRACKS_FILES:
        aria(f"{TRACKS_HTTP}/{TRACKS_BASE}.dbm.vctracks/{f}", tracks / f"{TRACKS_BASE}.dbm.vctracks" / f)

    # 2. umbilicus (published 2026-09-03 by the team)
    sh(f"aws s3 cp --no-sign-request {UMBILICUS_S3} {DS}/umbilicus.json")

    # 3. lasagna normals + grad_mag, pyramid group 2 only (the group the spec names)
    for store in ["nx", "ny", "grad_mag"]:
        dest = DS / "lasagna" / f"{SCROLL}_{store}.ome.zarr"
        sh(f"aws s3 cp --no-sign-request --only-show-errors {LASAGNA_S3}/{SCROLL}_{store}.ome.zarr/.zattrs {dest}/.zattrs")
        sh(f"aws s3 cp --no-sign-request --only-show-errors {LASAGNA_S3}/{SCROLL}_{store}.ome.zarr/.zgroup {dest}/.zgroup")
        sh(f"aws s3 cp --no-sign-request --only-show-errors --recursive {LASAGNA_S3}/{SCROLL}_{store}.ome.zarr/2 {dest}/2")

    # 4. repair the DBM mtime so the published crossings cache is accepted
    #    (fingerprint is name/size/st_mtime_ns; HTTP Last-Modified loses the ns).
    print(repair_crossings_mtime(str(tracks / f"{TRACKS_BASE}.dbm"),
                                 str(tracks / f"{TRACKS_BASE}.dbm.crossings.npz")))

    vol.commit()
    sh(f"du -sh {DS}/* {DS}/lasagna/* && find {DS} -maxdepth 2 | sort")
    print(f"fetch_dataset done in {(time.time()-t0)/60:.1f} min")


def repair_crossings_mtime(dbm_path: str, npz_path: str) -> str:
    """After ttendoscopie-creator/eligible-spiral-dataset (MIT): restore the
    producer's st_mtime_ns, which is recorded inside the sidecar."""
    import re, zipfile
    with zipfile.ZipFile(npz_path) as zf:
        name = next(n for n in zf.namelist() if n.startswith("metadata"))
        raw = zf.read(name)
    text = ""
    for enc in ("utf-32-le", "utf-8", "latin-1"):
        cand = raw.decode(enc, errors="ignore")
        if '"db_signature"' in cand:
            text = cand
            break
    m = re.search(r'"db_signature"\s*:\s*(\[\s*\[.*?\]\s*\])', text, re.S)
    if not m:
        return "sidecar carries no db_signature; left alone"
    sig = json.loads(m.group(1))
    entry = next((e for e in sig if e[0] == os.path.basename(dbm_path)), None)
    if entry is None:
        return "sidecar names a different DBM; left alone"
    _, want_size, want_ns = entry
    st = os.stat(dbm_path)
    if st.st_size != want_size:
        return f"size mismatch ({st.st_size} vs {want_size}); cache left alone"
    if st.st_mtime_ns == want_ns:
        return "mtime already matches"
    os.utime(dbm_path, ns=(st.st_atime_ns, want_ns))
    return f"restored st_mtime_ns {st.st_mtime_ns} -> {want_ns}"


def write_scroll_spec(sense: str) -> Path:
    """spiral-scroll.json per the 18 Aug workflow post template, with explicit
    paths (our files do not use the conventional names) and the two keys that
    are easy to drop: normal_zarr_group / lasagna_scale (group 2 == 4x here,
    read from the store's own .zattrs)."""
    spec = {
        "schema_version": 1,
        "name": SCROLL,
        "voxel_size_um": VOXEL_SIZE_UM,
        "spiral_outward_sense": sense,
        "normal_zarr_group": "2",
        "lasagna_scale": 4,
        "paths": {
            "tracks_dbm": f"tracks/{TRACKS_BASE}.dbm",
            "normal_x": f"lasagna/{SCROLL}_nx.ome.zarr",
            "normal_y": f"lasagna/{SCROLL}_ny.ome.zarr",
            "gradient_magnitude": f"lasagna/{SCROLL}_grad_mag.ome.zarr",
        },
    }
    p = DS / "spiral-scroll.json"
    p.write_text(json.dumps(spec, indent=2) + "\n")
    print(p.read_text())
    return p


@app.function(image=image, volumes={str(DATA): vol}, timeout=HOURS, cpu=4, memory=16384)
def render_slices(zs: str = "5000,9000,13000", level: int = 2, crop: int = 0) -> list[str]:
    """Axial slices at pyramid level 2 with the umbilicus marked, for the human
    CW/ACW read. Reads the CT volume straight from S3; nothing is downloaded."""
    script = VILLA / "spiral-fitting" / "_render_slices_pherc0211.py"
    script.write_text(RENDER_SCRIPT)
    out = DATA / "renders"
    out.mkdir(exist_ok=True)
    DS.mkdir(parents=True, exist_ok=True)
    if not (DS / "umbilicus.json").exists():
        sh(f"aws s3 cp --no-sign-request {UMBILICUS_S3} {DS}/umbilicus.json")
    sh(f"cd {VILLA}/spiral-fitting && uv run python {script} {DS}/umbilicus.json {out} {zs} {level} {crop}")
    vol.commit()
    return sorted(str(p) for p in out.glob("*.png"))


RENDER_SCRIPT = r'''
import json, sys, warnings
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, s3fs, zarr
warnings.filterwarnings("ignore")
VOLUME = "vesuvius-challenge-open-data/PHerc0211/volumes/20250821151803-9.362um-1.2m-113keV-masked.zarr"
umb_path, out_dir, zs = sys.argv[1], sys.argv[2], [int(z) for z in sys.argv[3].split(",")]
LEVEL = sys.argv[4] if len(sys.argv) > 4 else "2"
CROP = int(sys.argv[5]) if len(sys.argv) > 5 else 0   # half-width in level pixels around the umbilicus; 0 = full slice
SCALE = 2 ** int(LEVEL)
pts = sorted(json.load(open(umb_path))["control_points"], key=lambda p: p["z"])
def umb(z):
    if z <= pts[0]["z"]: return pts[0]["x"], pts[0]["y"]
    if z >= pts[-1]["z"]: return pts[-1]["x"], pts[-1]["y"]
    for a, b in zip(pts, pts[1:]):
        if a["z"] <= z <= b["z"]:
            t = (z - a["z"]) / max(b["z"] - a["z"], 1)
            return a["x"] + t*(b["x"]-a["x"]), a["y"] + t*(b["y"]-a["y"])
fs = s3fs.S3FileSystem(anon=True)
arr = zarr.open_group(zarr.storage.FsspecStore(fs, path=VOLUME), mode="r")[LEVEL]
print("level", LEVEL, arr.shape, arr.dtype)
for z in zs:
    zi = min(arr.shape[0]-1, round(z/SCALE))
    ux, uy = umb(z); cx, cy = ux/SCALE, uy/SCALE
    if CROP:
        y0, y1 = max(0, int(cy-CROP)), min(arr.shape[1], int(cy+CROP))
        x0, x1 = max(0, int(cx-CROP)), min(arr.shape[2], int(cx+CROP))
        slab = np.asarray(arr[zi, y0:y1, x0:x1]); extent = (x0, x1, y1, y0)
    else:
        slab = np.asarray(arr[zi]); extent = None
    lo, hi = np.percentile(slab[slab > 0], [1, 99]) if (slab > 0).any() else (0, 1)
    disp = np.clip((slab.astype(np.float32)-lo)/max(hi-lo, 1), 0, 1)
    fig, ax = plt.subplots(figsize=(10, 10.5), dpi=150)
    ax.imshow(disp, cmap="gray", origin="upper", extent=extent)
    ax.plot(cx, cy, "r+", ms=24, mew=2.5)
    ax.set_title(f"PHerc0211  z_full={z}  (level {LEVEL}, {SCALE}x down, z_idx={zi}{', crop' if CROP else ''})  red + = published umbilicus")
    ax.set_xlabel("x  (array column, increases to the right)"); ax.set_ylabel("y  (array row, increases downward)")
    fig.text(0.02, 0.01, "Viewed looking along +z with +x right, +y down (native array orientation, no flips). "
             "Read the winding sense from the umbilicus outward.", fontsize=8)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    p = f"{out_dir}/PHerc0211_z{z}_level{LEVEL}{'_crop'+str(CROP) if CROP else ''}.png"; fig.savefig(p); plt.close(fig); print("saved", p)
'''


@app.function(image=image, volumes={str(DATA): vol}, timeout=3 * HOURS, cpu=8, memory=32768)
def pack_pools() -> None:
    """Current villa main loads lasagna stores only through resident-pool
    sidecars (lasagna_data.py: 'there is no other loading path'). The 18 Aug
    workflow post predates this and does not mention it."""
    t0 = time.time()
    sh(f"cd {VILLA}/spiral-fitting && uv run python pack_resident_pools.py {DS}/lasagna "
       f"--what normals,grad_mag --normal-group 2 --io-threads 16 --verify 500")
    vol.commit()
    sh(f"du -sh {DS}/lasagna/*")
    print(f"pack_pools done in {(time.time()-t0)/60:.1f} min")


@app.function(image=image, gpu=GPU, volumes={str(DATA): vol}, timeout=6 * HOURS, cpu=8, memory=49152)
def spiral_fit(sense: str = "CW", steps: int = 1500, z_begin: int = 10000, z_end: int = 11000,
               run_tag: str = "") -> str:
    """Headless fit_spiral.py. Overrides follow the 18 Aug workflow post plus
    input_use_tracks=true (villa#1651 flipped the default to false on 30 Aug,
    which the post does not reflect; see villa#1716)."""
    t0 = time.time()
    sense = sense.upper()
    assert sense in ("CW", "ACW")
    write_scroll_spec(sense)
    tag = run_tag or f"{sense}_z{z_begin}-{z_end}_s{steps}"
    out_dir = DATA / "out" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    overrides = {
        "z_begin": z_begin,
        "z_end": z_end,
        "optimizer_num_training_steps": steps,
        "input_use_tracks": True,
        "input_disable_patches": True,
        "input_use_outer_shell": False,
        "loss_weight_shell_outer": 0,
        "loss_weight_shell_patch_radius": 0,
        "dense_spacing_mode": "grad_mag",
        "loss_weight_dense_spacing": 0,
    }
    env = dict(os.environ,
               FIT_SPIRAL_CONFIG_OVERRIDES=json.dumps(overrides),
               FIT_SPIRAL_OUT_DIR=str(out_dir),
               FIT_SPIRAL_CACHE_DIR=str(DATA / "cache" / tag),  # per-run: CW and ACW fits run concurrently
               WANDB_MODE="disabled")
    (out_dir / "overrides.json").write_text(json.dumps(overrides, indent=2))
    sh("nvidia-smi --query-gpu=name,memory.total --format=csv")
    log = out_dir / "fit.log"
    with open(log, "w") as lf:
        proc = subprocess.Popen(
            f"cd {VILLA}/spiral-fitting && uv run python fit_spiral.py --dataset {DS} 2>&1",
            shell=True, env=env, stdout=subprocess.PIPE, text=True)
        for line in proc.stdout:
            print(line, end="", flush=True)
            lf.write(line)
        rc = proc.wait()
    vol.commit()
    mins = (time.time() - t0) / 60
    (out_dir / "timing.json").write_text(json.dumps({"exit_code": rc, "wall_minutes": mins, "gpu": GPU}, indent=2))
    sh(f"find {out_dir} -maxdepth 3 | head -60")
    print(f"spiral_fit {tag}: exit {rc}, {mins:.1f} min on {GPU}")
    return tag


# ---------------------------------------------------------------------------
@app.function(image=ink_image, volumes={str(DATA): vol}, timeout=20 * 60)
def probe_ink() -> None:
    sh(f"cd {VILLA}/vesuvius && uv run --extra models python -c \"import torch, vesuvius.ink_detection.inference.infer as m; print('torch', torch.__version__, 'cuda build', torch.version.cuda); print('infer module', m.__file__)\"")
    sh("vc_render_tifxyz --help 2>&1 | head -60 || true")
    sh(f"cd {VILLA}/spiral-fitting && uv run python render_ink.py --help | head -40")


@app.function(image=ink_image, gpu=GPU, volumes={str(DATA): vol}, timeout=8 * HOURS, cpu=16, memory=65536)
def render_and_infer(run_tag: str, winding_min: int = -1, winding_max: int = -1,
                     cache_gb: int = 16) -> str:
    """Steps 4-6 of the Aug runbook in one GPU container:
    render_ink.py (full-scroll concat + lasagna flatten, GPU) ->
    vc_render_tifxyz (28-slice surface volume from S3, CPU) ->
    vesuvius.ink_detection.inference.infer (ink_9um, GPU, both directions)."""
    t0 = time.time()
    fit_root = DATA / "out" / run_tag
    runs = sorted(p for p in fit_root.iterdir() if p.is_dir() and (p / "meshes" / "fitted").is_dir())
    assert runs, f"no fit run with meshes/fitted under {fit_root}"
    run_dir = runs[-1]
    meshes = run_dir / "meshes" / "fitted"
    if winding_min >= 0:
        scoped = run_dir / "meshes" / f"fitted_scoped_w{winding_min:03d}-{winding_max:03d}"
        scoped.mkdir(exist_ok=True)
        n = 0
        for d in sorted(meshes.glob("w*_spliced")):
            widx = int(d.name[1:].split("_")[0])
            if winding_min <= widx <= winding_max:
                link = scoped / d.name
                if not link.exists():
                    link.symlink_to(d)
                n += 1
        assert n, "no windings in range"
        print(f"scoped {n} windings into {scoped}")
        meshes = scoped
    out = DATA / "ink" / run_tag / meshes.name
    out.mkdir(parents=True, exist_ok=True)
    cache = DATA / "volume-cache" / f"{SCROLL}.zarr"
    cache.mkdir(parents=True, exist_ok=True)

    # 4. concat + lasagna flatten (+ preview render) -- writes <meshes>/concat/*_flat
    sh(f"cd {VILLA}/spiral-fitting && uv run python render_ink.py {meshes} "
       f"--volume {cache} --remote-url {VOLUME_ZARR} --group-idx 1 --scale 0.25 --num-slices 5 "
       f"--no-strips --full-scroll --lasagna-device cuda 2>&1 | tee {out}/render_ink.log")
    flats = sorted((meshes / "concat").glob("*_flat"))
    assert flats, "render_ink.py produced no *_flat tifxyz"
    flat = flats[-1]
    print("flattened tifxyz:", flat)

    # 5. full-resolution 28-slice surface volume for the 9um ink model
    seg_zarr = out / "segment.zarr"
    sh(f"vc_render_tifxyz --volume {cache} --remote-url {VOLUME_ZARR} --group-idx 0 --scale 1 "
       f"--segmentation {flat} --num-slices 28 --slice-step 1 --cache-gb {cache_gb} "
       f"--zarr-output {seg_zarr} 2>&1 | tee {out}/vc_render_tifxyz.log")

    # 6. ink inference, both directions
    ckpt = DATA / "checkpoints" / "ink_9um" / CHECKPOINT_FILE
    if not ckpt.exists():
        sh(f"cd {VILLA}/vesuvius && uvx --from huggingface_hub hf download {CHECKPOINT_REPO} {CHECKPOINT_FILE} "
           f"--local-dir {DATA}/checkpoints/ink_9um")
    sh(f"cd {VILLA}/vesuvius && uv run --extra models python -m vesuvius.ink_detection.inference.infer "
       f"{seg_zarr} {ckpt} {out}/segment.tif --overlap 0.5 --blend-mode hann --batch-size 4 --direction both "
       f"2>&1 | tee {out}/infer.log")
    vol.commit()
    mins = (time.time() - t0) / 60
    (out / "timing.json").write_text(json.dumps({"wall_minutes": mins, "gpu": GPU, "fit_run": str(run_dir)}, indent=2))
    sh(f"ls -la {out}")
    print(f"render_and_infer done in {mins:.1f} min")
    return str(out)
