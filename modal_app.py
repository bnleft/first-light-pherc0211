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

# Third layer: the vc_* apps rebuilt from the pinned villa commit. The published
# ghcr `:main` image turned out to lag main (its vc_render_tifxyz has no
# --flip-normals, which main's source has), so for the writeup's "pinned versions"
# claim the binaries have to come from the same commit as the Python code.
vc_image = ink_image.run_commands(
    # The May image is Ubuntu 24.04; main's install_build_deps.sh now targets 26.04
    # (flang-21) and cannot run here. Install only what the apps build needs beyond
    # what the image already has.
    "apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq liblapacke-dev libopenblas-dev",
    # flatboi drags in PaStiX and is only used by render_ink --strips, which we do not use;
    # the Qt tracer GUI is not needed headless either.
    f"cd {VILLA}/volume-cartographer && cmake --preset ci-release-gcc -DVC_BUILD_FLATBOI=OFF -DVC_BUILD_UI_TRACER=OFF "
    f"&& cmake --build --preset ci-release-gcc -j 16 "
    f"&& cmake --install build/ci-release-gcc --prefix /usr/local --component vc_runtime "
    f"&& rm -rf build && vc_render_tifxyz --help | grep -q flip-normals",
    gpu=None,
)

GPU = "A10"          # 24 GB, same class as the RTX 3090 the survey tool measured on
CHECKPOINT_REPO = "scrollprize/ink_9um"
CHECKPOINT_FILE = "hybrid_3d2d-seed42/step-075000.pth"
HOURS = 3600


def sh(cmd: str, **kw) -> None:
    """Run under bash with pipefail so `... | tee log` cannot mask a failure
    (the first control run did exactly that: vc_render_tifxyz failed, tee
    returned 0, and inference ran against a store that did not exist)."""
    print(f"\n$ {cmd}", flush=True)
    subprocess.run(["bash", "-o", "pipefail", "-c", cmd], check=True, **kw)


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


@app.function(image=vc_image, gpu=GPU, volumes={str(DATA): vol}, timeout=8 * HOURS, cpu=16, memory=65536)
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


# ---------------------------------------------------------------------------
# Positive control: PHerc0139 w035, a segment in the ink_9um training set with
# published labels. The Aug team learned the HF `ink/0139/w035_2026031718` mesh is
# registered to a 2.4 um volume and renders all-zero against the 9.362 um volume;
# the open-data bucket publishes per-volume registrations, so we take the
# `-on-20250728140407-9.362um` mesh directly and also the team's own pre-rendered
# 9.362 um surface volume (zero geometry code of ours) as a second control path.
CONTROL_SEG_PREFIX = "s3://vesuvius-challenge-open-data/PHerc0139/segments/20260317000000-w035_2026031718"
CONTROL_MESH_S3 = f"{CONTROL_SEG_PREFIX}/mesh/20260317000000-on-20250728140407-9.362um.tifxyz"
CONTROL_SURFVOL_S3 = f"{CONTROL_SEG_PREFIX}/surface-volumes/9.362um-1.2m-113keV-volume-20250728140407.zarr"
CONTROL_VOLUME_ZARR = "s3://vesuvius-challenge-open-data/PHerc0139/volumes/20250728140407-9.362um-1.2m-113keV-masked.zarr"
CONTROL_LABELS_HF = "ink_9um/labels/native9-scrollprizeorg-21slices/w035"


@app.function(image=vc_image, gpu=GPU, volumes={str(DATA): vol}, timeout=4 * HOURS, cpu=16, memory=65536)
def control(flip_normals: bool = True, cache_gb: int = 16, use_published_render: bool = False) -> str:
    """Run the identical inference on the control. Path A (default): our own
    vc_render_tifxyz render of the published 9.362um-registered mesh. Path B
    (--use-published-render): the team's pre-rendered surface volume."""
    t0 = time.time()
    root = DATA / "control"
    out = root / ("out_published_render" if use_published_render else "out_own_render")
    out.mkdir(parents=True, exist_ok=True)
    ckpt = DATA / "checkpoints" / "ink_9um" / CHECKPOINT_FILE
    if not ckpt.exists():
        sh(f"cd {VILLA}/vesuvius && uvx --from huggingface_hub hf download {CHECKPOINT_REPO} {CHECKPOINT_FILE} "
           f"--local-dir {DATA}/checkpoints/ink_9um")
    if use_published_render:
        seg_zarr = root / "published_surface_volume.zarr"
        if not (seg_zarr / ".zgroup").exists() and not (seg_zarr / ".zarray").exists():
            sh(f"aws s3 cp --no-sign-request --only-show-errors --recursive {CONTROL_SURFVOL_S3} {seg_zarr}")
    else:
        mesh = root / "mesh_on_9362um.tifxyz"
        if not (mesh / "x.tif").exists():
            sh(f"aws s3 cp --no-sign-request --only-show-errors --recursive {CONTROL_MESH_S3} {mesh}")
        sh(f"ls -la {mesh}; cat {mesh}/meta.json")
        cache = DATA / "volume-cache" / "control-PHerc0139-9362.zarr"
        cache.mkdir(parents=True, exist_ok=True)
        seg_zarr = out / "control.zarr"
        flip = "--flip-normals" if flip_normals else ""
        sh(f"vc_render_tifxyz --volume {cache} --remote-url {CONTROL_VOLUME_ZARR} --group-idx 0 --scale 1 "
           f"--segmentation {mesh} --num-slices 28 --slice-step 1 --cache-gb {cache_gb} {flip} "
           f"--zarr-output {seg_zarr} 2>&1 | tee {out}/vc_render_tifxyz.log")
    sh(f"cd {VILLA}/vesuvius && uv run --extra models python -m vesuvius.ink_detection.inference.infer "
       f"{seg_zarr} {ckpt} {out}/control_prediction.tif --overlap 0.5 --blend-mode hann --batch-size 4 --direction both "
       f"2>&1 | tee {out}/infer.log")
    vol.commit()
    (out / "timing.json").write_text(json.dumps({"wall_minutes": (time.time()-t0)/60, "gpu": GPU,
                                                 "flip_normals": flip_normals, "published_render": use_published_render}, indent=2))
    sh(f"ls -la {out}")
    return str(out)


@app.function(image=vc_image, volumes={str(DATA): vol}, timeout=20 * 60)
def probe_vc() -> None:
    sh("vc_render_tifxyz --help | grep -n flip-normals")
    sh("ls -la /usr/local/bin | grep -E 'vc_|flatboi'")
    sh("cat /src/.git/HEAD 2>/dev/null; git -C /src rev-parse HEAD 2>/dev/null || echo 'no /src git'; ls /src 2>/dev/null | head")


# ---------------------------------------------------------------------------
LASAGNA_STORES = ["nx", "ny", "grad_mag"]


@app.function(image=image, volumes={str(DATA): vol}, timeout=3 * HOURS, cpu=16, memory=32768)
def pack_pools_fast(z_begin: int = 10000, z_end: int = 11000, margin: int = 1500) -> None:
    """Replace fetch_dataset's lasagna step + pack_pools.

    Copying the full group-2 stores (~100k tiny chunk files each) into a Modal
    Volume ran at ~2 chunk-rows/min, i.e. hours per store. The fitter only reads
    the sidecars pack_resident_pools.py produces (a handful of large files), and
    the packer only iterates over chunk files that exist. So: pull just the
    chunk rows covering [z_begin - margin, z_end + margin) to local disk, pack
    there, and write the sidecars plus the zarr metadata files to the volume.
    Rows outside the window are absent bricks, which read back as no-data."""
    import shutil
    t0 = time.time()
    scale, chunk = 4, 32
    row_lo = max(0, (z_begin - margin) // scale // chunk)
    row_hi = (z_end + margin) // scale // chunk + 1
    print(f"lasagna group 2 chunk rows {row_lo}..{row_hi} (full-res z {row_lo*scale*chunk}..{row_hi*scale*chunk})")
    local = Path("/tmp/lasagna")
    local.mkdir(parents=True, exist_ok=True)
    for st in LASAGNA_STORES:
        src = f"{LASAGNA_S3}/{SCROLL}_{st}.ome.zarr"
        dst = local / f"{SCROLL}_{st}.ome.zarr"
        (dst / "2").mkdir(parents=True, exist_ok=True)
        for f in [".zattrs", ".zgroup", "2/.zarray"]:
            sh(f"aws s3 cp --no-sign-request --only-show-errors {src}/{f} {dst}/{f}")
        rows = " ".join(str(r) for r in range(row_lo, row_hi))
        # 16 parallel row copies; each row is ~700 files / ~17 MB
        sh(f"cd {dst}/2 && echo {rows} | tr ' ' '\\n' | xargs -P 16 -I{{}} aws s3 cp --no-sign-request --only-show-errors --recursive {src}/2/{{}} {dst}/2/{{}}")
        n = sum(1 for _ in (dst / "2").rglob("*") if _.is_file())
        print(f"{st}: {n} chunk files on local disk")
    sh(f"cd {VILLA}/spiral-fitting && uv run python pack_resident_pools.py {local} "
       f"--what normals,grad_mag --normal-group 2 --io-threads 16 --verify 500")
    # publish: zarr metadata (so the fitter's directory checks pass) + sidecars
    vol_lasagna = DS / "lasagna"
    vol_lasagna.mkdir(parents=True, exist_ok=True)
    for st in LASAGNA_STORES:
        dst = vol_lasagna / f"{SCROLL}_{st}.ome.zarr"
        (dst / "2").mkdir(parents=True, exist_ok=True)
        for f in [".zattrs", ".zgroup", "2/.zarray"]:
            shutil.copy2(local / f"{SCROLL}_{st}.ome.zarr" / f, dst / f)
    for side in sorted(local.glob("*.respool_g2*")):
        target = vol_lasagna / side.name
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(side, target)
        print("published sidecar", target)
    # the DBM mtime repair that fetch_dataset never reached
    tracks = DS / "tracks"
    print(repair_crossings_mtime(str(tracks / f"{TRACKS_BASE}.dbm"), str(tracks / f"{TRACKS_BASE}.dbm.crossings.npz")))
    vol.commit()
    sh(f"du -sh {vol_lasagna}/* && cat {vol_lasagna}/{SCROLL}_nx.ome.zarr.respool_g2_pair/meta.json | head -c 600")
    print(f"pack_pools_fast done in {(time.time()-t0)/60:.1f} min")


# ---------------------------------------------------------------------------
OVERLAY_SCRIPT = r'''
import glob, json, os, re, sys, warnings
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, s3fs, zarr
from PIL import Image
warnings.filterwarnings("ignore")
VOLUME = "vesuvius-challenge-open-data/PHerc0211/volumes/20250821151803-9.362um-1.2m-113keV-masked.zarr"
runs = json.loads(sys.argv[1]); umb_path, out_dir = sys.argv[2], sys.argv[3]
zs = [int(z) for z in sys.argv[4].split(",")]; crop = int(sys.argv[5]); tol = float(sys.argv[6]); msz = float(sys.argv[7]) if len(sys.argv) > 7 else 6.0
LEVEL, SCALE = "1", 2
pts = sorted(json.load(open(umb_path))["control_points"], key=lambda p: p["z"])
def umb(z):
    for a, b in zip(pts, pts[1:]):
        if a["z"] <= z <= b["z"]:
            t = (z - a["z"]) / max(b["z"] - a["z"], 1)
            return a["x"] + t*(b["x"]-a["x"]), a["y"] + t*(b["y"]-a["y"])
    p = pts[0] if z < pts[0]["z"] else pts[-1]; return p["x"], p["y"]
def load_meshes(meshes_dir):
    out = []
    for d in sorted(glob.glob(os.path.join(meshes_dir, "w*_spliced"))):
        w = int(re.match(r"w(\d+)", os.path.basename(d)).group(1))
        x = np.array(Image.open(f"{d}/x.tif")); y = np.array(Image.open(f"{d}/y.tif")); z = np.array(Image.open(f"{d}/z.tif"))
        ok = (x != -1) & (y != -1) & (z != -1)
        out.append((w, x[ok].astype(np.float32), y[ok].astype(np.float32), z[ok].astype(np.float32)))
    return out
fs = s3fs.S3FileSystem(anon=True)
arr = zarr.open_group(zarr.storage.FsspecStore(fs, path=VOLUME), mode="r")[LEVEL]
meshes = {name: load_meshes(path) for name, path in runs.items()}
for name, ms in meshes.items():
    print(name, len(ms), "windings;", sum(len(m[1]) for m in ms), "valid vertices")
# quantitative sense check: mean CT intensity under the fitted vertices (full window, level 1)
for z in zs:
    zi = round(z / SCALE)
    full = np.asarray(arr[zi]).astype(np.float32)
    for name, ms in meshes.items():
        vals = []
        for w, x, y, zz in ms:
            sel = np.abs(zz - z) < tol
            if sel.any():
                xi = np.clip((x[sel]/SCALE).round().astype(int), 0, full.shape[1]-1); yi = np.clip((y[sel]/SCALE).round().astype(int), 0, full.shape[0]-1)
                vals.append(full[yi, xi])
        v = np.concatenate(vals) if vals else np.zeros(1)
        bg = full[full > 0]
        print(f"CTSCORE z={z} {name}: n={v.size} mean_ct={v.mean():.2f} median_ct={np.median(v):.1f} frac_bright(>{np.percentile(bg,75):.0f})={np.mean(v > np.percentile(bg,75)):.3f}  [scroll-voxel mean {bg.mean():.1f}]")
for z in zs:
    zi = round(z / SCALE); ux, uy = umb(z); cx, cy = ux/SCALE, uy/SCALE
    y0, y1 = max(0, int(cy-crop)), min(arr.shape[1], int(cy+crop)); x0, x1 = max(0, int(cx-crop)), min(arr.shape[2], int(cx+crop))
    slab = np.asarray(arr[zi, y0:y1, x0:x1]); lo, hi = np.percentile(slab[slab>0], [1, 99])
    disp = np.clip((slab.astype(np.float32)-lo)/max(hi-lo,1), 0, 1)
    fig, axes = plt.subplots(1, len(runs), figsize=(9*len(runs), 9.6), dpi=130)
    for ax, (name, ms) in zip(np.atleast_1d(axes), meshes.items()):
        ax.imshow(disp, cmap="gray", origin="upper", extent=(x0, x1, y1, y0))
        n = 0
        for w, x, y, zz in ms:
            sel = np.abs(zz - z) < tol
            if sel.any():
                # draw each winding's z-slab as a polyline ordered by angle about the umbilicus
                px, py = x[sel]/SCALE, y[sel]/SCALE
                ang = np.arctan2(py - cy, px - cx); order = np.argsort(ang)
                px, py, ang = px[order], py[order], ang[order]
                # break the line where consecutive points are far apart (gaps / wrap)
                d = np.hypot(np.diff(px), np.diff(py)); brk = np.where(d > 25)[0] + 1
                for seg_x, seg_y in zip(np.split(px, brk), np.split(py, brk)):
                    if len(seg_x) > 2:
                        ax.plot(seg_x, seg_y, "-", lw=1.0, color=plt.cm.turbo((w-10)/120), alpha=0.95)
                n += int(sel.sum())
        ax.plot(cx, cy, "r+", ms=20, mew=2)
        ax.set_xlim(x0, x1); ax.set_ylim(y1, y0)
        ax.set_title(f"{name}: fitted sheets within |dz|<{tol:g} vx of z={z}  ({n} vertices)  colour = winding index", fontsize=10)
    fig.text(0.01, 0.005, "Level 1 (2x). Red + = published umbilicus. A correct sense follows the bright papyrus sheets; a wrong sense cuts across them.", fontsize=9)
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    p = f"{out_dir}/overlay_z{z}_" + "_vs_".join(runs) + ".png"; fig.savefig(p); plt.close(fig); print("saved", p)
'''


@app.function(image=image, volumes={str(DATA): vol}, timeout=HOURS, cpu=4, memory=16384)
def overlay_fits(run_tags: str, zs: str = "10250,10500,10750", crop: int = 300, tol: float = 2.0, msz: float = 6.0) -> list[str]:
    """Overlay fitted meshes from one or more runs on CT slices, side by side.
    This is how the winding sense is actually decided (villa#1621: the
    satisfaction metric is periodic in the winding and cannot tell CW from ACW)."""
    runs = {}
    for tag in run_tags.split(","):
        fit_root = DATA / "out" / tag
        run_dir = sorted(p for p in fit_root.iterdir() if (p / "meshes" / "fitted").is_dir())[-1]
        runs[tag.split("_")[0]] = str(run_dir / "meshes" / "fitted")
    script = VILLA / "spiral-fitting" / "_overlay_fits.py"
    script.write_text(OVERLAY_SCRIPT)
    out = DATA / "renders"
    sh(f"cd {VILLA}/spiral-fitting && uv run python {script} '{json.dumps(runs)}' {DS}/umbilicus.json {out} {zs} {crop} {tol} {msz}")
    vol.commit()
    return sorted(str(p) for p in out.glob("overlay_*.png"))


# ---------------------------------------------------------------------------
CALIB_SCRIPT = r'''
import json, sys, numpy as np, zarr
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
labels_dir, pred_fwd, pred_rev, out_json = sys.argv[1:5]
lab = zarr.open_group(labels_dir, mode="r")["0"]           # (28, 5820, 5240) uint8
print("labels", lab.shape, lab.dtype)
L = np.asarray(lab[:]).max(axis=0) > 0                      # any-depth ink label
print("label pixels", int(L.sum()), "of", L.size, f"({L.mean()*100:.3f}%)")
res = {"label_pixels": int(L.sum())}
for name, path in [("forward", pred_fwd), ("reverse", pred_rev)]:
    P = np.asarray(Image.open(path)).astype(np.float32)  # PIL decodes the LZW TIFFs; tifffile needs imagecodecs
    assert P.shape == L.shape, (P.shape, L.shape)
    on, off = P[L], P[~L & (P > 0)]
    res[name] = {"median_on_label": float(np.median(on)), "mean_on_label": float(on.mean()),
                 "p90_off_label": float(np.percentile(off, 90)), "p99_off_label": float(np.percentile(off, 99)),
                 "frac_off_label_above_median_on": float((off >= np.median(on)).mean())}
    print(name, json.dumps(res[name]))
json.dump(res, open(out_json, "w"), indent=2)
'''


@app.function(image=image, volumes={str(DATA): vol}, timeout=HOURS, cpu=8, memory=32768)
def calibrate_control() -> dict:
    """Threshold T for the preregistered readout: the median predicted value
    over the control's published ink-label pixels (labels from the ink_9um
    training tree; the same 5820x5240x28 grid as our control render)."""
    root = DATA / "control"
    labels = root / "w035_inklabels.zarr"
    if not (labels / "0" / ".zarray").exists():
        (labels / "0").mkdir(parents=True, exist_ok=True)
        base = "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/native9-scrollprizeorg-21slices/w035/w035_inklabels.zarr"
        sh(f"curl -sfL {base}/.zgroup -o {labels}/.zgroup; curl -sfL {base}/.zattrs -o {labels}/.zattrs || true; curl -sfL {base}/0/.zarray -o {labels}/0/.zarray")
        meta = json.loads((labels / "0" / ".zarray").read_text())
        sep = meta.get("dimension_separator", ".")
        ny = -(-meta["shape"][1] // meta["chunks"][1]); nx = -(-meta["shape"][2] // meta["chunks"][2])
        keys = [f"0{sep}{y}{sep}{x}" for y in range(ny) for x in range(nx)]
        (root / "_keys.txt").write_text("\n".join(keys))
        # missing chunks are fill_value; curl -f skips 404s quietly
        sh(f"cd {labels}/0 && xargs -P 32 -I{{}} sh -c 'curl -sfL {base}/0/{{}} -o {{}} || true' < {root}/_keys.txt; ls | wc -l")
    script = VILLA / "spiral-fitting" / "_calibrate.py"
    script.write_text(CALIB_SCRIPT)
    out = root / "out_own_render"
    sh(f"cd {VILLA}/spiral-fitting && uv run python {script} {labels} {out}/control_prediction.tif {out}/control_prediction_reverse.tif {out}/calibration.json")
    vol.commit()
    return json.loads((out / "calibration.json").read_text())
