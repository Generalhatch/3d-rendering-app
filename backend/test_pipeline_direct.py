"""
Full end-to-end pipeline test — exercises runner.run_pipeline() directly.
Uses the RR scan in scan-only mode.
"""
import sys, time
sys.path.insert(0, "/Users/tannerhatch/3d-rendering-app/backend")

from pathlib import Path
import shutil, json

# ── Setup: create a fake job directory ────────────────────────────────────────
SOURCE = Path("/Users/tannerhatch/3d-rendering-app/source-files")
rr_src = [f for f in SOURCE.glob("*.laz copy") if "RR" in f.name][0]

# The pipeline reads from uploads_dir(job_id) / scan_filename
# We'll copy the file there with the correct name
from app.storage import uploads_dir, artifacts_dir, results_dir, init_db

JOB_ID = "test-run-001"

# Init DB and dirs
init_db()

# Ensure dirs exist
udir = uploads_dir(JOB_ID)
adir = artifacts_dir(JOB_ID)
rdir = results_dir(JOB_ID)
udir.mkdir(parents=True, exist_ok=True)
adir.mkdir(parents=True, exist_ok=True)
rdir.mkdir(parents=True, exist_ok=True)

# Copy scan with clean name (no " copy")
dest = udir / "RR-scan.laz"
if not dest.exists():
    shutil.copy2(rr_src, dest)
    print(f"Copied scan to {dest}")
else:
    print(f"Scan already at {dest}")

try:
    import sqlite3, os
    # Find DB via settings
    os.environ.setdefault("DATA_DIR", str(Path("/Users/tannerhatch/3d-rendering-app/data")))
    from app.storage import init_db
    init_db()
    from app.config import get_settings
    db_path = get_settings().db_path
    print(f"DB path: {db_path}")
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        INSERT OR REPLACE INTO jobs
            (job_id, status, created_at, updated_at, scan_filenames)
        VALUES (?, 'pending', datetime('now'), datetime('now'), ?)
    """, (JOB_ID, json.dumps(["RR-scan.laz"])))
    conn.commit()
    conn.close()
    print(f"Job registered: {JOB_ID}")
except Exception as e:
    print(f"DB insert skipped: {e}")

# ── Run pipeline ──────────────────────────────────────────────────────────────
print("▶  Running pipeline (scan-only mode)…\n")
from app.pipeline.runner import run_pipeline

t0 = time.time()
run_pipeline(
    job_id=JOB_ID,
    scan_paths=[dest],
    plan_path=None,  # scan-only!
)
dt = time.time() - t0
print(f"\n▶  Pipeline took {dt:.1f}s\n")

# ── Check results ─────────────────────────────────────────────────────────────
result_file = rdir / "aligned.json"
if result_file.exists():
    result = json.loads(result_file.read_text())
    print(f"✅  Result saved!")
    print(f"   scan_only: {result.get('scan_only')}")
    print(f"   plan_bounds: {result.get('plan_bounds')}")
    print(f"   rooms: {len(result.get('rooms', []))}")
    print(f"   fixtures: {len(result.get('fixtures', []))}")
    align = result.get('alignment', {})
    print(f"   alignment.mode: {align.get('mode')}")
    print(f"   alignment.num_wall_planes: {align.get('num_wall_planes')}")
else:
    print("❌  No result file found")

# Check decimated PLY
ply_path = adir / "scan_decimated.ply"
print(f"\n   scan_decimated.ply: {'✓ exists' if ply_path.exists() else '✗ missing'}")

print("\n✅  DONE\n")
