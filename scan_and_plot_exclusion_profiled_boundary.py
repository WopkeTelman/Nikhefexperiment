#!/usr/bin/env python3
"""Scan and plot the profiled FCC-ee 95% CL exclusion boundary for Model23 heavy W'. """

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Keep runtime deterministic/headless.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import matplotlib.pyplot as plt
import numpy as np
import yaml

try:
    import jax

    jax.config.update("jax_enable_x64", True)
except Exception:
    pass


MASSES = [3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0, 18.0, 20.0]

G_LOW = 1e-6
G_HIGH = 3.0
TARGET_DELTA = 3.84
DELTA_TOL = 0.1
MAX_BISECT_ITERS = 18

ROOT = Path(__file__).resolve().parents[1]

INJECT_TEMPLATE = ROOT / "runcards" / "bsm_closure_inject_model23_uvmass_clean.yaml"
UV_FIT_TEMPLATE = ROOT / "runcards" / "discovery_benchmarks" / "fit_discovery_uv_m10_g03.yaml"
DISCOVERY_BOUNDARY_CSV = (
    ROOT / "pseudo_data" / "discovery_benchmarks" / "fccee_discovery_5sigma_scan.csv"
)

PSEUDO_BASE = ROOT / "pseudo_data" / "exclusion_benchmarks"
LEGACY_PSEUDO_BASE = ROOT / "pseudo_data" / "discovery_benchmarks"
STAGING_BASE = PSEUDO_BASE / "_staging_scan"
RUN_CARD_BASE = ROOT / "runcards" / "exclusion_benchmarks" / "scan_runtime"
RESULTS_BASE = ROOT / "closure_results" / "exclusion_benchmarks_scan"
LEGACY_RESULTS_BASE_DISCOVERY_SCAN = ROOT / "closure_results" / "discovery_benchmarks_scan"
LOG_BASE = ROOT / "logs" / "exclusion_profiled_scan"

OUT_CSV = PSEUDO_BASE / "fccee_exclusion_profiled_scan.csv"
OUT_PDF = ROOT / "plots" / "paper_ready" / "fccee_exclusion_profiled_scan.pdf"
OUT_PNG = ROOT / "plots" / "paper_ready" / "fccee_exclusion_profiled_scan.png"

POINT_DIR_RE = re.compile(r"^m_([-+0-9.eE]+)g([-+0-9.eE]+)$")
SEED_DIR_RE = re.compile(r"^seed_(\d+)$")
UV_COUPLINGS = ("gWH", "gWLf11", "gWLf22", "gWLf33", "gWqf33")


@dataclass
class EvalPoint:
    mass: float
    g: float
    seed_dir: Path
    sm_fit_result: Path
    uv_fit_result: Path
    chi2_sm: float
    chi2_uv: float
    delta_chi2: float


@dataclass
class BoundaryPoint:
    mass: float
    g_95cl: float
    status: str


@dataclass
class ScanContext:
    inject_template_cfg: dict
    uv_fit_template_cfg: dict
    point_dir_index: Dict[Tuple[float, float], Path]
    cache_eval: Dict[Tuple[float, float], EvalPoint]


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML mapping: {path}")
    return data


def dump_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def run_cmd(cmd: List[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    log_txt = (
        f"$ {' '.join(cmd)}\n\n"
        f"returncode: {proc.returncode}\n\n"
        f"STDOUT:\n{proc.stdout}\n\n"
        f"STDERR:\n{proc.stderr}\n"
    )
    log_path.write_text(log_txt, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)} (see {log_path})")


def point_key(mass: float, g: float) -> Tuple[float, float]:
    return (round(float(mass), 10), round(float(g), 14))


def format_mass(mass: float) -> str:
    return f"{mass:.1f}"


def format_g(g: float) -> str:
    if g >= 1.0 and abs(g - round(g)) < 1e-14:
        return f"{g:.1f}"
    if g >= 1e-3:
        s = f"{g:.8f}".rstrip("0").rstrip(".")
        return s if s else "0"
    return f"{g:.8g}"


def number_token(x: float) -> str:
    s = format_g(x)
    return s.replace("-", "m").replace(".", "p").replace("+", "")


def point_tag(mass: float, g: float) -> str:
    return f"m_{format_mass(mass)}g{format_g(g)}"


def deterministic_seed(mass: float, g: float) -> int:
    raw = f"{mass:.12g}_{g:.16g}".encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    return 100000 + int(digest[:8], 16) % 800000


def uv_result_id(mass: float, g: float) -> str:
    return f"fit_exclusion_profiled_uv_m{number_token(mass)}_g{number_token(g)}"


def legacy_uv_result_id(mass: float, g: float) -> str:
    return f"fit_discovery_scan_m{number_token(mass)}_g{number_token(g)}"


def sm_result_id(mass: float, g: float) -> str:
    return f"fit_exclusion_profiled_sm_m{number_token(mass)}_g{number_token(g)}"


def scan_existing_point_dirs(base: Path) -> Dict[Tuple[float, float], Path]:
    out: Dict[Tuple[float, float], Path] = {}
    if not base.exists():
        return out
    for child in sorted(base.iterdir()):
        if not child.is_dir():
            continue
        if child.name.startswith("_"):
            continue
        m = POINT_DIR_RE.match(child.name)
        if not m:
            continue
        mass = float(m.group(1))
        g = float(m.group(2))
        out[point_key(mass, g)] = child
    return out


def has_projected_payload(seed_dir: Path) -> bool:
    return any(seed_dir.glob("*_proj.yaml")) or any(
        p.is_file() for p in seed_dir.glob("FCCee_*.yaml")
    )


def choose_seed_dir(point_dir: Path) -> Optional[Path]:
    if not point_dir.exists():
        return None
    candidates = []
    for child in sorted(point_dir.iterdir()):
        if child.is_dir() and SEED_DIR_RE.match(child.name):
            if has_projected_payload(child):
                candidates.append(child)
    return candidates[0] if candidates else None


def ensure_plain_yaml_symlinks(seed_dir: Path) -> None:
    for proj_yaml in sorted(seed_dir.glob("*_proj.yaml")):
        plain_yaml = seed_dir / proj_yaml.name.replace("_proj.yaml", ".yaml")
        if plain_yaml.exists():
            continue
        try:
            plain_yaml.symlink_to(proj_yaml.name)
        except OSError:
            shutil.copy2(proj_yaml, plain_yaml)


def build_injection_card(cfg_template: dict, mass: float, g: float, proj_path: Path) -> dict:
    cfg = copy.deepcopy(cfg_template)
    cfg["result_ID"] = f"proj_exclusion_scan_m{number_token(mass)}_g{number_token(g)}"
    cfg["projections_path"] = str(proj_path.resolve())

    coeffs = cfg.get("coefficients")
    if not isinstance(coeffs, dict):
        raise ValueError("Injection template missing 'coefficients' block.")

    for name in UV_COUPLINGS:
        block = coeffs.get(name, {})
        if not isinstance(block, dict):
            block = {}
        block["constrain"] = True
        block["value"] = float(g)
        block["min"] = float(g)
        block["max"] = float(g)
        coeffs[name] = block

    m_block = coeffs.get("m", {})
    if not isinstance(m_block, dict):
        m_block = {}
    m_block["is_mass"] = True
    m_block["constrain"] = True
    m_block["value"] = float(mass)
    m_block["min"] = float(mass)
    m_block["max"] = float(mass)
    coeffs["m"] = m_block

    return cfg


def find_existing_seed_dir(mass: float, g: float) -> Optional[Tuple[Path, Path]]:
    tag = point_tag(mass, g)
    for base in (PSEUDO_BASE, LEGACY_PSEUDO_BASE):
        point_dir = base / tag
        seed_dir = choose_seed_dir(point_dir)
        if seed_dir is not None:
            ensure_plain_yaml_symlinks(seed_dir)
            return point_dir, seed_dir
    return None


def ensure_pseudodata(ctx: ScanContext, mass: float, g: float) -> Path:
    k = point_key(mass, g)
    point_dir = ctx.point_dir_index.get(k)
    if point_dir is not None:
        existing = choose_seed_dir(point_dir)
        if existing is not None:
            ensure_plain_yaml_symlinks(existing)
            return existing

    existing_pair = find_existing_seed_dir(mass, g)
    if existing_pair is not None:
        point_dir, seed_dir = existing_pair
        ctx.point_dir_index[k] = point_dir
        return seed_dir

    tag = point_tag(mass, g)
    seed = deterministic_seed(mass, g)
    staging_dir = STAGING_BASE / tag
    point_dir = PSEUDO_BASE / tag
    staging_dir.mkdir(parents=True, exist_ok=True)
    point_dir.mkdir(parents=True, exist_ok=True)

    card_cfg = build_injection_card(ctx.inject_template_cfg, mass, g, staging_dir)
    card_path = RUN_CARD_BASE / f"inject_{tag}.yaml"
    dump_yaml(card_path, card_cfg)

    log_path = LOG_BASE / "proj" / f"{tag}_seed_{seed}.log"
    run_cmd(
        [
            "smefit",
            "PROJ",
            "--noise",
            "L0",
            "--lumi",
            "5e6",
            "--seed",
            str(seed),
            str(card_path),
        ],
        log_path,
    )

    src_seed = staging_dir / f"seed_{seed}"
    if not src_seed.exists():
        seeds = sorted(staging_dir.glob("seed_*"))
        if not seeds:
            raise FileNotFoundError(f"No PROJ seed directory created in {staging_dir}")
        src_seed = seeds[-1]

    dst_seed = point_dir / src_seed.name
    if not dst_seed.exists():
        shutil.move(str(src_seed), str(dst_seed))

    ensure_plain_yaml_symlinks(dst_seed)
    meta = {
        "mass": mass,
        "g": g,
        "seed": seed,
        "injection_card": str(card_path.resolve()),
        "proj_command": (
            f"smefit PROJ --noise L0 --lumi 5e6 --seed {seed} {card_path.resolve()}"
        ),
    }
    dump_yaml(dst_seed / "scan_meta.yaml", meta)

    ctx.point_dir_index[k] = point_dir
    return dst_seed


def build_uv_fit_card(cfg_template: dict, seed_dir: Path, mass: float, g: float) -> Tuple[Path, str]:
    cfg = copy.deepcopy(cfg_template)
    rid = uv_result_id(mass, g)
    cfg["result_ID"] = rid
    cfg["data_path"] = str(seed_dir.resolve())
    cfg["result_path"] = str(RESULTS_BASE.resolve())
    card = RUN_CARD_BASE / f"fit_uv_{point_tag(mass, g)}.yaml"
    dump_yaml(card, cfg)
    return card, rid


def build_sm_fit_card(cfg_template: dict, seed_dir: Path, mass: float, g: float) -> Tuple[Path, str]:
    cfg = copy.deepcopy(cfg_template)
    rid = sm_result_id(mass, g)
    cfg["result_ID"] = rid
    cfg["data_path"] = str(seed_dir.resolve())
    cfg["result_path"] = str(RESULTS_BASE.resolve())

    coeffs = cfg.get("coefficients")
    if not isinstance(coeffs, dict):
        raise ValueError("SM fit template missing 'coefficients' block.")

    for name in UV_COUPLINGS:
        block = coeffs.get(name, {})
        if not isinstance(block, dict):
            block = {}
        block["constrain"] = True
        block["value"] = 0.0
        block["min"] = 0.0
        block["max"] = 0.0
        coeffs[name] = block

    # Keep one profiled direction for NS robustness. With all parameters fixed,
    # UltraNest receives a 0-dimensional problem and fails.
    m_block = coeffs.get("m", {})
    if not isinstance(m_block, dict):
        m_block = {}
    m_block["is_mass"] = True
    m_block.pop("constrain", None)
    m_block.pop("value", None)
    m_block.setdefault("min", 0.5)
    m_block.setdefault("max", 20.0)
    coeffs["m"] = m_block

    card = RUN_CARD_BASE / f"fit_sm_{point_tag(mass, g)}.yaml"
    dump_yaml(card, cfg)
    return card, rid


def extract_chi2_from_fit_json(path: Path) -> float:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if "max_loglikelihood" in payload:
        return float(-2.0 * float(payload["max_loglikelihood"]))
    if "chi2_min" in payload:
        return float(payload["chi2_min"])
    if "chi2" in payload:
        return float(payload["chi2"])
    raise KeyError(f"No chi2-compatible key in {path}")


def ensure_uv_fit(ctx: ScanContext, seed_dir: Path, mass: float, g: float) -> Path:
    rid = uv_result_id(mass, g)
    fit_json = RESULTS_BASE / rid / "fit_results.json"
    if fit_json.exists():
        return fit_json

    legacy_json = LEGACY_RESULTS_BASE_DISCOVERY_SCAN / legacy_uv_result_id(mass, g) / "fit_results.json"
    if legacy_json.exists():
        return legacy_json

    card, rid = build_uv_fit_card(ctx.uv_fit_template_cfg, seed_dir, mass, g)
    log_path = LOG_BASE / "ns_uv" / f"{rid}.log"
    run_cmd(["smefit", "NS", str(card)], log_path)

    fit_json = RESULTS_BASE / rid / "fit_results.json"
    if not fit_json.exists():
        raise FileNotFoundError(f"Missing UV fit output after NS: {fit_json}")
    return fit_json


def ensure_sm_fit(ctx: ScanContext, seed_dir: Path, mass: float, g: float) -> Path:
    rid = sm_result_id(mass, g)
    fit_json = RESULTS_BASE / rid / "fit_results.json"
    if fit_json.exists():
        return fit_json

    card, rid = build_sm_fit_card(ctx.uv_fit_template_cfg, seed_dir, mass, g)
    log_path = LOG_BASE / "ns_sm" / f"{rid}.log"
    run_cmd(["smefit", "NS", str(card)], log_path)

    fit_json = RESULTS_BASE / rid / "fit_results.json"
    if not fit_json.exists():
        raise FileNotFoundError(f"Missing SM fit output after NS: {fit_json}")
    return fit_json


def evaluate_point(ctx: ScanContext, mass: float, g: float) -> EvalPoint:
    k = point_key(mass, g)
    cached = ctx.cache_eval.get(k)
    if cached is not None:
        return cached

    seed_dir = ensure_pseudodata(ctx, mass, g)
    sm_fit_json = ensure_sm_fit(ctx, seed_dir, mass, g)
    uv_fit_json = ensure_uv_fit(ctx, seed_dir, mass, g)

    chi2_sm = extract_chi2_from_fit_json(sm_fit_json)
    chi2_uv = extract_chi2_from_fit_json(uv_fit_json)
    delta = chi2_sm - chi2_uv

    out = EvalPoint(
        mass=mass,
        g=g,
        seed_dir=seed_dir,
        sm_fit_result=sm_fit_json,
        uv_fit_result=uv_fit_json,
        chi2_sm=chi2_sm,
        chi2_uv=chi2_uv,
        delta_chi2=delta,
    )
    ctx.cache_eval[k] = out
    print(
        f"[eval] m={mass:.1f}, g={g:.8g} -> chi2_SM={chi2_sm:.6f}, "
        f"chi2_UV={chi2_uv:.6f}, delta={delta:.6f}"
    )
    return out


def bisection_g95(ctx: ScanContext, mass: float) -> BoundaryPoint:
    lo = float(G_LOW)
    hi = float(G_HIGH)
    e_lo = evaluate_point(ctx, mass, lo)
    e_hi = evaluate_point(ctx, mass, hi)
    f_lo = e_lo.delta_chi2 - TARGET_DELTA
    f_hi = e_hi.delta_chi2 - TARGET_DELTA

    if f_lo >= 0.0:
        return BoundaryPoint(mass=mass, g_95cl=lo, status="already_above_at_glow")
    if f_hi < 0.0:
        return BoundaryPoint(mass=mass, g_95cl=float("nan"), status="no_crossing_in_bracket")

    for it in range(1, MAX_BISECT_ITERS + 1):
        mid = math.sqrt(lo * hi)
        e_mid = evaluate_point(ctx, mass, mid)
        f_mid = e_mid.delta_chi2 - TARGET_DELTA
        print(
            f"  [bisect] m={mass:.1f} iter={it} g={mid:.8g} "
            f"delta={e_mid.delta_chi2:.6f} (target={TARGET_DELTA:.2f})"
        )

        if abs(f_mid) <= DELTA_TOL:
            return BoundaryPoint(mass=mass, g_95cl=mid, status="converged")

        if f_mid >= 0.0:
            hi = mid
        else:
            lo = mid

    # Conservative fallback: upper bound where Delta-chi2 is expected >= target.
    return BoundaryPoint(mass=mass, g_95cl=hi, status="max_iter_upper_bound")


def write_boundary_csv(rows: List[BoundaryPoint], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mass", "g_95cl"])
        for row in sorted(rows, key=lambda r: r.mass):
            gtxt = "nan" if not math.isfinite(row.g_95cl) else f"{row.g_95cl:.16e}"
            writer.writerow([f"{row.mass:.12g}", gtxt])


def load_boundary(path: Path, g_columns: Tuple[str, ...]) -> Tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        return np.array([]), np.array([])

    xs: List[float] = []
    ys: List[float] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return np.array([]), np.array([])

        g_col = None
        for name in g_columns:
            if name in reader.fieldnames:
                g_col = name
                break
        if g_col is None:
            return np.array([]), np.array([])

        for row in reader:
            m = float(row["mass"])
            try:
                g = float(row[g_col])
            except ValueError:
                continue
            if math.isfinite(g) and g > 0.0:
                xs.append(m)
                ys.append(g)

    if not xs:
        return np.array([]), np.array([])

    idx = np.argsort(np.asarray(xs))
    return np.asarray(xs)[idx], np.asarray(ys)[idx]


def plot_boundaries(rows: List[BoundaryPoint], discovery_csv: Path, out_pdf: Path, out_png: Path) -> None:
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    valid = sorted((r for r in rows if math.isfinite(r.g_95cl) and r.g_95cl > 0.0), key=lambda r: r.mass)
    if not valid:
        raise RuntimeError("No finite exclusion boundary points found; cannot plot.")

    m_exc = np.asarray([r.mass for r in valid], dtype=float)
    g_exc = np.asarray([r.g_95cl for r in valid], dtype=float)

    m_disc, g_disc = load_boundary(discovery_csv, ("g_5sigma", "g"))

    fig, ax = plt.subplots(figsize=(8.8, 6.2))
    ax.loglog(
        m_exc,
        g_exc,
        color="#0B5CAD",
        marker="o",
        linewidth=2.3,
        label="Profiled 95% CL exclusion (Delta-chi2 = 3.84)",
    )
    if m_disc.size > 0:
        ax.loglog(
            m_disc,
            g_disc,
            color="#C46B00",
            marker="s",
            linestyle="--",
            linewidth=2.0,
            label="Profiled 5sigma discovery (Delta-chi2 = 25)",
        )

    ax.set_xlabel("m [TeV]", fontsize=14)
    ax.set_ylabel("g", fontsize=14)
    ax.set_title("FCC-ee Heavy W' Profiled Exclusion vs Discovery", fontsize=15)
    ax.grid(False)
    ax.legend(frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=260)
    plt.close(fig)


def prepare_context() -> ScanContext:
    inject_cfg = load_yaml(INJECT_TEMPLATE)
    uv_fit_cfg = load_yaml(UV_FIT_TEMPLATE)

    point_dir_index: Dict[Tuple[float, float], Path] = {}
    for base in (LEGACY_PSEUDO_BASE, PSEUDO_BASE):
        # Let the dedicated exclusion base override legacy entries.
        point_dir_index.update(scan_existing_point_dirs(base))

    return ScanContext(
        inject_template_cfg=inject_cfg,
        uv_fit_template_cfg=uv_fit_cfg,
        point_dir_index=point_dir_index,
        cache_eval={},
    )


def print_boundary_table(rows: List[BoundaryPoint]) -> None:
    print("\nProfiled 95% CL exclusion boundary (Delta-chi2 = chi2_SM - chi2_UV = 3.84)")
    print("mass_TeV,g_95cl,status")
    for row in sorted(rows, key=lambda r: r.mass):
        gtxt = "nan" if not math.isfinite(row.g_95cl) else f"{row.g_95cl:.8g}"
        print(f"{row.mass:.1f},{gtxt},{row.status}")


def main() -> None:
    for p in (RUN_CARD_BASE, RESULTS_BASE, LOG_BASE, STAGING_BASE, PSEUDO_BASE):
        p.mkdir(parents=True, exist_ok=True)

    ctx = prepare_context()
    boundary_rows: List[BoundaryPoint] = []

    print("Starting profiled 95% CL exclusion scan")
    print(f"Mass grid: {MASSES}")
    print(f"g bracket: [{G_LOW}, {G_HIGH}]")
    print(f"Delta-chi2 target: {TARGET_DELTA} (tolerance ±{DELTA_TOL})")
    print("NS settings and likelihood flags are inherited from:")
    print(f"  {UV_FIT_TEMPLATE}")
    print()

    for mass in MASSES:
        print(f"[mass] m={mass:.1f} TeV")
        row = bisection_g95(ctx, mass)
        boundary_rows.append(row)
        print(f"  -> m={mass:.1f}: g_95cl={row.g_95cl} ({row.status})")
        print()

    write_boundary_csv(boundary_rows, OUT_CSV)
    plot_boundaries(boundary_rows, DISCOVERY_BOUNDARY_CSV, OUT_PDF, OUT_PNG)

    print_boundary_table(boundary_rows)
    print("\nFigure legend description:")
    print(
        "Solid blue circles: profiled 95% CL exclusion boundary "
        "(Delta-chi2 = chi2_SM - chi2_UV = 3.84, both from nested sampling)."
    )
    print(
        "Dashed orange squares: profiled 5sigma discovery boundary "
        "(Delta-chi2 = 25) from fccee_discovery_5sigma_scan.csv."
    )

    print("\nScan complete.")
    print(f"Boundary CSV: {OUT_CSV}")
    print(f"Plot PDF:     {OUT_PDF}")
    print(f"Plot PNG:     {OUT_PNG}")


if __name__ == "__main__":
    main()
