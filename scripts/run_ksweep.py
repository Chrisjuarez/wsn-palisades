"""End-to-end pipeline: AOI → DEM/DSM/CHM → packs → K-sweep → save.

Example:
    python scripts/run_ksweep.py \\
        --aoi sample/aoi_palisades.geojson \\
        --data data \\
        --scenario all --k-min 10 --k-max 60 --k-step 10 \\
        --out results/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from wsn_palisades import SensorParams, SolarParams
from wsn_palisades.candidates import load_aoi, precompute_scenario_loky
from wsn_palisades.ksweep import run_k_sweep_all
from wsn_palisades.persistence import save_ksweep, save_metrics_csv
from wsn_palisades.surfaces import DEMManager, warp_surfaces_to_utm

SCENARIO_LABELS = {"flat": "FLAT", "dem": "DEM", "dsm_chm": "DSM/CHM"}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--aoi", required=True, help="Path to AOI geojson polygon")
    ap.add_argument("--data", default="data", help="Directory holding the .tif rasters")
    ap.add_argument(
        "--scenario",
        choices=("flat", "dem", "dsm_chm", "all"),
        default="all",
        help="Which scenario(s) to precompute and sweep over",
    )
    ap.add_argument("--k-min", type=int, default=10)
    ap.add_argument("--k-max", type=int, default=60)
    ap.add_argument("--k-step", type=int, default=10)
    ap.add_argument("--grid-size", type=int, default=30, help="Candidate grid size")
    ap.add_argument("--cov-grid-size", type=int, default=80, help="Coverage grid size")
    ap.add_argument("--n-jobs", type=int, default=None, help="Parallel workers for precompute")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results", help="Output directory")
    ap.add_argument("--tag", default=None, help="Suffix for output filenames")
    return ap.parse_args()


def main():
    load_dotenv()
    args = parse_args()

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "csv").mkdir(parents=True, exist_ok=True)

    aoi_poly = load_aoi(args.aoi)
    print(f"AOI: {args.aoi}  bounds={aoi_poly.bounds}")

    data = Path(args.data).resolve()
    dtm = data / "palisadesoutput.dtm.tif"
    dsm = data / "palisadesoutput.dsm.tif"
    chm = data / "palisadesCHM.tif"
    if not dtm.exists():
        print(f"error: {dtm} not found. Run scripts/fetch_data.py first.", file=sys.stderr)
        sys.exit(2)

    print("loading rasters and warping to UTM...")
    SP = SensorParams()
    solar = SolarParams()
    dem = DEMManager.from_files(
        aoi_poly,
        dtm_path=str(dtm),
        dsm_path=str(dsm) if dsm.exists() else None,
        chm_path=str(chm) if chm.exists() else None,
    )
    warp_surfaces_to_utm(dem, aoi_poly, target_res_m=2.0)

    scenarios = ["flat", "dem", "dsm_chm"] if args.scenario == "all" else [args.scenario]
    packs_by_label = {}
    for s in scenarios:
        print(f"precompute scenario={s}")
        packs = precompute_scenario_loky(
            aoi_poly, dem, s, SP,
            grid_size=args.grid_size,
            cov_grid_size=args.cov_grid_size,
            n_jobs=args.n_jobs,
            batch_size=4,
            solar_params=solar,
        )
        packs_by_label[SCENARIO_LABELS[s]] = packs

    K_values = list(range(args.k_min, args.k_max + 1, args.k_step))
    print(f"running k-sweep over K = {K_values}")
    df, results = run_k_sweep_all(
        packs_by_label,
        SP=SP,
        K_values=K_values,
        random_seed=args.seed,
    )

    tag = args.tag or f"k{args.k_min}to{args.k_max}"
    pkl_path = out / f"res_k_sweep_{tag}.pkl.gz"
    csv_path = out / "csv" / f"df_k_sweep_{tag}.csv"
    save_ksweep(results, pkl_path)
    save_metrics_csv(df, csv_path)
    print(f"saved {pkl_path}")
    print(f"saved {csv_path}")


if __name__ == "__main__":
    main()
