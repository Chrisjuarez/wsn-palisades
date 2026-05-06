# wsn-palisades

Terrain-, vegetation-, and solar-aware **NSGA-III sensor placement** for wireless sensor networks. Selects K sensor locations under realistic line-of-sight, canopy attenuation, minimum-separation, and solar-irradiance constraints, and compares random / greedy / simple-NSGA-III / seeded-NSGA-III across FLAT, DEM, and DSM/CHM scenarios.

Case study: **Pacific Palisades**, using local 0.5 m LiDAR-derived DTM/DSM/CHM rasters.

## Layout

```
src/wsn_palisades/         # the Python package (pure functions, no module globals)
notebooks/                 # one clean demo notebook walking through the pipeline
scripts/                   # CLI: fetch_data.py, run_ksweep.py, make_figures.py
app/streamlit_app.py       # Streamlit web app for exploring results
results/                   # canonical small outputs that ship with the repo
sample/aoi_palisades.geojson  # canonical AOI used for the bundled run
tests/                     # smoke + synthetic tests
```

## Quick start (no large downloads)

The bundled `.pkl.gz` and `.csv` in `results/` are sufficient to walk through the demo notebook and the Streamlit app — neither path requires the multi-GB GeoTIFF rasters.

```bash
git clone https://github.com/chrisjuarez/wsn-palisades.git
cd wsn-palisades
python -m venv .venv && source .venv/bin/activate
pip install -e .

# Demo notebook (figures + interactive map)
jupyter notebook notebooks/sensor_nsga_fast_palisades.ipynb

# Streamlit app
streamlit run app/streamlit_app.py
```

The Explore Results page works against `results/res_k_sweep_k=10to60_PalisadesFinal.pkl.gz`. The Live AOI page draws a small AOI on a map, fetches a DEM via OpenTopography, and runs greedy + random selection — it requires `OPENTOPO_API_KEY` in `.env`.

## Reproducing the full study

The Palisades DTM / DSM / CHM rasters are 1–1.8 GB each and are hosted on **S3** rather than committed to git.

```bash
cp .env.example .env
# fill in OPENTOPO_API_KEY, NSRDB_API_KEY, NSRDB_EMAIL, WSN_DATA_BUCKET

python scripts/fetch_data.py                     # downloads the .tif rasters into ./data/

python scripts/run_ksweep.py \                   # full pipeline (~hours)
    --aoi sample/aoi_palisades.geojson \
    --scenario all --k-min 10 --k-max 60 \
    --out results/

python scripts/make_figures.py \                 # renders the IEEE figures from the CSV
    --csv results/csv/df_k_sweep_k10to60.csv \
    --out results/figures/
```

## Method

For each candidate sensor location, the pipeline computes a per-azimuth visibility γ(θ) by combining

- **terrain occlusion** — distance-limited openness from the local DEM/DSM with a configurable horizon threshold and sensor height,
- **vegetation attenuation** — canopy intersected with the line of sight, density-weighted against a reference height.

That gives a per-direction effective radius `r_eff(θ)`, which is rasterized onto an AOI grid to produce per-sensor coverage masks. K-subset selection then optimises four objectives — coverage %, mean γ, mean pairwise spacing, mean solar exposure — subject to a minimum-separation constraint.

Selection methods:

- `random_select` — feasible-only random baseline.
- `greedy_select` — single-pass weighted-sum greedy.
- `simple_nsga_select` — NSGA-III (pymoo) with random feasible sampling and a `GeometricRepair` operator.
- `nsga_select` — same NSGA-III, additionally seeded with greedy solutions across a balanced weight simplex.

Solar exposure is modelled with `pvlib` clearsky + Perez transposition, masked by the per-azimuth horizon profile.

## Streamlit deployment

Local: `streamlit run app/streamlit_app.py`.

The Explore Results path needs no S3 / API access — the saved `.pkl.gz` (~13 MB) ships with the repo, so the app is deployable on Streamlit Community Cloud directly. The Live AOI path needs `OPENTOPO_API_KEY` set as a Streamlit secret.

## Citing

Paper forthcoming. BibTeX entry will be added on publication.

## License

MIT — see [LICENSE](LICENSE).
