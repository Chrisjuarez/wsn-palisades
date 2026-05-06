"""Streamlit MVP for exploring saved K-sweep results and running small live AOIs.

Run locally:
    streamlit run app/streamlit_app.py

The Explore Results page is the MVP path — it reads bundled `.pkl.gz`/`.csv`
files in ``results/`` and needs no S3 / API access. The Live AOI page needs
``OPENTOPO_API_KEY`` and is restricted to small AOIs (≤2 km²).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from shapely.geometry import shape

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
SAMPLE = REPO / "sample"

DEFAULT_KSWEEP = RESULTS / "res_k_sweep_k=10to60_PalisadesFinal.pkl.gz"
DEFAULT_CSV = RESULTS / "csv" / "df_k_sweep_k=10to60_PalisadesFinal.csv"


@st.cache_resource(show_spinner="Loading saved K-sweep...")
def cached_load_ksweep(path: str):
    from wsn_palisades.persistence import load_ksweep
    return load_ksweep(path)


@st.cache_data(show_spinner=False)
def cached_load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def cached_load_aoi(path: str):
    from wsn_palisades.candidates import load_aoi
    return load_aoi(path)


# ============================================================================
# Sidebar
# ============================================================================

def sidebar():
    st.sidebar.title("WSN Palisades")
    st.sidebar.caption("Terrain/vegetation/solar-aware NSGA-III sensor placement")
    page = st.sidebar.radio(
        "Page",
        ("Home", "Explore Results", "Live AOI"),
        index=1,
    )
    st.sidebar.markdown("---")
    return page


# ============================================================================
# Pages
# ============================================================================


def page_home():
    st.title("WSN Palisades")
    st.write(
        "This tool selects K wireless-sensor locations under realistic "
        "line-of-sight, canopy attenuation, minimum-separation, and solar "
        "constraints, comparing four selection methods: random, greedy, "
        "simple NSGA-III, and seeded NSGA-III."
    )
    fig_path = RESULTS / "figures" / "ieee_fig1_coverage_vs_k.png"
    if fig_path.exists():
        st.image(str(fig_path), caption="Coverage vs K — across terrains and optimizers")
    st.markdown(
        """
        **Pages**
        - **Explore Results** — load the bundled saved K-sweep and inspect
          per-(scenario, K, optimizer) trade-offs and sensor placements.
        - **Live AOI** — draw a small AOI and run the FLAT/DEM scenarios on
          the fly (requires `OPENTOPO_API_KEY`).

        See [README](https://github.com/chrisjuarez/wsn-palisades) for full
        reproduction instructions.
        """
    )


def page_explore():
    from wsn_palisades import maps as wp_maps
    from wsn_palisades import plotting as wp_plotting

    st.title("Explore saved results")
    st.caption("Bundled canonical Palisades sweep, K = 10..60.")

    if not DEFAULT_KSWEEP.exists():
        st.error(f"Missing {DEFAULT_KSWEEP}. Run scripts/run_ksweep.py first.")
        return

    res = cached_load_ksweep(str(DEFAULT_KSWEEP))
    df = cached_load_csv(str(DEFAULT_CSV)) if DEFAULT_CSV.exists() else None
    aoi = cached_load_aoi(str(SAMPLE / "aoi_palisades.geojson"))

    scenarios = wp_maps.list_scenarios(res)
    col1, col2, col3 = st.columns(3)
    scen = col1.selectbox("Scenario", scenarios, index=0)
    Ks = wp_maps.list_k_values(res, scen)
    K = col2.selectbox("K", Ks, index=len(Ks) - 1 if Ks else 0)
    opts = wp_maps.list_optimizers(res, scen, int(K))
    opt_label_to_key = {label: key for label, key in opts}
    opt_label = col3.selectbox("Optimizer", list(opt_label_to_key.keys()))
    opt_key = opt_label_to_key.get(opt_label)

    n_sensors = st.slider(
        "Number of sensors to render",
        1, int(K), int(K), 1,
    )

    tab_map, tab_pareto, tab_lines = st.tabs(["Map", "Pareto", "Metric vs K"])

    with tab_map:
        try:
            from streamlit_folium import st_folium
            fmap = wp_maps.build_folium_map(
                aoi, res, scenario=scen, K=int(K),
                optimizer_key=opt_key, n_sensors=int(n_sensors),
                grid_size=30,
            )
            st_folium(fmap, width=900, height=560, returned_objects=[])
        except ImportError:
            st.warning("Install `streamlit-folium` to enable interactive maps.")

    with tab_pareto:
        if df is not None:
            x_metric = st.selectbox(
                "X axis", ["coverage_pct", "gamma_mean", "d_mean_m", "solar_mean"], index=0
            )
            y_metric = st.selectbox(
                "Y axis", ["gamma_mean", "coverage_pct", "d_mean_m", "solar_mean"], index=0
            )
            fig = wp_plotting.plot_pareto_tradeoff(df, x=x_metric, y=y_metric)
            st.pyplot(fig)
        else:
            st.info("Metrics CSV not bundled — Pareto chart unavailable.")

    with tab_lines:
        if df is not None:
            metric = st.selectbox(
                "Metric",
                ["coverage_pct", "gamma_mean", "d_mean_m", "solar_mean"],
                index=0,
            )
            fig = wp_plotting.plot_metric_vs_k(df, metric=metric, ylabel=metric)
            st.pyplot(fig)
        else:
            st.info("Metrics CSV not bundled.")

    st.markdown("---")
    if df is not None:
        with st.expander("Raw metrics for the selected slice"):
            sub = df[(df["scenario"].astype(str).str.strip() == scen) & (df["K"] == int(K))]
            st.dataframe(sub, use_container_width=True)


def page_live():
    from wsn_palisades.candidates import precompute_scenario_loky
    from wsn_palisades.optimizers import greedy_select, random_select
    from wsn_palisades.params import SensorParams, SolarParams
    from wsn_palisades.surfaces import DEMManager, warp_surfaces_to_utm

    st.title("Live AOI — quick FLAT/DEM run")
    st.caption(
        "Draw a small AOI (≤2 km²), fetch DEM via OpenTopography, "
        "and run greedy + random for a single K. DSM/CHM is unavailable for "
        "arbitrary AOIs and is omitted here."
    )

    api_key = os.environ.get("OPENTOPO_API_KEY")
    if not api_key:
        st.warning("`OPENTOPO_API_KEY` not set. Live DEM fetch will be skipped.")

    try:
        import folium
        from streamlit_folium import st_folium
        from folium.plugins import Draw
    except ImportError:
        st.error("Install `folium` and `streamlit-folium` to enable AOI drawing.")
        return

    col_left, col_right = st.columns([3, 2])
    with col_left:
        m = folium.Map(location=(34.06, -118.54), zoom_start=14,
                       tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                       attr="Esri")
        Draw(export=False, draw_options={"polyline": False, "circlemarker": False, "marker": False}).add_to(m)
        out = st_folium(m, width=700, height=520, returned_objects=["last_active_drawing"])

    drawn: Optional[dict] = out.get("last_active_drawing") if out else None
    aoi = None
    area_km2 = None
    if drawn and drawn.get("geometry"):
        aoi = shape(drawn["geometry"])
        # quick lat/lon area approx
        from pyproj import Geod
        try:
            area, _ = Geod(ellps="WGS84").geometry_area_perimeter(aoi)
            area_km2 = abs(area) / 1e6
        except Exception:
            area_km2 = None

    with col_right:
        st.write("**Selected AOI**")
        if aoi is None:
            st.info("Draw a polygon or rectangle on the map.")
        else:
            st.write(f"Bounds: `{aoi.bounds}`")
            if area_km2 is not None:
                st.write(f"Area: **{area_km2:.2f} km²**")
                if area_km2 > 2.0:
                    st.error("AOI too large — keep it under 2 km² for a live run.")

        K = st.number_input("K (sensors)", min_value=5, max_value=80, value=20)
        run = st.button("Run", disabled=(aoi is None or (area_km2 is not None and area_km2 > 2.0)))

    if not run or aoi is None:
        return

    if not api_key:
        st.error("Live DEM fetch requires `OPENTOPO_API_KEY` in `.env`.")
        return

    with st.spinner("Fetching DEM and computing visibility..."):
        import requests
        minx, miny, maxx, maxy = aoi.bounds
        params = {
            "demtype": "SRTMGL1",
            "south": miny, "north": maxy, "west": minx, "east": maxx,
            "outputFormat": "GTiff",
            "API_Key": api_key,
        }
        r = requests.get("https://portal.opentopography.org/API/globaldem", params=params, timeout=60)
        if r.status_code != 200:
            st.error(f"OpenTopography fetch failed (HTTP {r.status_code}): {r.text[:200]}")
            return

        dmgr = DEMManager(r.content)
        dmgr.calculate_slope_and_aspect()
        warp_surfaces_to_utm(dmgr, aoi, target_res_m=2.0)

        SP = SensorParams(R_m=300.0, az_step_deg=2, min_sep_m=200.0)
        solar = SolarParams()
        packs = precompute_scenario_loky(
            aoi, dmgr, "dem", SP,
            grid_size=20, cov_grid_size=40, n_jobs=4,
            solar_params=solar, verbose=False,
        )
        g = greedy_select(packs, int(K), SP)
        r_res = random_select(packs, int(K), SP)

    st.success("Done.")
    cmp = pd.DataFrame(
        [
            {"optimizer": "Random", "coverage_pct": r_res["coverage_pct"], "gamma_mean": r_res["gamma_mean"], "d_mean_m": r_res["d_mean_m"], "solar_mean": r_res["solar_mean"]},
            {"optimizer": "Greedy", "coverage_pct": g["coverage_pct"], "gamma_mean": g["gamma_mean"], "d_mean_m": g["d_mean_m"], "solar_mean": g["solar_mean"]},
        ]
    )
    st.dataframe(cmp, use_container_width=True)

    # Render greedy placement on the map
    cands = packs["candidates"]
    m2 = folium.Map(location=(aoi.centroid.y, aoi.centroid.x), zoom_start=15,
                    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                    attr="Esri")
    folium.Polygon(locations=[(c[1], c[0]) for c in aoi.exterior.coords], color="#FFFFFF", weight=3, fill=False).add_to(m2)
    for rank, idx in enumerate(g["idxs"], start=1):
        lon, lat = cands[int(idx)]
        folium.CircleMarker(location=(lat, lon), radius=6, color="#F58518",
                            fill=True, fill_color="#F58518", fill_opacity=0.9, weight=1,
                            popup=f"Greedy rank {rank}").add_to(m2)
    st.write("**Greedy placement**")
    st_folium(m2, width=900, height=460, returned_objects=[])


# ============================================================================
# Entrypoint
# ============================================================================


def main():
    load_dotenv()
    st.set_page_config(page_title="WSN Palisades", layout="wide")
    page = sidebar()
    if page == "Home":
        page_home()
    elif page == "Explore Results":
        page_explore()
    else:
        page_live()


if __name__ == "__main__":
    main()
