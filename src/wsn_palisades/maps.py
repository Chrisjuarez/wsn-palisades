"""Interactive maps for visualising sensor placements from saved results.

Two backends:

- ``build_folium_map`` — pure folium HTML map. Works in Streamlit (via
  ``streamlit-folium``) and can be saved to standalone HTML.
- ``build_ipyleaflet_map`` — full notebook-only interactive viewer with
  scenario / K / optimizer dropdowns. Mirrors the legacy
  ``saved_sensor_map.py`` widget.

Both backends consume the saved ``res_k_sweep`` dict directly and regenerate
candidate coordinates from the AOI, so they don't need the full ``packs_*``
objects.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
from shapely.geometry import Point, Polygon

SCENARIO_COLORS = {
    "FLAT": "#4C78A8",
    "DEM": "#F58518",
    "DSM/CHM": "#54A24B",
    "DSMCHM": "#54A24B",
    "CHM": "#54A24B",
}

OPTIMIZER_PRETTY = {
    "random": "Random",
    "greedy": "Greedy",
    "nsga": "NSGA",
    "nsga3": "NSGA-III",
    "simple_nsga": "Simple-NSGA",
    "seeded_nsga": "Seeded-NSGA",
}


# -- pure helpers ------------------------------------------------------------


def regenerate_coverage_grid(polygon: Polygon, grid_size: int = 30) -> List[Tuple[float, float]]:
    """Same candidate-generation order used during optimization."""
    min_x, min_y, max_x, max_y = polygon.bounds
    xs = np.linspace(min_x, max_x, grid_size)
    ys = np.linspace(min_y, max_y, grid_size)
    pts: List[Tuple[float, float]] = []
    for x in xs:
        for y in ys:
            if polygon.contains(Point(x, y)):
                pts.append((float(x), float(y)))
    return pts


def list_scenarios(res_k_sweep: dict) -> List[str]:
    return [s for s in res_k_sweep.keys() if list_k_values(res_k_sweep, s)]


def list_k_values(res_k_sweep: dict, scenario: str) -> List[int]:
    out: list[int] = []
    for k in res_k_sweep.get(scenario, {}).keys():
        try:
            out.append(int(k))
        except Exception:
            pass
    return sorted(set(out))


def list_optimizers(res_k_sweep: dict, scenario: str, k: int) -> List[Tuple[str, str]]:
    res_pack = res_k_sweep.get(scenario, {}).get(int(k))
    if res_pack is None:
        res_pack = res_k_sweep.get(scenario, {}).get(str(k))
    if res_pack is None:
        return []
    opts = []
    for key, value in res_pack.items():
        if not isinstance(value, dict):
            continue
        if "idxs" in value:
            opts.append(key)
            continue
        for cand in ("best_compromise", "best", "solution", "best_solution"):
            if isinstance(value.get(cand), dict) and "idxs" in value[cand]:
                opts.append(key)
                break
    pref = ["seeded_nsga", "simple_nsga", "nsga3", "nsga", "greedy", "random"]
    ordered = [k for k in pref if k in opts] + [k for k in opts if k not in pref]
    return [(OPTIMIZER_PRETTY.get(k, k), k) for k in ordered]


def extract_solution(res_pack: dict, optimizer_key: str) -> dict:
    if optimizer_key not in res_pack:
        raise KeyError(f"Optimizer '{optimizer_key}' not found. Available: {list(res_pack.keys())}")
    sol = res_pack[optimizer_key]
    if isinstance(sol, dict) and "idxs" in sol:
        return sol
    if isinstance(sol, dict):
        for cand in ("best_compromise", "best", "solution", "best_solution"):
            if cand in sol and isinstance(sol[cand], dict) and "idxs" in sol[cand]:
                return sol[cand]
    raise ValueError(f"Could not find 'idxs' for optimizer '{optimizer_key}'.")


def _parse_idxs(idxs) -> List[int]:
    return [int(x) for x in np.asarray(idxs).reshape(-1)]


# -- folium backend (Streamlit-compatible) -----------------------------------


def build_folium_map(
    aoi_poly: Polygon,
    res_k_sweep: dict,
    scenario: str,
    K: int,
    optimizer_key: str,
    grid_size: int = 30,
    n_sensors: int | None = None,
    marker_radius: int = 7,
    tiles: str = "Esri.WorldImagery",
    range_m: float | None = 300.0,
    show_range: bool = True,
    contours: dict | None = None,
):
    """Return a folium.Map with the AOI polygon and selected sensor placements.

    Parameters
    ----------
    range_m
        Sensor coverage radius in meters. When ``show_range`` is True and no
        ``contours`` are provided, a uniform ``folium.Circle`` of this radius
        is drawn around each sensor (FLAT-style fallback). Pass ``None`` or
        set ``show_range=False`` to suppress.
    show_range
        Whether to draw the coverage halos / footprints.
    contours
        Optional ``{scenario: {idx: [(lon, lat), ...closed polygon]}}`` lookup.
        When provided, the per-azimuth ``r_eff`` polygon for each sensor is
        rendered instead of a uniform circle — irregular for DEM and DSM/CHM.
        Build with ``scripts/save_contours.py``.
    """
    import folium

    candidates = regenerate_coverage_grid(aoi_poly, grid_size=grid_size)
    if not candidates:
        raise ValueError("No candidate points generated; check aoi_poly / grid_size.")

    res_pack = res_k_sweep.get(scenario, {}).get(int(K))
    if res_pack is None:
        res_pack = res_k_sweep.get(scenario, {}).get(str(K))
    if res_pack is None:
        raise KeyError(f"No results for scenario={scenario}, K={K}.")

    sol = extract_solution(res_pack, optimizer_key)
    idxs = _parse_idxs(sol["idxs"])
    if n_sensors is not None:
        idxs = idxs[: int(n_sensors)]

    bad = [i for i in idxs if i < 0 or i >= len(candidates)]
    if bad:
        raise ValueError(
            f"Saved idx {bad[0]} out of range for grid_size={grid_size} "
            f"(candidates={len(candidates)}). Use the grid_size from the original run."
        )

    cx, cy = aoi_poly.centroid.x, aoi_poly.centroid.y
    if tiles == "Esri.WorldImagery":
        m = folium.Map(
            location=(cy, cx), zoom_start=13,
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri",
        )
    else:
        m = folium.Map(location=(cy, cx), zoom_start=13, tiles=tiles)

    aoi_latlon = [(c[1], c[0]) for c in aoi_poly.exterior.coords]
    folium.Polygon(
        locations=aoi_latlon, color="#FFFFFF", weight=3, fill=False,
        popup=f"AOI ({scenario})",
    ).add_to(m)

    color = SCENARIO_COLORS.get(scenario, "#4C78A8")
    pretty = OPTIMIZER_PRETTY.get(optimizer_key, optimizer_key)
    show_halo = bool(show_range)

    # Per-scenario contour lookup (irregular footprints) takes precedence over
    # the uniform circle fallback.
    scenario_contours: dict[int, list] = {}
    if isinstance(contours, dict):
        # Try the exact scenario name and a few common aliases.
        for key in (scenario, scenario.replace("/", ""), scenario.replace("/", "_")):
            if key in contours:
                scenario_contours = contours[key] or {}
                break

    for rank, idx in enumerate(idxs, start=1):
        lon, lat = candidates[idx]
        # Coverage footprint
        if show_halo:
            poly = scenario_contours.get(int(idx)) or scenario_contours.get(idx)
            if poly:
                folium.Polygon(
                    locations=[(la, lo) for lo, la in poly],
                    color=color, weight=1.5,
                    fill=True, fill_color=color, fill_opacity=0.12, opacity=0.55,
                ).add_to(m)
            elif range_m is not None and range_m > 0:
                folium.Circle(
                    location=(lat, lon),
                    radius=float(range_m),
                    color=color,
                    weight=1.5,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.10,
                    opacity=0.55,
                ).add_to(m)
        # Centroid dot (radius in pixels)
        popup_extra = ""
        if show_halo and scenario_contours.get(int(idx)):
            popup_extra = "<br>footprint: per-azimuth r_eff"
        elif show_halo and range_m:
            popup_extra = f"<br>range: {range_m:.0f} m (uniform fallback)"
        folium.CircleMarker(
            location=(lat, lon),
            radius=marker_radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.9,
            weight=1,
            popup=folium.Popup(
                f"<b>{scenario} · {pretty} · K={K}</b><br>"
                f"rank: <b>{rank}</b><br>candidate idx: <b>{idx}</b><br>"
                f"({lon:.6f}, {lat:.6f})" + popup_extra,
                max_width=260,
            ),
        ).add_to(m)

    minx, miny, maxx, maxy = aoi_poly.bounds
    m.fit_bounds([(miny, minx), (maxy, maxx)])
    return m


# -- ipyleaflet backend (notebook only) --------------------------------------


def build_ipyleaflet_map(
    aoi_poly: Polygon,
    res_k_sweep: dict,
    grid_size: int = 30,
):
    """Interactive widget viewer (notebook only). Returns a VBox with map+controls."""
    from ipyleaflet import (
        CircleMarker,
        LayerGroup,
        LayersControl,
        Map,
        Polygon as LPolygon,
        WidgetControl,
        basemaps,
    )
    from ipywidgets import HTML, Button, Dropdown, HBox, IntSlider, Layout, Output, VBox

    candidates = regenerate_coverage_grid(aoi_poly, grid_size=grid_size)
    scenarios = list_scenarios(res_k_sweep)
    if not scenarios:
        raise ValueError("No valid scenarios found in res_k_sweep.")

    cx, cy = aoi_poly.centroid.x, aoi_poly.centroid.y
    minx, miny, maxx, maxy = aoi_poly.bounds
    m = Map(center=(cy, cx), zoom=13, basemap=basemaps.Esri.WorldImagery)
    m.add_control(LayersControl(position="topright"))
    m.fit_bounds([(miny, minx), (maxy, maxx)])

    scenario_dd = Dropdown(options=[(s, s) for s in scenarios], value=scenarios[0], description="Scenario:")
    k_dd = Dropdown(description="K:")
    optimizer_dd = Dropdown(description="Optimizer:", layout=Layout(width="240px"))
    n_sensors_sl = IntSlider(value=10, min=1, max=120, description="#Sensors:")
    marker_sl = IntSlider(value=7, min=3, max=16, description="Marker:")
    render_btn = Button(description="Render", button_style="success")
    clear_btn = Button(description="Clear")
    out = Output()

    def _refresh(*_):
        scn = scenario_dd.value
        ks = list_k_values(res_k_sweep, scn)
        k_dd.options = ks
        if ks and k_dd.value not in ks:
            k_dd.value = ks[0]
        if k_dd.value:
            opts = list_optimizers(res_k_sweep, scn, int(k_dd.value))
            optimizer_dd.options = opts
            if opts and optimizer_dd.value not in [v for _, v in opts]:
                optimizer_dd.value = opts[0][1]

    scenario_dd.observe(_refresh, names="value")
    k_dd.observe(_refresh, names="value")
    _refresh()

    def _clear_layers():
        for lyr in [lyr for lyr in m.layers if isinstance(lyr, LayerGroup)]:
            m.remove_layer(lyr)

    def _render(_):
        with out:
            out.clear_output()
            scn = scenario_dd.value
            k = int(k_dd.value)
            opt = optimizer_dd.value
            color = SCENARIO_COLORS.get(scn, "#4C78A8")
            res_pack = res_k_sweep.get(scn, {}).get(k) or res_k_sweep.get(scn, {}).get(str(k))
            if res_pack is None:
                print(f"No results for {scn}, K={k}")
                return
            try:
                sol = extract_solution(res_pack, opt)
            except Exception as e:
                print(f"Error: {e}")
                return
            idxs = _parse_idxs(sol["idxs"])[: int(n_sensors_sl.value)]
            layer = LayerGroup(name=f"{scn} {opt} K={k}")
            layer.add_layer(LPolygon(locations=[(c[1], c[0]) for c in aoi_poly.exterior.coords],
                                     color="#FFFFFF", weight=3, fill=False))
            for rank, idx in enumerate(idxs, start=1):
                if idx < 0 or idx >= len(candidates):
                    continue
                lon, lat = candidates[idx]
                marker = CircleMarker(
                    location=(lat, lon), radius=int(marker_sl.value),
                    color=color, fill_color=color, fill_opacity=0.9, weight=1,
                )
                marker.popup = HTML(f"<b>rank {rank}, idx {idx}</b><br>({lon:.5f}, {lat:.5f})")
                layer.add_layer(marker)
            _clear_layers()
            m.add_layer(layer)
            print(f"Rendered {len(idxs)} sensors")

    render_btn.on_click(_render)
    clear_btn.on_click(lambda _: (_clear_layers(), out.clear_output()))

    controls = VBox([
        HBox([scenario_dd, k_dd, optimizer_dd]),
        HBox([n_sensors_sl, marker_sl]),
        HBox([render_btn, clear_btn]),
        out,
    ])
    return VBox([controls, m])


# -- multi-file loader (drop-in replacement for legacy helper) ---------------


def merge_res_k_sweeps(*res_dicts: dict) -> dict:
    merged: dict = {}
    for d in res_dicts:
        for scenario, k_dict in d.items():
            merged.setdefault(scenario, {})
            merged[scenario].update(k_dict)
    return merged


def load_and_merge(files: Iterable[str | Path]) -> dict:
    """Load one or more saved ``res_k_sweep`` pickles and merge them into one dict."""
    from .persistence import load_ksweep
    chunks = [load_ksweep(p) for p in files]
    if not chunks:
        raise ValueError("No files provided.")
    return merge_res_k_sweeps(*chunks)


__all__ = [
    "regenerate_coverage_grid",
    "list_scenarios",
    "list_k_values",
    "list_optimizers",
    "extract_solution",
    "build_folium_map",
    "build_ipyleaflet_map",
    "merge_res_k_sweeps",
    "load_and_merge",
    "SCENARIO_COLORS",
    "OPTIMIZER_PRETTY",
]
