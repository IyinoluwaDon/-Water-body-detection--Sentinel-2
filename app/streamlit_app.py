"""Water body detection explorer. Reads the small files in app/data/ written by the notebook's Step 8."""

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

DATA = Path(os.environ.get("WATERDET_APP_DATA", Path(__file__).parent / "data"))
st.set_page_config(page_title="Water body detection", layout="wide")

if not (DATA / "summary.json").exists():
    st.error("No results found in app/data/. Run the notebook to Step 8, then copy the contents of "
             "outputs/streamlit_data/ into app/data/.")
    st.stop()


@st.cache_data
def load_json(name):
    return json.loads((DATA / name).read_text(encoding="utf-8"))


@st.cache_data
def load_image(name):
    return Image.open(DATA / name).convert("RGBA")


def f(v, nd=1):
    """Format a number, or n/a when the pipeline left it blank."""
    return "n/a" if v is None else f"{v:.{nd}f}"


summary, meta = load_json("summary.json"), load_json("meta.json")
periods, labels = meta["periods"], meta["period_labels"]
primary = meta["primary_mode"]
model = summary["modes"][primary]


def blend(period, opacity, with_water=True):
    base = load_image(f"rgb_{period}.png")
    if not with_water:
        return base
    over = load_image(f"water_{period}.png").copy()
    over.putalpha(over.getchannel("A").point(lambda a: int(a * opacity)))
    return Image.alpha_composite(base, over)


st.title("Water body detection from satellite images")
st.caption(f"Sentinel-2 imagery, NDWI/MNDWI features and a Decision Tree. Study area: {meta['aoi_name'].replace('_', ' ')} "
           f"{meta['bbox']} (west, south, east, north).")

tabs = st.tabs(["Overview", "Map explorer", "Seasonal change", "How the model decides", "Accuracy and limits", "Data"])

with tabs[0]:
    cols = st.columns(len(periods) + (1 if meta["has_change"] else 0))
    for col, p in zip(cols, periods):
        a = summary["areas"][p]
        col.metric(f"Water, {p} season", f"{f(a.get('water_km2'))} km²", f"{f(a.get('water_pct_of_valid'))}% of mapped area", delta_color="off")
    if meta["has_change"]:
        ch = next(iter(summary["change"].values()))
        gain, loss = ch.get("water gain (land -> water)"), ch.get("water loss (water -> land)")
        net = "n/a" if gain is None or loss is None else f"{gain - loss:+.2f} km²"
        cols[-1].metric("Net water change", net, f"+{f(gain, 2)} gained / -{f(loss, 2)} lost", delta_color="off")
    st.markdown(
        "**What this does.** The satellite measures light in several colours. Water absorbs near-infrared and shortwave-infrared light "
        "and reflects green, so two simple ratios (NDWI and MNDWI) stand out over water. A Decision Tree learns the yes/no rules that "
        "separate water from land, is tested on areas it never saw, and is then used to draw a map for each season.")
    first = periods[0]
    c1, c2 = st.columns(2)
    c1.image(blend(first, 0, False), caption="What the satellite sees")
    c2.image(blend(first, 0.7), caption="Water found by the model (blue)")

with tabs[1]:
    c = st.columns([1, 1, 2])
    period = c[0].radio("Season", periods, format_func=lambda p: labels[p])
    opacity = c[1].slider("Water overlay strength", 0, 100, 70) / 100
    side = c[2].checkbox("Show satellite image and detection side by side", value=False)
    if side:
        a, b = st.columns(2)
        a.image(blend(period, 0, False), caption="Satellite image")
        b.image(blend(period, opacity), caption="With detected water")
    else:
        st.image(blend(period, opacity), caption=labels[period])
    a = summary["areas"][period]
    st.write(f"Water: **{f(a.get('water_km2'), 2)} km²** ({f(a.get('water_pct_of_valid'))}% of mapped area). "
             f"Grey areas ({f(a.get('nodata_pct_of_aoi'))}% of the study area) had no clear satellite image and are not classified.")

with tabs[2]:
    if not meta["has_change"]:
        st.info("Only one season was available, so there is no change map.")
    else:
        ch = next(iter(summary["change"].values()))
        base_p = st.radio("Background", periods, format_func=lambda p: labels[p], horizontal=True)
        over = load_image("change.png")
        st.image(Image.alpha_composite(load_image(f"rgb_{base_p}.png"), over), caption="Green: water gained. Orange: water lost.")
        df = pd.DataFrame({"km²": {k: v for k, v in ch.items() if k != "compared_km2"}})
        st.bar_chart(df.fillna(0))
        st.dataframe(df.style.format("{:.2f}", na_rep="n/a"))
        st.caption("Changes of a few pixels along shorelines can be noise. Treat small values with care.")

with tabs[3]:
    st.write(f"The final tree has **{model['tree_leaves']} leaves** and depth **{model['tree_depth']}**. Each line below is a rule.")
    if (DATA / "tree_rules.txt").exists():
        st.code((DATA / "tree_rules.txt").read_text(encoding="utf-8"))
    c1, c2 = st.columns(2)
    if (DATA / "tree.png").exists():
        c1.image(str(DATA / "tree.png"), caption="The decision tree")
    if (DATA / "cv_depth_curve.png").exists():
        c2.image(str(DATA / "cv_depth_curve.png"), caption="How tree size was chosen (tested on unseen areas)")
    fi = pd.DataFrame({"importance": model["feature_importance"]})
    st.bar_chart(fi)

with tabs[4]:
    t = model["test_spatial_holdout"]
    st.subheader("Agreement with the JRC Global Surface Water map, on areas the model never saw")
    rows = {"Decision Tree": t}
    rows.update(model.get("baselines_on_same_test_points", {}))
    keep = ["overall_accuracy", "kappa", "water_precision", "water_recall", "water_f1", "water_iou"]
    df = pd.DataFrame({k: {m: v.get(m) for m in keep} for k, v in rows.items()}).T.dropna(axis=1, how="all")
    st.dataframe(df.style.format("{:.3f}", na_rep="n/a"))
    st.info("Important: these scores compare the model with a global map (JRC, built from 1984 to 2021 Landsat at 30 m), not with the ground. "
            "They show the model is consistent with that map. They are not proof the map matches reality today.")
    ref = summary.get("reference_validation")
    if ref and ref.get("area_weighted"):
        st.subheader("Independent reference validation")
        st.write(f"{ref.get('n_used')} points labelled by looking at satellite images.")
        st.json(ref["methods"])
    else:
        st.caption("No independent reference points were used. Labelling about 200 points gives a true accuracy figure.")
    for n in summary.get("notes", []):
        st.caption(f"Note: {n}")

with tabs[5]:
    st.write("Downloads: the water maps are GeoTIFF files you can open in QGIS (0 land, 1 water, 255 no data).")
    for p in periods:
        f = DATA / f"water_{p}.tif"
        if f.exists():
            st.download_button(f"Water map, {p} season (GeoTIFF)", f.read_bytes(), file_name=f.name)
    if (DATA / "change.tif").exists():
        st.download_button("Change map (GeoTIFF)", (DATA / "change.tif").read_bytes(), file_name="change.tif")
    if (DATA / "key_numbers.csv").exists():
        st.download_button("Key numbers (CSV)", (DATA / "key_numbers.csv").read_bytes(), file_name="key_numbers.csv")
    st.caption("Data: Sentinel-2 (Copernicus/ESA), Cloud Score+ (Google), JRC Global Surface Water v1.4 (Pekel et al. 2016).")
