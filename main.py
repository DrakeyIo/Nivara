"""NIVARA - Accessibility Intelligence Dashboard

Run: streamlit run main.py
Keep best.pt and the assets/ folder next to this file.
"""
import base64
import os
import sys
from pathlib import Path

import folium
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image
from streamlit_folium import st_folium

PROJECT_DIR = Path(__file__).resolve().parent
ASSETS_DIR = PROJECT_DIR / "assets"
LOGO = (
    next(
        (
            path
            for path in sorted(ASSETS_DIR.iterdir())
            if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        ),
        None,
    )
    if ASSETS_DIR.is_dir()
    else None
)
TRAIN_DIR = Path(
    os.environ.get("YOLO_TRAIN_DIR", str(Path(sys.prefix).parent / "training"))
).expanduser().resolve()
SAMPLE_DIR = TRAIN_DIR / "dataset" / "test" / "images"
MODEL_PATH = PROJECT_DIR / "best.pt"

st.set_page_config(
    page_title="NIVARA",
    page_icon=LOGO if LOGO else "♿",
    layout="wide",
)

CITIES = {
    "Mumbai": (19.0760, 72.8777),
    "Kolkata": (22.5726, 88.3639),
    "Delhi": (28.6139, 77.2090),
    "Bengaluru": (12.9716, 77.5946),
}
BARRIERS = ["Construction", "Encroachment", "Parked vehicle", "Damaged footpath", "Waterlogging"]
GROUPS = ["Wheelchair users", "Visually impaired", "Elderly", "Children"]
IMPACT = pd.DataFrame(
    [
        [0.9, 0.8, 0.7, 0.6],
        [0.8, 0.9, 0.6, 0.5],
        [0.7, 0.8, 0.5, 0.6],
        [1.0, 0.9, 0.9, 0.7],
        [0.9, 0.6, 0.8, 0.8],
    ],
    index=BARRIERS,
    columns=GROUPS,
)
CLASS_TO_BARRIER = {
    "car": "Parked vehicle",
    "motobike": "Parked vehicle",
    "motorbike": "Parked vehicle",
    "motorcycle": "Parked vehicle",
    "street vendor": "Encroachment",
    "advertisement board": "Encroachment",
    "garbage bag": "Encroachment",
    "construction material": "Construction",
}
BANDS = ["Low", "Medium", "High", "Critical"]
BAND_COLORS = {"Low": "#2ecc71", "Medium": "#f1c40f", "High": "#e67e22", "Critical": "#e74c3c"}

st.markdown(
    """
<style>
.block-container{padding-top:2rem;max-width:1200px}
.hero{display:flex;align-items:center;gap:16px;margin-bottom:.2rem}
.hero h1{margin:0;font-size:2.2rem;padding:0}
.hero p{margin:0;color:#6b7280}
.steps{color:#6b7280;margin:.2rem 0 1.2rem}
</style>
""",
    unsafe_allow_html=True,
)
if LOGO:
    mime = "image/png" if LOGO.suffix.lower() == ".png" else "image/jpeg"
    logo_html = (
        f'<img src="data:{mime};base64,{base64.b64encode(LOGO.read_bytes()).decode()}" '
        'alt="NIVARA logo" style="height:56px;border-radius:10px">'
    )
else:
    logo_html = '<span aria-hidden="true" style="font-size:2.2rem">♿</span>'
st.markdown(
    f'<div class="hero">{logo_html}<div><h1>NIVARA</h1>'
    f"<h6>By ERROR 404: Winner Not Found</h6></div></div>"
    f"<p>AI-powered accessibility intelligence: from data to action</p></div></div>"
    f'<div class="steps">Detect barriers &nbsp;→&nbsp; Map them &nbsp;→&nbsp; '
    f'Rank by who is affected &nbsp;→&nbsp; Fix the worst first</div>',
    unsafe_allow_html=True,
)


@st.cache_data
def load_data(city: str, n: int = 150) -> pd.DataFrame:
    rng = np.random.default_rng(sorted(CITIES).index(city) + 1)
    lat0, lon0 = CITIES[city]
    return pd.DataFrame(
        {
            "id": [f"NVR-{i:04d}" for i in range(1, n + 1)],
            "type": rng.choice(BARRIERS, n, p=[0.2, 0.25, 0.2, 0.2, 0.15]),
            "lat": lat0 + rng.normal(0, 0.03, n),
            "lon": lon0 + rng.normal(0, 0.03, n),
            "severity": rng.integers(1, 6, n),
            "footfall": rng.integers(1, 6, n),
            "days_open": rng.integers(0, 60, n),
            "status": rng.choice(["Open", "In progress", "Resolved"], n, p=[0.55, 0.25, 0.2]),
        }
    )


def priority(df: pd.DataFrame, groups: list[str]) -> pd.Series:
    """Score barriers by severity, group impact, footfall, and days unresolved."""
    weight = IMPACT.loc[df["type"], groups].mean(axis=1).values
    score = df["severity"] * weight * (0.5 + df["footfall"] / 5) * (
        1 + np.minimum(df["days_open"], 30) / 30
    )
    return (score / 15 * 100).round(1)


@st.cache_resource
def load_model():
    if MODEL_PATH.is_file():
        from ultralytics import YOLO

        return YOLO(str(MODEL_PATH))
    return None


city_col, group_col = st.columns(2)
city = city_col.selectbox("City", list(CITIES))
choice = group_col.selectbox("Prioritise for", ["Everyone"] + GROUPS)
groups = GROUPS if choice == "Everyone" else [choice]

df = load_data(city)
df = df[df["status"] != "Resolved"].copy()
df["priority"] = priority(df, groups)
df["band"] = pd.cut(df["priority"], [-1, 20, 35, 50, 101], labels=BANDS).astype(str)
burden = pd.Series(
    {group: (df["severity"] * df["type"].map(IMPACT[group])).sum() for group in GROUPS}
)

kpis = [
    ("Open barriers", int(len(df))),
    ("Critical", int((df["band"] == "Critical").sum())),
    ("Avg days unresolved", round(float(df["days_open"].mean()), 1)),
    ("Hit hardest", burden.idxmax()),
]
for col, (label, value) in zip(st.columns(4), kpis):
    with col.container(border=True):
        st.metric(label, value)

tab_map, tab_ai, tab_queue = st.tabs(["🗺️ City map", "📷 Detect a barrier", "🚨 Fix first"])

with tab_map:
    map_view = folium.Map(location=CITIES[city], zoom_start=12, tiles="OpenStreetMap")
    for row in df.itertuples():
        folium.CircleMarker(
            [row.lat, row.lon],
            radius=4 + row.severity * 1.5,
            color=BAND_COLORS[row.band],
            fill=True,
            fill_opacity=0.75,
            tooltip=f"{row.type} · priority {row.priority}",
        ).add_to(map_view)
    st_folium(map_view, height=500, width=1100, returned_objects=[])
    st.caption("🟢 Low  🟡 Medium  🟠 High  🔴 Critical · larger dot = more severe · demo data")

with tab_ai:
    st.write("Upload a street photo, or try a sample. NIVARA finds barriers and names their type.")
    upload = st.file_uploader("Street photo", type=["jpg", "jpeg", "png"])
    samples = (
        sorted(
            (
                path
                for path in SAMPLE_DIR.iterdir()
                if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
            ),
            key=lambda path: path.name.lower(),
        )[:12]
        if SAMPLE_DIR.is_dir()
        else []
    )
    selected_sample = None
    if samples:
        selected_sample = st.selectbox(
            "...or try a sample",
            [None] + samples,
            format_func=lambda path: "Choose a sample" if path is None else path.name,
        )

    source = upload if upload is not None else selected_sample
    if source is not None:
        image = Image.open(source).convert("RGB")
        model = load_model()
        if model is None:
            st.image(image, alt="Uploaded street scene")
            st.info("Add best.pt next to main.py to turn on detection.")
        else:
            result = model(image, conf=0.15)[0]
            left, right = st.columns([3, 2])
            left.image(
                np.ascontiguousarray(result.plot()[..., ::-1]),
                alt="Street image with detected objects",
            )
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                right.success("No barriers detected.")
            else:
                names = [str(model.names[int(class_id)]).strip() for class_id in boxes.cls]
                barriers = [
                    CLASS_TO_BARRIER.get(name.casefold(), "Other") for name in names
                ]
                counts = pd.Series(barriers).value_counts()
                right.markdown("**Found:** " + ", ".join(f"{count}× {kind}" for kind, count in counts.items()))
                right.dataframe(
                    pd.DataFrame(
                        {
                            "Detected": names,
                            "Barrier type": barriers,
                            "Confidence": [f"{confidence:.0%}" for confidence in boxes.conf.tolist()],
                        }
                    ),
                    hide_index=True,
                    alt="Detected object names, barrier categories, and confidence scores",
                )


def explain_priority(row: pd.Series) -> str:
    most_affected = IMPACT.loc[row["type"], groups].idxmax()
    return f"Hits {most_affected.lower()} hardest · open {row['days_open']} days"


with tab_queue:
    top = df.sort_values("priority", ascending=False).head(10).copy()
    top["Why"] = top.apply(explain_priority, axis=1)
    top.insert(0, "Rank", range(1, len(top) + 1))
    left, right = st.columns([3, 2])

    left.subheader("Fix these first")
    left.dataframe(
        top[["Rank", "type", "Why", "priority"]].rename(
            columns={"type": "Barrier", "priority": "Priority"}
        ),
        hide_index=True,
        alt="Barriers ranked by priority with explanations",
        column_config={
            "Priority": st.column_config.ProgressColumn(
                "Priority", min_value=0, max_value=100, format="%.0f"
            )
        },
    )
    left.download_button(
        "⬇️ Export list (CSV)",
        top[["Rank", "type", "Why", "priority", "lat", "lon"]].to_csv(index=False),
        "nivara_fix_first.csv",
        "text/csv",
    )

    right.subheader("Who is hit hardest?")
    burden_index = (
        (burden / burden.max() * 100)
        .round()
        .rename_axis("Group")
        .reset_index(name="Impact")
        .sort_values("Impact")
    )
    figure = px.bar(burden_index, x="Impact", y="Group", orientation="h")
    figure.update_layout(
        height=300,
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis_title="Impact index",
        yaxis_title=None,
    )
    right.plotly_chart(figure, key="burden", alt="Impact index by mobility group")

    with st.expander("How is priority calculated?"):
        st.write(
            "Severity × how much the barrier affects the chosen group × foot traffic "
            "× days left unresolved."
        )

st.caption("Prototype · map and ranking use simulated data · detection uses your trained YOLO model.")
