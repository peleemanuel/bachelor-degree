# app.py
import streamlit as st
from streamlit_folium import folium_static
import folium

from trace_viewer.drones import DroneRegistry, DroneSpec, DroneType
from trace_viewer.trace import TraceCreator
from trace_viewer.overlapping_algo import compare_overlapping_zones

import matplotlib.pyplot as plt

def _plot_gray(arr):
    fig, ax = plt.subplots()
    ax.imshow(arr, cmap="gray")
    ax.axis("off")
    return fig

st.set_page_config(page_title="Flight Trace Manager", layout="wide")

# Initialize session state
if "traces" not in st.session_state:
    # folder_path -> TraceCreator
    st.session_state.traces = {}

# --- Sidebar: Drone Configuration ---
st.sidebar.header("Drone Configuration")
spec_names = list(DroneRegistry._specs.keys())
choice     = st.sidebar.selectbox("Select Drone Type", spec_names)
spec = DroneRegistry.get(choice)
st.sidebar.markdown(
    f"**Spec:**\n"
    f"- Sensor width (mm): {spec.sensor_width_mm}\n"
    f"- Focal length (mm): {spec.focal_length_mm}"
)


# allow registering a brand-new spec
st.sidebar.subheader("➕ Add New Drone Spec")
new_name   = st.sidebar.text_input("Name (unique)")
new_sens   = st.sidebar.number_input("Sensor width (mm)",   value=spec.sensor_width_mm)
new_focal  = st.sidebar.number_input("Focal length (mm)",   value=spec.focal_length_mm)
if st.sidebar.button("Register Drone"):
    if not new_name:
        st.sidebar.error("Please give it a non‐empty name.")
    elif new_name in DroneRegistry._specs:
        st.sidebar.error("Name already exists.")
    else:
        DroneRegistry.register(DroneSpec(new_name, new_sens, new_focal))
        st.sidebar.success(f"Registered '{new_name}'")
        # update the select‐list
        spec_names.append(new_name)

# --- Sidebar: Add New Flight ---
st.sidebar.header("Load New Flight")
new_folder = st.sidebar.text_input("Image Folder Path")
cases = []
if st.sidebar.button("➕ Add Flight") and new_folder:
    try:
        # 1) Build the new trace
        tc_new = TraceCreator(new_folder, spec)
        tc_new.build()

        # 2) Clip overlaps against all existing traces
        removed_from = []
        for path, tc_old in list(st.session_state.traces.items()):

            new_cases = compare_overlapping_zones(tc_old, tc_new)
            cases.extend(new_cases)

            if tc_old.overlaps(tc_new):
                pct = tc_old.remove_overlap(tc_new)
                removed_from.append((path, pct))
                # if the old trace is now <5%, drop it
                if tc_old.total_area_percent < 5.0:
                    st.sidebar.info(f"Removing '{path}' (only {tc_old.total_area_percent:.1f}% left)")
                    st.session_state.traces.pop(path)

        # 3) Finally store the new trace
        st.session_state.traces[new_folder] = tc_new

        # 4) Feedback to user
        st.success(f"Loaded '{new_folder}'")
        if removed_from:
            msgs = [f"– clipped {p:.1f}% from {fld!r}" for fld, p in removed_from]
            st.warning("Overlap adjustments:\n" + "\n".join(msgs))

    except Exception as e:
        st.error(f"Failed to load '{new_folder}': {e}")

# --- Sidebar: Manage Loaded Flights ---
st.sidebar.header("Manage Flights")
to_delete = st.sidebar.multiselect(
    "Select flights to remove",
    options=list(st.session_state.traces.keys())
)
if st.sidebar.button("🗑️ Delete Selected"):
    for f in to_delete:
        st.session_state.traces.pop(f, None)
    st.sidebar.success(f"Removed {len(to_delete)} flight(s).")

# --- Main Map Display ---
st.title("📍 Flight Trace Map")

# Determine center: first trace or Romania
if st.session_state.traces:
    first_tc = next(iter(st.session_state.traces.values()))
    center = first_tc.center()
else:
    center = (45.9432, 24.9668)  # Romania centroid

m = folium.Map(location=center, zoom_start=13)

# Draw all the traces (already clipped as needed)
for folder, tc in st.session_state.traces.items():
    tc.add_to_map(m)
    folium.map.Marker(
        location=tc.center(),
        icon=folium.DivIcon(html=f"""<div style="font-size:12px;color:black">{folder}</div>""")
    ).add_to(m)

folium_static(m)
for idx, case in enumerate(cases):
    st.markdown(f"### Overlap candidate #{idx+1}")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(case["crop1"], caption="Overlap Crop 1")
        st.write("Mask 1")
        st.pyplot(_plot_gray(case["mask1"]))  # helper to plot a gray image
    with col2:
        st.image(case["crop2"], caption="Overlap Crop 2")
        st.write("Mask 2")
        st.pyplot(_plot_gray(case["mask2"]))
    with col3:
        st.write("Filtered Difference")
        st.pyplot(_plot_gray(case["diff"]))

    # ask the user
    key = f"confirm_{idx}"
    confirm = st.radio(
        "Does this represent a real change?",
        ("Yes", "No"),
        key=key
    )
    if confirm == "Yes":
        st.success(f"Change confirmed at lat={case['lat']:.6f}, lon={case['lon']:.6f}")
    else:
        st.info("Change rejected.")