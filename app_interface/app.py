# app.py
import streamlit as st
from streamlit_folium import folium_static
import folium

from gps_info import DroneType
from image import get_drone
from trace_creating import TraceCreator  # your module

st.set_page_config(page_title="Flight Trace Viewer", layout="wide")

# --- Sidebar for new flight input ---
st.sidebar.header("New Flight Info")
folder = st.sidebar.text_input("Image Folder Path:")
if st.sidebar.button("Load New Flight Info"):
    if folder:
        try:
            # Initialize TraceCreator
            drone = get_drone(DroneType.DJI_MINI_4K)
            tc = TraceCreator(folder, drone)
            tc.generate_informations()
            tc.generate_trace()
            tc.calculate_points_direction()
            tc.get_coords_for_each_image()
            tc.get_one_polygon()

            # Store in session for persistence
            st.session_state['trace_creator'] = tc
            st.success("Flight loaded successfully!")
        except Exception as e:
            st.error(f"Error: {e}")

# --- Main area: display map ---
st.title("📍 Flight Trace Map")

# Create base map centered on last loaded trace or default
if 'trace_creator' in st.session_state:
    tc = st.session_state['trace_creator']
    # Center on first GPS point
    center = tc.trace[0] if tc.trace else (0, 0)
else:
    center = (0, 0)

m = folium.Map(location=center, zoom_start=15)

# If a flight is loaded, overlay its trace & polygon
if 'trace_creator' in st.session_state:
    tc.get_trace_map(m)

# Render the map in Streamlit
folium_static(m)
