# app.py
import streamlit as st
from streamlit_folium import folium_static
import folium

from trace_viewer.drones import get_drone, DroneType
from trace_viewer.trace import TraceCreator

def load_trace(folder: str) -> TraceCreator:
    drone = get_drone(DroneType.DJI_MINI_4K)
    tc = TraceCreator(folder, drone)
    tc.build_all()   # see trace.py suggestion
    return tc

st.sidebar.header("New Flight Info")
folder = st.sidebar.text_input("Image Folder Path:")
if st.sidebar.button("Load New…") and folder:
    try:
        st.session_state.tc = load_trace(folder)
        st.success("Loaded!")
    except Exception as e:
        st.error(e)

st.title("📍 Flight Trace Map")
center = st.session_state.tc.center() if "tc" in st.session_state else (0,0)
m = folium.Map(location=center, zoom_start=15)
if "tc" in st.session_state:
    st.session_state.tc.add_to_map(m)
folium_static(m)
