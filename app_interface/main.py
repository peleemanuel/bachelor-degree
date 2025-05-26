import os
import sys
import uuid
from pathlib import Path

from nicegui import ui
from fastapi.responses import FileResponse

# add this import
import folium

# --- imports for TraceCreator and its dependencies ---
from gps_info import GPSInfo, DroneType
from image import get_drone
from trace_creating import TraceCreator  # your refactored class

# Directory to dump uploads (and exported maps)
BASE_UPLOAD = Path('uploads')
BASE_UPLOAD.mkdir(exist_ok=True)

# In-memory list of TraceCreator instances
traces: list[TraceCreator] = []

# Pre-select a drone for all traces
drone = get_drone(DroneType.DJI_MINI_4K)

def render_folium(map_obj):
    """Convert Folium Map to UI HTML."""
    iframe = map_obj.get_root()._repr_html_()
    return ui.html(iframe).classes('w-full h-full')

def create_map():
    """Build folium.Map and layer all traces."""
    m = folium.Map(location=[45.9432, 24.9668], zoom_start=7,
                   width='100%', height='100%')            # full screen :contentReference[oaicite:10]{index=10}
    m.get_root().width  = '100%'
    m.get_root().height = '100%'

    for tc in traces:
        # draw polyline via Folium
        folium.PolyLine(
            locations=tc.trace,
            color='blue', weight=3, opacity=0.7
        ).add_to(m)                                         # Folium PolyLine 

        # draw unified polygon via Folium
        poly_coords = [(lat, lon) for lon, lat in tc.unified_polygon.exterior.coords]
        folium.Polygon(
            locations=poly_coords,
            color='green', fill=True, fill_opacity=0.2
        ).add_to(m)
    return m

@ui.page('/')
def index():
    with ui.element().classes('w-full h-screen flex flex-col'):
        # display the Folium map iframe
        render_folium(create_map())

    # controls
    with ui.row().classes('p-4 bg-gray-100'):
            folder_input = ui.input('Folder path', placeholder='Path to image folder')\
                              .style('width: 300px')
            ui.button('Add Trace from Folder',
                      on_click=lambda: handle_folder(folder_input.value))
            ui.button('Refresh Map',
                      on_click=lambda: render_folium(create_map()))

def handle_folder(folder_path: str):
    if os.path.isdir(folder_path):
        tc = TraceCreator(folder_path, drone)
        tc.generate_informations(); tc.generate_trace()
        tc.calculate_points_direction(); tc.get_coords_for_each_image()
        tc.get_one_polygon()
        traces.append(tc)
    else:
        ui.notify(f"'{folder_path}' is not a valid directory.", color='negative')

def export_folium_map():
    """
    Build a standalone Folium map with ALL traces and save it to HTML.
    """
    # 1) create a new Folium map centered on Romania
    m = folium.Map(location=[45.9432, 24.9668], zoom_start=7)
    # 2) layer each trace onto it using our helper
    for tc in traces:
        m = tc.get_trace_map(m)
    # 3) save to disk
    out_path = BASE_UPLOAD / 'all_traces_map.html'
    m.save(str(out_path))
    ui.notify(f'Folium map exported to {out_path}', color='positive')


@ui.page('/all_traces_map.html')
def serve_exported_map():
    """
    Serve the exported Folium HTML
    """
    return FileResponse(BASE_UPLOAD / 'all_traces_map.html')


@ui.page('/image/{filename}')
def serve_image(filename):
    return FileResponse(BASE_UPLOAD / filename)


# Allow multiprocessing guard for NiceGUI
if __name__ in {"__main__", "__mp_main__"}:
    ui.run(title='Trace Dashboard', native=True, reload=False)
