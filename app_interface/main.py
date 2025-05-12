from nicegui import ui
from fastapi.responses import FileResponse
import os

# Store uploads
UPLOAD_DIR = 'uploads'
os.makedirs(UPLOAD_DIR, exist_ok=True)

# In-memory list of (lat, lng, filename)
photos = []

@ui.page('/')
def dashboard():
    # Layout: map full-width, upload panel overlayed
    with ui.element().classes('w-full relative'):
        # Center map on initial coords
        leaflet = ui.leaflet(center=(0, 0), zoom=2)  # :contentReference[oaicite:11]{index=11}

        # Overlay upload button top-right
        ui.upload(
            on_upload=lambda files: handle_upload(files, leaflet)
        ).classes('absolute top-2 right-2 z-[1000] bg-white p-2 rounded')

def handle_upload(files, leaflet):
    for file in files:
        # Save file
        path = os.path.join(UPLOAD_DIR, file.name)
        with open(path, 'wb') as f:
            f.write(file.content)
        # Simulate geolocation: click at marker coords or use metadata
        lat, lng = 51.5, -0.09  # replace with real coords
        photos.append((lat, lng, file.name))
        # Add a marker with an HTML popup containing the image
        leaflet.marker(lat, lng).bind_popup(
            f'<img src="/image/{file.name}" style="max-width:200px;" />'
        )

# Serve uploaded images
@ui.page('/image/{filename}')
def serve_image(filename):
    return FileResponse(os.path.join(UPLOAD_DIR, filename))

ui.run(title='Photo Dashboard', native=True)
