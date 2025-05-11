from nicegui import ui

# Home page
@ui.page('/', title='My App')
def main_page():
    ui.button('Upload', on_click=lambda: ui.open('/upload'))
    ui.button('Map',    on_click=lambda: ui.open('/map'))

# Upload page
@ui.page('/upload')
def upload_page():
    ui.label('Upload Context').classes('text-2xl')
    ui.button('Back', on_click=lambda: ui.open('/'))

# Map page
@ui.page('/map')
def map_page():
    ui.label('Map Context').classes('text-2xl')
    ui.button('Back', on_click=lambda: ui.open('/'))

# Launch as a native window without auto-reload
ui.run(native=True)
