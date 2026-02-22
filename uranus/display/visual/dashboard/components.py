import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

def render_system_form(system_id, pi_options, sub_options, name="", desc=""):
    return dbc.Card([
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    dbc.Label("System Name"),
                    dbc.Input(id={"type": "sys-name", "index": system_id}, type="text", value=name),
                ], width=5),
                dbc.Col([
                    dbc.Label("System Description"),
                    dbc.Input(id={"type": "sys-desc", "index": system_id}, type="text", value=desc),
                ], width=5),
                dbc.Col([
                    dbc.ButtonGroup([
                        dbc.Button(html.I(className="bi bi-trash"),      id={"type": "remove-sys",    "index": system_id}, color="danger", size="sm"),
                    ], className="mt-4")
                ], width=2)
            ]),
            html.Div(id={"type": "subsystems-container", "index": system_id}, className="ms-4 mt-3", children=[]),
            dbc.Button("Add Subsystem", id={"type": "add-sub-btn", "index": system_id}, size="sm", color="info", className="mt-2 ms-4"),
        ])
    ], id={"type": "sys-card", "index": system_id}, className="mb-3 border-primary")

def render_subsystem_form(sub_id, pi_options, sub_options, name="", desc="", parents=[], children=[]):
    return dbc.Card([
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    dbc.Label("Subsystem Name"),
                    dbc.Input(id={"type": "sub-name", "index": sub_id}, type="text", value=name),
                ], width=5),
                dbc.Col([
                    dbc.Label("Subsystem Description"),
                    dbc.Input(id={"type": "sub-desc", "index": sub_id}, type="text", value=desc),
                ], width=5),
                dbc.Col([
                    dbc.ButtonGroup([
                        dbc.Button(html.I(className="bi bi-trash"),      id={"type": "remove-sub",    "index": sub_id}, color="danger", size="sm"),
                    ], className="mt-4")
                ], width=2)
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Label("Parents"),
                    dcc.Dropdown(id={"type": "sub-parents", "index": sub_id}, options=sub_options, multi=True, value=parents)
                ], width=6),
                dbc.Col([
                    dbc.Label("Children"),
                    dcc.Dropdown(id={"type": "sub-children", "index": sub_id}, options=sub_options, multi=True, value=children)
                ], width=6),
            ], className="mb-2"),
            html.Div(id={"type": "sub-sensors-container", "index": sub_id}, className="ms-4 mt-2", children=[]),
            dbc.Button("Add Sensor to Subsystem", id={"type": "add-sub-sensor-btn", "index": sub_id}, size="sm", color="warning", className="mt-2 ms-4"),
        ])
    ], id={"type": "sub-card", "index": sub_id}, className="mb-2 border-info")

def render_sensor_form(sensor_id, pi_options, name="", tag="", desc="", pi_id=None):
    return dbc.Card([
        dbc.CardBody([
            dbc.Row([
                dbc.Col([dbc.Label("Sensor Name"), dbc.Input(id={"type": "sensor-name", "index": sensor_id}, type="text", value=name)], width=3),
                dbc.Col([dbc.Label("Tag"), dbc.Input(id={"type": "sensor-tag", "index": sensor_id}, type="text", value=tag)], width=3),
                dbc.Col([dbc.Label("PI Model"), dcc.Dropdown(id={"type": "sensor-pi", "index": sensor_id}, options=pi_options, value=pi_id)], width=3),
                dbc.Col([
                    dbc.ButtonGroup([
                        dbc.Button(html.I(className="bi bi-trash"),      id={"type": "remove-sensor",    "index": sensor_id}, color="danger", size="sm"),
                    ], className="mt-4")
                ], width=3)
            ]),
            dbc.Label("Description"),
            dbc.Input(id={"type": "sensor-desc", "index": sensor_id}, type="text", value=desc)
        ])
    ], id={"type": "sensor-card", "index": sensor_id}, className="mb-2 bg-light")
