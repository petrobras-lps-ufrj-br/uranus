import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from models import SessionLocal, Product, System, Subsystem, Sensor, PI, DRIVER_OPTIONS
from .components import render_system_form, render_subsystem_form, render_sensor_form

try:
    import dash_cytoscape as cyto
except ImportError:
    cyto = None

def build_product_graph(p, systems, subsystems, sensors):
    if not cyto:
        return dbc.Alert("dash-cytoscape not installed.", color="warning")
    
    elements = []
    # Root: Product
    product_node = f"prod-{p.product_id}"
    elements.append({'data': {'id': product_node, 'label': p.name, 'type': 'product'}})

    for sys in systems:
        sys_node = f"sys-{sys.system_id}"
        elements.append({'data': {'id': sys_node, 'label': sys.name, 'type': 'system'}})
        elements.append({'data': {'source': product_node, 'target': sys_node}})

        # Filter subsystems for this system
        current_subs = [s for s in subsystems if s.system_id == sys.system_id]
        for sub in current_subs:
            sub_node = f"sub-{sub.subsystem_id}"
            elements.append({'data': {'id': sub_node, 'label': sub.name, 'type': 'subsystem'}})
            elements.append({'data': {'source': sys_node, 'target': sub_node}})

            # Subsystem to Subsystem relationships (parents/children are IDs)
            for child_id in sub.children:
                elements.append({'data': {'source': sub_node, 'target': f"sub-{child_id}", 'rel': 'sub-child'}})
            
            # Sensors for this subsystem
            current_sensors = [s for s in sensors if s.subsystem_id == sub.subsystem_id]
            for s in current_sensors:
                sensor_node = f"sensor-{s.sensor_id}"
                elements.append({'data': {'id': sensor_node, 'label': s.name, 'type': 'sensor'}})
                elements.append({'data': {'source': sub_node, 'target': sensor_node}})

    # Stylesheet for cytoscape
    stylesheet = [
        {'selector': 'node', 'style': {'label': 'data(label)', 'color': 'white', 'text-outline-width': 2, 'text-outline-color': '#888'}},
        {'selector': '[type = "product"]', 'style': {'background-color': '#007bff', 'shape': 'rectangle', 'width': 60, 'height': 60}},
        {'selector': '[type = "system"]', 'style': {'background-color': '#28a745', 'shape': 'ellipse'}},
        {'selector': '[type = "subsystem"]', 'style': {'background-color': '#ffc107', 'shape': 'diamond'}},
        {'selector': '[type = "sensor"]', 'style': {'background-color': '#dc3545', 'shape': 'triangle', 'width': 20, 'height': 20}},
        {'selector': 'edge', 'style': {'width': 2, 'line-color': '#ccc', 'target-arrow-color': '#ccc', 'target-arrow-shape': 'triangle', 'curve-style': 'bezier'}}
    ]

    return cyto.Cytoscape(
        id={'type': 'product-graph', 'index': p.product_id},
        layout={'name': 'breadthfirst', 'directed': True, 'padding': 10},
        style={'width': '100%', 'height': '400px', 'background-color': '#f8f9fa'},
        elements=elements,
        stylesheet=stylesheet
    )

def render_list_products(requests_pathname_prefix):
    db = SessionLocal()
    products = db.query(Product).all()
    # Manual hierarchical fetching
    all_systems = db.query(System).all()
    all_subsystems = db.query(Subsystem).all()
    all_sensors = db.query(Sensor).all()
    
    # Pre-map all subsystems for name resolution
    sub_map = {s.subsystem_id: s.name for s in all_subsystems}
    db.close()
    
    cards = []
    for p in products:
        sys_items = []
        p_systems = [s for s in all_systems if s.product_id == p.product_id]
        
        for sys in p_systems:
            sub_list = []
            sys_subs = [s for s in all_subsystems if s.system_id == sys.system_id]
            
            for sub in sys_subs:
                sub_sensors_list = [s for s in all_sensors if s.subsystem_id == sub.subsystem_id]
                sub_sensors_html = [html.Li(f"Sensor: {s.name} (Tag: {s.tag})") for s in sub_sensors_list]
                
                rel_info = []
                if sub.parents:
                    p_names = [sub_map.get(pid, pid) for pid in sub.parents]
                    rel_info.append(html.Div([
                        dbc.Badge("Parents", color="info", className="me-1"),
                        html.Span(", ".join(p_names), className="text-muted small")
                    ], className="mb-1"))
                if sub.children:
                    c_names = [sub_map.get(cid, cid) for cid in sub.children]
                    rel_info.append(html.Div([
                        dbc.Badge("Children", color="success", className="me-1"),
                        html.Span(", ".join(c_names), className="text-muted small")
                    ], className="mb-1"))

                sub_list.append(html.Li([
                    html.Div([
                        html.Strong(f"Subsystem: {sub.name}"),
                        html.Div(rel_info, className="ms-3 mt-1") if rel_info else None,
                    ], className="mb-1"),
                    html.Ul(sub_sensors_html, className="mt-1") if sub_sensors_html else html.P("No sensors", className="text-muted small ms-3")
                ], className="mb-3"))
            
            sys_items.append(dbc.AccordionItem([
                html.P(sys.description, className="text-muted"),
                html.H6("Subsystems"),
                html.Ul(sub_list) if sub_list else html.P("No subsystems", className="text-muted small"),
            ], title=f"System: {sys.name}"))

        hierarchy_list_view = dbc.Accordion(sys_items, start_collapsed=True) if sys_items else html.P("No systems defined.", className="text-muted")
        # Extract only relevant ones for graph view
        p_subsystems = [s for s in all_subsystems if s.system_id in [sys.system_id for sys in p_systems]]
        p_sensors = [s for s in all_sensors if s.subsystem_id in [sub.subsystem_id for sub in p_subsystems]]
        hierarchy_graph_view = build_product_graph(p, p_systems, p_subsystems, p_sensors)

        tabs_view = dbc.Tabs([
            dbc.Tab(hierarchy_list_view, label="List View", tab_id="list", label_style={"cursor": "pointer"}),
            dbc.Tab(html.Div(hierarchy_graph_view, className="p-2 border rounded bg-light"), label="Graph View", tab_id="graph", label_style={"cursor": "pointer"}),
        ], active_tab="list")

        cards.append(dbc.Card([
            dbc.CardHeader([
                dbc.Row([
                    dbc.Col(html.H4(p.name, className="mb-0 text-primary"), width=8),
                    dbc.Col([
                        dbc.Button([html.I(className="bi bi-pencil-square"), " Edit"], 
                                  href=f"{requests_pathname_prefix}edit-product/{p.product_id}",
                                  color="primary", size="sm", className="me-2 shadow-sm"),
                        dbc.Button([html.I(className="bi bi-trash"), " Delete"], 
                                  id={"type": "delete-product-btn", "index": p.product_id},
                                  color="danger", size="sm", className="shadow-sm"),
                    ], width=4, className="text-end")
                ], align="center")
            ]),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.P([html.Strong("Location: "), (p.location_type or "N/A").capitalize()]),
                        html.P([html.Strong("Country: "), p.country]),
                    ], width=4),
                    dbc.Col([
                        html.P([html.Strong("State: "), p.state]),
                        html.P([html.Strong("Number/ID: "), p.number]),
                    ], width=4),
                    dbc.Col([
                        html.P([html.Strong("Description: "), p.description]),
                    ], width=4),
                ]),
                html.Hr(),
                html.H5([html.I(className="bi bi-diagram-3 me-2"), "Hierarchy Breakdown"]),
                tabs_view
            ]),
            dbc.CardFooter([
                dbc.Row([
                    dbc.Col([
                        html.Small([
                            html.Strong("Catalog Summary: "),
                            f"{len(p_systems)} Systems, ",
                            f"{len(p_subsystems)} Subsystems, ",
                            f"{len(p_sensors)} Sensors"
                        ], className="text-muted")
                    ])
                ])
            ], className="bg-light border-top-0")
        ], className="mb-4 shadow-sm border-0"))

    return html.Div([
        html.H3("Products Catalog", className="mb-4"),
        html.Div(id="delete-product-status", style={"display": "none"}),
        html.Div(cards) if cards else dbc.Alert("No products found.", color="info")
    ])

def render_list_pis(requests_pathname_prefix):
    db = SessionLocal()
    pis = db.query(PI).all()
    db.close()
    
    rows = []
    for pi in pis:
        rows.append(html.Tr([
            html.Td(pi.pi_id),
            html.Td(pi.name),
            html.Td(pi.driver),
            html.Td(pi.description),
            html.Td(str(pi.configuration)),
            html.Td(dbc.Button([html.I(className="bi bi-pencil-square")], 
                              href=f"{requests_pathname_prefix}edit-pi/{pi.pi_id}",
                              color="warning", size="sm"))
        ]))

    table = dbc.Table([
        html.Thead(html.Tr([
            html.Th("PI ID"), html.Th("Name"), html.Th("Driver"), 
            html.Th("Description"), html.Th("Config"), html.Th("Action")
        ])),
        html.Tbody(rows)
    ], bordered=True, hover=True, striped=True)

    return html.Div([
        html.H3("PI Entities List"),
        table
    ])

def render_add_pi(pi_id=None):
    initial_name = ""
    initial_driver = "cognite"
    initial_desc = ""
    initial_project = ""
    initial_token = ""
    title = "Add New PI Model"

    if pi_id:
        db = SessionLocal()
        pi = db.query(PI).filter(PI.pi_id == pi_id).first()
        if pi:
            initial_name = pi.name
            initial_driver = pi.driver
            initial_desc = pi.description
            config = pi.configuration or {}
            initial_project = config.get("project_name", "")
            initial_token = config.get("token", "")
            title = f"Edit PI Model: {pi.name}"
        db.close()

    return dbc.Form([
        dcc.Input(id="pi-id", type="hidden", value=pi_id if pi_id else ""),
        html.H3(title),
        dbc.Label("Name"),
        dbc.Input(id="pi-name", type="text", placeholder="Enter PI Name", value=initial_name),
        dbc.Label("Driver"),
        dcc.Dropdown(
            id="pi-driver",
            options=[{"label": opt.capitalize(), "value": opt} for opt in DRIVER_OPTIONS],
            value=initial_driver
        ),
        dbc.Label("Description"),
        dbc.Input(id="pi-desc", type="text", placeholder="Enter PI Description", value=initial_desc),
        
        html.Div(id="cognite-config-section", children=[
            dbc.Label("Project Name"),
            dbc.Input(id="pi-cognite-project", type="text", placeholder="e.g. my-project", value=initial_project),
            dbc.Label("Token"),
            dbc.Input(id="pi-cognite-token", type="password", placeholder="Bearer Token...", value=initial_token),
        ], style={"display": "none"}),
        
        html.Div(id="not-implemented-msg", children=[
            html.P("driver not implement yet", className="text-warning mt-2 fw-bold")
        ], style={"display": "none"}),
        
        dbc.Button("Save PI", id="save-pi-btn", color="primary", className="mt-3"),
        html.Div(id="save-pi-status", className="mt-2")
    ])

def render_add_product(prod_id=None):
    initial_name = ""
    initial_desc = ""
    initial_location = "on-shore"
    initial_country = ""
    initial_state = ""
    initial_number = ""
    initial_systems_nodes = []
    title = "Add New Product"

    db = SessionLocal()
    pis = db.query(PI).all()
    pi_options = [{"label": pi.name, "value": pi.pi_id} for pi in pis]
    
    all_subs = db.query(Subsystem).all()
    sub_options = [{"label": s.name, "value": s.subsystem_id} for s in all_subs]

    if prod_id:
        prod = db.query(Product).filter(Product.product_id == prod_id).first()
        if prod:
            initial_name = prod.name
            initial_desc = prod.description
            initial_location = prod.location_type
            initial_country = prod.country
            initial_state = prod.state
            initial_number = prod.number
            title = f"Edit Product: {prod.name}"
            
            # Manual hierarchical fetch for existing product
            systems = db.query(System).filter(System.product_id == prod.product_id).all()
            for sys in systems:
                sys_id = sys.system_id
                sys_node = render_system_form(sys_id, pi_options, sub_options, sys.name, sys.description)
                
                sub_nodes = []
                subsystems = db.query(Subsystem).filter(Subsystem.system_id == sys.system_id).all()
                for sub in subsystems:
                    # Enforce hierarchical ID for UI components even if DB has old format
                    if '/' not in sub.subsystem_id:
                        sub_id_for_ui = f"{sys_id}/{sub.subsystem_id}"
                    else:
                        sub_id_for_ui = sub.subsystem_id
                        
                    sub_node_form = render_subsystem_form(sub_id_for_ui, pi_options, sub_options, sub.name, sub.description, 
                                                   sub.parents, sub.children)
                    
                    s_nodes = []
                    sensors = db.query(Sensor).filter(Sensor.subsystem_id == sub.subsystem_id).all()
                    for s in sensors:
                        # Enforce hierarchical sensor ID
                        if '/' not in s.sensor_id:
                            sensor_id_for_ui = f"{sub_id_for_ui}/{s.sensor_id}"
                        else:
                            sensor_id_for_ui = s.sensor_id
                        
                        s_nodes.append(render_sensor_form(sensor_id_for_ui, pi_options, s.name, s.tag, s.description, s.pi_id))
                    
                    sub_node_form.children[0].children[2].children = s_nodes
                    sub_nodes.append(sub_node_form)
                
                sys_node.children[0].children[1].children = sub_nodes
                initial_systems_nodes.append(sys_node)
    db.close()

    return html.Div([
        dcc.Input(id="prod-id", type="hidden", value=prod_id if prod_id else ""),
        html.H3(title),
        
        dbc.Card([
            dbc.CardBody([
                dbc.Label("Product Name"),
                dbc.Input(id="prod-name", type="text", className="mb-3", value=initial_name),
                dbc.Label("Product Description"),
                dbc.Textarea(id="prod-desc", className="mb-3", value=initial_desc),

                dbc.Row([
                    dbc.Col([
                        dbc.Label("Location Type"),
                        dbc.RadioItems(
                            options=[{"label": "On-shore", "value": "on-shore"}, {"label": "Off-shore", "value": "off-shore"}],
                            value=initial_location, id="prod-location-type", inline=True, className="mb-3"
                        ),
                    ]),
                    dbc.Col([
                        dbc.Label("Country"), dbc.Input(id="prod-country", type="text", className="mb-3", value=initial_country),
                    ]),
                ]),
                dbc.Row([
                    dbc.Col([
                        dbc.Label("State"), dbc.Input(id="prod-state", type="text", className="mb-3", value=initial_state),
                    ]),
                    dbc.Col([
                        dbc.Label("Number / ID"), dbc.Input(id="prod-number", type="text", className="mb-3", value=initial_number),
                    ]),
                ]),
                
                html.H5("Systems"),
                html.Div(id="systems-container", children=initial_systems_nodes),
                dbc.Button("Add System", id="add-system-btn", color="secondary", className="mt-2 text-white"),
                
                html.Hr(),
                dbc.Button("Save Product Hierarchy", id="save-product-btn", color="success", className="mt-3 me-2"),
                html.Div(id="save-product-status")
            ])
        ])
    ])

def render_display_page():
    db = SessionLocal()
    products = db.query(Product).all()
    db.close()
    
    product_options = [{"label": p.name, "value": p.product_id} for p in products]
    
    return html.Div([
        html.H3("Display Console"),
        dbc.Tabs([
            dbc.Tab(label="Sensor Selection", tab_id="display-tab-selection", children=[
                dbc.Card([
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Step 1: Select Product"),
                                dcc.Dropdown(
                                    id="display-product-dropdown",
                                    options=product_options,
                                    placeholder="Choose a product to see sensors..."
                                ),
                            ], width=6),
                        ], className="mb-4"),
                        html.Div(id="display-sensors-table-container")
                    ])
                ], className="border-top-0")
            ]),
            dbc.Tab(label="Graph Visualization", tab_id="display-tab-graph", children=[
                dbc.Card([
                    dbc.CardBody([
                        html.Div(id="display-graph-content", children=[
                            dbc.Alert("Please select sensors to generate graphs.", color="info", className="mt-3")
                        ])
                    ])
                ], className="border-top-0")
            ]),
        ], id="display-tabs", active_tab="display-tab-selection", className="mt-3")
    ])

def render_admin_page():
    return html.Div([
        html.H3("Administration & Data Management", className="mb-4"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("Export Data", className="mb-0")),
                    dbc.CardBody([
                        html.P("Download the entire database state (Products, PIs, Systems, etc.) as a JSON file.", className="text-muted"),
                        dbc.Button([html.I(className="bi bi-download me-2"), "Export to JSON"], id="export-full-db-btn", color="primary"),
                        dcc.Download(id="download-full-db-json")
                    ])
                ], className="mb-4 shadow-sm")
            ], width=6),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("Import Data", className="mb-0")),
                    dbc.CardBody([
                        html.P("Upload a JSON file to restore or add data to the database. We will check for existing IDs.", className="text-muted"),
                        dcc.Upload(
                            id="upload-full-db-json",
                            children=html.Div(["Drag and Drop or ", html.A("Select JSON File")]),
                            style={
                                'width': '100%', 'height': '60px', 'lineHeight': '60px',
                                'borderWidth': '1px', 'borderStyle': 'dashed',
                                'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px 0'
                            },
                            multiple=False
                        ),
                        html.Div(id="import-full-db-status")
                    ])
                ], className="mb-4 shadow-sm")
            ], width=6)
        ]),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("Maintenance", className="mb-0 text-danger")),
                    dbc.CardBody([
                        html.P("Dangerous: This will permanently delete all Products, PIs, Systems, and Sensors from the database.", className="text-muted"),
                        dbc.Button([html.I(className="bi bi-exclamation-triangle-fill me-2"), "Clear Entire Database"], 
                                  id="clear-db-btn", color="danger", className="mt-2"),
                        html.Div(id="clear-db-status", className="mt-2")
                    ])
                ], className="shadow-sm border-danger")
            ], width=12)
        ]),
        
        # Confirmation Modal for clearing DB
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("Confirm Clear Database")),
            dbc.ModalBody("Are you absolutely sure you want to clear EVERYTHING? This action cannot be undone."),
            dbc.ModalFooter([
                dbc.Button("Cancel", id="cancel-clear-btn", className="ms-auto", n_clicks=0),
                dbc.Button("Yes, Clear All", id="confirm-clear-btn", color="danger", n_clicks=0),
            ]),
        ], id="clear-db-modal", is_open=False),
    ])
