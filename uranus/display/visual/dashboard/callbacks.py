import dash
from dash import html, dcc, Input, Output, State, ALL, MATCH, callback_context, no_update
import dash_bootstrap_components as dbc
from models import SessionLocal, Product, System, Subsystem, Sensor, PI, DRIVER_OPTIONS, generate_id
import json
import uuid
import base64
from .components import render_system_form, render_subsystem_form, render_sensor_form

def register_dashboard_callbacks(dash_app):
    
    @dash_app.callback(
        [Output("cognite-config-section", "style"),
         Output("not-implemented-msg", "style"),
         Output("save-pi-btn", "disabled")],
        Input("pi-driver", "value")
    )
    def toggle_driver_fields(driver):
        if driver == "cognite":
            return {"display": "block"}, {"display": "none"}, False
        else:
            return {"display": "none"}, {"display": "block"}, True

    @dash_app.callback(
        Output("save-pi-status", "children"),
        Input("save-pi-btn", "n_clicks"),
        State("pi-id", "value"),
        State("pi-name", "value"),
        State("pi-driver", "value"),
        State("pi-desc", "value"),
        State("pi-cognite-project", "value"),
        State("pi-cognite-token", "value")
    )
    def save_pi(n_clicks, pi_id, name, driver, desc, cognite_project, cognite_token):
        if not n_clicks: return ""
        try:
            config = {}
            if driver == "cognite":
                config = {
                    "project_name": cognite_project,
                    "token": cognite_token
                }
            
            db = SessionLocal()
            if pi_id:
                pi = db.query(PI).filter(PI.pi_id == pi_id).first()
                if pi:
                    pi.name = name
                    pi.driver = driver
                    pi.description = desc
                    pi.configuration = config
                else:
                    db.close()
                    return dbc.Alert("PI not found for update!", color="danger")
            else:
                new_pi = PI(pi_id=generate_id(), name=name, driver=driver, description=desc, configuration=config)
                db.add(new_pi)
            
            db.commit()
            db.close()
            return dbc.Alert(f"PI '{name}' saved successfully!", color="success")
        except Exception as e:
            if 'db' in locals(): db.close()
            return dbc.Alert(f"Error: {str(e)}", color="danger")

    @dash_app.callback(
        Output("systems-container", "children"),
        [Input("add-system-btn", "n_clicks"),
         Input({"type": "remove-sys", "index": ALL}, "n_clicks")],
        State("systems-container", "children"),
        prevent_initial_call=True
    )
    def manage_systems(add_clicks, remove_clicks, current_systems):
        if current_systems is None: current_systems = []
        ctx = callback_context
        if not ctx.triggered: return current_systems
        trigger_id = ctx.triggered_id
        
        if trigger_id == "add-system-btn":
            try:
                db = SessionLocal()
                pis = db.query(PI).all()
                all_subs = db.query(Subsystem).all()
                db.close()
                pi_options = [{"label": pi.name, "value": pi.pi_id} for pi in pis]
                sub_options = [{"label": s.name, "value": s.subsystem_id} for s in all_subs]
                new_id = generate_id()
                current_systems.append(render_system_form(new_id, pi_options, sub_options))
                return current_systems
            except Exception as e:
                print(f"DEBUG Error in manage_systems (add): {e}")
                return current_systems

        try:
            idx = trigger_id['index']
            t_type = trigger_id['type']
            pos = -1
            for i, child in enumerate(current_systems):
                if isinstance(child, dict) and 'props' in child and 'id' in child['props'] and child['props']['id'] == {"type": "sys-card", "index": idx}:
                    pos = i
                    break
            if pos == -1: return current_systems

            if t_type == "remove-sys":
                current_systems.pop(pos)
        except Exception as e:
            print(f"DEBUG Error in manage_systems (action): {e}")
        return current_systems

    @dash_app.callback(
        Output({"type": "subsystems-container", "index": MATCH}, "children"),
        [Input({"type": "add-sub-btn", "index": MATCH}, "n_clicks"),
         Input({"type": "remove-sub", "index": ALL}, "n_clicks")],
        State({"type": "subsystems-container", "index": MATCH}, "children"),
        prevent_initial_call=True
    )
    def manage_subsystems(add_clicks, r_clicks, current_subs):
        if current_subs is None: current_subs = []
        ctx = callback_context
        if not ctx.triggered: return current_subs
        trigger_id = ctx.triggered_id
        
        # In pattern matching, triggered_id index for the MATCH element is the system_id
        if trigger_id.get('type') == "add-sub-btn":
            try:
                system_id = trigger_id['index']
                new_sub_id = f"{system_id}/{generate_id()}"
                db = SessionLocal()
                pis = db.query(PI).all()
                all_subs = db.query(Subsystem).all()
                db.close()
                pi_options = [{"label": pi.name, "value": pi.pi_id} for pi in pis]
                sub_options = [{"label": s.name, "value": s.subsystem_id} for s in all_subs]
                current_subs.append(render_subsystem_form(new_sub_id, pi_options, sub_options))
                return current_subs
            except Exception as e:
                print(f"DEBUG Error in manage_subsystems (add): {e}")
                return current_subs

        try:
            idx = trigger_id['index']
            t_type = trigger_id['type']
            pos = -1
            for i, child in enumerate(current_subs):
                if isinstance(child, dict) and 'props' in child and 'id' in child['props'] and child['props']['id'] == {"type": "sub-card", "index": idx}:
                    pos = i
                    break
            if pos == -1: return current_subs

            if t_type == "remove-sub":
                current_subs.pop(pos)
        except Exception as e:
            print(f"DEBUG Error in manage_subsystems (action): {e}")
        return current_subs

    @dash_app.callback(
        Output({"type": "sub-sensors-container", "index": MATCH}, "children"),
        [Input({"type": "add-sub-sensor-btn", "index": MATCH}, "n_clicks"),
         Input({"type": "remove-sensor", "index": ALL}, "n_clicks")],
        State({"type": "sub-sensors-container", "index": MATCH}, "children"),
        prevent_initial_call=True
    )
    def manage_sensors(add_sub_clicks, r_clicks, sub_sensors):
        if sub_sensors is None: sub_sensors = []
        ctx = callback_context
        if not ctx.triggered: return no_update
        trigger_id = ctx.triggered_id
        
        db = SessionLocal()
        pis = db.query(PI).all()
        db.close()
        pi_options = [{"label": pi.name, "value": pi.pi_id} for pi in pis]
        
        if trigger_id.get('type') == "add-sub-sensor-btn":
            subsystem_id = trigger_id['index']
            new_sensor_id = f"{subsystem_id}/{generate_id()}"
            sub_sensors.append(render_sensor_form(new_sensor_id, pi_options))
            return sub_sensors

        try:
            idx = trigger_id['index']
            t_type = trigger_id['type']
            for i, s in enumerate(sub_sensors):
                if isinstance(s, dict) and 'props' in s and 'id' in s['props'] and s['props']['id'] == {"type": "sensor-card", "index": idx}:
                    if t_type == "remove-sensor":
                        sub_sensors.pop(i)
                    return sub_sensors
        except Exception as e:
            print(f"DEBUG Error in manage_sensors (action): {e}")
        return no_update

    @dash_app.callback(
        [Output({"type": "sub-parents", "index": ALL}, "options"),
         Output({"type": "sub-children", "index": ALL}, "options")],
        [Input({"type": "sub-name", "index": ALL}, "value"),
         Input({"type": "sub-name", "index": ALL}, "id")],
        State({"type": "sub-parents", "index": ALL}, "options"),
        prevent_initial_call=False
    )
    def update_sub_options(sub_names, sub_ids, current_options):
        new_options = []
        for name, id_dict in zip(sub_names, sub_ids):
            label = name if name else f"Unnamed Subsystem ({id_dict['index']})"
            new_options.append({"label": label, "value": id_dict['index']})
        
        db = SessionLocal()
        all_db_subs = db.query(Subsystem).all()
        db.close()
        for s in all_db_subs:
            if not any(opt['value'] == s.subsystem_id for opt in new_options):
                new_options.append({"label": f"DB: {s.name}", "value": s.subsystem_id})
        return [new_options] * len(sub_ids), [new_options] * len(sub_ids)

    @dash_app.callback(
        Output("save-product-status", "children"),
        Input("save-product-btn", "n_clicks"),
        State("prod-id", "value"),
        State("prod-name", "value"),
        State("prod-desc", "value"),
        State("prod-location-type", "value"),
        State("prod-country", "value"),
        State("prod-state", "value"),
        State("prod-number", "value"),
        State({"type": "sys-name", "index": ALL}, "value"),
        State({"type": "sys-name", "index": ALL}, "id"),
        State({"type": "sys-desc", "index": ALL}, "value"),
        State({"type": "sub-name", "index": ALL}, "value"),
        State({"type": "sub-name", "index": ALL}, "id"),
        State({"type": "sub-desc", "index": ALL}, "value"),
        State({"type": "sub-parents", "index": ALL}, "value"),
        State({"type": "sub-children", "index": ALL}, "value"),
        State({"type": "sensor-name", "index": ALL}, "value"),
        State({"type": "sensor-name", "index": ALL}, "id"),
        State({"type": "sensor-tag", "index": ALL}, "value"),
        State({"type": "sensor-desc", "index": ALL}, "value"),
        State({"type": "sensor-pi", "index": ALL}, "value"),
    )
    def save_entire_product(n_clicks, 
                            prod_id, 
                            p_name, 
                            p_desc, 
                            p_location, 
                            p_country, 
                            p_state, 
                            p_number, 
                            sys_names, 
                            sys_ids, 
                            sys_descs, 
                            sub_names, 
                            sub_ids, 
                            sub_descs, 
                            sub_parents_vals, 
                            sub_children_vals,
                            s_names, 
                            s_ids, 
                            s_tags, 
                            s_descs, 
                            s_pis
        ):
        
        if not n_clicks: return ""
        try:
            db = SessionLocal()
            if prod_id:
                product = db.query(Product).filter(Product.product_id == prod_id).first()
                if not product:
                    db.close()
                    return dbc.Alert("Product not found!", color="danger")
                product.name = p_name
                product.description = p_desc
                product.location_type = p_location
                product.country = p_country
                product.state = p_state
                product.number = p_number
                
                # Manual cascade equivalent: delete systems first
                db.query(System).filter(System.product_id == product.product_id).delete()
            else:
                product = Product(
                    product_id=generate_id(), name=p_name, description=p_desc, location_type=p_location,
                    country=p_country, state=p_state, number=p_number
                )
                db.add(product)
            db.flush()

            system_map = {}
            for name, sid_dict, desc in zip(sys_names, sys_ids, sys_descs):
                sid = sid_dict['index']
                print( f"System: {name}, ID: {sid}, Description: {desc}")
                system = System(system_id=sid, name=name, description=desc, product_id=product.product_id)
                db.add(system)
                system_map[sid] = system
            db.flush()

            subsystem_map = {}
            for name, subid_dict, desc, parents, children in zip(sub_names, sub_ids, sub_descs, sub_parents_vals, sub_children_vals):
                sid = subid_dict['index']
                # Determine which system this subsystem belongs to
                # If hierarchical, the prefix before the FIRST slash is the system_id
                sys_id = sid.split('/')[0] if '/' in sid else None
                
                print( f"Subsystem: {name}, ID: {sid}, Description: {desc}, Parents: {parents}, Children: {children}, ParentSystem: {sys_id}")
                
                if sys_id and sys_id in system_map:
                    sub = Subsystem(
                        subsystem_id=sid, name=name, description=desc, system_id=sys_id,
                        parents=parents or [], children=children or []
                    )
                    db.add(sub)
                    subsystem_map[sid] = sub
                else:
                    print(f"WARNING: Subsystem {name} skipped because parent system {sys_id} not found in current session.")

            db.flush()
            for name, s_id_dict, tag, desc, pi_id in zip(s_names, s_ids, s_tags, s_descs, s_pis):
                sid = s_id_dict['index']
                # Hierarchical index: system_id/subsystem_id/sensor_id
                # So the parent subsystem ID is the combination of the first two parts
                parts = sid.split('/')
                if len(parts) >= 3:
                    sub_id = "/".join(parts[:2])
                elif len(parts) == 2:
                    sub_id = parts[0]
                else:
                    sub_id = None
                
                print(f"Sensor: {name}, ID: {sid}, ParentSubsystem: {sub_id}")
                
                if sub_id and sub_id in subsystem_map:
                    sensor = Sensor(
                        sensor_id=sid, name=name, tag=tag, description=desc, 
                        pi_id=pi_id, subsystem_id=sub_id
                    )
                    db.add(sensor)
                else:
                    print(f"WARNING: Sensor {name} skipped because parent subsystem {sub_id} not found.")

            db.commit()
            db.close()
            return dbc.Alert("Hierarchy saved successfully!", color="success")
        except Exception as e:
            print(f"ERROR Saving: {e}")
            if 'db' in locals(): db.close()
            return dbc.Alert(f"Error: {str(e)}", color="danger")

    @dash_app.callback(
        Output("delete-product-status", "children"),
        Input({"type": "delete-product-btn", "index": ALL}, "n_clicks"),
        prevent_initial_call=True
    )
    def delete_product(n_clicks):
        ctx = callback_context
        if not ctx.triggered or not any(n_clicks): return no_update
        trigger_id = ctx.triggered_id
        prod_id = trigger_id['index']
        try:
            db = SessionLocal()
            product = db.query(Product).filter(Product.product_id == prod_id).first()
            if product:
                db.delete(product)
                db.commit()
            db.close()
            return dcc.Location(id="refresh-catalog", href=dash_app.config.requests_pathname_prefix, refresh=True)
        except Exception as e:
            if 'db' in locals(): db.close()
            return dbc.Alert(f"Delete failed: {e}", color="danger")

    @dash_app.callback(
        Output("display-sensors-table-container", "children"),
        Input("display-product-dropdown", "value"),
        prevent_initial_call=True
    )
    def update_display_sensors_table(prod_id):
        if not prod_id: return ""
        db = SessionLocal()
        product = db.query(Product).filter(Product.product_id == prod_id).first()
        if not product:
            db.close()
            return dbc.Alert("Product not found.", color="danger")
        
        systems = db.query(System).filter(System.product_id == product.product_id).all()
        
        # Mappings for name resolution
        all_subs = db.query(Subsystem).all()
        sub_map = {s.subsystem_id: s.name for s in all_subs}
        all_pis = db.query(PI).all()
        pi_map = {p.pi_id: p.name for p in all_pis}

        rows = []
        for sys in systems:
            subsystems = db.query(Subsystem).filter(Subsystem.system_id == sys.system_id).all()
            for sub in subsystems:
                sensors = db.query(Sensor).filter(Sensor.subsystem_id == sub.subsystem_id).all()
                for s in sensors:
                    p_names = [sub_map.get(pid, pid) for pid in sub.parents]
                    c_names = [sub_map.get(cid, cid) for cid in sub.children]
                    pi_name = pi_map.get(s.pi_id, s.pi_id) if s.pi_id else "N/A"

                    rows.append(html.Tr([
                        html.Td(dbc.Checkbox(id={"type": "display-sensor-checkbox", "index": s.sensor_id})),
                        html.Td(sub.name),
                        html.Td(", ".join(p_names) if p_names else "-"),
                        html.Td(", ".join(c_names) if c_names else "-"),
                        html.Td(s.description),
                        html.Td(pi_name),
                        html.Td(s.tag),
                    ]))
        db.close()
        if not rows: return dbc.Alert("No sensors found for this product.", color="warning")
        
        table = dbc.Table([
            html.Thead(html.Tr([
                html.Th("Select"), html.Th("Subsystem"), html.Th("Parents"), html.Th("Children"),
                html.Th("Description"), html.Th("PI Name"), html.Th("Sensor Tag"),
            ])),
            html.Tbody(rows)
        ], bordered=True, hover=True, striped=True)
        
        return html.Div([
            html.H5(f"Sensors for {product.name}"),
            table,
            dbc.Button("Apply Selection", id="apply-display-btn", color="primary", className="mt-3"),
            html.Div(id="apply-display-status", className="mt-2")
        ])

    @dash_app.callback(
        [Output("apply-display-status", "children"),
         Output("display-graph-content", "children"),
         Output("display-tabs", "active_tab")],
        Input("apply-display-btn", "n_clicks"),
        State({"type": "display-sensor-checkbox", "index": ALL}, "value"),
        State({"type": "display-sensor-checkbox", "index": ALL}, "id"),
        prevent_initial_call=True
    )
    def apply_sensor_selection(n_clicks, values, ids):
        if not n_clicks: return no_update, no_update, no_update
        selected_ids = [id_dict['index'] for val, id_dict in zip(values, ids) if val if isinstance(id_dict, dict)]
        if not selected_ids:
            return dbc.Alert("No sensors selected.", color="warning"), no_update, "display-tab-selection"
        db = SessionLocal()
        sensors = db.query(Sensor).filter(Sensor.sensor_id.in_(selected_ids)).all()
        db.close()
        
        graph_elements = [
            html.H4(f"Visualization: {len(sensors)} Sensors Selected", className="mb-4"),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(f"Sensor: {s.name}"),
                        dbc.CardBody([
                            html.P(f"Tag: {s.tag}", className="text-muted small"),
                            dcc.Graph(
                                figure={
                                    'data': [{'x': [1, 2, 3], 'y': [4, 1, 2], 'type': 'line', 'name': s.tag}],
                                    'layout': {'title': f'Live Data: {s.name}', 'height': 300}
                                }
                            )
                        ])
                    ], className="mb-3")
                ], width=6) for s in sensors
            ])
        ]
        return dbc.Alert(f"Generated visualizations for {len(sensors)} sensors.", color="success"), html.Div(graph_elements), "display-tab-graph"

    # --- Admin Callbacks ---
    
    @dash_app.callback(
        Output("clear-db-modal", "is_open"),
        [Input("clear-db-btn", "n_clicks"), Input("cancel-clear-btn", "n_clicks"), Input("confirm-clear-btn", "n_clicks")],
        [State("clear-db-modal", "is_open")],
    )
    def toggle_clear_modal(n_open, n_cancel, n_confirm, is_open):
        if n_open or n_cancel or n_confirm:
            return not is_open
        return is_open

    @dash_app.callback(
        Output("clear-db-status", "children"),
        Input("confirm-clear-btn", "n_clicks"),
        prevent_initial_call=True
    )
    def clear_database(n_clicks):
        if not n_clicks: return ""
        try:
            db = SessionLocal()
            db.query(Sensor).delete()
            db.query(Subsystem).delete()
            db.query(System).delete()
            db.query(Product).delete()
            db.query(PI).delete()
            db.commit()
            db.close()
            return dbc.Alert("Database cleared successfully!", color="success")
        except Exception as e:
            if 'db' in locals(): db.close()
            return dbc.Alert(f"Clear failed: {e}", color="danger")

    @dash_app.callback(
        Output("download-full-db-json", "data"),
        Input("export-full-db-btn", "n_clicks"),
        prevent_initial_call=True
    )
    def export_all_data(n_clicks):
        if not n_clicks: return no_update
        
        db = SessionLocal()
        products = db.query(Product).all()
        pis = db.query(PI).all()
        systems = db.query(System).all()
        subsystems = db.query(Subsystem).all()
        sensors = db.query(Sensor).all()
        db.close()
        
        data = {
            "version": "1.0",
            "pis": [
                {
                    "pi_id": pi.pi_id,
                    "name": pi.name,
                    "driver": pi.driver,
                    "description": pi.description,
                    "configuration": pi.configuration
                } for pi in pis
            ],
            "products": [
                {
                    "product_id": p.product_id,
                    "name": p.name,
                    "description": p.description,
                    "location_type": p.location_type,
                    "country": p.country,
                    "state": p.state,
                    "number": p.number
                } for p in products
            ],
            "systems": [
                {
                    "system_id": s.system_id,
                    "product_id": s.product_id,
                    "name": s.name,
                    "description": s.description
                } for s in systems
            ],
            "subsystems": [
                {
                    "subsystem_id": sub.subsystem_id,
                    "system_id": sub.system_id,
                    "name": sub.name,
                    "description": sub.description,
                    "parents": sub.parents,
                    "children": sub.children
                } for sub in subsystems
            ],
            "sensors": [
                {
                    "sensor_id": s.sensor_id,
                    "subsystem_id": s.subsystem_id,
                    "pi_id": s.pi_id,
                    "name": s.name,
                    "description": s.description,
                    "tag": s.tag
                } for s in sensors
            ]
        }
        
        return dict(content=json.dumps(data, indent=2), filename="full_db_export.json")

    @dash_app.callback(
        Output("import-full-db-status", "children"),
        Input("upload-full-db-json", "contents"),
        State("upload-full-db-json", "filename"),
        prevent_initial_call=True
    )
    def import_all_data(contents, filename):
        if not contents: return ""
        
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        try:
            data = json.loads(decoded.decode('utf-8'))
            db = SessionLocal()
            
            collisions = []
            
            # 1. PIs
            for pi_data in data.get('pis', []):
                if db.query(PI).filter(PI.pi_id == pi_data['pi_id']).first():
                    collisions.append(f"PI ID {pi_data['pi_id']}")
                else:
                    db.add(PI(
                        pi_id=pi_data['pi_id'], name=pi_data['name'], 
                        driver=pi_data['driver'], description=pi_data['description'],
                        configuration=pi_data['configuration']
                    ))
            
            # 2. Products
            for p_data in data.get('products', []):
                if db.query(Product).filter(Product.product_id == p_data['product_id']).first():
                    collisions.append(f"Product ID {p_data['product_id']}")
                else:
                    db.add(Product(
                        product_id=p_data['product_id'], name=p_data['name'],
                        description=p_data['description'], location_type=p_data['location_type'],
                        country=p_data['country'], state=p_data['state'], number=p_data['number']
                    ))
            
            # Flush to satisfy following FKs
            db.flush() 
            
            # 3. Systems (Depends on Products)
            for s_data in data.get('systems', []):
                if db.query(System).filter(System.system_id == s_data['system_id']).first():
                    collisions.append(f"System ID {s_data['system_id']}")
                else:
                    db.add(System(
                        system_id=s_data['system_id'], product_id=s_data['product_id'],
                        name=s_data['name'], description=s_data['description']
                    ))
            
            db.flush()

            # 4. Subsystems (Depends on Systems)
            for sub_data in data.get('subsystems', []):
                if db.query(Subsystem).filter(Subsystem.subsystem_id == sub_data['subsystem_id']).first():
                    collisions.append(f"Subsystem ID {sub_data['subsystem_id']}")
                else:
                    db.add(Subsystem(
                        subsystem_id=sub_data['subsystem_id'], system_id=sub_data['system_id'],
                        name=sub_data['name'], description=sub_data['description'],
                        parents=sub_data['parents'], children=sub_data['children']
                    ))

            db.flush()

            # 5. Sensors (Depends on Subsystems and PIs)
            for sens_data in data.get('sensors', []):
                if db.query(Sensor).filter(Sensor.sensor_id == sens_data['sensor_id']).first():
                    collisions.append(f"Sensor ID {sens_data['sensor_id']}")
                else:
                    db.add(Sensor(
                        sensor_id=sens_data['sensor_id'], subsystem_id=sens_data['subsystem_id'],
                        pi_id=sens_data['pi_id'], name=sens_data['name'],
                        description=sens_data['description'], tag=sens_data['tag']
                    ))

            if collisions:
                db.rollback()
                db.close()
                return dbc.Alert([
                    html.Strong("Import failed! The following IDs already exist in the database:"),
                    html.Ul([html.Li(c) for c in collisions])
                ], color="danger")
            
            db.commit()
            db.close()
            return dbc.Alert(f"Data from '{filename}' imported successfully!", color="success")
            
        except Exception as e:
            if 'db' in locals(): db.close()
            return dbc.Alert(f"Import failed: {str(e)}", color="danger")
