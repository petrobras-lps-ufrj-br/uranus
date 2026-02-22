from dash import html, dcc
import dash_bootstrap_components as dbc

def create_layout(requests_pathname_prefix):
    return dbc.Container([
        dbc.Row([
            # Sidebar
            dbc.Col([
                html.H2("Cognite Display", className="display-7"),
                html.Hr(),
                dbc.Nav([
                    dbc.NavLink([html.I(className="bi bi-display me-2"), "Display"], href=f"{requests_pathname_prefix}display", active="exact", id="nav-display"),
                    dbc.NavLink([html.I(className="bi bi-list-task me-2"), "List Products"], href=f"{requests_pathname_prefix}list-products", active="exact", id="nav-list-products"),
                    dbc.NavLink([html.I(className="bi bi-plus-circle me-2"), "Add Product"], href=f"{requests_pathname_prefix}add-product"  , active="exact", id="nav-add-product"  ),
                    dbc.NavLink([html.I(className="bi bi-hdd-network me-2"), "List PIs"]   , href=f"{requests_pathname_prefix}list-pis"     , active="exact", id="nav-list-pis"     ),
                    dbc.NavLink([html.I(className="bi bi-plus-square me-2"), "Add PI"]     , href=f"{requests_pathname_prefix}add-pi"       , active="exact", id="nav-add-pi"       ),
                    html.Hr(),
                    dbc.NavLink([html.I(className="bi bi-gear-fill me-2"), "Admin"], href=f"{requests_pathname_prefix}admin", active="exact", id="nav-admin"),
                ], vertical=True, pills=True),
            ], width=3, className="bg-light p-4", style={"height": "100vh"}),

            # Content Area
            dbc.Col([
                dcc.Location(id="url"),
                html.Div(id="page-content", className="p-4")
            ], width=9)
        ])
    ], fluid=True)
