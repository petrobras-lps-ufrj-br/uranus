import dash
from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc
from .layout import create_layout
from .pages import render_list_products, render_list_pis, render_add_pi, render_add_product, render_display_page
from .callbacks import register_dashboard_callbacks

def create_dashboard(requests_pathname_prefix, server=True):
    dash_app = dash.Dash(
        __name__,
        server=server,
        requests_pathname_prefix=requests_pathname_prefix,
        routes_pathname_prefix='/',
        external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP]
    )

    dash_app.layout = create_layout(requests_pathname_prefix)

    @dash_app.callback(
        Output("page-content", "children"),
        Input("url", "pathname")
    )
    def render_page_content(pathname):
        if not pathname:
            return render_list_products(requests_pathname_prefix)
            
        # Normalize pathname by stripping leading/trailing slashes for easier matching
        # requests_pathname_prefix is typically "/dashboard/"
        prefix = requests_pathname_prefix.strip("/")
        path = pathname.strip("/")
        
        # Split into parts
        parts = path.split("/")
        
        # If the path is just the prefix or list-products
        if path == prefix or (len(parts) > 0 and parts[-1] == "list-products"):
            return render_list_products(requests_pathname_prefix)
        
        if len(parts) >= 2 and parts[0] == prefix:
            action = parts[1]
            if action == "display":
                return render_display_page()
            elif action == "add-product":
                return render_add_product()
            elif action == "list-pis":
                return render_list_pis(requests_pathname_prefix)
            elif action == "add-pi":
                return render_add_pi()
            elif action == "edit-pi" and len(parts) >= 3:
                return render_add_pi(pi_id=parts[2])
            elif action == "edit-product" and len(parts) >= 3:
                return render_add_product(prod_id=parts[2])
        return html.Div([
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognized...")
        ])

    register_dashboard_callbacks(dash_app)

    return dash_app
