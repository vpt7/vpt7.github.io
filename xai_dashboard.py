from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

# Load Iris dataset
iris = load_iris(as_frame=True)
X = iris.data
y = iris.target

# Train a RandomForest model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Generate SHAP values with additivity check disabled
explainer = shap.Explainer(model, X)
shap_values = explainer(X, check_additivity=False)

# Aggregate SHAP values across classes by averaging
shap_values_avg = shap_values.values.mean(axis=2)  # shape becomes (150, 4)

# Sum the base values across classes
base_values_sum = shap_values.base_values.sum(axis=1)

# Create a DataFrame for SHAP values (averaged across classes)
shap_df = pd.DataFrame(shap_values_avg, columns=X.columns)
shap_df['output'] = base_values_sum + shap_values_avg.sum(axis=1)

# Dash application
app = Dash(__name__)
app.title = "Interactive Model Explanation Dashboard"

# Layout
app.layout = html.Div([
    html.H1("Model Explanations Dashboard"),
    html.Label("Select Feature to Visualize:"),
    dcc.Dropdown(
        id="feature-dropdown",
        options=[{"label": col, "value": col} for col in X.columns],
        value=X.columns[0]
    ),
    dcc.Graph(id="shap-graph"),
    html.Br(),
    dcc.Markdown("""
    **Note**: This dashboard visualizes SHAP values (feature importance) for each feature in the dataset, 
    showing how much each feature contributes to the model's prediction.
    """)
])

# Callback for interactive visualization
@app.callback(
    Output("shap-graph", "figure"),
    Input("feature-dropdown", "value")
)
def update_graph(selected_feature):
    fig = px.histogram(
        shap_df, 
        x=selected_feature, 
        title=f"SHAP Values for {selected_feature}",
        labels={"x": "SHAP Value", "count": "Frequency"}
    )
    return fig

# Run the Dash app
if __name__ == "__main__":
    app.run_server(debug=True)
