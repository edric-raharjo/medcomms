# Loading the library
import pandas as pd
import plotly.express as px

# Load the Data
plot_df = pd.read_csv('plot_df.csv')
plot_df_3d = pd.read_csv('plot_df_3d.csv')

# First Plot
from dash import Dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output

import plotly.express as px

from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import numpy as np
import pandas as pd

# Initialize the Dash app
app = Dash(__name__)

# List of dates
dates = [
    "2021-07-18", "2021-08-15", "2021-09-12", "2021-10-10", "2021-11-07", "2021-12-05",
    "2022-01-02", "2022-01-30", "2022-02-27", "2022-03-27", "2022-04-24", "2022-05-22",
    "2022-06-19", "2022-07-17", "2022-08-14", "2022-09-11", "2022-10-09", "2022-11-06",
    "2022-12-04", "2023-01-01", "2023-01-29", "2023-02-26", "2023-03-26", "2023-04-23",
    "2023-05-21", "2023-06-18", "2023-07-16", "2023-08-13", "2023-09-10", "2023-10-08",
    "2023-11-05", "2023-12-03"
]

# App layout with improved styling
app.layout = html.Div(style={'padding': '30px'}, children=[
    html.H1("3D Embeddings Visualization", style={'textAlign': 'center', 'fontSize': '32px', 'fontFamily': 'Arial'}),
    
    html.Div([
        html.Label("Select Mode:", style={'fontSize': '18px', 'paddingRight': '10px', 'fontFamily': 'Arial'}),
        dcc.RadioItems(
            id='mode-select',
            options=[
                {'label': 'Normal', 'value': 'normal'},
                {'label': 'Centroid', 'value': 'centroid'}
            ],
            value='normal',  # Default value
            labelStyle={'display': 'inline-block', 'marginRight': '20px'}
        ),
    ], style={'paddingBottom': '30px', 'textAlign': 'center'}),
    
    html.Div([
        html.Label("Legend :", style={'fontSize': '18px', 'paddingRight': '10px', 'fontFamily': 'Arial'}),
        dcc.RadioItems(
            id='legend',
            options=[
                {'label': 'Yes', 'value': 'True'},
                {'label': 'No', 'value': 'False'}
            ],
            value='True',  # Default value
            labelStyle={'display': 'inline-block', 'marginRight': '20px'}
        ),
    ], style={'paddingBottom': '30px', 'textAlign': 'center'}),
    
    html.Div([
        html.Label("Select Dates (up to 3):", style={'fontSize': '18px', 'paddingRight': '10px', 'fontFamily': 'Arial'}),
        dcc.Checklist(
            id='date-selector',
            options=[{'label': date, 'value': date} for date in dates],
            value=[dates[0]],  # Default to first date
            inline=True,
            style={'fontSize': '16px', 'fontFamily': 'Arial'}
        ),
    ], style={'paddingBottom': '30px'}),
    
    html.Div([
        html.Label("Select Countries:", style={'fontSize': '18px', 'paddingRight': '10px', 'fontFamily': 'Arial'}),
        dcc.Checklist(
            id='country-filter',
            options=[{'label': country, 'value': country} for country in plot_df_3d['country'].unique()],
            value=['taiwan', 'japan', 'hong-kong', 'thailand', 'vietnam', 'south-korea'],  # Default to all countries selected
            inline=True,
            style={'fontSize': '16px', 'fontFamily': 'Arial'}
        ),
    ], style={'paddingBottom': '30px'}),
    
    dcc.Graph(id='3d-plot', style={'width': '90%', 'height': '800px', 'margin': '0 auto'})
])

# Callback to update the plot based on selected mode, dates, and countries
@app.callback(
    Output('3d-plot', 'figure'),
    [Input('mode-select', 'value'),
     Input('date-selector', 'value'),
     Input('country-filter', 'value'),
     Input('legend','value')]
)
def update_plot(mode, selected_dates, selected_countries, LEGEND):
    LEGEND = eval(LEGEND)
    # Ensure the user selects up to 3 dates
    if len(selected_dates) > 3 and mode == 'centroid':
        # Sort dates chronologically and limit to the first 3
        selected_dates = sorted(selected_dates)[:3]
    else:
        # Sort dates chronologically if there are fewer than 3
        selected_dates = sorted(selected_dates)
    
    # Filter the DataFrame based on selected countries
    filtered_df = plot_df_3d[plot_df_3d['country'].isin(selected_countries)]
    
    # Set up 
    fig = go.Figure()
    
    if mode == 'normal':
        color_map = {'taiwan':'pink','japan':'red','hong-kong':'green','thailand':'purple','vietnam':'yellow','south-korea':'blue'}
        # Create a 3D scatter plot for normal mode
        for country in selected_countries:
            country_data = filtered_df[filtered_df['country'] == country]
            for date in selected_dates:
                country_date_df = country_data[country_data['date'] == date]
                colors = [color_map[cat] for cat in country_date_df['country']]
                print(filtered_df.shape)
                print(filtered_df['country'].unique())
                print(filtered_df['date'].unique())
                print (colors)

                print (country_date_df.shape)
                print(fig.data)
                print(f"Trace for {country}: {len(country_date_df)} points")

                fig.add_trace(go.Scatter3d(
                        x=country_date_df['PC1'],
                        y=country_date_df['PC2'],
                        z=country_date_df['PC3'],
                        mode='markers',
                        marker=dict(size=15, color=colors),
                        name=country    ,
                        hoverinfo='text',
                        text=country_date_df['title'].tolist(),
                        showlegend=LEGEND
                    )
                )
        
        fig.update_layout(title=f"Embeddings for {', '.join(selected_dates)}")
    
    elif mode == 'centroid':
        # Color Map
        color_map = {'taiwan':'pink','japan':'red','hong-kong':'green','thailand':'purple','vietnam':'yellow','south-korea':'blue'}
        # Create a 3D scatter plot for centroid mode
        for country in selected_countries:
            country_data = filtered_df[filtered_df['country'] == country]
            for date in selected_dates:
                country_date_df = country_data[country_data['date'] == date]
                if not country_date_df.empty:
                    # Calculate centroid
                    centroid = country_date_df[['PC1', 'PC2', 'PC3']].mean()
                    colors = [color_map[cat] for cat in country_date_df['country']]

                    # Plot
                    fig.add_trace(go.Scatter3d(
                            x=[centroid['PC1']],
                            y=[centroid['PC2']],
                            z=[centroid['PC3']],
                            mode='markers',
                            marker=dict(size=24, color=colors, symbol='x'),
                            name=f"Centroid {country} {date}",
                            hoverinfo='text',
                            text=f"Centroid of {country} on {date}",
                            showlegend=LEGEND
                        )
                    )

        fig.update_layout(title=f"Centroids for {', '.join(selected_dates)}")
    
        # Draw lines between consecutive dates (if multiple dates are selected)
        if len(selected_dates) > 1:
            print (selected_dates)
            color_map = {'taiwan':'pink','japan':'red','hong-kong':'green','thailand':'purple','vietnam':'yellow','south-korea':'blue'}
            for country in selected_countries:
                country_data = filtered_df[filtered_df['country'] == country]
                for i in range(len(selected_dates) - 1):
                    start_date = selected_dates[i]
                    end_date = selected_dates[i + 1]
                    # print (start_date, "to", end_date)
            
                    # Filter data for the two dates
                    start_data = country_data[country_data['date'] == start_date][['PC1', 'PC2', 'PC3']].mean().tolist()
                    end_data = country_data[country_data['date'] == end_date][['PC1', 'PC2', 'PC3']].mean().tolist()
                    # print (start_data,"\n",start_date)
                    # print (end_data,"\n",end_date)

                    # Calculate direction vector
                    dx = end_data[0] - start_data[0]
                    dy = end_data[1] - start_data[1]
                    dz = end_data[2] - start_data[2]

                    # Shorten the line by 10% from both ends
                    shrink_factor_start = 0.02
                    shrink_factor_end = 0.06

                    new_start = [
                        start_data[0] + shrink_factor_start * dx,
                        start_data[1] + shrink_factor_start * dy,
                        start_data[2] + shrink_factor_start * dz
                    ]
                    new_end = [
                        end_data[0] - shrink_factor_end * dx,
                        end_data[1] - shrink_factor_end * dy,
                        end_data[2] - shrink_factor_end * dz
                    ]
            
                    if start_data and end_data:
                        colors = [color_map[cat] for cat in country_data['country']]
                        # Add the shortened line
                        fig.add_trace(go.Scatter3d(
                                x=[new_start[0], new_end[0]],
                                y=[new_start[1], new_end[1]],
                                z=[new_start[2], new_end[2]],
                                mode='lines',
                                line=dict(color=colors, width=15),
                                name=f"Line {start_date} to {end_date}",
                                showlegend=LEGEND
                            )
                        )

                        # Add a dummy trace for the legend
                        fig.add_trace(go.Scatter3d(
                                x=[None],  # No actual points, just a placeholder
                                y=[None],
                                z=[None],
                                mode='markers',
                                marker=dict(size=10, color=colors[0]),  # Use the same color as the line
                                name=f"Line {start_date} to {end_date}",  # Legend label
                                showlegend=LEGEND
                            )
                        )

                        # Add the arrowhead (cone)
                        fig.add_trace(
                            go.Cone(
                                x=[new_end[0]],
                                y=[new_end[1]],
                                z=[new_end[2]],
                                u=[dx],
                                v=[dy],
                                w=[dz],
                                colorscale=[[0, colors[0]], [1, colors[0]]],
                                sizemode="absolute",
                                sizeref=0.5,  # Size of the arrowhead
                                showscale=False
                            )
                        )
    
    # Set consistent ticks for axes and layout
    tick_range = np.arange(-20, 30, 5).tolist()
    fig.update_layout(
        
        scene=dict(
            xaxis=dict(tickvals=tick_range, range=[-20, 30]),
            yaxis=dict(tickvals=tick_range, range=[-20, 30]),
            zaxis=dict(tickvals=tick_range, range=[-20, 30]),
            aspectmode="cube"
        ),
        height=1100,  # Larger plot height
        width=1200,  # Larger plot width
    )
    
    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
