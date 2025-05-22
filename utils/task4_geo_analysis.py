import pandas as pd
import plotly.express as px
import os

def analyze_and_visualize(csv_path='dataset/Dataset.csv', map_html_path='static/map.html'):
    df = pd.read_csv(csv_path)

    # Filter rows with coordinates
    df = df.dropna(subset=['Latitude', 'Longitude'])

    # Basic scatter map
    fig = px.scatter_mapbox(
        df,
        lat="Latitude",
        lon="Longitude",
        color="Aggregate rating",
        hover_name="Restaurant Name",
        hover_data=["City", "Cuisines", "Price range"],
        zoom=3,
        height=700
    )

    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

    os.makedirs(os.path.dirname(map_html_path), exist_ok=True)
    fig.write_html(map_html_path)
    print(f"Interactive map saved to {map_html_path}")

def statistical_summary(csv_path='dataset/Dataset.csv'):
    df = pd.read_csv(csv_path)

    # Group by City
    summary = df.groupby('City').agg({
        'Restaurant Name': 'count',
        'Aggregate rating': 'mean',
        'Price range': 'mean'
    }).rename(columns={
        'Restaurant Name': 'Restaurant Count',
        'Aggregate rating': 'Avg Rating',
        'Price range': 'Avg Price Range'
    }).sort_values('Restaurant Count', ascending=False)

    print("\nCity-Level Summary:\n")
    print(summary.head(10))

if __name__ == '__main__':
    analyze_and_visualize()
    statistical_summary()
