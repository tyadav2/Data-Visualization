import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime

def parse_data(year: str):
    """Parse basketball player data from basketball-reference.com"""
    try:
        url = f"https://www.basketball-reference.com/leagues/NBA_{year}_per_game.html"
        parsed_df = pd.read_html(url, header=0)[0]
        parsed_df = parsed_df.drop(parsed_df[parsed_df['Age'] == 'Age'].index)
        parsed_df = parsed_df.fillna(0)
        parsed_df = parsed_df.drop(['Rk'], axis=1)
        
        # Convert percentage columns to string
        percentage_cols = ['FG%', '3P%', '2P%', 'eFG%', 'FT%']
        for col in percentage_cols:
            if col in parsed_df.columns:
                parsed_df[col] = parsed_df[col].astype(str)
        
        # Convert Age to integer
        parsed_df['Age'] = pd.to_numeric(parsed_df['Age'], errors='coerce').fillna(0).astype(int)
        
        # Ensure Team column is string type
        parsed_df['Team'] = parsed_df['Team'].astype(str)
        
        return parsed_df, None
    except Exception as e:
        return None, f"Error: {str(e)}"

def get_teams_for_year(year):
    """Get list of teams for the selected year"""
    df, error = parse_data(str(year))
    if df is not None and error is None:  # Make sure we have valid data
        # Convert team values to strings before sorting to avoid type comparison issues
        return sorted(df['Team'].astype(str).unique().tolist())
    return []

def create_correlation_heatmap(df):
    if df is None or df.empty:
        return None
    
    # Drop non-numeric columns and percentage columns that were converted to strings
    non_numeric_columns = df.select_dtypes(exclude=[np.number]).columns
    df_numeric = df.drop(columns=non_numeric_columns)
    
    if not df_numeric.empty:
        plt.figure(figsize=(12, 8))
        corr = df_numeric.corr()
        mask = np.zeros_like(corr)
        mask[np.triu_indices_from(mask)] = True
        
        with sns.axes_style("white"):
            sns.heatmap(corr, mask=mask, vmax=1, square=True, cmap="YlOrRd")
            plt.title("Correlation Heatmap of Player Statistics")
            plt.tight_layout()
        return plt
    return None

def update_teams(year):
    """Update team dropdown based on selected year"""
    teams = get_teams_for_year(year)
    return gr.Dropdown(choices=teams, value=None if not teams else teams[0])

def update_interface(year, team, positions, min_age, max_age):
    """Update interface based on user inputs"""
    # Get data
    df, error = parse_data(str(year))
    
    if error:
        return gr.DataFrame(value=None), None, None, f"Error: {error}"
    
    if df is None:
        return gr.DataFrame(value=None), None, None, "Error: Could not fetch data"

    try:
        # Filter positions if provided
        if positions and len(positions) > 0:
            df = df[df['Pos'].isin(positions)]
        
        # Filter team if provided
        if team:
            df = df[df['Team'] == team]
        
        # Filter age range using separate min and max values
        df = df[df['Age'].between(min_age, max_age)]
        
        # Create heatmap
        heatmap = create_correlation_heatmap(df)
        
        # Create download link
        if not df.empty:
            csv = df.to_csv(index=False)
            download_link = f'<a href="data:text/csv;base64,{csv}" download="player_stats.csv">Download CSV</a>'
        else:
            download_link = "No data available for download"

        return df, heatmap, download_link, None
        
    except Exception as e:
        return gr.DataFrame(value=None), None, None, f"Error processing data: {str(e)}"

def create_gradio_interface():
    # Create list of years from 1950 to 2024 in reverse order
    current_year = datetime.now().year
    years = list(range(current_year, 1949, -1))
    
    with gr.Blocks(theme=gr.themes.Monochrome()) as demo:
        gr.Markdown("# NBA Player Statistics Explorer")
        
        with gr.Row():
            with gr.Column():
                year = gr.Dropdown(
                    choices=years,
                    value=current_year-1,  # Default to previous year
                    label="Season Year",
                    info="Select NBA season year"
                )
                
                team = gr.Dropdown(
                    choices=[],  # Will be populated when year is selected
                    label="Team",
                    info="Select a team to filter players",
                    value=None
                )

            with gr.Column():
                positions = gr.CheckboxGroup(
                    choices=["C", "PF", "SF", "PG", "SG"],
                    label="Position Filter",
                    info="Select multiple positions to filter",
                    value=[]
                )
                
                # Replace single slider with two separate sliders
                min_age = gr.Slider(
                    minimum=18,
                    maximum=40,
                    value=18,
                    step=1,
                    label="Minimum Age",
                    info="Filter players by minimum age"
                )
                
                max_age = gr.Slider(
                    minimum=18,
                    maximum=40,
                    value=40,
                    step=1,
                    label="Maximum Age",
                    info="Filter players by maximum age"
                )

        gr.Markdown("## Player Statistics")
        error_output = gr.Markdown("")
        stats_df = gr.DataFrame(interactive=False)
        
        gr.Markdown("## Statistics Correlation Heatmap")
        heatmap = gr.Plot()
        
        download_link = gr.HTML("Download the dataset:")

        # Update teams when year changes
        year.change(
            fn=update_teams,
            inputs=year,
            outputs=team
        )

        # Update interface when any input changes
        inputs = [year, team, positions, min_age, max_age]
        outputs = [stats_df, heatmap, download_link, error_output]
        
        # Update interface when any input changes
        for input_comp in inputs:
            input_comp.change(
                fn=update_interface,
                inputs=inputs,
                outputs=outputs
            )

    return demo

if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch()