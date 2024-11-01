import base64

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

st.title('# EDA BasketBall')

st.sidebar.header("Input Parameters")
selected_year = st.sidebar.selectbox("Year", list(reversed(range(1950, 2020))))

"""
parsing basketball players info from https://www.basketball-reference.com
"""
@st.cache_data
def parse_data(year: str):
    url = "https://www.basketball-reference.com/leagues/NBA_" + year + "_per_game.html"
    parsed_df = pd.read_html(url, header=0)[0]
    parsed_df = parsed_df.drop(parsed_df[parsed_df['Age'] == 'Age'].index)
    parsed_df = parsed_df.fillna(0)
    parsed_df = parsed_df.drop(['Rk'], axis=1)
    # PyArrow bug
    # https://stackoverflow.com/a/69581451/11105356
    parsed_df['FG%'] = parsed_df['FG%'].astype(str)
    parsed_df['3P%'] = parsed_df['3P%'].astype(str)
    parsed_df['2P%'] = parsed_df['2P%'].astype(str)
    parsed_df['eFG%'] = parsed_df['eFG%'].astype(str)
    parsed_df['FT%'] = parsed_df['FT%'].astype(str)
    # Convert 'Age' to integer, with errors handled
    parsed_df['Age'] = pd.to_numeric(parsed_df['Age'], errors='coerce').fillna(0).astype(int)
    return parsed_df


df_player_stat_dataset = parse_data(str(selected_year))

df_player_stat_dataset['Team'] = df_player_stat_dataset['Team'].astype(str)  # Ensure Team values are strings
sorted_dataset_by_team = sorted(df_player_stat_dataset.Team.unique())

# team filter
selected_team = st.sidebar.multiselect("Team", sorted_dataset_by_team, sorted_dataset_by_team)

# position filter
player_positions = ['C', 'PF', 'SF', 'PG', 'SG']
selected_position = st.sidebar.multiselect("Position", player_positions, player_positions)

# multi slider
# https://github.com/streamlit/streamlit/issues/855
unique_age_values = df_player_stat_dataset.Age.unique()
minValue, maxValue = min(unique_age_values), max(unique_age_values)


# age filter
selected_age = st.sidebar.slider("Age", int(minValue), int(maxValue), (int(minValue), int(maxValue)), 1)
min_age, max_age = selected_age

# filtered dataset
df_selected_dataset = df_player_stat_dataset[
    (df_player_stat_dataset.Team.isin(selected_team) &
     df_player_stat_dataset.Pos.isin(selected_position) &
     df_player_stat_dataset['Age'].between(min_age, max_age))]

# Convert specific non-numeric columns to string type
if 'Awards' in df_selected_dataset.columns:
    df_selected_dataset.loc[:, 'Awards'] = df_selected_dataset['Awards'].astype(str)

# display dataframe
st.header('Display Player Stats of Selected Team(s)')
st.write(f'Data Dimension  Row : {df_selected_dataset.shape[0]} and Col : {df_selected_dataset.shape[1]}')
st.dataframe(df_selected_dataset)


def download_dataset(dataset): # download the dataset as csv file
    csv = dataset.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings to bytes conversions
    href_link = f'<a href="data:file/csv;base64,{b64}" download="player_stats.csv">Download CSV File</a>'
    return href_link


st.markdown(download_dataset(df_selected_dataset), unsafe_allow_html=True)

if st.button("Inter-correlation Heatmap"):
    st.header("Inter-correlation Heatmap")

    df_selected_dataset.to_csv('output.csv', index=False)
    df = pd.read_csv('output.csv')

    # Drop non-numeric columns to avoid serialization issues
    non_numeric_columns = df.select_dtypes(exclude=[np.number]).columns
    df = df.drop(columns=non_numeric_columns)

    # Check if there are any numeric columns to avoid errors
    if not df.empty:
        corr = df.corr()

        mask = np.zeros_like(corr)
        mask[np.triu_indices_from(mask)] = True

        # plot the Inter-correlation Heatmap
        with sns.axes_style("white"):
            f, ax = plt.subplots(figsize=(7, 5))
            try:
                sns.heatmap(corr, mask=mask, vmax=1, square=True)
            except ValueError:
                pass
        st.pyplot(f)
    else:
        st.write("No numeric data available for correlation.")