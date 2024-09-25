import streamlit as st
import pandas as pd
import numpy as np
import math as mt

# Page settings
st.set_page_config(page_title='WT20I Performance Analysis Portal', layout='wide')
st.title('WT20I Performance Analysis Portal')

# Load data
pdf = pd.read_csv("Dataset/up_com_wt20i.csv")
idf = pd.read_csv("Dataset/updated_wt20i.csv")
ldf = pd.read_csv("Dataset/squads.csv")  # Load squads.csv for batting type

# Preprocess the debut column to extract the year
idf['debut_year'] = idf['debut_year'].str.split('/').str[0]  # Extract the year from "YYYY/YY"

# Convert the relevant columns to integers
columns_to_convert = ['runs', 'hundreds', 'fifties', 'thirties', 'highest_score']
idf[columns_to_convert] = idf[columns_to_convert].astype(int)

# Allowed countries
allowed_countries = ['India', 'England', 'Australia', 'Pakistan', 'Bangladesh', 
                     'West Indies', 'Scotland', 'South Africa', 'New Zealand', 'Sri Lanka']

# Sidebar for selecting between "Player Profile" and "Matchup Analysis"
sidebar_option = st.sidebar.radio(
    "Select an option:",
    ("Player Profile", "Matchup Analysis")
)

# If "Player Profile" is selected
if sidebar_option == "Player Profile":
    st.header("Player Profile")

    # Player search input (selectbox)
    player_name = st.selectbox("Search for a player", idf['batsman'].unique())

    # Filter the data for the selected player
    player_info = idf[idf['batsman'] == player_name].iloc[0]

    # Filter to get batting type from squads.csv
    player_batting_type = ldf[ldf['player_name'] == player_name]['batting_hand']

    # Check for existence of batting type
    if not player_batting_type.empty:
        batting_type_display = player_batting_type.iloc[0]  # Get the batting type
    else:
        batting_type_display = "N/A"  # Default if no type is found

    # Tabs for "Overview", "Career Statistics", and "Current Form"
    tab1, tab2, tab3 = st.tabs(["Overview", "Career Statistics", "Current Form"])

    with tab1:
        st.header("Overview")

        # Create columns for the first row (full name, country, age)
        col1, col2, col3 = st.columns(3)

        # Display player profile information
        with col1:
            st.markdown("FULL NAME:")
            st.markdown(f"<span style='font-size: 20px; font-weight: bold;'>{player_info['batsman']}</span>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("COUNTRY:")
            st.markdown(f"<span style='font-size: 20px; font-weight: bold;'>{player_info['batting_team'].upper()}</span>", unsafe_allow_html=True)
        
        with col3:
            st.markdown("AGE:")  # Placeholder for age
            st.markdown("<span style='font-size: 20px; font-weight: bold;'>N/A</span>", unsafe_allow_html=True)  # Placeholder for future age data

        # Create columns for the second row (batting style, bowling style, playing role)
        col4, col5, col6 = st.columns(3)

        with col4:
            st.markdown("BATTING STYLE:")
            st.markdown(f"<span style='font-size: 20px; font-weight: bold;'>{batting_type_display}</span>", unsafe_allow_html=True)

        with col5:
            st.markdown("BOWLING STYLE:")
            st.markdown("<span style='font-size: 20px; font-weight: bold;'>N/A</span>", unsafe_allow_html=True)  # Placeholder for bowling style

        with col6:
            st.markdown("PLAYING ROLE:")
            st.markdown("<span style='font-size: 20px; font-weight: bold;'>N/A</span>", unsafe_allow_html=True)  # Placeholder for playing role

    with tab2:
        st.header("Career Statistics")
        # Display player statistics here

    with tab3:
        st.header("Current Form")
        # Display player current form statistics here

    # New section for filtering by bowling team
    st.subheader("Batsman Performance Against Teams")
    team_performance_df = idf[idf['batsman'] == player_name].copy()
    
    # Create a dataframe to store results for each team
    team_results = {}

    for team in allowed_countries:
        team_stats = team_performance_df[team_performance_df['bowling_team'] == team]
        if not team_stats.empty:
            # Aggregate statistics for the team
            runs = team_stats['runs'].sum()
            matches = team_stats['match_id'].nunique()
            dismissals = team_stats['player_dismissed'].count()
            highest_score = team_stats['batsman_runs'].max()

            team_results[team] = {
                'Runs': runs,
                'Matches': matches,
                'Dismissals': dismissals,
                'Highest Score': highest_score
            }

    # Convert team results to DataFrame
    results_df = pd.DataFrame(team_results).T.fillna(0)

    # Display the results in a table format
    st.dataframe(results_df)

# The Matchup Analysis section can be added here if needed
