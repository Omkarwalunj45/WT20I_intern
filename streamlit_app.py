import streamlit as st
import pandas as pd

# Page settings
st.set_page_config(page_title='WT20I Performance Analysis Portal', layout='wide')
st.title('WT20I Performance Analysis Portal')

# Load data
pdf = pd.read_csv("Dataset/WT20I_Bat.csv")
idf = pd.read_csv("Dataset/squads.csv")

# Sidebar for selecting between "Player Profile" and "Matchup Analysis"
sidebar_option = st.sidebar.radio(
    "Select an option:",
    ("Player Profile", "Matchup Analysis")
)

# If "Player Profile" is selected
if sidebar_option == "Player Profile":
    st.header("Player Profile")

    # Player search input (selectbox)
    player_name = st.selectbox("Search for a player", idf['player_name'].unique())

    # Filter the data for the selected player
    player_info = idf[idf['player_name'] == player_name].iloc[0]

    # Tabs for "Overview", "Career Statistics", and "Current Form"
    tab1, tab2, tab3 = st.tabs(["Overview", "Career Statistics", "Current Form"])

    with tab1:
        st.header("Overview")
        
        # Create columns for horizontal layout
        col1, col2, col3 = st.columns(3)

        # Display player profile information
        with col1:
            st.markdown("FULL NAME:")
            st.markdown(f"**{player_info['player_name'].upper()}**")
        
        with col2:
            st.markdown("COUNTRY:")
            st.markdown(f"**{player_info['team_name'].upper()}**")
        
        with col3:
            st.markdown("AGE:")  # Placeholder for age
            st.markdown("**N/A**")  # Placeholder for future age data

        # Below the horizontal layout for batting style, bowling style, and role
        st.markdown("**BATTING STYLE:** N/A")  # Placeholder for batting style
        st.markdown("**BOWLING STYLE:** N/A")  # Placeholder for bowling style
        st.markdown("**PLAYING ROLE:** N/A")  # Placeholder for playing role

    with tab2:
        st.header("Career Statistics")
        # Add career statistics content here

    with tab3:
        st.header("Current Form")
        # Add current form content here

# If "Matchup Analysis" is selected
elif sidebar_option == "Matchup Analysis":
    st.header("Matchup Analysis")
    # Add content for Matchup Analysis here
