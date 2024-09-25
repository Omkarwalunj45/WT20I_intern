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
        
        # Display player profile information in the Overview tab, formatted like Cricinfo
        
        st.markdown(f"### **FULL NAME:** {player_info['player_name'].upper()}")
        st.markdown(f"### **COUNTRY:** {player_info['team_name'].upper()}")
        st.markdown(f"### **BATTING STYLE:** {player_info['batting_hand'].upper()}")
        
        # Placeholder values for Bowling Style, Playing Role, Age (to be added if data is available)
        # st.write(f"**Bowling Style:** {player_info['bowling_hand'].upper()}")
        # st.write(f"**Playing Role:** {player_info['playing_role'].upper()}")
        # st.write(f"**Age:** {player_info['age']}")  # Assuming age is available in dataset

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
