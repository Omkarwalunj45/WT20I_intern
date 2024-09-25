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

        # Create columns for the first row (full name, country, age)
        col1, col2, col3 = st.columns(3)

        # Display player profile information
        with col1:
            st.markdown("FULL NAME:")
            st.markdown(f"<span style='font-size: 20px; font-weight: bold;'>{player_info['player_name'].upper()}</span>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("COUNTRY:")
            st.markdown(f"<span style='font-size: 20px; font-weight: bold;'>{player_info['team_name'].upper()}</span>", unsafe_allow_html=True)
        
        with col3:
            st.markdown("AGE:")
            age = player_info.get('age', 'N/A')  # Use 'N/A' if age is not available
            st.markdown(f"<span style='font-size: 20px; font-weight: bold;'>{age}</span>", unsafe_allow_html=True)

        # Create columns for the second row (batting style, bowling style, playing role)
        col4, col5, col6 = st.columns(3)

        # Below the first row for batting style, bowling style, and role
        with col4:
            st.markdown("BATTING STYLE:")
            batting_style = player_info.get('batting_hand', 'N/A')  # Use 'N/A' if batting style is not available
            st.markdown(f"<span style='font-size: 20px; font-weight: bold;'>{batting_style.upper()}</span>", unsafe_allow_html=True)
        
        with col5:
            st.markdown("BOWLING STYLE:")
            bowling_style = player_info.get('bowling_hand', 'N/A')  # Use 'N/A' if bowling style is not available
            st.markdown(f"<span style='font-size: 20px; font-weight: bold;'>{bowling_style.upper()}</span>", unsafe_allow_html=True)

        with col6:
            st.markdown("PLAYING ROLE:")
            playing_role = player_info.get('playing_role', 'N/A')  # Use 'N/A' if playing role is not available
            st.markdown(f"<span style='font-size: 20px; font-weight: bold;'>{playing_role.upper()}</span>", unsafe_allow_html=True)

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
