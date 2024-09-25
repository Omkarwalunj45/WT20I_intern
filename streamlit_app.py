import streamlit as st
import pandas as pd

# Page settings
st.set_page_config(page_title='WT20I Performance Analysis Portal', layout='wide')
st.title('WT20I Performance Analysis Portal')

# Load data
pdf = pd.read_csv("Dataset/women_bbb_t20_compressed.csv")
idf = pd.read_csv("Dataset/updated_wt20i.csv")
# ldf = pd.read_csv("Dataset/squads.csv")

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
    # ldata = ldf[ldf['player_name'] == player_name].iloc[0]
    
    

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

        # Below the first row for batting style, bowling style, and role
        with col4:
            st.markdown("BATTING STYLE:")
            # st.markdown(f"<span style='font-size: 20px; font-weight: bold;'>{ldata['batting_hand'].upper()}</span>", unsafe_allow_html=True)
            st.markdown("<span style='font-size: 20px; font-weight: bold;'>N/A</span>", unsafe_allow_html=True)  # Placeholder for future age data
        
        with col5:
            st.markdown("BOWLING STYLE:")
            st.markdown("<span style='font-size: 20px; font-weight: bold;'>N/A</span>", unsafe_allow_html=True)  # Placeholder for bowling style
        
        with col6:
            st.markdown("PLAYING ROLE:")
            st.markdown("<span style='font-size: 20px; font-weight: bold;'>N/A</span>", unsafe_allow_html=True)  # Placeholder for playing role

    with tab2:
        st.header("Career Statistics")

        # Dropdown for Batting or Bowling selection
        option = st.selectbox("Select Career Stat Type", ("Batting", "Bowling"))

        # Show Career Averages based on the dropdown
        st.subheader("Career Averages")

        # Filter data from pdf for the selected player
        player_data = idf[idf['batsmans'] == player_name]

        # If batting is selected, show batting stats
        if option == "Batting":
            st.write("Batting Career Averages")

            # Display the player's batting statistics in a table format
            if not player_data.empty:
                # Get the entire row for the player
                batting_stats = player_data

                # Create a custom header for the DataFrame
                header = batting_stats.columns.str.upper()  # Capitalize column names
                header = [f"**{col}**" for col in header]  # Make headers bold
                
                # Display header in a markdown format
                st.markdown(f"<h4>{' | '.join(header)}</h4>", unsafe_allow_html=True)

                # Displaying the player's stats row
                st.dataframe(batting_stats.style.hide_index())  # Hide index for a cleaner look
            else:
                st.write("No Batting Data Available.")

        # If bowling is selected, show bowling stats (if available in the dataset)
        if option == "Bowling":
            st.write("Bowling Career Averages")

            # Placeholder for bowling data – assuming future data will include bowling stats
            st.write("No Bowling Data Available.")

    with tab3:
        st.header("Current Form")
        # Add current form content here

# If "Matchup Analysis" is selected
elif sidebar_option == "Matchup Analysis":
    st.header("Matchup Analysis")
    # Add content for Matchup Analysis here
