import streamlit as st
import pandas as pd

# Page settings
st.set_page_config(page_title='WT20I Performance Analysis Portal', layout='wide')
st.title('WT20I Performance Analysis Portal')

# Load data
pdf = pd.read_csv("Dataset/women_bbb_t20_compressed.csv")
idf = pd.read_csv("Dataset/updated_wt20i.csv")
ldf = pd.read_csv("Dataset/squads.csv")  # Load squads.csv for batting type

# Preprocess the debut column to extract the year
idf['debut_year'] = idf['debut_year'].str.split('/').str[0]  # Extract the year from "YYYY/YY"

# Convert the relevant columns to integers
columns_to_convert = ['runs', 'hundreds', 'fifties', 'thirties', 'highest_score']
idf[columns_to_convert] = idf[columns_to_convert].astype(int)

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

        # Dropdown for Batting or Bowling selection
        option = st.selectbox("Select Career Stat Type", ("Batting", "Bowling"))

        # Show Career Averages based on the dropdown
        st.subheader("Career Performance")

        # Display Career Averages based on selection
        if option == "Batting":
            # Create a temporary DataFrame and filter the player's row
            temp_df = idf.drop(columns=['Unnamed: 0', 'final_year', 'matches_x', 'matches_y', 'surname', 'initial'])
            player_stats = temp_df[temp_df['batsman'] == player_name]  # Filter for the selected player

            # Convert column names to uppercase and replace underscores with spaces
            player_stats.columns = [col.upper().replace('_', ' ') for col in player_stats.columns]

            # Display the player's statistics in a table format with bold headers
            st.markdown("### Batting Statistics")
            st.table(player_stats.style.set_table_attributes("style='font-weight: bold;'"))  # Display the filtered DataFrame as a table

        elif option == "Bowling":
            # Similar logic can be added here for bowling statistics if needed
            st.write("Bowling statistics feature is not yet implemented.")

    with tab3:
        st.header("Current Form")
        # Add current form content here

# If "Matchup Analysis" is selected
elif sidebar_option == "Matchup Analysis":
    st.header("Matchup Analysis")
    # Add content for Matchup Analysis here
