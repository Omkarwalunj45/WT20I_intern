import streamlit as st
import pandas as pd
import math as mt
import numpy as np

# Page settings
st.set_page_config(page_title='WT20I Performance Analysis Portal', layout='wide')
st.title('WT20I Performance Analysis Portal')

# Load data
pdf = pd.read_csv("Dataset/up_com_wt20i.csv")
idf = pd.read_csv("Dataset/updated_wt20i.csv")
ldf = pd.read_csv("Dataset/squads.csv")  # Load squads.csv for batting type
def cumulator(temp_df):       
    # Create new columns for counting runs
    temp_df['is_dot'] = temp_df['total_runs'].apply(lambda x: 1 if x == 0 else 0)
    temp_df['is_one'] = temp_df['batsman_runs'].apply(lambda x: 1 if x == 1 else 0)
    temp_df['is_two'] = temp_df['batsman_runs'].apply(lambda x: 1 if x == 2 else 0)
    temp_df['is_three'] = temp_df['batsman_runs'].apply(lambda x: 1 if x == 3 else 0)
    temp_df['is_four'] = temp_df['batsman_runs'].apply(lambda x: 1 if x == 4 else 0)
    temp_df['is_six'] = temp_df['batsman_runs'].apply(lambda x: 1 if x == 6 else 0)
    
    # Calculate runs, balls faced, innings, and dismissals
    runs = temp_df.groupby(['batsman'])['batsman_runs'].sum().reset_index().rename(columns={'batsman_runs': 'runs'})
    balls = temp_df.groupby(['batsman'])['ball'].count().reset_index()
    inn = temp_df.groupby(['batsman'])['match_id'].apply(lambda x: len(list(np.unique(x)))).reset_index().rename(columns={'match_id': 'innings'})
    matches = temp_df.groupby(['batsman'])['match_id'].nunique().reset_index().rename(columns={'match_id': 'matches'})
    dis = temp_df.groupby(['batsman'])['player_dismissed'].count().reset_index().rename(columns={'player_dismissed': 'dismissals'})
    sixes = temp_df.groupby(['batsman'])['is_six'].sum().reset_index().rename(columns={'is_six': 'sixes'})
    fours = temp_df.groupby(['batsman'])['is_four'].sum().reset_index().rename(columns={'is_four': 'fours'})
    dots = temp_df.groupby(['batsman'])['is_dot'].sum().reset_index().rename(columns={'is_dot': 'dots'})
    ones = temp_df.groupby(['batsman'])['is_one'].sum().reset_index().rename(columns={'is_one': 'ones'})
    twos = temp_df.groupby(['batsman'])['is_two'].sum().reset_index().rename(columns={'is_two': 'twos'})
    threes = temp_df.groupby(['batsman'])['is_three'].sum().reset_index().rename(columns={'is_three': 'threes'})
    bat_team = temp_df.groupby(['batsman'])['batting_team'].unique().reset_index()
    
    # Convert the array of countries to a string without brackets
    bat_team['batting_team'] = bat_team['batting_team'].apply(lambda x: ', '.join(x)).str.replace('[', '').str.replace(']', '')
    
    match_runs = temp_df.groupby(['batsman', 'match_id'])['batsman_runs'].sum().reset_index()
    
    # Count 100s, 50s, and 30s
    hundreds = match_runs[match_runs['batsman_runs'] >= 100].groupby('batsman').size().reset_index(name='hundreds')
    fifties = match_runs[(match_runs['batsman_runs'] >= 50) & (match_runs['batsman_runs'] < 100)].groupby('batsman').size().reset_index(name='fifties')
    thirties = match_runs[(match_runs['batsman_runs'] >= 30) & (match_runs['batsman_runs'] < 50)].groupby('batsman').size().reset_index(name='thirties')
    
    # Calculate the highest score for each batsman
    highest_scores = match_runs.groupby('batsman')['batsman_runs'].max().reset_index(name='highest_score')
    
    # Fill NaNs with 0 for counts in case a batsman has none
    hundreds['hundreds'] = hundreds['hundreds'].fillna(0)
    fifties['fifties'] = fifties['fifties'].fillna(0)
    thirties['thirties'] = thirties['thirties'].fillna(0)
    
    # Merge all stats into a single DataFrame
    bat_rec = (
        inn.merge(runs, on='batsman')
        .merge(bat_team, on='batsman')
        .merge(balls, on='batsman')
        .merge(dis, on='batsman')
        .merge(sixes, on='batsman')
        .merge(fours, on='batsman')
        .merge(dots, on='batsman')
        .merge(ones, on='batsman')
        .merge(twos, on='batsman')
        .merge(threes, on='batsman')
        .merge(hundreds, on='batsman', how='left').fillna(0)
        .merge(fifties, on='batsman', how='left').fillna(0)
        .merge(thirties, on='batsman', how='left').fillna(0)
        .merge(highest_scores, on='batsman', how='left').fillna(0)
    )
    
    # Ensure to count matches as well
    matches = temp_df.groupby('batsman')['match_id'].nunique().reset_index(name='matches')
    
    # Merging matches data
    bat_rec = bat_rec.merge(matches, on='batsman', how='left')
    
    # Reset index for the final DataFrame
    bat_rec.reset_index(inplace=True, drop=True)
    
    # Calculating additional columns
    def bpd(balls, dis):
        return balls if dis == 0 else balls / dis
    
    def bpb(balls, bdry):
        return balls if bdry == 0 else balls / bdry
    
    def avg(runs, dis, inn):
        return runs / inn if dis == 0 else runs / dis
    
    def DP(balls, dots):
        return (dots / balls) * 100
    
    bat_rec['SR'] = bat_rec.apply(lambda x: (x['runs'] / x['ball']) * 100, axis=1)
    bat_rec['BPD'] = bat_rec.apply(lambda x: bpd(x['ball'], x['dismissals']), axis=1)
    bat_rec['BPB'] = bat_rec.apply(lambda x: bpb(x['ball'], (x['fours'] + x['sixes'])), axis=1)
    bat_rec['nbdry_sr'] = bat_rec.apply(lambda x: (
        (x['dots'] * 0 + x['ones'] * 1 + x['twos'] * 2 + x['threes'] * 3) / 
        (x['dots'] + x['ones'] + x['twos'] + x['threes']) * 100) 
        if (x['dots'] + x['ones'] + x['twos'] + x['threes']) > 0 else 0, 
        axis=1
    )
    bat_rec['AVG'] = bat_rec.apply(lambda x: avg(x['runs'], x['dismissals'], x['innings']), axis=1)
    bat_rec['dot_percentage'] = (bat_rec['dots'] / bat_rec['ball']) * 100
    
    # Adding career span based on 'season' column
    debut_year = temp_df.groupby('batsman')['season'].min().reset_index()
    final_year = temp_df.groupby('batsman')['season'].max().reset_index()
    debut_year.rename(columns={'season': 'debut_year'}, inplace=True)
    final_year.rename(columns={'season': 'final_year'}, inplace=True)
    
    # Merging career span into bat_rec DataFrame
    bat_rec = bat_rec.merge(debut_year, on='batsman').merge(final_year, on='batsman')
    
    # Reset index for the final DataFrame
    bat_rec.reset_index(inplace=True, drop=True)
    return bat_rec


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
            allowed_countries = ['India', 'England', 'Australia', 'Pakistan', 'Bangladesh', 
                     'West Indies', 'Scotland', 'South Africa', 'New Zealand', 'Sri Lanka']
            pdf['total_runs'] = pdf['runs_off_bat'] + pdf['extras']
            pdf = pdf.rename(columns={'runs_off_bat': 'batsman_runs', 'wicket_type': 'dismissal_kind', 'striker': 'batsman', 'innings': 'inning'})
            pdf = pdf.dropna(subset=['ball'])
            # Convert the 'ball' column to numeric if it's not already (optional but recommended)
            pdf['ball'] = pd.to_numeric(pdf['ball'], errors='coerce') 
            # Applying the lambda function to calculate the 'over'
            pdf['over'] = pdf['ball'].apply(lambda x: mt.floor(x) + 1 if pd.notnull(x) else None)
            for country in allowed_countries:
                temp_df = pdf[(pdf['batsman'] == player_name) & (pdf['bowling_team'] == country)]
                
                temp_df=cumulator(temp_df)
                 
                temp_df = temp_df.drop(columns=['final_year'])
                temp_df.columns = [col.upper().replace('_', ' ') for col in temp_df.columns]
                
                st.markdown("### vs 'country'")
                st.table(temp_df.style.set_table_attributes("style='font-weight: bold;'")) 
                    
            
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
