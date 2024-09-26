import streamlit as st
import pandas as pd
import math as mt
import numpy as np

# Page settings
st.set_page_config(page_title='WT20I Performance Analysis Portal', layout='wide')
st.title('WT20I Performance Analysis Portal')

# Load data
pdf = pd.read_csv("Dataset/up_com_wt20i.csv",low_memory=False)
idf = pd.read_csv("Dataset/lifesaver.csv",low_memory=False)
info_df=pd.read_csv("Dataset/w_info_final.csv",low_memory=False)
bpdf=pd.read_csv("Dataset/final_cum_wbowl.csv",low_memory=False)
bidf=pd.read_csv("Dataset/lifesaver_bowl.csv",low_memory=False)
def round_up_floats(df, decimal_places=2):
    # Round up only for float columns
    float_cols = df.select_dtypes(include=['float'])
    df[float_cols.columns] = np.ceil(float_cols * (10 ** decimal_places)) / (10 ** decimal_places)
    return df
def standardize_season(season):
    if '/' in season:  # Check if the season is in 'YYYY/YY' format
          year = season.split('/')[0]  # Get the first part
    else:
          year = season  # Use as is if already in 'YYYY' format
    return year.strip()  # Return the year stripped of whitespace
# Define the columns related to runs
columns_to_convert = ['runs', 'hundreds', 'fifties', 'thirties', 'highest_scores']
ldf = pd.read_csv("Dataset/squads.csv",low_memory=False)  # Load squads.csv for batting type
# pdf = pdf.drop_duplicates(subset=['match_id', 'ball'], keep='first')

def cumulator(temp_df):
    # First, remove duplicates based on match_id and ball within the same match
    print(f"Before removing duplicates based on 'match_id' and 'ball': {temp_df.shape}")
    temp_df = temp_df.drop_duplicates(subset=['match_id', 'ball'], keep='first')
    print(f"After removing duplicates based on 'match_id' and 'ball': {temp_df.shape}")

    # Ensure 'total_runs' exists
    if 'total_runs' not in temp_df.columns:
        raise KeyError("Column 'total_runs' does not exist in temp_df.")
    
    # Create new columns for counting runs
    temp_df['is_dot'] = temp_df['total_runs'].apply(lambda x: 1 if x == 0 else 0)
    temp_df['is_one'] = temp_df['batsman_runs'].apply(lambda x: 1 if x == 1 else 0)
    temp_df['is_two'] = temp_df['batsman_runs'].apply(lambda x: 1 if x == 2 else 0)
    temp_df['is_three'] = temp_df['batsman_runs'].apply(lambda x: 1 if x == 3 else 0)
    temp_df['is_four'] = temp_df['batsman_runs'].apply(lambda x: 1 if x == 4 else 0)
    temp_df['is_six'] = temp_df['batsman_runs'].apply(lambda x: 1 if x == 6 else 0)

    # Calculate runs, balls faced, innings, dismissals, etc.
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
    print(1)
    matches = temp_df.groupby(['batsman'])['match_id'].nunique().reset_index(name='matches')
    print(0)

    # Convert the array of countries to a string without brackets
    bat_team['batting_team'] = bat_team['batting_team'].apply(lambda x: ', '.join(x)).str.replace('[', '').str.replace(']', '')

    match_runs = temp_df.groupby(['batsman', 'match_id'])['batsman_runs'].sum().reset_index()

    # Count 100s, 50s, and 30s
    hundreds = match_runs[match_runs['batsman_runs'] >= 100].groupby('batsman').size().reset_index(name='hundreds')
    fifties = match_runs[(match_runs['batsman_runs'] >= 50) & (match_runs['batsman_runs'] < 100)].groupby('batsman').size().reset_index(name='fifties')
    thirties = match_runs[(match_runs['batsman_runs'] >= 30) & (match_runs['batsman_runs'] < 50)].groupby('batsman').size().reset_index(name='thirties')

    # Calculate the highest score for each batsman
    highest_scores = match_runs.groupby('batsman')['batsman_runs'].max().reset_index().rename(columns={'batsman_runs': 'highest_score'})

    # Merge all the calculated metrics into a single DataFrame
    summary_df = runs.merge(balls, on='batsman', how='left')
    summary_df = summary_df.merge(inn, on='batsman', how='left')
    summary_df = summary_df.merge(matches, on='batsman', how='left')
    summary_df = summary_df.merge(dis, on='batsman', how='left')
    summary_df = summary_df.merge(sixes, on='batsman', how='left')
    summary_df = summary_df.merge(fours, on='batsman', how='left')
    summary_df = summary_df.merge(dots, on='batsman', how='left')
    summary_df = summary_df.merge(ones, on='batsman', how='left')
    summary_df = summary_df.merge(twos, on='batsman', how='left')
    summary_df = summary_df.merge(threes, on='batsman', how='left')
    summary_df = summary_df.merge(bat_team, on='batsman', how='left')
    summary_df = summary_df.merge(hundreds, on='batsman', how='left')
    summary_df = summary_df.merge(fifties, on='batsman', how='left')
    summary_df = summary_df.merge(thirties, on='batsman', how='left')
    summary_df = summary_df.merge(highest_scores, on='batsman', how='left')
    summary_df = summary_df.merge(matches, on='batsman', how='left')
          # Calculating additional columns
    def bpd(balls, dis):
      return balls if dis == 0 else balls / dis
    
    def bpb(balls, bdry):
      return balls if bdry == 0 else balls / bdry
    
    def avg(runs, dis, inn):
      return runs / inn if dis == 0 else runs / dis
    
    def DP(balls, dots):
      return (dots / balls) * 100
    
    summary_df['SR'] = summary_df.apply(lambda x: (x['runs'] / x['ball']) * 100, axis=1)
    
    summary_df['BPD'] = summary_df.apply(lambda x: bpd(x['ball'], x['dismissals']), axis=1)
    summary_df['BPB'] = summary_df.apply(lambda x: bpb(x['ball'], (x['fours'] + x['sixes'])), axis=1)
    summary_df['nbdry_sr'] = summary_df.apply(lambda x: ((x['dots'] * 0 + x['ones'] * 1 + x['twos'] * 2 + x['threes'] * 3) /(x['dots'] + x['ones'] + x['twos'] + x['threes']) * 100) if (x['dots'] + x['ones'] + x['twos'] + x['threes']) > 0 else 0,axis=1)
    summary_df['AVG'] =summary_df.apply(lambda x: avg(x['runs'], x['dismissals'], x['innings']), axis=1)
    summary_df['dot_percentage'] = (summary_df['dots'] / summary_df['ball']) * 100


    debut_year = temp_df.groupby('batsman')['season'].min().reset_index()
    final_year = temp_df.groupby('batsman')['season'].max().reset_index()
    debut_year.rename(columns={'season': 'debut_year'}, inplace=True)
    final_year.rename(columns={'season': 'final_year'}, inplace=True)
    summary_df = summary_df.merge(debut_year, on='batsman').merge(final_year, on='batsman')


    # Merging matches data
    summary_df = summary_df.merge(matches, on='batsman', how='left')

    return summary_df

def bcum(df):
    # First, remove duplicates based on match_id and ball within the same match
    print(f"Before removing duplicates based on 'match_id' and 'ball': {df.shape}")
    df = df.drop_duplicates(subset=['match_id', 'ball'], keep='first')
    print(f"After removing duplicates based on 'match_id' and 'ball': {df.shape}")

    # Define helper columns for various runs
    df['is_dot'] = df['total_runs'].apply(lambda x: 1 if x == 0 else 0)
    df['is_one'] = df['batsman_runs'].apply(lambda x: 1 if x == 1 else 0)
    df['is_two'] = df['batsman_runs'].apply(lambda x: 1 if x == 2 else 0)
    df['is_three'] = df['batsman_runs'].apply(lambda x: 1 if x == 3 else 0)
    df['is_four'] = df['batsman_runs'].apply(lambda x: 1 if x == 4 else 0)
    df['is_six'] = df['batsman_runs'].apply(lambda x: 1 if x == 6 else 0)

    # Create various aggregates
    runs = pd.DataFrame(df.groupby(['bowler'])['batsman_runs'].sum()).reset_index().rename(columns={'batsman_runs': 'runs'})
    innings = pd.DataFrame(df.groupby(['bowler'])['match_id'].nunique()).reset_index().rename(columns={'match_id': 'innings'})
    balls = pd.DataFrame(df.groupby(['bowler'])['ball'].count()).reset_index().rename(columns={'ball': 'balls'})
    wkts = pd.DataFrame(df.groupby(['bowler'])['bowler_wkt'].sum()).reset_index().rename(columns={'bowler_wkt': 'wkts'})
    dots = pd.DataFrame(df.groupby(['bowler'])['is_dot'].sum()).reset_index().rename(columns={'is_dot': 'dots'})
    ones = pd.DataFrame(df.groupby(['bowler'])['is_one'].sum()).reset_index().rename(columns={'is_one': 'ones'})
    twos = pd.DataFrame(df.groupby(['bowler'])['is_two'].sum()).reset_index().rename(columns={'is_two': 'twos'})
    threes = pd.DataFrame(df.groupby(['bowler'])['is_three'].sum()).reset_index().rename(columns={'is_three': 'threes'})
    fours = pd.DataFrame(df.groupby(['bowler'])['is_four'].sum()).reset_index().rename(columns={'is_four': 'fours'})
    sixes = pd.DataFrame(df.groupby(['bowler'])['is_six'].sum()).reset_index().rename(columns={'is_six': 'sixes'})

    # Calculate 3W or more hauls by bowler
    dismissals_count = df.groupby(['bowler', 'match_id'])['bowler_wkt'].sum()
    three_wicket_hauls = dismissals_count[dismissals_count >= 3].groupby('bowler').count().reset_index().rename(columns={'bowler_wkt': 'three_wicket_hauls'})

    # Identify maiden overs (group by match and over, check if total_runs == 0)
    df['over'] = df['ball'].apply(lambda x: int(x))  # Assuming ball represents the ball within an over
    maiden_overs = df.groupby(['bowler', 'match_id', 'over']).filter(lambda x: x['total_runs'].sum() == 0)
    maiden_overs_count = maiden_overs.groupby('bowler')['over'].count().reset_index().rename(columns={'over': 'maiden_overs'})

    # Merge all metrics into a single DataFrame
    bowl_rec = pd.merge(innings, balls, on='bowler')\
                 .merge(runs, on='bowler')\
                 .merge(wkts, on='bowler')\
                 .merge(sixes, on='bowler')\
                 .merge(fours, on='bowler')\
                 .merge(dots, on='bowler')\
                 .merge(three_wicket_hauls, on='bowler', how='left')\
                 .merge(maiden_overs_count, on='bowler', how='left')

    # Fill NaN values for bowlers with no 3W hauls or maiden overs
    bowl_rec['three_wicket_hauls'] = bowl_rec['three_wicket_hauls'].fillna(0)
    bowl_rec['maiden_overs'] = bowl_rec['maiden_overs'].fillna(0)

    # Calculate additional metrics
    bowl_rec['dot%'] = (bowl_rec['dots'] / bowl_rec['balls']) * 100
    bowl_rec = bowl_rec[bowl_rec.innings >= 10]
    bowl_rec['avg'] = bowl_rec['runs'] / bowl_rec['wkts']
    bowl_rec['sr'] = bowl_rec['balls'] / bowl_rec['wkts']
    bowl_rec['econ'] = (bowl_rec['runs'] * 6 / bowl_rec['balls'])

    return bowl_rec


# Preprocess the debut column to extract the year
idf['debut_year'] = idf['debut_year'].str.split('/').str[0]  # Extract the year from "YYYY/YY"

# Convert the relevant columns to integers
columns_to_convert = ['runs', 'hundreds', 'fifties', 'thirties', 'highest_score']
idf[columns_to_convert] = idf[columns_to_convert].astype(int)

# Sidebar for selecting between "Player Profile" and "Matchup Analysis"
sidebar_option = st.sidebar.radio(
    "Select an option:",
    ("Player Profile", "Matchup Analysis","Strength vs Weakness","Team Builder")
)

if sidebar_option == "Player Profile":
    st.header("Player Profile")

    # Player search input (selectbox)
    player_name = st.selectbox("Search for a player", idf['batsman'].unique())

    # Filter the data for the selected player
    player_info = idf[idf['batsman'] == player_name].iloc[0]

    # Check if the player exists in info_df
    matching_rows = info_df[info_df['Player_name'] == player_name]

    if not matching_rows.empty:
        # If there is a matching row, access the first one
        p_info = matching_rows.iloc[0]
    else:
        # st.write(f"No player found with the name '{player_name}'")
        p_info = None  # Set a fallback

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
            st.markdown("AGE:")
            if p_info is not None:
                st.markdown(f"<span style='font-size: 20px; font-weight: bold;'>{p_info['age']}</span>", unsafe_allow_html=True)
            else:
                st.markdown("<span style='font-size: 20px; font-weight: bold;'>N/A</span>", unsafe_allow_html=True)

        # Create columns for the second row (batting style, bowling style, playing role)
        col4, col5, col6 = st.columns(3)

        with col4:
            st.markdown("BATTING STYLE:")
            if p_info is not None:
                st.markdown(f"<span style='font-size: 20px; font-weight: bold;'>{p_info['batting_style'].upper()}</span>", unsafe_allow_html=True)
            else:
                st.markdown("<span style='font-size: 20px; font-weight: bold;'>N/A</span>", unsafe_allow_html=True)

        with col5:
            st.markdown("BOWLING STYLE:")
            if p_info is not None:
                if p_info['bowling_style'] == 'Wicketkeeper':
                    st.markdown("<span style='font-size: 20px; font-weight: bold;'>NONE</span>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<span style='font-size: 20px; font-weight: bold;'>{p_info['bowling_style'].upper()}</span>", unsafe_allow_html=True)
            else:
                st.markdown("<span style='font-size: 20px; font-weight: bold;'>N/A</span>", unsafe_allow_html=True)

        with col6:
            st.markdown("PLAYING ROLE:")
            if p_info is not None:
                st.markdown(f"<span style='font-size: 20px; font-weight: bold;'>{p_info['role'].upper()}</span>", unsafe_allow_html=True)
            else:
                st.markdown("<span style='font-size: 20px; font-weight: bold;'>N/A</span>", unsafe_allow_html=True)

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
            player_stats=round_up_floats(player_stats)
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
               temp_df = pdf[pdf['batsman'] == player_name]
               if not temp_df[temp_df['batting_team'] == country].empty:
                        continue
               temp_df = temp_df[(temp_df['bowling_team'] == country)]
               temp_df = cumulator(temp_df)
                    
                    # If temp_df is empty after applying cumulator, skip to the next iteration
               if len(temp_df) == 0:
                   continue  
                    
                    # Drop the specified columns and modify the column names
               temp_df = temp_df.drop(columns=['final_year', 'batsman', 'batting_team','debut_year','matches_x','matches_y'])
            #    # Convert specific columns to integers
            #    # Round off the remaining float columns to 2 decimal places
            #    float_cols = temp_df.select_dtypes(include=['float']).columns
            #    temp_df[float_cols] = temp_df[float_cols].round(2) 
               temp_df = round_up_floats(temp_df) 
               columns_to_convert = ['runs', 'hundreds', 'fifties', 'thirties', 'highest_score']

               # Fill NaN values with 0
               temp_df[columns_to_convert] = temp_df[columns_to_convert].fillna(0)
                
               # Convert the specified columns to integer type
               temp_df[columns_to_convert] = temp_df[columns_to_convert].astype(int)
 
                
               temp_df.columns = [col.upper().replace('_', ' ') for col in temp_df.columns]
               cols = temp_df.columns.tolist()

               # Specify the desired order with 'year' first
               new_order = ['MATCHES'] + [col for col in cols if col != 'MATCHES']
                         
               # Reindex the DataFrame with the new column order
               temp_df =temp_df[new_order]
 
                    
                    # Display the results
               st.markdown(f"### vs **{country.upper()}**")
               
               # Display the table with bold font
               st.table(temp_df.style.set_table_attributes("style='font-weight: bold;'"))

        
            tdf = pdf[pdf['batsman'] == player_name]

            def standardize_season(season):
                if '/' in season:  # Check if the season is in 'YYYY/YY' format
                    year = season.split('/')[0]  # Get the first part
                else:
                    year = season  # Use as is if already in 'YYYY' format
                return year.strip()  # Return the year stripped of whitespace
            tdf['season'] =tdf['season'].apply(standardize_season)
            
            # Populate an array of unique seasons
            unique_seasons = tdf['season'].unique()
            
            # Optional: Convert to a sorted list (if needed)
            unique_seasons = sorted(set(unique_seasons))
            # print(unique_seasons)
            tdf=pd.DataFrame(tdf)
            # print(temp_df.head(50))
            tdf['batsman_runs'] = tdf['batsman_runs'].astype(int)
            tdf['total_runs'] = tdf['total_runs'].astype(int)
            # Run a for loop and pass temp_df to a cumulative function
            i=0
            for season in unique_seasons:
                print(i)
                temp_df = tdf[(tdf['season'] == season)]
                print(temp_df.head())
                temp_df = cumulator(temp_df)
                if i==0:
                    result_df = temp_df  # Initialize with the first result_df
                    i=1+i
                else:
                    result_df = pd.concat([result_df, temp_df], ignore_index=True)
                result_df = result_df.drop(columns=['batsman', 'batting_team','debut_year','matches_x','matches_y'])
                # Convert specific columns to integers
                # Round off the remaining float columns to 2 decimal places
                float_cols = result_df.select_dtypes(include=['float']).columns
                result_df[float_cols] = result_df[float_cols].round(2)
                columns_to_convert = ['runs', 'hundreds', 'fifties', 'thirties', 'highest_score']

               # Fill NaN values with 0
                temp_df[columns_to_convert] = temp_df[columns_to_convert].fillna(0)
                
               # Convert the specified columns to integer type
                temp_df[columns_to_convert] = temp_df[columns_to_convert].astype(int)
            columns_to_convert = ['runs', 'hundreds', 'fifties', 'thirties', 'highest_score']

               # Fill NaN values with 0
            temp_df[columns_to_convert] = temp_df[columns_to_convert].fillna(0)
                
               # Convert the specified columns to integer type
            temp_df[columns_to_convert] = temp_df[columns_to_convert].astype(int)
            result_df=result_df.rename(columns={'final_year':'year'})
            result_df.columns = [col.upper().replace('_', ' ') for col in result_df.columns]
            result_df = round_up_floats(result_df)
            columns_to_convert = ['RUNS', 'HUNDREDS', 'FIFTIES', 'THIRTIES', 'HIGHEST SCORE']

               # Fill NaN values with 0
            result_df[columns_to_convert] = result_df[columns_to_convert].fillna(0)
                
               # Convert the specified columns to integer type
            result_df[columns_to_convert] = result_df[columns_to_convert].astype(int)
                    
            # Display the results
            st.markdown(f"### **Yearwise Performnce**")
            cols = result_df.columns.tolist()

            # Specify the desired order with 'year' first
            new_order = ['YEAR'] + [col for col in cols if col != 'YEAR']
                     
            # Reindex the DataFrame with the new column order
            result_df = result_df[new_order]
            st.table(result_df.style.set_table_attributes("style='font-weight: bold;'"))

            tdf = pdf[pdf['batsman'] == player_name]
            temp_df=tdf[(tdf['inning']==1)]
            temp_df=cumulator(temp_df)
            temp_df['inning']=1
            cols = temp_df.columns.tolist()
            new_order = ['inning'] + [col for col in cols if col != 'inning']          
            # Reindex the DataFrame with the new column order
            temp_df =temp_df[new_order] 
            result_df = temp_df
            temp_df=tdf[(tdf['inning']==2)]
            temp_df=cumulator(temp_df)
            temp_df['inning']=2
            cols = temp_df.columns.tolist()
            new_order = ['inning'] + [col for col in cols if col != 'inning']          
            # Reindex the DataFrame with the new column order
            temp_df =temp_df[new_order] 
            result_df = pd.concat([result_df, temp_df], ignore_index=True)
            result_df = result_df.drop(columns=['batsman', 'batting_team','debut_year','matches_x','matches_y','final_year'])
            # Convert specific columns to integers
            # Round off the remaining float columns to 2 decimal places
            float_cols = result_df.select_dtypes(include=['float']).columns
            result_df[float_cols] = result_df[float_cols].round(2)
            
            result_df=result_df.rename(columns={'final_year':'year'})
            result_df.columns = [col.upper().replace('_', ' ') for col in result_df.columns]
            columns_to_convert = ['RUNS', 'HUNDREDS', 'FIFTIES', 'THIRTIES', 'HIGHEST SCORE']

               # Fill NaN values with 0
            result_df[columns_to_convert] = result_df[columns_to_convert].fillna(0)
                
               # Convert the specified columns to integer type
            result_df[columns_to_convert] = result_df[columns_to_convert].astype(int)
                    
            # Display the results
            result_df = result_df.drop(columns=['MATCHES'])
            st.markdown(f"### **Inningwise Performnce**")
            st.table(result_df.style.set_table_attributes("style='font-weight: bold;'"))

            venue_country_map = {
            'Melbourne Cricket Ground': 'Australia',
            'Simonds Stadium, South Geelong': 'Australia',
            'Adelaide Oval': 'Australia',
            'Sinhalese Sports Club Ground': 'Sri Lanka',
            'Saxton Oval': 'New Zealand',
            'Asian Institute of Technology Ground': 'Thailand',
            'North Sydney Oval': 'Australia',
            'Manuka Oval': 'Australia',
            'Coolidge Cricket Ground': 'Antigua',
            'Sharjah Cricket Stadium': 'UAE',
            'Senwes Park': 'South Africa',
            'Buffalo Park': 'South Africa',
            'The Wanderers Stadium': 'South Africa',
            'SuperSport Park': 'South Africa',
            'Newlands': 'South Africa',
            'The Cooper Associates County Ground': 'England',
            'County Ground': 'England',
            'Brabourne Stadium': 'India',
            'Bay Oval': 'New Zealand',
            'Pukekura Park': 'New Zealand',
            'Seddon Park': 'New Zealand',
            'Nondescripts Cricket Club Ground': 'Sri Lanka',
            'Mangaung Oval': 'South Africa',
            'Allan Border Field': 'Australia',
            'VRA Ground': 'Netherlands',
            'Kinrara Academy Oval': 'Malaysia',
            'Royal Selangor Club': 'Malaysia',
            'Providence Stadium': 'Guyana',
            'Daren Sammy National Cricket Stadium, Gros Islet': 'St Lucia',
            'Sir Vivian Richards Stadium, North Sound': 'Antigua',
            'Westpac Stadium': 'New Zealand',
            'Eden Park': 'New Zealand',
            'Brian Lara Stadium, Tarouba': 'Trinidad and Tobago',
            'Colts Cricket Club Ground': 'Sri Lanka',
            'Colombo Cricket Club Ground': 'Sri Lanka',
            'Chilaw Marians Cricket Club Ground': 'Sri Lanka',
            'County Ground, Hove': 'England',
            'Barsapara Cricket Stadium': 'India',
            'Southend Club Cricket Stadium': 'Pakistan',
            'Sydney Showground Stadium': 'Australia',
            'W.A.C.A. Ground': 'Australia',
            'Junction Oval': 'Australia',
            'Sydney Cricket Ground': 'Australia',
            'P Sara Oval': 'Sri Lanka',
            'LC de Villiers Oval': 'South Africa',
            'City Oval': 'South Africa',
            'Willowmoore Park': 'South Africa',
            'Basin Reserve': 'New Zealand',
            'Forthill': 'Scotland',
            'Kensington Oval, Barbados': 'Barbados',
            'Lalabhai Contractor Stadium': 'India',
            'Darren Sammy National Cricket Stadium, St Lucia': 'St Lucia',
            'Gaddafi Stadium': 'Pakistan',
            'Kingsmead': 'South Africa',
            'Sky Stadium': 'New Zealand',
            'McLean Park': 'New Zealand',
            'Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium': 'India',
            'County Ground, Northampton': 'England',
            'County Ground, Chelmsford': 'England',
            'The Cooper Associates County Ground, Taunton': 'England',
            'Carrara Oval': 'Australia',
            'Coolidge Cricket Ground, Antigua': 'Antigua',
            'John Davies Oval, Queenstown': 'New Zealand',
            'Edgbaston, Birmingham': 'England',
            'Kinrara Academy Oval, Kuala Lumpur': 'Malaysia',
            'County Ground, New Road, Worcester': 'England',
            'County Ground, Derby': 'England',
            'Riverside Ground, Chester-le-Street': 'England',
            'County Ground, Bristol': 'England',
            'Bready Cricket Club, Magheramason, Bready': 'Ireland',
            'Rangiri Dambulla International Stadium': 'Sri Lanka',
            'Sheikh Zayed Stadium, Abu Dhabi': 'UAE',
            'Sylhet International Cricket Stadium, Academy Ground': 'Bangladesh',
            'Kennington Oval, London': 'England',
            "Lord's, London": 'England',
            'St George\'s Park, Gqeberha': 'South Africa',
            'Hagley Oval, Christchurch': 'New Zealand',
            'Bellerive Oval, Hobart': 'Australia',
            'Dr DY Patil Sports Academy, Mumbai': 'India',
            'National Stadium, Karachi': 'Pakistan',
            'Shere Bangla National Stadium, Mirpur': 'Bangladesh',
            'Diamond Oval, Kimberley': 'South Africa',
            'Headingley, Leeds': 'England',
            'The Rose Bowl, Southampton': 'England',
            'Trent Bridge': 'England',
            'Wankhede Stadium, Mumbai': 'India',
            'Eden Gardens': 'India',
            # Add more as needed
            }
            
            # Creating a DataFrame to display venues and their corresponding countries
            pdf['country'] = pdf['venue'].map(venue_country_map)
            allowed_countries = ['India', 'England', 'Australia', 'Pakistan', 'Bangladesh',
                     'West Indies', 'Scotland', 'South Africa', 'New Zealand', 'Sri Lanka']
            i=0
            for country in allowed_countries:
                temp_df = pdf[pdf['batsman'] == player_name]
                # print(temp_df.match_id.unique())
                # print(temp_df.head(20))
                temp_df = temp_df[(temp_df['country'] == country)]
                temp_df = cumulator(temp_df)
                temp_df['country']=country.upper()
                cols = temp_df.columns.tolist()
                new_order = ['country'] + [col for col in cols if col != 'country']
                # Reindex the DataFrame with the new column order
                temp_df =temp_df[new_order]
                # print(temp_df)
             # If temp_df is empty after applying cumulator, skip to the next iteration
                if len(temp_df) == 0:
                   continue
                elif i==0:
                    result_df = temp_df
                    i=i+1
                else:
                    result_df = result_df.reset_index(drop=True)
                    temp_df = temp_df.reset_index(drop=True)
                    result_df = result_df.loc[:, ~result_df.columns.duplicated()]
            
                    result_df = pd.concat([result_df, temp_df],ignore_index=True)
                    
            
                result_df = result_df.drop(columns=['batsman', 'batting_team','debut_year','final_year','matches_x','matches_y'])
                # Round off the remaining float columns to 2 decimal places
                float_cols = result_df.select_dtypes(include=['float']).columns
                result_df[float_cols] = result_df[float_cols].round(2)
            result_df.columns = [col.upper().replace('_', ' ') for col in result_df.columns]
            result_df = round_up_floats(result_df)
            columns_to_convert = ['RUNS', 'HUNDREDS', 'FIFTIES', 'THIRTIES', 'HIGHEST SCORE']

               # Fill NaN values with 0
            result_df[columns_to_convert] = result_df[columns_to_convert].fillna(0)
                
               # Convert the specified columns to integer type
            result_df[columns_to_convert] = result_df[columns_to_convert].astype(int)
            cols = result_df.columns.tolist()
            new_order = ['COUNTRY'] + [col for col in cols if col != 'COUNTRY']
            result_df = result_df[new_order]
            # result_df = result_df.loc[:, ~result_df.columns.duplicated()]
            result_df = result_df.drop(columns=['MATCHES'])
            st.markdown(f"### **In Host Country**")
            st.table(result_df.style.set_table_attributes("style='font-weight: bold;'"))
          

            

            

        elif option == "Bowling":
            # Prepare the DataFrame for displaying player-specific bowling statistics
            temp_df = bidf
                
                # Filter for the selected player
            player_stats = temp_df[temp_df['bowler'] == player_name]  # Assuming bidf has bowler data
                
                # Convert column names to uppercase and replace underscores with spaces
            player_stats.columns = [col.upper().replace('_', ' ') for col in player_stats.columns]
                
                # Function to round float values if necessary (assuming round_up_floats exists)
            player_stats = round_up_floats(player_stats)
                
                # Display the player's bowling statistics in a table format with bold headers
            player_stats = player_stats.drop(columns=['UNNAMED: 0','BOWLER'])
            st.markdown("### Bowling Statistics")
            st.table(player_stats.style.set_table_attributes("style='font-weight: bold;'"))  # Display the filtered DataFrame as a table

            allowed_countries = ['India', 'England', 'Australia', 'Pakistan', 'Bangladesh', 
                         'West Indies', 'Scotland', 'South Africa', 'New Zealand', 'Sri Lanka']
    
            # Calculating total runs and renaming relevant columns
            bpdf['total_runs'] = bpdf['runs_off_bat'] + bpdf['extras']
            bpdf = bpdf.rename(columns={'runs_off_bat': 'batsman_runs', 'wicket_type': 'dismissal_kind', 'striker': 'batsman', 'innings': 'inning', 'bowler': 'bowler_name'})
            bpdf = bpdf.dropna(subset=['ball'])
            
            # Convert the 'ball' column to numeric if it's not already
            bpdf['ball'] = pd.to_numeric(bpdf['ball'], errors='coerce')
            
            # Calculate 'over' by applying lambda function
            bpdf['over'] = bpdf['ball'].apply(lambda x: mt.floor(x) + 1 if pd.notnull(x) else None)
        
            # Iterate over allowed countries for bowling analysis
            for country in allowed_countries:
                temp_df = pdf[pdf['bowler_name'] == player_name]  # Filter data for selected bowler
                
                # Check if the bowler has bowled against the current country
                if not temp_df[temp_df['bowling_team'] == country].empty:
                    continue  # Skip if no data found for this country
                
                temp_df = temp_df[temp_df['batting_team'] == country]  # Filter for the country team faced
                
                # Apply the bowler cumulation function (bcum)
                temp_df = bcum(temp_df)
                
                # If the DataFrame is empty after applying `bcum`, skip this iteration
                if len(temp_df) == 0:
                    continue
                
                # Drop unwanted columns
                # temp_df = temp_df.drop(columns=['final_year', 'bowler_name', 'bowling_team', 'debut_year', 'matches_x', 'matches_y'])
                
                # Round up float columns (assuming `round_up_floats()` is already defined)
                temp_df = round_up_floats(temp_df)
                
                # Define columns to convert to integer type
                columns_to_convert = ['runs', 'wickets', 'maidens', 'economy_rate', 'best_figures']
        
                # Fill NaN values with 0 and convert specified columns to integers
                temp_df[columns_to_convert] = temp_df[columns_to_convert].fillna(0)
                temp_df[columns_to_convert] = temp_df[columns_to_convert].astype(int)
                
                # Convert column names to uppercase and replace underscores with spaces
                temp_df.columns = [col.upper().replace('_', ' ') for col in temp_df.columns]
                
                # Reorder columns (you can adjust the column names as needed)
                cols = temp_df.columns.tolist()
                new_order = ['MATCHES'] + [col for col in cols if col != 'MATCHES']
                temp_df = temp_df[new_order]  # Reindex the DataFrame with new order
                
                # Display the results for the current country
                st.markdown(f"### vs **{country.upper()}**")
                st.table(temp_df.style.set_table_attributes("style='font-weight: bold;'"))
                    

            

    with tab3:
        st.header("Current Form")
        # Add current form content here

# If "Matchup Analysis" is selected
elif sidebar_option == "Matchup Analysis":
    pdf['total_runs'] = pdf['runs_off_bat'] + pdf['extras']
    pdf = pdf.rename(columns={'runs_off_bat': 'batsman_runs', 'wicket_type': 'dismissal_kind', 'striker': 'batsman', 'innings': 'inning'})
    pdf = pdf.dropna(subset=['ball'])
     # Convert the 'ball' column to numeric if it's not already (optional but recommended)
    pdf['ball'] = pd.to_numeric(pdf['ball'], errors='coerce') 
           # Applying the lambda function to calculate the 'over'
    pdf['over'] = pdf['ball'].apply(lambda x: mt.floor(x) + 1 if pd.notnull(x) else None)
            
    st.header("Matchup Analysis")
    # Filter unique batters and bowlers from the DataFrame
    unique_batters = pdf['batsman'].unique()  # Adjust the column name as per your PDF data structure
    unique_bowlers = pdf['bowler'].unique()    # Adjust the column name as per your PDF data structure

    # Search box for Batters
    batter_name = st.selectbox("Select a Batter", unique_batters)

    # Search box for Bowlers
    bowler_name = st.selectbox("Select a Bowler", unique_bowlers)

    # Dropdown for grouping options
    grouping_option = st.selectbox("Group By", ["Year", "Match", "Venue", "Inning"])
    matchup_df = pdf[(pdf['batsman'] == batter_name) & (pdf['bowler'] == bowler_name)]

    # Step 3: Create a download option for the DataFrame
    if not matchup_df.empty:
        # Convert the DataFrame to CSV format
        csv = matchup_df.to_csv(index=False)  # Generate CSV string
        
        # Step 4: Create the download button
        st.download_button(
            label="Download Matchup Data as CSV",
            data=csv,  # Pass the CSV string directly
            file_name=f"{batter_name}_vs_{bowler_name}_matchup.csv",
            mime="text/csv"  # Specify the MIME type for CSV
        )
    else:
        st.warning("No data available for the selected matchup.")
    if grouping_option == "Year":
        tdf = pdf[(pdf['batsman'] == batter_name) & (pdf['bowler'] == bowler_name)]

        def standardize_season(season):
            if '/' in season:  # Check if the season is in 'YYYY/YY' format
                year = season.split('/')[0]  # Get the first part
            else:
                year = season  # Use as is if already in 'YYYY' format
            return year.strip()  # Return the year stripped of whitespace

        tdf['season'] = tdf['season'].apply(standardize_season)

        # Populate an array of unique seasons
        unique_seasons = tdf['season'].unique()
        
        # Optional: Convert to a sorted list (if needed)
        unique_seasons = sorted(set(unique_seasons))

        # Ensure tdf is a DataFrame
        tdf = pd.DataFrame(tdf)
        tdf['batsman_runs'] = tdf['batsman_runs'].astype(int)
        tdf['total_runs'] = tdf['total_runs'].astype(int)

        # Initialize an empty result DataFrame
        result_df = pd.DataFrame()
        i=0
        # Run a for loop and pass temp_df to a cumulative function
        for season in unique_seasons:
            temp_df = tdf[tdf['season'] == season]
            temp_df = cumulator(temp_df)

            if i==0:
                    result_df = temp_df  # Initialize with the first result_df
                    i=1+i
            else:
                    result_df = pd.concat([result_df, temp_df], ignore_index=True)
        # Drop unnecessary columns related to performance metrics
        columns_to_drop = ['batsman', 'bowler', 'batting_team', 'debut_year', 'matches_x', 'matches_y', 'fifties', 'hundreds', 'thirties', 'highest_score','matches']
        result_df = result_df.drop(columns=columns_to_drop, errors='ignore')

        # Convert specific columns to integers and fill NaN values
        columns_to_convert = ['runs','dismissals']
        for col in columns_to_convert:
            result_df[col] = result_df[col].fillna(0).astype(int)

        result_df = result_df.rename(columns={'final_year': 'year'})
        result_df.columns = [col.upper().replace('_', ' ') for col in result_df.columns]

        # Display the results
        st.markdown("### **Yearwise Performance**")
        cols = result_df.columns.tolist()

        # Specify the desired order with 'year' first
        new_order = ['YEAR'] + [col for col in cols if col != 'YEAR']
                  
        # Reindex the DataFrame with the new column order
        result_df = result_df[new_order]
        st.table(result_df.style.set_table_attributes("style='font-weight: bold;'"))
    elif grouping_option == "Match":
        tdf = pdf[(pdf['batsman'] == batter_name) & (pdf['bowler'] == bowler_name)]

        # Populate an array of unique match IDs
        unique_matches = sorted(set(tdf['match_id'].unique()))

        # Ensure tdf is a DataFrame
        tdf = pd.DataFrame(tdf)
        tdf['batsman_runs'] = tdf['batsman_runs'].astype(int)
        tdf['total_runs'] = tdf['total_runs'].astype(int)

        # Initialize an empty result DataFrame
        result_df = pd.DataFrame()
        i = 0

        # Run a for loop and pass temp_df to a cumulative function
        for match_id in unique_matches:
            temp_df = tdf[tdf['match_id'] == match_id]
            current_match_id = match_id
            temp_df = cumulator(temp_df)
            temp_df.insert(0, 'MATCH_ID', current_match_id)

            if i == 0:
                result_df = temp_df  # Initialize with the first result_df
                i = 1 + i
            else:
                result_df = pd.concat([result_df, temp_df], ignore_index=True)

        # Drop unnecessary columns related to performance metrics
        # result_df['last_year']=result_df['last_year'].apply(standardize_season)
        # result_df = result_df.rename(columns={'final_year': 'year'})
        columns_to_drop = ['batsman', 'bowler', 'batting_team', 'debut_year', 'matches_x', 'matches_y', 
                           'fifties', 'hundreds', 'thirties', 'highest_score', 'season','matches']
        result_df = result_df.drop(columns=columns_to_drop, errors='ignore')

        # Convert specific columns to integers and fill NaN values
        columns_to_convert = ['runs', 'dismissals']
        for col in columns_to_convert:
            result_df[col] = result_df[col].fillna(0).astype(int)

        # Rename columns for better presentation
        result_df = result_df.rename(columns={'match_id': 'MATCH ID'})
        
        
        result_df.columns = [col.upper().replace('_', ' ') for col in result_df.columns]
        result_df['FINAL YEAR']=result_df['FINAL YEAR'].apply(standardize_season)
        
        result_df = result_df.rename(columns={'FINAL YEAR': 'YEAR'})  

        # Display the results
        st.markdown("### **Matchwise Performance**")
        cols = result_df.columns.tolist()

        # Reindex the DataFrame with the new column order
        result_df=result_df.sort_values('YEAR',ascending=True)
        result_df=result_df[['MATCH ID'] + ['YEAR'] + [col for col in result_df.columns if col not in ['MATCH ID','YEAR']]]
        st.table(result_df.style.set_table_attributes("style='font-weight: bold;'"))
    elif grouping_option == "Venue":
        # Filter the DataFrame for the selected batsman and bowler
        tdf = pdf[(pdf['batsman'] == batter_name) & (pdf['bowler'] == bowler_name)]
    
        # Ensure tdf is a DataFrame and populate unique venue values
        tdf = pd.DataFrame(tdf)
        tdf['batsman_runs'] = tdf['batsman_runs'].astype(int)
        tdf['total_runs'] = tdf['total_runs'].astype(int)
    
        # Initialize an empty result DataFrame
        result_df = pd.DataFrame()
        i = 0
    
        # Populate an array of unique venues
        unique_venues = tdf['venue'].unique()
        
        for venue in unique_venues:
            # Filter temp_df for the current venue
            temp_df = tdf[tdf['venue'] == venue]
    
            # Store the current venue in a variable
            current_venue = venue
    
            # Call the cumulator function
            temp_df = cumulator(temp_df)
    
            # Insert the current venue as the first column in temp_df
            temp_df.insert(0, 'VENUE', current_venue)
    
            # Concatenate results
            if i == 0:
                result_df = temp_df  # Initialize with the first result_df
                i += 1
            else:
                result_df = pd.concat([result_df, temp_df], ignore_index=True)
    
        # Drop unnecessary columns related to performance metrics
        columns_to_drop = ['batsman', 'bowler', 'batting_team', 'debut_year', 'matches_x', 'matches_y', 'fifties', 'hundreds', 'thirties', 'highest_score', 'matches']
        result_df = result_df.drop(columns=columns_to_drop, errors='ignore')
    
        # Convert specific columns to integers and fill NaN values
        columns_to_convert = ['runs', 'dismissals']
        for col in columns_to_convert:
            result_df[col] = result_df[col].fillna(0).astype(int)
    
        # Rename and format columns
        # result_df['last_year']=result_df['last_year'].apply(standardize_season)
        
        # result_df = result_df.rename(columns={'final_year': 'year'})    
        result_df.columns = [col.upper().replace('_', ' ') for col in result_df.columns]
        result_df['FINAL YEAR']=result_df['FINAL YEAR'].apply(standardize_season)
        
        result_df = result_df.rename(columns={'FINAL YEAR': 'YEAR'})   
    
        # Display the results
        st.markdown("### **Venuewise Performance**")
        cols = result_df.columns.tolist()
    
        # Specify the desired order with 'venue' first
        new_order = ['VENUE'] + [col for col in cols if col != 'VENUE']
        
                      
        # Reindex the DataFrame with the new column order
        result_df = result_df[new_order]
        result_df=result_df.sort_values('YEAR',ascending=True)
        result_df=result_df[['VENUE'] + ['YEAR'] + [col for col in result_df.columns if col not in ['VENUE','YEAR']]]
        st.table(result_df.style.set_table_attributes("style='font-weight: bold;'"))
    else:
        # Assuming pdf is your main DataFrame
        # Filter for innings 1 and 2 and prepare to accumulate results
        innings = [1, 2]
        result_df = pd.DataFrame()  # Initialize an empty DataFrame for results
        
        for inning in innings:
            # Filter for the specific inning
            tdf = pdf[(pdf['batsman'] == batter_name) & (pdf['bowler'] == bowler_name) & (pdf['inning'] == inning)]
            
            # Check if there's any data for the current inning
            if not tdf.empty:
                # Call the cumulator function
                temp_df = cumulator(tdf)
        
                # Add the inning as the first column in temp_df
                temp_df.insert(0, 'INNING', inning)
        
                # Concatenate to the main result DataFrame
                result_df = pd.concat([result_df, temp_df], ignore_index=True)
        
        # After processing both innings, drop unnecessary columns if needed
        columns_to_drop = ['batsman', 'bowler', 'batting_team', 'debut_year', 'matches_x', 'matches_y', 'fifties', 'hundreds', 'thirties', 'highest_score', 'matches']
        result_df = result_df.drop(columns=columns_to_drop, errors='ignore')
        
        # Convert specific columns to integers and fill NaN values
        columns_to_convert = ['runs', 'dismissals']
        for col in columns_to_convert:
            result_df[col] = result_df[col].fillna(0).astype(int)
        
        result_df.columns = [col.upper().replace('_', ' ') for col in result_df.columns]
        result_df['FINAL YEAR']=result_df['FINAL YEAR'].apply(standardize_season)
        
        result_df = result_df.rename(columns={'FINAL YEAR': 'YEAR'})   
        result_df=result_df.sort_values('YEAR',ascending=True)
        
        # Display the results
        st.markdown("### **Innings Performance**")
        result_df=result_df[['INNING'] + ['YEAR'] + [col for col in result_df.columns if col not in ['INNING','YEAR']]]
        st.table(result_df.style.set_table_attributes("style='font-weight: bold;'"))

        
    
    
    
    
