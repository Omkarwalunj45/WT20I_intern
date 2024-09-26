import streamlit as st
import pandas as pd
import math as mt
import numpy as np

# Page settings
st.set_page_config(page_title='WT20I Performance Analysis Portal', layout='wide')
st.title('WT20I Performance Analysis Portal')

# Load data
pdf = pd.read_csv("Dataset/up_com_wt20i.csv",low_memory=False)
idf = pd.read_csv("Dataset/updated_wt20i.csv",low_memory=False)
ldf = pd.read_csv("Dataset/squads.csv",low_memory=False)  # Load squads.csv for batting type
idf[['runs', 'hundreds', 'fifties', 'thirties', 'highest_score']] = idf[['runs', 'hundreds', 'fifties', 'thirties', 'highest_score']].astype(int)
pdf = pdf.drop_duplicates(subset=['match_id', 'ball'], keep='first')


def cumulator(df):
  print(f"Before removing duplicates based on 'match_id' and 'ball': {df.shape}")
  df = df.drop_duplicates(subset=['match_id', 'ball'], keep='first')
    
  print(f"After removing duplicates based on 'match_id' and 'ball': {temp_df.shape}")
  import pandas as pd
  import math as mt
  import numpy as np
  # Create new columns for counting runs
  df['is_dot'] = df['total_runs'].apply(lambda x: 1 if x == 0 else 0)
  df['is_one'] = df['batsman_runs'].apply(lambda x: 1 if x == 1 else 0)
  df['is_two'] = df['batsman_runs'].apply(lambda x: 1 if x == 2 else 0)
  df['is_three'] = df['batsman_runs'].apply(lambda x: 1 if x == 3 else 0)
  df['is_four'] = df['batsman_runs'].apply(lambda x: 1 if x == 4 else 0)
  df['is_six'] = df['batsman_runs'].apply(lambda x: 1 if x == 6 else 0)

  # Calculate runs, balls faced, innings, and dismissals
  runs = df.groupby(['batsman'])['batsman_runs'].sum().reset_index().rename(columns={'batsman_runs': 'runs'})
  balls = df.groupby(['batsman'])['ball'].count().reset_index()
  inn = df.groupby(['batsman'])['match_id'].apply(lambda x: len(list(np.unique(x)))).reset_index().rename(columns={'match_id': 'innings'})
  matches = df.groupby(['batsman'])['match_id'].nunique().reset_index().rename(columns={'match_id': 'matches'})
  dis = df.groupby(['batsman'])['player_dismissed'].count().reset_index().rename(columns={'player_dismissed': 'dismissals'})
  sixes = df.groupby(['batsman'])['is_six'].sum().reset_index().rename(columns={'is_six': 'sixes'})
  fours = df.groupby(['batsman'])['is_four'].sum().reset_index().rename(columns={'is_four': 'fours'})
  dots = df.groupby(['batsman'])['is_dot'].sum().reset_index().rename(columns={'is_dot': 'dots'})
  ones = df.groupby(['batsman'])['is_one'].sum().reset_index().rename(columns={'is_one': 'ones'})
  twos = df.groupby(['batsman'])['is_two'].sum().reset_index().rename(columns={'is_two': 'twos'})
  threes = df.groupby(['batsman'])['is_three'].sum().reset_index().rename(columns={'is_three': 'threes'})
  bat_team = df.groupby(['batsman'])['batting_team'].unique().reset_index()

  # Convert the array of countries to a string without brackets
  bat_team['batting_team'] = bat_team['batting_team'].apply(lambda x: ', '.join(x)).str.replace('[', '').str.replace(']', '')

  match_runs = df.groupby(['batsman', 'match_id'])['batsman_runs'].sum().reset_index()

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
  matches = df.groupby('batsman')['match_id'].nunique().reset_index(name='matches')

  # Merging matches data
  bat_rec = bat_rec.merge(matches, on='batsman', how='left')

  # Reset index for the final DataFrame
  bat_rec.reset_index(inplace=True, drop=True)
  # Ensure to count matches as well
  matches = df.groupby('batsman')['match_id'].nunique().reset_index(name='matches')

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
  debut_year = df.groupby('batsman')['season'].min().reset_index()
  final_year = df.groupby('batsman')['season'].max().reset_index()
  debut_year.rename(columns={'season': 'debut_year'}, inplace=True)
  final_year.rename(columns={'season': 'final_year'}, inplace=True)

  # Merging career span into bat_rec DataFrame
  bat_rec = bat_rec.merge(debut_year, on='batsman').merge(final_year, on='batsman')
  bat_rec['hundreds'] = bat_rec['hundreds'].fillna(0).astype(int)
    
    # Replace NaN values with 0 and convert to int for 'fifties'
  bat_rec['fifties'] = bat_rec['fifties'].fillna(0).astype(int)

    # Replace NaN values with 0 and convert to int for 'thirties'
  bat_rec['thirties'] = bat_rec['thirties'].fillna(0).astype(int)

    # Replace NaN values with 0 and convert to int for 'highest_score'
  bat_rec['highest_score'] = bat_rec['highest_score'].fillna(0).astype(int)
  bat_rec['runs'] = bat_rec['runs'].fillna(0).astype(int)  


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
               temp_df = pdf[pdf['batsman'] == player_name]
               if not temp_df[temp_df['batting_team'] == country].empty:
                        continue
               temp_df = temp_df[(temp_df['bowling_team'] == country)]
               temp_df = cumulator(temp_df)
                    
                    # If temp_df is empty after applying cumulator, skip to the next iteration
               if len(temp_df) == 0:
                   continue  
                    
                    # Drop the specified columns and modify the column names
               temp_df = temp_df.drop(columns=['final_year', 'batsman', 'batting_team','debut_year'])
               # Convert specific columns to integers
               # Round off the remaining float columns to 2 decimal places
               float_cols = temp_df.select_dtypes(include=['float']).columns
               temp_df[float_cols] = temp_df[float_cols].round(2) 
                
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
                result_df = result_df.drop(columns=['batsman', 'batting_team','debut_year'])
                # Convert specific columns to integers
                # Round off the remaining float columns to 2 decimal places
                float_cols = result_df.select_dtypes(include=['float']).columns
                result_df[float_cols] = result_df[float_cols].round(2)
            result_df=result_df.rename(columns={'final_year':'year'})
            result_df.columns = [col.upper().replace('_', ' ') for col in result_df.columns]
                    
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
            result_df = result_df.drop(columns=['batsman', 'batting_team','debut_year','matches','final_year'])
            # Convert specific columns to integers
            # Round off the remaining float columns to 2 decimal places
            float_cols = result_df.select_dtypes(include=['float']).columns
            result_df[float_cols] = result_df[float_cols].round(2)
            result_df=result_df.rename(columns={'final_year':'year'})
            result_df.columns = [col.upper().replace('_', ' ') for col in result_df.columns]
                    
            # Display the results
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
            i=0
            allowed_countries = ['India', 'England', 'Australia', 'Pakistan', 'Bangladesh', 
                                 'West Indies', 'Scotland', 'South Africa', 'New Zealand', 'Sri Lanka']
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
             # If temp_df is empty after applying cumulator, skip to the next iteration
                if len(temp_df) == 0:
                   continue  
                elif i==0:
                    result_df = temp_df
                    i=i+1
                else:
                    result_df = pd.concat([result_df, temp_df], ignore_index=True)
                    
                result_df = result_df.drop(columns=['batsman', 'batting_team','debut_year','final_year','matches'])
                # # Round off the remaining float columns to 2 decimal places
                # float_cols = result_df.select_dtypes(include=['float']).columns
                # result_df[float_cols] = result_df[float_cols].round(2)
            result_df.columns = [col.upper().replace('_', ' ') for col in result_df.columns]
            cols = result_df.columns.tolist()
            new_order = ['COUNTRY'] + [col for col in cols if col != 'COUNTRY']
            result_df = result_df[new_order]
            # print(result_df)
            st.table(result_df.style.set_table_attributes("style='font-weight: bold;'"))

            

            

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
