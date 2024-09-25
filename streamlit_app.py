import streamlit as st
import pandas as pd
st.set_page_config(page_title='WT20I Performance Analysis Portal', layout='wide')
st.title('WT20I Performance Analysis Portal')

pdf = pd.read_csv("Dataset/WT20I_Bat.csv")
idf = pd.read_csv("Dataset/squads.csv")

# Player Profile sidebar
with col1:
    st.header("Player Profile")
    profile_option = st.radio(
        "Select an option:",
        ("Overview", "Career Statistics", "Current Form")
    )

# Matchup Analysis sidebar
with col2:
    st.header("Matchup Analysis")
    # Add content for Matchup Analysis sidebar here

# Main content area
if profile_option == "Overview":
    st.header("Overview")
    # Add overview content here
elif profile_option == "Career Statistics":
    st.header("Career Statistics")
    # Add career statistics content here
elif profile_option == "Current Form":
    st.header("Current Form")
    # Add current form content here
