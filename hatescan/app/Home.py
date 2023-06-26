import streamlit as st
import pandas as pd
from google.oauth2 import service_account
from google.cloud import bigquery
from PIL import Image

# Set the background color and font style
st.markdown(
    """
    <style>
    body {
        color: #333333;
        font-family: Arial, sans-serif;
        background-color: #F5F5F5;
    }
    .container {
        max-width: 800px;
        padding: 20px;
    }
    .title {
        font-size: 36px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 40px;
    }
    .subtitle {
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .text {
        font-size: 18px;
        margin-bottom: 20px;
    }
    .logo-image {
        display: block;
        margin: 0 auto;
        width: 300px;
        padding: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load the JavaScript code to detect the theme mode
st.markdown(
    """
    <script>
    document.addEventListener("DOMContentLoaded", function() {
        var themeMode = window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
        document.getElementById("theme-mode").innerText = themeMode;
    });
    </script>
    """,
    unsafe_allow_html=True
)

# Get the detected theme mode
theme_mode = st.markdown("Getting theme mode...").empty()

# Display the appropriate logo based on the theme mode
if theme_mode == "dark":
    st.image('hatescan/app/logo_hscan_dark.png', use_column_width=True)
else:
    st.image('hatescan/app/logo_hscan.png', use_column_width=True)

# Display the layout
# st.image(logo_path, use_column_width=True)
st.markdown("<h2>Welcome to HateScan!</h2>", unsafe_allow_html=True)
st.markdown("<p>HateScan offers you the ability to analyze language usage from real Twitter accounts, including those of your friends, celebrities, and public figures. Gain a deeper understanding of speech patterns and explore prevalent topics by comparing tweets, accounts, and global scores.</p>", unsafe_allow_html=True)
st.markdown("<p>To get started, simply click on the navigation icon located on your left.</p>", unsafe_allow_html=True)
