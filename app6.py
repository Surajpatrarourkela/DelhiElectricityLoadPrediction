import streamlit as st
import pandas as pd
import joblib
import numpy as np
import datetime
import matplotlib.pyplot as plt
from dateutil import parser
import mysql.connector
import re

# Load trained model
xgb_model = joblib.load("electricity_load_xgb.pkl")

# Define features
FEATURE_ORDER = ['BRPL', 'BYPL', 'NDPL', 'NDMC', 'MES', 'hour', 'day', 'month', 'weekday', 'weekend']

# MySQL Connection
def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="442005",  # Replace with your MySQL password
        database="electricity_forecast"
    )

# Prepare Input for Forecast
def prepare_input(date, hour):
    """Generate input features for the selected date and hour."""
    date = pd.to_datetime(date)
    return pd.DataFrame({
        'BRPL': [1], 'BYPL': [0], 'NDPL': [0], 'NDMC': [0], 'MES': [0],
        'hour': [hour],
        'day': [date.day],
        'month': [date.month],
        'weekday': [date.weekday()],
        'weekend': [1 if date.weekday() >= 5 else 0]
    })[FEATURE_ORDER]

def forecast_load(date_input, days_to_forecast, hour):
    # Ensure the date_input is valid
    if date_input.year != 2025:
        # Force the year to 2025 if it's not already set
        date_input = date_input.replace(year=2025)
    # Generate a date range starting from the corrected date_input
    forecast_results = []
    dates = pd.date_range(start=date_input, periods=days_to_forecast)
    for forecast_date in dates:
        input_features = prepare_input(forecast_date, hour)
        prediction = xgb_model.predict(input_features)[0]
        forecast_results.append(prediction)
    # Create a DataFrame with the results
    df_results = pd.DataFrame({"Date": dates.strftime('%Y-%m-%d'), "Predicted Load (MW)": forecast_results})
    return df_results

# Plot Forecast
def plot_forecast(df_results):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df_results['Date'], df_results['Predicted Load (MW)'], marker='o', linestyle='-', color='b', label='Predicted Load')
    ax.set_xlabel("Date")
    ax.set_ylabel("Load (MW)")
    ax.set_title("Electricity Load Forecast")
    ax.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig)

# Chatbot Response
def chat_response(user_input):
    user_input = user_input.lower()
    days_to_forecast = 1
    hour = 12
    date_input = None
    # Greetings
    greetings = ["hi", "hello", "hey", "hola", "good morning", "good evening"]
    if any(greet in user_input for greet in greetings):
        return f"Hello {st.session_state.username}! How can I assist you with electricity load forecasting?"
    # General Questions
    general_questions = {
        # About the Bot
        "who are you": "I am an AI-driven assistant designed to help with electricity load forecasting and answer general queries.",
        "what is your name": "My name is AI Assistant. I assist users with electricity load forecasting and general information.",
        "how are you": "I'm just a program, so I don't have feelings, but thanks for asking!",
        "what can you do": "I can forecast electricity loads, answer industry-specific questions, and provide insights into energy consumption trends.",
        "what else can you do": "In addition to forecasting, I can answer questions about renewable energy, peak demand, and energy-saving tips.",
        "what is your work": "My primary role is to assist with electricity load forecasting and provide insights into energy-related topics.",
        "do you know me": f"Yes, I know you. Your username is {st.session_state.username}.",
        "tell me my name": f"Your name is {st.session_state.username}.",
        "what is my name": f"Your name is {st.session_state.username}.",
        "what are you showing": "I display electricity load forecasts and predictions based on your queries.",
        # Industry-Specific Questions
        "what is electricity load forecasting": "Electricity load forecasting predicts the amount of electricity that will be consumed at a given time. It helps utilities plan generation and distribution efficiently.",
        "why is load forecasting important": "Load forecasting ensures grid stability, optimizes energy generation, and reduces costs by preventing overproduction or shortages.",
        "what factors affect electricity demand": "Factors include weather conditions, time of day, seasonality, economic activity, and population growth.",
        "what is peak demand": "Peak demand refers to the highest level of electricity consumption during a specific period. It often occurs during extreme weather conditions.",
        "how is renewable energy impacting the grid": "Renewable energy sources like solar and wind reduce reliance on fossil fuels but introduce variability due to their dependence on weather conditions.",
        "what is baseload power": "Baseload power refers to the minimum level of electricity demand required over a 24-hour period. It is typically met by stable energy sources like coal, nuclear, or hydro.",
        "how does seasonality affect electricity demand": "Electricity demand is higher in summer due to air conditioning and in winter due to heating requirements.",
        "what are the challenges in forecasting": "Challenges include unpredictable weather, sudden changes in consumer behavior, and integrating renewable energy sources.",
        "tell me about delhi's electricity demand": "Delhi's electricity demand peaks during summer due to high air conditioning usage. The city also experiences fluctuations based on industrial and commercial activity.",
        "what is demand-side management": "Demand-side management involves strategies to influence consumer behavior to reduce peak demand and improve grid efficiency.",
        "how can i reduce energy consumption": "You can reduce energy consumption by using energy-efficient appliances, optimizing HVAC systems, and adopting renewable energy sources.",
    }
    # Check for general questions first
    for question, response in general_questions.items():
        if question in user_input:
            return response
    # Forecast Requests
    forecast_keywords = ["forecast", "predict", "load for", "electricity demand"]
    if any(keyword in user_input for keyword in forecast_keywords):
        date_input = parse_date_from_input(user_input)
    if "days" in user_input:
        words = user_input.split()
        for i, word in enumerate(words):
            if word.isdigit() and "days" in words[i + 1:]:
                days_to_forecast = int(word)
                break
    if not date_input:
        date_input = datetime.date.today()
    # Call forecast_load with the corrected date_input
    df_results = forecast_load(date_input, days_to_forecast, hour)
    if days_to_forecast == 1:
        return f"Forecasted Load for {date_input}: {df_results['Predicted Load (MW)'].values[0]:.2f} MW"
    elif days_to_forecast > 1:
        return df_results
    # Handle Vague Inputs
    vague_inputs = ["show me", "give me", "can you show", "display"]
    if any(vague in user_input for vague in vague_inputs):
        return "Could you please clarify what you'd like me to show? For example, ask about electricity load forecasting or energy trends."
    # Fallback to GenAI Model for Unrecognized Questions
    try:
        # Pass the input to your GenAI model
        genai_response = generate_response(user_input)  # Replace with your GenAI model's function
        return genai_response
    except Exception as e:
        print(f"Error generating GenAI response: {e}")
    return "I'm still learning and couldn't understand that. Please try rephrasing your question."

# Parse Date from Input
def parse_date_from_input(user_input):
    try:
        # Attempt to parse the date from the input
        parsed_date = parser.parse(user_input, fuzzy=True).date()
        return parsed_date
    except Exception as e:
        print(f"Error parsing date: {e}")
        return None

# Initialize Session State
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = None
    st.session_state.chat_history = []

# Logout Functionality
def logout():
    st.session_state.logged_in = False
    st.session_state.username = None
    st.session_state.chat_history = []
    st.rerun()  # Refresh the app to reflect logout

# Signup Form
def signup():
    st.subheader("Signup")
    username = st.text_input("Username")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")
    if st.button("Signup"):
        if password != confirm_password:
            st.error("Passwords do not match!")
        elif not re.match(r"[^@]+@[^@]+\.[^@]+", email):  # Validate email format using regex
            st.error("Invalid email format!")
        else:
            connection = get_db_connection()
            cursor = connection.cursor()
            cursor.execute("INSERT INTO users (username, email, password) VALUES (%s, %s, %s)",
                           (username, email, password))
            connection.commit()
            cursor.close()
            connection.close()
            st.success("Account created successfully!")

# Login Form
def login():
    st.subheader("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)
        cursor.execute("SELECT * FROM users WHERE username = %s AND password = %s", (username, password))
        user = cursor.fetchone()
        cursor.close()
        connection.close()
        if user:
            st.session_state.logged_in = True
            st.session_state.username = user["username"]
            st.success("Logged in successfully!")
        else:
            st.error("Invalid username or password!")

# Main App
def main():
    st.set_page_config(page_title="AI-Driven Electricity Forecast", layout="wide", page_icon="⚡")
    st.title("⚡ AI-Driven Delhi Electricity Load Forecast with Chatbot")

    # Add theme selection
    theme = st.sidebar.selectbox("Select Theme", ["System Default", "Light", "Dark"])
    if theme == "Light":
        st.config.set_option("theme.base", "light")
    elif theme == "Dark":
        st.config.set_option("theme.base", "dark")

    # Sidebar for Login/Signup
    if not st.session_state.logged_in:
        menu = ["Login", "Signup"]
        choice = st.sidebar.selectbox("Menu", menu)
        if choice == "Login":
            login()
        elif choice == "Signup":
            signup()
    else:
        # Logout Button
        if st.sidebar.button("Logout", key="logout_button"):
            logout()

        # User Profile Display
        with st.sidebar:
            st.markdown(f"""
            <div style="text-align: right;">
                <img src="https://upload.wikimedia.org/wikipedia/commons/7/7c/User_font_awesome.svg" 
                     alt="Profile" width="50" height="50" style="border-radius: 50%;">
                <div style="display: none;" id="tooltip">{st.session_state.username}</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("View Account Details"):
                st.write(f"Username: {st.session_state.username}")
                connection = get_db_connection()
                cursor = connection.cursor(dictionary=True)
                cursor.execute("SELECT * FROM users WHERE username = %s", (st.session_state.username,))
                user_details = cursor.fetchone()
                cursor.close()
                connection.close()
                st.write(f"Email: {user_details['email']}")
        # Chatbot Interface
        chat_history = st.session_state.get('chat_history', [])
        user_input = st.sidebar.text_input("Ask the chatbot anything about electricity load forecasting:")
        if user_input:
            chat_history.append(("User", user_input))
            bot_response = chat_response(user_input)
            if isinstance(bot_response, pd.DataFrame):
                st.sidebar.text("Forecasted Load (MW):")
                st.sidebar.dataframe(bot_response)
                plot_forecast(bot_response)
            else:
                chat_history.append(("Bot", bot_response))
                st.sidebar.text_area("Chatbot Response:", bot_response)
            st.session_state.chat_history = chat_history

        # Display Chat History
        for sender, message in chat_history:
            if sender == "User":
                st.sidebar.write(f"**{sender}:** {message}")
            else:
                st.sidebar.write(f"{sender}: {message}")

        # Today's Load Prediction
        st.subheader("📊 Today's Predicted Load for Delhi")
        today = datetime.date.today()
        today_forecast = forecast_load(today, 1, 12)
        st.metric(label="Today's Load (MW)", value=f"{today_forecast['Predicted Load (MW)'].values[0]:.2f}")

        # Next 7 Days Load Prediction
        st.subheader("📈 Predicted Load for the Next 7 Days")
        nxt_7_days_forecast = forecast_load(today, 7, 12)
        st.dataframe(nxt_7_days_forecast)
        plot_forecast(nxt_7_days_forecast)

        # Download Forecast Data
        st.subheader("📥 Download Forecast Data")
        forecast_csv = nxt_7_days_forecast.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", forecast_csv, "electricity_forecast.csv", "text/csv", key='download-csv')

if __name__ == "__main__":
    main()