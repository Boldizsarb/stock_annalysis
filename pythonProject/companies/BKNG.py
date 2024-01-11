
import streamlit as st
from correlation import calculate_bkng_correlation
from EDA.EDA_BKNG import BKNG_EDA
from forecast_prediction.bkng_prediction import lstm_prediction
from forecast_prediction.bkng_prediction import lstm_prediction_forecast
from forecast_prediction.bkng_prediction import linear_regression_forecast
from forecast_prediction.bkng_prediction import linear_regression
from forecast_prediction.bkng_prediction import facebook2_model
from forecast_prediction.bkng_prediction import arima


from selling_index.bkng_selling_index import sell_prediction_forecast2

### This file is for ASML company computations only
## this file is BEING IMPORTED TO GUI  to the choices

def initialize_session_state():
    if 'show_correlation23' not in st.session_state:
        st.session_state['show_correlation23'] = False
    if 'show_eda23' not in st.session_state:
        st.session_state['show_eda23'] = False
    if 'show_prediction23' not in st.session_state:
        st.session_state['show_prediction23'] = False
    if 'show_additional_analysis23' not in st.session_state:  # New session state variable
        st.session_state['show_additional_analysis23'] = False
    if 'days_to_forecast23' not in st.session_state:
        st.session_state['days_to_forecast23'] = 30
    if 'days_to_forecast223' not in st.session_state:
        st.session_state['days_to_forecast223'] = 30
    if 'months_to_forecast23' not in st.session_state:
        st.session_state['months_to_forecast23'] = 3
    if 'timeframe23' not in st.session_state:
        st.session_state['timeframe23'] = '7 Days'

        # Initialize session state variables
initialize_session_state()


def bkng():
    st.write(
        "**About the company:** ASML Holding is a Dutch multinational company specializing in the development and production of advanced semiconductor lithography machines. Founded in 1984, ASML has become a leading player in the semiconductor industry, providing critical equipment that enables the production of smaller and more powerful microchips used in a wide range of electronic devices. The company's cutting-edge technology plays a crucial role in advancing the capabilities of modern electronics, contributing to the continuous evolution of computing and communication technologies worldwide.")
    initialize_session_state() ## initializing the state variables!!!
    # Layout for buttons
    button_col1, button_col2, button_col3, button_col4 = st.columns(4)  # Added a fourth column

    # Button interactions
    if button_col1.button("Show Correlations"):
        st.session_state['show_correlation23'] = True
        st.session_state['show_eda23'] = False
        st.session_state['show_prediction23'] = False
        st.session_state['show_additional_analysis23'] = False

    if button_col2.button("Perform EDA"):
        st.session_state['show_eda23'] = True
        st.session_state['show_correlation23'] = False
        st.session_state['show_prediction23'] = False
        st.session_state['show_additional_analysis23'] = False

    if button_col3.button("Prediction / Forecasting"):
        st.session_state['show_prediction23'] = True
        st.session_state['show_correlation23'] = False
        st.session_state['show_eda23'] = False
        st.session_state['show_additional_analysis23'] = False

    if button_col4.button("Sell or Buy "):
        st.session_state['show_additional_analysis23'] = True
        st.session_state['show_correlation23'] = False
        st.session_state['show_eda23'] = False
        st.session_state['show_prediction23'] = False

    # Correlation Analysis Section
    if st.session_state['show_correlation23']:
        st.title("BKNG Correlation Analysis")
        positive_correlations, negative_correlations = calculate_bkng_correlation()
        st.subheader("Top 10 Positive Correlations for BKNG ")
        st.bar_chart(positive_correlations)
        st.subheader("Top 10 Negative Correlations for BKNG ")
        st.bar_chart(negative_correlations)

    # EDA Section
    if st.session_state['show_eda23']:

        BKNG_EDA()

    # Prediction / Forecasting Section
    if st.session_state['show_prediction23']:
        st.write("The data is being trained on the neural network... it might take a minute or two!")
        with st.expander("LSTM Past Prediction"):
            lstm_prediction()
        with st.expander("LSTM Neural Network for Forecasting"):
            st.session_state['days_to_forecast23'] = st.number_input(
                'Enter the number of days to forecast *(min: 5 days, max: 90 days)*:',
                min_value=5, max_value=90, value=st.session_state['days_to_forecast23'],
                key='input_days_to_forecast')

            if st.session_state['days_to_forecast23'] > 1:
                lstm_prediction_forecast(st.session_state['days_to_forecast23'])
        with st.expander("Linear Regression Prediction"):
            linear_regression()
        with st.expander("Linear Regression Forecast"):
            st.session_state['days_to_forecast223'] = st.number_input(# Use days_to_forecast223
                'Enter the number of days to forecast using Linear Regression *(min: 5 days, max: 90 days)*:',
                min_value=5, max_value=90, value=st.session_state['days_to_forecast223'],# Use days_to_forecast223
                key='input_days_to_forecast2')
            linear_regression_forecast(days_to_forecast=st.session_state['days_to_forecast223'])  # Use days_to_forecast223
        with st.expander("Facebook Prophet model"):
            facebook2_model()  ## need to understand it first
        with st.expander("ARIMA Forecasting"):
            st.session_state['months_to_forecast23'] = st.number_input(
                'Enter the number of months to forecast using ARIMA *(min: 1 month, max: 12 months)*:',
                min_value=1, max_value=12, value=st.session_state['months_to_forecast23'],
                key='input_months_to_forecast')
            arima(forecast_months=st.session_state['months_to_forecast23'])
        with st.expander("Random Forest Model"):
            from forecast_prediction.asml_prediction import forest_model
            st.write("this is the forest model")
            forest_model()   ## forecast needed!

        ## sell buy index
    if st.session_state['show_additional_analysis23']:
        st.subheader("Sell, Buy or Hold?")
        # Adding a select box for choosing the timeframe
        timeframe_selection = st.selectbox(
            "Choose the Time Frame for Prediction:",
            ('7 Days', '14 Days', '30 Days'),
            key='timeframe'
        )

        # Conditional execution based on the selected timeframe
        if timeframe_selection == '7 Days':
            sell_prediction_forecast2(7)
        elif timeframe_selection == '14 Days':
            sell_prediction_forecast2(14)
        elif timeframe_selection == '30 Days':
            sell_prediction_forecast2(30)






