import streamlit as st
from datetime import date

from prediction_per_hours import get_prediction
from utils import plot

import yfinance as yf
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Stock Forecast App')

stocks = ('SAN.MC', 'IAG.MC', 'BBVA.MC', '^IBEX')
selected_stock = st.selectbox('Select dataset for prediction', stocks)

n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365


@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

	
data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

st.subheader('Raw data')
st.write(data.tail())

# Plot raw data
def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
	fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)
	
plot_raw_data()

# Predict forecast with Prophet.
df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Show and plot forecast
st.subheader('Forecast data')
st.write(forecast.tail())
    
st.write(f'Forecast plot for {n_years} years')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)

targets=['Close','High','Low','Close']

st.subheader('Prediction per hour')
check1=st.checkbox('Show', key='0')
if check1:
    target=st.selectbox('Target',targets,key='1')
    hour=st.slider('Hours of prediction',1,5,key='2')
    epochs=st.slider('Epochs',300,1000,key='3')
    prediction=st.button('Make prediction',key='4')
    if prediction:
        prediction,test_data, predicted_prices,prediction,loss=get_prediction.start(company=selected_stock,period='1wk',prediction_days=hour, target=target,interval='1h', epochs=epochs)
        plot(test_data,target)
        st.write(f'Prediction: {prediction}')
        st.write(f'Loss: {loss}')

st.subheader('Prediction per day')
check2=st.checkbox('Show', key='5')
if check2:
    target2=st.selectbox('Target',targets,key='6')
    hour2=st.slider('Days of prediction',1,3,key='7')
    epochs2=st.slider('Epochs',300,1000,key='8')
    prediction2=st.button('Make prediction',key='9')
    if prediction2:
        prediction,test_data, predicted_prices,prediction,loss=get_prediction.start(company=selected_stock,period='1wk',prediction_days=hour, target=target,interval='1h', epochs=epochs)
        plot(test_data,target2)
        st.write(f'Prediction: {prediction}')
        st.write(f'Loss: {loss}')

st.subheader('Prediction per weeks')
check3=st.checkbox('Show', key='10')
if check3:
    target3=st.selectbox('Target',targets,key='11')
    hour3=st.slider('Weeks of prediction',1,3,key='12')
    epochs3=st.slider('Epochs',300,1000,key='13')
    prediction3=st.button('Make prediction',key='14')
    hour3=hour3*5
    if prediction3:
        prediction,test_data, predicted_prices,prediction,loss=get_prediction.start(company=selected_stock,period='1wk',prediction_days=hour, target=target,interval='1h', epochs=epochs)
        plot(test_data,target3)
        st.write(f'Prediction: {prediction}')
        st.write(f'Loss: {loss}')

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
st.write('')
st.markdown('If prediction = [[nan]] contact with the programmer')