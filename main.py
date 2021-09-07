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

st.title('Aplicación predicción bolsa de valores')

stocks = ('SAN.MC', 'IAG.MC', 'BBVA.MC', '^IBEX')
selected_stock = st.selectbox('Seleccione la compañía para hacer la predicción', stocks)

st.subheader('Predicción por años')
check4=st.checkbox('Mostrar', key='15')
if check4:
	n_years = st.slider('Predición por años', 1, 4)
	period = n_years * 365

	@st.cache
	def load_data(ticker):
	    data = yf.download(ticker, START, TODAY)
	    data.reset_index(inplace=True)
	    return data


	data_load_state = st.text('Cargando datos...')
	data = load_data(selected_stock)
	data_load_state.text('Datos cargados... Hecho!')

	#st.subheader('Tabla de datos')
	#st.write(data.tail())

	# Plot raw data
	def plot_raw_data():
		fig = go.Figure()
		fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_apertura"))
		fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_cierre"))
		fig.layout.update(title_text='Datos mostrados de forma gráfica', xaxis_rangeslider_visible=True)
		st.plotly_chart(fig)

	# Predict forecast with Prophet.
	df_train = data[['Date','Close']]
	df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

	@st.cache
	def infer():
	    m = Prophet()
	    m.fit(df_train)
	    future = m.make_future_dataframe(periods=period)
	    forecast = m.predict(future)
	    return forecast,m

	# Show and plot forecast
	st.subheader('Datos de predicción')
	forecast,m=infer()
	#st.write(forecast.tail())

	st.write(f'Predición hecha para {n_years} año(s)')
	fig1 = plot_plotly(m, forecast)
	st.plotly_chart(fig1)

targets=['Open','High','Low','Close']

st.subheader('Predicción por horas')
check1=st.checkbox('Mostrar', key='0')
if check1:
    target=st.selectbox('Etiqueta',targets,key='1')
    hour=st.slider('Horas de la predicción',1,5,key='2')
    epochs=st.slider('Pasos',300,1000,key='3')
    prediction=st.button('Hacer predicción',key='4')
    if prediction:
        prediction,test_data, predicted_prices,prediction,loss=get_prediction.start(company=selected_stock,period='2wk',prediction_days=hour, target=target,interval='1h', epochs=epochs)
        #plot(test_data,target)
        st.write(f'Predicción: {prediction}')
        st.write(f'Pérdida en el entrenamiento: {loss}')

st.subheader('Predicción por día')
check2=st.checkbox('Mostrar', key='5')
if check2:
    target2=st.selectbox('Etiqueta',targets,key='6')
    hour2=st.slider('Días de la predicción',1,3,key='7')
    epochs2=st.slider('Pasos',300,1000,key='8')
    prediction2=st.button('Hacer predicción',key='9')
    if prediction2:
        prediction,test_data, predicted_prices,prediction,loss=get_prediction.start(company=selected_stock,period='2mo',prediction_days=hour2, target=target2,interval='1d', epochs=epochs2)
        #plot(test_data,target2)
        st.write(f'Predicción: {prediction}')
        st.write(f'Pérdida en el entrenamiento: {loss}')

st.subheader('Predicción por semanas')
check3=st.checkbox('Mostrar', key='10')
if check3:
    target3=st.selectbox('Etiqueta',targets,key='11')
    hour3=st.slider('Semanas de predicción',1,3,key='12')
    epochs3=st.slider('Pasos',300,1000,key='13')
    prediction3=st.button('Hacer predicción',key='14')
    hour3=hour3*5
    if prediction3:
        prediction,test_data, predicted_prices,prediction,loss=get_prediction.start(company=selected_stock,period='3mo',prediction_days=hour3, target=target3,interval='1d', epochs=epochs3)
        #plot(test_data,target3)
        st.write(f'Predicción: {prediction}')
        st.write(f'Pérdida en el entrenamiento: {loss}')

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
st.write('')
st.markdown('Si la predicción = [[nan]], contactar al programador')
st.markdown('Última actualización (07-09-2021 / 11:00): mejora de la perdida de entrenamiento aumentando la tasa de obtención de datos')
