from plotly import graph_objs as go
import numpy as np
import pandas as pd
import streamlit as st

def plot(data,target):
    print(data)
    data=data.rename_axis('Date').reset_index()
    print(data)
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data[target], name='Test Data'))
    fig.layout.update(title_text='Time Series data prediction',xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

#https://github.com/Sauvageduck24/ai_forecast