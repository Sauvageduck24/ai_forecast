import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

import yfinance as yf

import tensorflow_addons as tfa
import tensorflow as tf

class get_prediction():
    
    def start(company,period, prediction_days,target,interval,epochs,test_period,interval_period):

        data=yf.download(company,period=period,interval=interval)

        scaler=MinMaxScaler(feature_range=(0,1))
        scaled_data=scaler.fit_transform(data[target].values.reshape(-1,1))

        x_train=[]
        y_train=[]

        for x in range(prediction_days, len(scaled_data)):
            x_train.append(scaled_data[x-prediction_days:x,0])
            y_train.append(scaled_data[x,0])

        x_train, y_train=np.array(x_train), np.array(y_train)
        x_train=np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        print(x_train.shape)
        lr_decayed_fn=tf.keras.experimental.CosineDecay(0.1,10)

        model=Sequential()
        model.add(tf.keras.layers.Bidirectional(LSTM(50,return_sequences=True,input_shape=(x_train.shape[1],1))))
        model.add(Dropout(.2))
        model.add(tf.keras.layers.Bidirectional(LSTM(50,return_sequences=True)))
        model.add(Dropout(.2))
        model.add(tf.keras.layers.Bidirectional(LSTM(50,return_sequences=True)))
        model.add(Dropout(.2))
        #model.add(tf.keras.layers.Bidirectional(LSTM(50)))
        model.add(LSTM(50))
        model.add(Dropout(.2))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error',optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

        #companyy=company.replace('.','')
        #companyy=companyy.lower()

        #model_ckpt=tf.keras.callbacks.ModelCheckpoint(f'checkpoints/model_pred_hour_{companyy}.h5',monitor='loss',verbose=1,save_best_only=True, save_weights_only=True,mode='max')

        #model.load_weights(f'checkpoints/model_pred_hour_{companyy}.h5')

        history=model.fit(x_train, y_train,epochs=epochs, batch_size=64, verbose=1)

        test_data=yf.download(company,period=test_period, interval=interval_period)

        actual_prices=test_data[target].values
        total_dataset=pd.concat((data[target], test_data[target]),axis=1)

        model_inputs=total_dataset[len(total_dataset)-len(test_data)-prediction_days:].values
        model_inputs=model_inputs.reshape(-1, 1)
        model_inputs=scaler.transform(model_inputs)

        x_test=[]
        for x in range(prediction_days, len(model_inputs)):
            x_test.append(model_inputs[x-prediction_days:x, 0])
        
        x_test=np.array(x_test)
        x_test=np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        predicted_prices=model.predict(x_test)
        predicted_prices=scaler.inverse_transform(predicted_prices)

        real_data = [model_inputs[len(model_inputs) - prediction_days:len(model_inputs+1), 0]]
        real_data = np.array(real_data)
        real_data=np.reshape(real_data, (real_data.shape[0], real_data.shape[1] ,1))

        prediction=model.predict(real_data)
        prediction=scaler.inverse_transform(prediction)
        print(f'Prediction: {prediction}')
        
        loss_return=history.history['loss']
        loss_return=loss_return.pop()

        return prediction,test_data, predicted_prices,prediction, loss_return
