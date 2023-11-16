import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import time
from iqoptionapi.stable_api import IQ_Option

# Function to load a trained joblib model
def load_joblib(path):
    return joblib.load(path)

# Function to handle login
def login():
    #email = st.text_input('Enter Email')
    #password = st.text_input('Enter Password', type='password',)
    api = IQ_Option("anuragkumar37970@gmail.com","anuragkumar37970@gmail.")
    Logged, reason = api.connect()
    if Logged == True:
        st.success('You are now Logged in....')
        
    else:
        st.error('Something went Wrong Retry again !!!!')
        st.error(reason)
        
    
    return api
    

# Function to load data from IQ Option
def Load_Data(api, symbol, timeframe):
    api.start_candles_stream(symbol, timeframe, 2)
    data = api.get_realtime_candles(symbol, timeframe)
    data = pd.DataFrame(data).transpose()
    return data


# Function to process data
def Process_data(data):
    data = data[data['volume'] != 0].copy()
    data['volop'] = np.array(data['volume'] / data['open'])
    data['opclose'] = np.array(data['close'] / data['open'])
    data['mean'] = np.array(data['volume'] / np.array((data['open'] + data['close'] + data['min'] + data['max']).mean()))
    df = data.iloc[:-1].copy()
    df2 = data[1:].copy()
    df['open1'] = np.array(df2['open'])
    df = df[['open', 'close', 'min', 'max', 'volume', 'volop','opclose', 'mean', 'open1']]
    return df




def predict_exacute_order(api,Pipeline,linear_model,DecisionTreeRegressor):
    
    balance_type="PRACTICE"
    #balance_type= "REAL"
    api.change_balance(balance_type)
    symbol = "EURUSD"
    amount = 1000
    
    
    
    data = Load_Data(api, symbol, 60)
    
    pre_data = Process_data(data)
    st.text(pre_data)

    scaled_X = Pipeline.transform(np.array(pre_data))

    Linear_pred = linear_model.predict(scaled_X)[-1]
    DecisionTreeRegressor_pred = DecisionTreeRegressor.predict(scaled_X)[-1]



    
    if Linear_pred < pre_data['open1'].iloc[-1] and DecisionTreeRegressor_pred < pre_data['open1'].iloc[-1] :
        api.buy_digital_spot(symbol,amount,'put',1)
        st.success('Order has solded  on : bearish Signals  ')
        
    elif Linear_pred > pre_data['open1'].iloc[-1] and DecisionTreeRegressor_pred > pre_data['open1'].iloc[-1] :
        api.buy_digital_spot(symbol,amount,'call',1)
        st.error('Order has bought on : bullish Signals')
    else :
        st.text('Hold')
        
    c = 'Currant Balance '+ str(api.get_balance() ) 
    l = 'linear pridicted :' + str(Linear_pred)
    d = 'DecisionTreeRegressor pridicted :' +  str(DecisionTreeRegressor_pred)
    o = 'open at :' + str(pre_data['open1'].iloc[-1])
    st.text(c)
    st.text(l)
    st.text(d)
    st.text(o)





def start(api,Pipeline,linear_model,DecisionTreeRegressor):

    balance = api.get_balance()
    while True  :
    
        current_time = time.localtime()
        if current_time.tm_sec ==  1 :
            st.text('*'*29)
            if current_time.tm_min % 1 == 0:
                balance = api.get_balance()
        
                if balance > 100 :
                    predict_exacute_order(api,Pipeline,linear_model,DecisionTreeRegressor)
                    time.sleep(45)

                else:
                    print('Your balance Less then 400')
            else:
                time.sleep(1)
        else:
    
            time.sleep(1)
            















# Main function
def main():
    st.title('Trad X')
    linear_model = load_joblib('new-linear-model-v2.0.joblib')
    DecisionTreeRegressor = load_joblib('new-DecisionTreeRegressor-v2.0.joblib')
    Pipeline = load_joblib('Pipeline-v2.0.joblib')

    api = login()


    start(api,Pipeline,linear_model,DecisionTreeRegressor)









if __name__ == "__main__":
    main()
