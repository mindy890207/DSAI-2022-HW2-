# You can write code above the if-main block.


if __name__ == '__main__':
    # You should not modify this part.
    import argparse


    parser = argparse.ArgumentParser()
    parser.add_argument('--training',
                       default='training_data.csv',
                       help='input training data file name')
    parser.add_argument('--testing',
                        default='testing_data.csv',
                        help='input testing data file name')
    parser.add_argument('--output',
                        default='output.csv',
                        help='output file name')
    args = parser.parse_args()
    
    # The following part is an example.
    # You can modify it at will.
    import pandas as pd
    import numpy as np
    import statsmodels.api as sm
    import itertools
    import warnings
    import matplotlib.pyplot as plt 
    import csv
    from statsmodels.tsa.arima_model import ARIMA
    from pmdarima.arima import ndiffs
    from pmdarima.arima import ADFTest
    from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
    from pmdarima.arima import auto_arima
    from sklearn.preprocessing import MinMaxScaler 
    import keras
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM
    from keras.layers import Dropout,BatchNormalization

    # preprocessing
    train_data = pd.read_csv(args.training,header=None)
    train_open = pd.read_csv(args.training,usecols=[0],header=None)
    train_close = pd.read_csv(args.training,usecols=[3],header=None)
    test_open = pd.read_csv(args.testing,usecols=[0],header=None)
    test_close = pd.read_csv(args.testing,usecols=[3],header=None)
    train = train_open
    
    sc = MinMaxScaler(feature_range = (0, 1))
    train= train.values.reshape(-1,1)
    train_scaled = sc.fit_transform(train)
    X_train = [] 
    y_train = []
    for i in range(10,len(train)):
        X_train.append(train_scaled[i-10:i-1, 0]) 
        y_train.append(train_scaled[i, 0]) 
    X_train, y_train = np.array(X_train), np.array(y_train) 
    X_train = np.reshape(X_train,(X_train.shape[0], X_train.shape[1], 1))

    #model
    keras.backend.clear_session()
    regressor = Sequential()
    regressor.add(LSTM(units = 100, input_shape = (X_train.shape[1], 1)))
    regressor.add(Dense(units = 1))
    regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
    history = regressor.fit(X_train, y_train, epochs = 100, batch_size = 16)
    dataset_total = pd.concat((train_open, test_open), axis = 0)
    inputs = dataset_total[len(dataset_total) - len(test_open) - 10:].values
    inputs = inputs.reshape(-1,1)
    inputs = sc.transform(inputs)
    X_test = []
    for i in range(10, len(inputs)):
        X_test.append(inputs[i-10:i-1, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    predicted_stock_price = regressor.predict(X_test)
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)

    #?????????????????????
    df_result = pd.DataFrame(predicted_stock_price)
    df_result.to_csv('predict.csv', index=0)

    
    #??????
    test_open = pd.read_csv(args.testing,usecols=[0],header=None)
    test_close = pd.read_csv(args.testing,usecols=[3],header=None)
    predict_data = pd.read_csv('predict.csv')
    
    #??????????????????short???long???????????????
    flag = 0
    state_change = 0
    #????????????action ??? type
    action = 0
    result = []
    set = 0
    #state ?????????????????????????????????0 unit
    state = 0
    #???????????????
    cost = 0
    profit = 0

    for i in range(len(test_open)-1):
        ##?????????????????????????????????????????????????????????????????????????????????sma_sshort???sma_slong
        predict_data.values[i] = test_open.values[i]

        # sma_short??????i??????????????????3???????????????????????????
        sma_short = pd.DataFrame()
        sma_short['open'] = predict_data.rolling(window=3).mean()

        #sma_long??????i??????????????????13???????????????????????????
        sma_long = pd.DataFrame()
        sma_long['open'] = predict_data.rolling(window=13).mean()

        #????????????????????????action????????????
        if((action == 1) & (state_change == 1)):
          cost = cost + test_open.values[i]
          state = state + 1
          state_change = 0
          #print("buy:")
          #print(test_open.values[i])
        elif((action == -1 )& (state_change == 1)):
          profit = profit + test_open.values[i] 
          state = state - 1 
          state_change = 0
          #print("sell:")
          #print(test_open.values[i])
       
        
        #???????????????sma_short>sma_long????????????<=sma_long(?????????)??????????????????
        if((sma_short['open'].values[i+1] > sma_long['open'].values[i+1]) & (flag != 1)):
          #state !=1????????????
          if(state != 1):
               flag = 1
               #action:??????
               action = 1
               state_change = 1
               result.append(1)
          elif(state == 1):
               print("already have 1 unit")

        #???????????????sma_short<sma_long????????????>=sma_long(?????????)??????????????????
        elif((sma_short['open'].values[i+1] < sma_long['open'].values[i+1]) & (flag != -1)):
          if(state != -1):
              flag = -1
              #action:??????
              action = -1
              state_change = 1
              result.append(-1)
          elif(state == -1):
              print("already short 1 unit")
        #??????????????????
        else:
           result.append(0)


    #??????????????????????????????????????????(?
    #??????1????????????
    if(state == 1):
      profit = profit + test_close.values[19]  
      #print(test_close.values[19])
      state = state - 1     
      #print("-1:")  
      #print("action:")
      #print(action)
    #short 1????????????
    elif(state == -1):
      cost = cost + test_close.values[19]  
      #print(test_close.values[19])
      state = state + 1     
      #print("1:")       
      #print("action:")
      #print(action)
    
    #total = profit - cost
    #print("total = ")
    #print(total)
    
    with open('output.csv', 'w', newline='') as csvfile:
        # ?????????????????????????????? CSV ????????????
        for i in range(len(test_open)-1):
          writer = csv.writer(csvfile, delimiter=' ')
          writer.writerow([result[i]])