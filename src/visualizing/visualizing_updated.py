import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from LSTM import shifting
from LSTM import separate_data
from LSTM import LSTM

dataframe = pd.read_csv('global_data.csv')
training_set = dataframe.iloc[:,1:2].values
train_size = int(len(training_set) * 0.80)

#Parameters
input_dim = 1;
hidden_layer_size = 2
num_layers = 1
output_dim = 1
seq_length = 4

model = LSTM(input_dim, output_dim, hidden_layer_size, num_layers)
model.load_state_dict(torch.load('LSTM.pkl'))
model.train()
model.eval()

sc = MinMaxScaler()
training_data = sc.fit_transform(training_set)

X, y = shifting(training_data, seq_length)
dataX = torch.Tensor(X)
dataY = torch.Tensor(y)

train_X, train_Y, test_X, test_Y = separate_data(X, y)

train_predict = model(dataX)
data_predict = train_predict.data.numpy()
data_predict = sc.inverse_transform(data_predict)

actual_data = dataY.data.numpy()
actual_data = sc.inverse_transform(actual_data)

plt.plot(data_predict, c='r', label='LSTM Predictions')
plt.plot(actual_data, c='b', label='Actual Data')
plt.axvline(x=train_size, linestyle='--')
plt.ylabel('# of Confirmed Cases')
plt.xlabel('# of Days From First Case')
plt.legend(loc='upper left')
plt.savefig('GlobalCases.png')

plt.show()
