import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

dataframe = pd.read_csv('global_data.csv')
dataframe = dataframe.iloc[:,1:2].values

def shifting(data, seq_length):
    x = []
    y = []

    for i in range(len(data)-seq_length-1):
        _x = data[i:i+seq_length]
        _y = data[i+seq_length]
        x.append(_x)
        y.append(_y)

    return np.array(x), np.array(y)

sc = MinMaxScaler()
training_data = sc.fit_transform(dataframe)

seq_length = 4
X, y = shifting(training_data, seq_length)

def separate_data(X, y):
    train_size = int(len(y) * 0.9)
    test_size = len(y) - train_size

    train_X = torch.Tensor(X[0:train_size])
    train_Y = torch.Tensor(y[0:train_size])

    test_X = torch.Tensor(X[train_size:])
    test_Y = torch.Tensor(y[train_size:])

    return train_X, train_Y, test_X, test_Y

train_X, train_Y, test_X, test_Y = separate_data(X, y)

class LSTM(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layer_size, num_layers):
        super(LSTM, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_layer_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_layer_size, output_dim)

    def forward(self, input):
        hidden_state = torch.zeros(self.num_layers, input.size(0), self.hidden_layer_size)
        cell_state = torch.zeros(self.num_layers, input.size(0), self.hidden_layer_size)

        output, (hidden_state, cell_state) = self.lstm(input, (hidden_state, cell_state))
        out = hidden_state.view(-1, 2)

        out = self.fc(out)

        return out

#Parameters
input_dim = 1;
hidden_layer_size = 2
num_layers = 1
output_dim = 1

num_of_epochs = 2000
display_step  = 100
learning_rate = 0.01

model = LSTM(input_dim, output_dim, hidden_layer_size, num_layers)

MSE = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_of_epochs):

    outputs = model(train_X)
    loss = MSE(outputs, train_Y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % display_step == 0:
        print('Epoch: {:d}, Loss: {:.4f}'.format(epoch, loss.item()))

torch.save(model.state_dict(), 'LSTM.pkl')
