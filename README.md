# Predicting Coronavirus Growth Rate using LSTM Time Series Forecasting
*Last Updated: April 9, 2020*

## What is a LSTM?
LSTM stands for Long-Short Term Neural Network and is a type of recurrent neural network that uses past data to make a prediction about the future. These types of neural networks could be used in all kinds of complex problems such as speech recognition and anomaly detection. 

## How does a LSTM work?
A typical LSTM is built up with different memory blocks called cells. Between each cell, there are two states being transferred: hidden and cell state. Each state holds information that is used in making the next prediction. A LSTM uses three different gates (input, forget, output) to regulate which information to use when making the prediction. 

## Predicting Global Coronavirus Growth
<img src="src/GlobalCases.png" width="500" height="400">

*The dashed line separates the 80% training and 20% testing data.* Although the LSTM comes close to almost zero training error, there is still considerable error in the testing as it is unable to predict the sharp spike in cases.

## Predicting US Coronavirus Growth
<img src="src/USCases.png" width="500" height="400">

*The dashed line separates the 80% training and 20% testing data.* In this example, as the LSTM captures the sharp increase before the testing data, it is able to better predict the trend with less error.
