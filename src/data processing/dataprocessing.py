import pandas as pd
import numpy as np

url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
dataframe = pd.read_csv(url)
dataframe = dataframe.drop(['Province/State', 'Country/Region', 'Lat', 'Long'], axis=1)

global_numbers = dataframe.sum(axis = 0, skipna=True)
global_numbers.to_csv('global_data.csv')
