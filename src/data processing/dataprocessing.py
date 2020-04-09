import pandas as pd

url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
dataframe = pd.read_csv(url)
dataframe.to_csv('total_data.csv')
dataframe = dataframe.drop(['Province/State', 'Lat', 'Long'], axis=1)

global_numbers = dataframe.drop(['Country/Region'], axis=1)
global_numbers = global_numbers.sum(axis = 0, skipna=True)
global_numbers.to_csv('global_data.csv')

us_numbers = dataframe.loc[dataframe['Country/Region'] == 'US']
us_numbers = us_numbers.drop(['Country/Region'], axis=1)
us_numbers.to_csv('us_data.csv')
