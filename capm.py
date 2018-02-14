import numpy as np
import pandas as pd
import datetime
import pickle
from pandas import DataFrame
import math
import os
from tqdm import tqdm


# This converter makes sure all the unavailable returns "B", "C".. are removed
def converter(num):
    try:
        return np.float(num)
    except:
        return np.nan


date_parser = lambda x: pd.datetime.strptime(x, '%Y%m%d')

class Capm_validation:

    def __init__(self):
        return

    # Data parsing and alignment

    # line_rou, line_mu_m, line_sigma should all be array of size num of years
    # self.line_rou, self.line_mu_m, self.line_sigma = find_CML()

    def readRetDfFromPickle(self, year):
        with open('yearly_pickles/' + str(year) + '_returns_pivoted.pickle', 'rb') as handle:
            r = pickle.load(handle)
        with open('yearly_pickles/' + str(year) + '_cap_pivoted.pickle', 'rb') as handle:
            c = pickle.load(handle)
        return r, c

    def _TestreadRetDfFromPickle(self, year):
        with open('test_yearly_pickles/' + str(year) + '_returns_pivoted.pickle', 'rb') as handle:
            r = pickle.load(handle)
        with open('test_yearly_pickles/' + str(year) + '_cap_pivoted.pickle', 'rb') as handle:
            c = pickle.load(handle)
        return r, c

    def simulate(self, universe_size, sample_size):
        returns_result = {}
        vol_result = {}
        print('[INFO] Using Universe Size:', universe_size)
        for year in tqdm(range(1970, 2017 + 1)):

            returns_result[year] = np.zeros(sample_size)
            vol_result[year] = np.zeros(sample_size)

            # Extract returns of that year and turn into numpy array for more reasonable runing time
            a, b = self.readRetDfFromPickle(year)
            current = a.as_matrix()
            cap = b.as_matrix()

            # Drop a column if it is all nan that year
            # current = np.array([[1,np.nan,3], [2, np.nan, 4], [1,np.nan,3]])
            current = current[:, ~np.all(np.isnan(current), axis=0)]
            trade_days = np.count_nonzero(~np.isnan(current), axis=0)
            median_trade_day = np.median(trade_days)

            current = current[:, trade_days >= median_trade_day]
            cap = cap[:, trade_days >= median_trade_day]
            universe_size = min(universe_size, current.shape[1])
            # Find the companies of highest cap
            universe = current[:, cap[0].argsort()[-universe_size:]]

            # Remove the days that have nan
            universe = universe[~np.isnan(universe).any(axis=1)]
            trade_days = universe.shape[0]
            for i in range(sample_size):
                # Generate the random weights
                ws = np.random.randint(1, 1000 + 1, universe_size)
                weights = 1.0 * ws / sum(ws)

                annual_returns = np.sum((np.prod(universe + 1, axis=0) - 1) * weights)

                P = np.sum(weights.reshape(1, universe_size) * universe, axis=1)
                MusStar = np.sum(P) / trade_days
                vol = (trade_days / (trade_days - 1)) * np.sum((P - MusStar) ** 2)
                returns_result[year][i] = annual_returns
                vol_result[year][i] = vol
        with open('returns_' + str(universe_size) + '.pickle', 'wb') as handle:
            pickle.dump(returns_result, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('vol_' + str(universe_size) + '.pickle', 'wb') as handle:
            pickle.dump(vol_result, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def readRetDf(self):
        print('[INFO] Reading and Preprocessing CSV...')
        df = pd.read_csv('56b9e978ebcbcbc7.csv', usecols=['date', 'RET', 'EXCHCD', 'TICKER', 'PRC', 'SHROUT'],
                         parse_dates=['date'], date_parser=date_parser,
                         converters={'RET': converter, 'EXCHCD': converter, 'PRC': converter, 'SHROUT': converter})

        df = df.dropna()
        df['CAP'] = abs(df['PRC'] * df['SHROUT'])

        # remove private companies
        df = df[df['EXCHCD'] != -2]
        df = df[df['EXCHCD'] != -1]
        df = df[df['EXCHCD'] != 0]
        if not os.path.isdir('yearly_pickles'):
            os.mkdir('yearly_pickles')
        for year in tqdm(range(1970, 2017 + 1)):
            yearRet = df[df['date'].dt.year == year]
            p = yearRet.pivot_table(index='date', columns='TICKER', values='RET', aggfunc='mean')
            with open('yearly_pickles/' + str(year) + '_returns_pivoted.pickle', 'wb') as handle:
                pickle.dump(p, handle, protocol=pickle.HIGHEST_PROTOCOL)
            p = yearRet.pivot_table(index='date', columns='TICKER', values='CAP', aggfunc='mean')
            with open('yearly_pickles/' + str(year) + '_cap_pivoted.pickle', 'wb') as handle:
                pickle.dump(p, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print("finish pickling year " + str(year))

    def _TestreadRetDf(self):
        df = pd.read_csv('medium.csv', usecols=['date', 'RET', 'EXCHCD', 'TICKER', 'PRC', 'SHROUT'],
                         parse_dates=['date'], date_parser=date_parser,
                         converters={'RET': converter, 'EXCHCD': converter, 'PRC': converter, 'SHROUT': converter})

        df = df.dropna()
        df['CAP'] = math.abs(df['PRC'] * df['SHROUT'])

        # remove private companies  (Not sure whether this is necessary)
        df = df[df['EXCHCD'] != -2]
        df = df[df['EXCHCD'] != -1]
        df = df[df['EXCHCD'] != 0]

        for year in range(1970, 2017 + 1):
            yearRet = df[df['date'].dt.year == year]
            p = yearRet.pivot_table(index='date', columns='TICKER', values='RET', aggfunc='mean')
            with open('test_yearly_pickles/' + str(year) + '_returns_pivoted.pickle', 'wb') as handle:
                pickle.dump(p, handle, protocol=pickle.HIGHEST_PROTOCOL)
            p = yearRet.pivot_table(index='date', columns='TICKER', values='CAP', aggfunc='mean')
            with open('test_yearly_pickles/' + str(year) + '_cap_pivoted.pickle', 'wb') as handle:
                pickle.dump(p, handle, protocol=pickle.HIGHEST_PROTOCOL)


Capm_validation().readRetDf()
# print(df.loc[df['TICKER'] == 'DGSE'])
# Capm_validation().TestreadRetDf()

# p = Capm_validation().TestreadRetDfFromPickle(1992)
# print(p['1992'])
# print(p['1992'].as_matrix())
Capm_validation().simulate(30, 50000)
Capm_validation().simulate(100, 50000)
Capm_validation().simulate(500, 50000)
Capm_validation().simulate(1000, 50000)
