import pickle
from matplotlib import pyplot as plt
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
plt.style.use('seaborn')


def calculateCML(csv_path):
    cml = pd.read_csv(csv_path, header=0)
    # print(cml.loc[cml['year'] == 1970])
    cmlDict = {}
    for year in range(1970, 2017+1):
        line = pd.DataFrame(cml.loc[cml['year'] == year]).reset_index().to_dict()
        cmlDict[year] = {'rou': line['rf'][0] / 100, 'mu': line['mu'][0], 'sigma': line['sigma'][0]}
    return cmlDict


def load_pickle(portfolio_size):
    return_yearly = pickle.load(open("returns_" + str(portfolio_size) + '.pickle', 'rb'))
    vol_yearly = pickle.load(open("vol_" + str(portfolio_size) + '.pickle', 'rb'))
    return return_yearly, vol_yearly


def generate_overview_scatter(return_yearly, vol_yearly, universe_size, cml):
    if return_yearly.keys() == vol_yearly.keys():
        total_years = list(return_yearly.keys())
        fig, ax = plt.subplots(8, 6, figsize=(30, 30))
        for i, ax_row in enumerate(ax):
            for j, axes in enumerate(ax_row):
                year = total_years[i * 6 + j]
                x_sigma = vol_yearly[year]
                y_sigma = return_yearly[year]
                axes.set_title(str(year))
                axes.set_xlim(left=0, right=max(x_sigma) + 0.05)

                axes.scatter(x=x_sigma, y=y_sigma, s=3)

                x = np.array(np.arange(0, max(x_sigma) + 0.1, 0.05))
                rou = str(cml[year]['rou'])
                mu = str(cml[year]['mu'])
                sigma = str(cml[year]['sigma'])
                formula = rou + '+' + '((' + mu + '-' + rou + ') / ' + sigma + ') * x'
                y = eval(formula)
                axes.plot(x, y, color='black')
        fig.suptitle("Comparisons of Random Portfolios from 1970-2017")
        plt.tight_layout()
        plt.savefig('scatter_' + str(universe_size) + '_overview.png', dpi=300)


def generate_individual_scatter(return_yearly, vol_yearly, universe_size, cml):
    folderPath = "scatter_individual_" + str(universe_size)
    if not os.path.isdir(folderPath):
        os.mkdir(folderPath)
    if return_yearly.keys() == vol_yearly.keys():
        total_years = list(return_yearly.keys())
        for year in total_years:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            x_sigma = vol_yearly[year]
            y_sigma = return_yearly[year]
            ax.set_xlim(left=0, right=max(x_sigma) + 0.05)

            ax.scatter(x=x_sigma, y=y_sigma, s=3)

            x = np.array(np.arange(0, max(x_sigma) + 0.1, 0.05))
            rou = str(cml[year]['rou'])
            mu = str(cml[year]['mu'])
            sigma = str(cml[year]['sigma'])
            formula = rou + '+' + '((' + mu + '-' + rou +') / ' + sigma +') * x'
            y = eval(formula)
            ax.plot(x, y, color='black')

            plt.xlabel("Standard Deviation")
            plt.ylabel("Expected Return")
            plt.title("Random Portfolio Year " + str(year) + ", Portfolio Size = " + str(universe_size))
            plt.savefig(os.path.join(folderPath, 'scatter_' + str(year) + '.png'), dpi=300)

def calculate_percentage(return_yearly, vol_yearly, universe_size, cml):
    if return_yearly.keys() == vol_yearly.keys():
        total_years = sorted(list(return_yearly.keys()))
        percent_year = []
        for year in (total_years):
            x_sigma = vol_yearly[year]
            y_mu = return_yearly[year]
            counts = 0
            for vol, return_mu in zip(x_sigma, y_mu):
                x = vol
                rou = str(cml[year]['rou'])
                mu = str(cml[year]['mu'])
                sigma = str(cml[year]['sigma'])
                formula = rou + '+' + '((' + mu + '-' + rou + ') / ' + sigma + ') * x'
                y = eval(formula)
                if return_mu >= y:
                    counts += 1
            percent_year.append(counts / len(y_mu))
        # visualize data
        plt.figure()
        plt.plot(total_years, percent_year, linestyle='--', marker='o')
        plt.title("Percent Random Portfolio Above CML by Year, Portfolio Size = " + str(universe_size))
        plt.xlabel("Year")
        plt.ylabel("Percentage")
        # plt.show()
        plt.savefig("percentage_" + str(universe_size) + '.png', dpi=300)
        # write data to csv
        p_dict = {}
        for year, data in zip(total_years, percent_year):
            p_dict[year] = data
        df = pd.DataFrame.from_dict(p_dict, orient='index')
        df.to_csv('percentage_' + str(universe_size) + '.csv')
        return total_years, percent_year

cml = calculateCML('cml.csv')
for uni_size in tqdm([30, 100, 500, 1000]):
    print('[INFO] Now processing: ', uni_size)
    r, v = load_pickle(uni_size)
    print('[INFO] Finished loading pickle.')
    generate_overview_scatter(r, v, uni_size, cml)
    print('[INFO] Finished overview scatter plot.')
    generate_individual_scatter(r, v, uni_size, cml)
    print('[INFO] Finished individual scatter.')
    # years, percentage = calculate_percentage(r, v, uni_size, cml)
    # print('[INFO] Finished Percentage Calculation.')
    # exit(0)
    # print(percentage)


