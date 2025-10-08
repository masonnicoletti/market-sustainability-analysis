#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 22:48:47 2025

@author: masonnicoletti
"""

'''
Midterm Project!
DS 2023
Design 1: Communicating with Data


Data Sets:
    1. ESG Data
    2. S&P 500 Data (2015-2024)
        i.  Companies
        ii. Index Performance
    3. Magnificent 7 Stocks
        i.   AAPl
        ii.  AMZN
        iii. GOOGL
        iv.  META
        v.   MSFT
        vi.  NVDA
        vii. TSLA


Research Questions:
    1. How do ESG Scores Range across different industries?
    2. How do ESG Scores of Mag Seven Companies compare to aggregate ESG score of other S&P Companies?


'''

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
pio.renderers.default="browser"




# 1. Bring in Data

# Load S&P Data
esg = pd.read_csv("./data/s&p_data/sp500_esg_data.csv")
sp500_companies = pd.read_csv("./data/s&p_data/sp500_price_data.csv")
sp500 = pd.read_csv("./data/s&p_data/sp500_index.csv")

# Load Mag Seven Data
aapl = pd.read_csv('./data/mag_seven_data/AAPL1424.csv')
amzn = pd.read_csv('./data/mag_seven_data/AMZN1424.csv')
googl = pd.read_csv('./data/mag_seven_data/GOOGL1424.csv')
meta = pd.read_csv('./data/mag_seven_data/META1424.csv')
msft = pd.read_csv('./data/mag_seven_data/MSFT1424.csv')
nvda = pd.read_csv('./data/mag_seven_data/NVDA1424.csv')
tsla = pd.read_csv('./data/mag_seven_data/TSLA1424.csv')

# Add symbol column to Mag Seven Data
aapl["Symbol"] = "AAPL"
amzn["Symbol"] = "AMZN"
googl["Symbol"] = "GOOGL"
meta["Symbol"] = "META"
msft["Symbol"] = "MSFT"
nvda["Symbol"] = "NVDA"
tsla["Symbol"] = "TSLA"

# Stack Mag Seven Data
mag7 = pd.concat([aapl, amzn, googl, meta, msft, nvda, tsla])




# 2. Data Cleaning

#print(esg.info())
#print(esg.describe())
#print(esg.isna().sum())
#print(esg['GICS Sector'].value_counts())
#sns.displot(esg, x='governanceScore')
#plt.show()
esg['GICS Sector'] = esg['GICS Sector'].astype('category')

mag7_list = ["AAPL", "AMZN", "GOOGL", "META", "MSFT", "NVDA", "TSLA"]
esg_filtered = esg.copy()
esg_filtered['Symbol'] = esg.Symbol.apply(lambda x: x if x in mag7_list else "Other")

#print(sp500.info())
#print(sp500.describe())
#sns.boxplot(sp500, x="S&P500")
#plt.show()
sp500.rename(columns={"S&P500": "Value"}, inplace=True)

#print(mag7.info())
#print(mag7.isna().sum())
#print(mag7.describe())
#print(mag7.Symbol.value_counts())

