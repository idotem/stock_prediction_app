from django.shortcuts import render
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import seaborn as sns
import datetime
import missingno as msno
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.impute import KNNImputer
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Input
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, OneHotEncoder

def default_stocks(request):
    symbol = "AAPL"
    stock_symbols = ['AAPL', 'NVDA', 'SNOW']
    context = {"stocks": stock_symbols,
               "prediction": get_stock_pred(symbol),
               "selected_symbol": symbol}
    return render(request, "stocks.html", context)


def stocks(request, symbol):
    stock_symbols = ['AAPL', 'NVDA', 'SNOW']
    context = {"stocks": stock_symbols,
               "prediction": get_stock_pred(symbol),
               "selected_symbol": symbol}
    return render(request, "stocks.html", context)


def get_stock_pred(symbol):
    prediction = "NO_VALUE"
    if symbol == 'AAPL':
        prediction = "BUY"
    elif symbol == 'NVDA':
        prediction = "SELL"
    elif symbol == 'SNOW':
        prediction = "SELL"
    return prediction
