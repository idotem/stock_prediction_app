from django.shortcuts import render
import yfinance as yf
import numpy as np
import pandas as pd
import datetime
import missingno as msno
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, OneHotEncoder

# stock_symbols = ['AAPL', 'NVDA', 'SNOW', 'AMD', 'CAT', 'CHWY', 'ETSY', 'GOOGL', 'MSFT', 'NVDA', 'PAYC',
#                  'PAYO', 'PEP', 'PFE', 'PLTR', 'SHOP', 'SQ', 'TSLA', 'VRNT']
stock_symbols = ['AAPL', 'NVDA', 'SNOW']
percentage_limit = 5


def default_stocks(request):
    symbol = "AAPL"
    context = {"stocks": stock_symbols,
               "prediction": get_insider_pred(symbol),
               "selected_symbol": symbol}
    return render(request, "stocks.html", context)


def stocks(request, symbol):
    context = {"stocks": stock_symbols,
               "insider": get_insider_pred(symbol),
               "analyst_rcmd": get_analyst_recommendation_pred(symbol),
               "analyst_est": get_analyst_estimations_pred(symbol),
               "selected_symbol": symbol}
    return render(request, "stocks.html", context)


def get_analyst_estimations_pred(stock_symbol):
    analyst_est = pd.read_json('./data/analyst_data/' + stock_symbol + '_EST.json')
    analyst_est_df = pd.DataFrame(analyst_est)
    analyst_est_df['dym'] = analyst_est_df['date'].dt.to_period('M')

    income = pd.read_json('./data/analyst_data/' + stock_symbol + '_INCOME_STATEMENT.json')
    income_df = pd.DataFrame(income)
    income_df['fillingDate'] = pd.to_datetime(income_df['fillingDate'])
    income_df['dym'] = income_df['date'].dt.to_period('M')
    income_df['fillingDym'] = income_df['fillingDate'].dt.to_period('M')

    columns_to_keep = ['revenue', 'sellingGeneralAndAdministrativeExpenses', 'ebitda', 'epsdiluted', 'eps', 'netIncome',
                       'dym', 'fillingDym']

    income_df = income_df[columns_to_keep]
    est_min_date = analyst_est_df['date'].min()
    today = datetime.date.today()
    stock_prices_df = yf.download(stock_symbol, est_min_date, today)

    stock_prices_df.drop(columns=['Open', 'High', 'Low', 'Volume', 'Close'], inplace=True)
    monthly_max_df = stock_prices_df['Adj Close'].resample('M').max().to_frame()
    monthly_max_df['max_next_3'] = (
        monthly_max_df['Adj Close']
        .shift(-1)
        .rolling(window=3, min_periods=1)
        .max()
    )

    monthly_max_df['dym'] = monthly_max_df.index.to_period('M')
    monthly_max_df = monthly_max_df.drop(columns='Adj Close')
    merged_est_income_df = pd.merge(analyst_est_df, income_df, how='left', left_on='dym', right_on='dym')
    merged_df = pd.merge(merged_est_income_df, monthly_max_df, how='left', left_on='fillingDym', right_on='dym')
    merged_df = merged_df.sort_values('fillingDym', ascending=False)
    merged_df = merged_df.dropna(subset=['revenue'])
    merged_df.reset_index(inplace=True)

    merged_df['pct_change'] = (merged_df['max_next_3'].shift(1) - merged_df['max_next_3']) / merged_df[
        'max_next_3'].shift(1) * 100

    # merged_df['trend'] = pd.cut(merged_df['pct_change'], bins=[-float('inf'), -percentage_limit, percentage_limit,
    #                                                            float('inf')], labels=['Sell', 'Hold', 'Buy'])
    merged_df['trend'] = pd.cut(merged_df['pct_change'], bins=[-float('inf'), percentage_limit,
                                                               float('inf')], labels=['Hold', 'Buy'])

    ready_df = merged_df.drop(
        columns=['date', 'dym_x', 'dym_y', 'fillingDym', 'symbol', 'pct_change', 'max_next_3', 'index'], inplace=False)
    ready_df = ready_df.dropna(subset=['trend'])
    label_encoder = LabelEncoder()
    ready_df['trend'] = label_encoder.fit_transform(ready_df['trend'])

    new_data = merged_df.head(1).drop(
        columns=['date', 'dym_x', 'dym_y', 'fillingDym', 'symbol', 'pct_change', 'max_next_3', 'index', 'trend'],
        inplace=False)

    X = ready_df.drop(columns='trend', inplace=False)
    y = ready_df['trend']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', enable_categorical=True)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    xg_acc = accuracy_score(y_test, y_pred)
    xg_predicted_trend_encoded = model.predict(new_data)
    xg_pred = label_encoder.inverse_transform(xg_predicted_trend_encoded)[0]

    Dense = tf.keras.layers.Dense
    Input = tf.keras.layers.Input
    model = tf.keras.models.Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(2, activation='softmax')  # 2 classes: Buy, Hold
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=50, batch_size=10, validation_split=0.2)
    seq_y_pred_prob = model.predict(X_test)
    seq_y_pred = np.argmax(seq_y_pred_prob, axis=1)

    seq_acc = accuracy_score(y_test, seq_y_pred)

    seq_predicted_trend_encoded = model.predict(new_data)
    seq_y_pred_new_data = np.argmax(seq_predicted_trend_encoded, axis=1)
    seq_pred = label_encoder.inverse_transform(y_pred)[0]

    # smote = SMOTE()
    # X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    model = RandomForestClassifier(n_estimators=100, max_depth=43, random_state=42)
    # model.fit(X_train_resampled, y_train_resampled)
    model.fit(X_train, y_train)

    dt_y_pred = model.predict(X_test)

    # print(classification_report(y_test, y_pred, target_names=['Buy', 'Hold', 'Hold']))
    dt_acc = accuracy_score(y_test, dt_y_pred)

    dt_predicted_trend_encoded = model.predict(new_data)
    dt_pred = label_encoder.inverse_transform(dt_predicted_trend_encoded)[0]
    return {"xg_classifier": {"trend": xg_pred, "accuracy": round(xg_acc, 2)},
            "sequential": {"trend": seq_pred, "accuracy": round(seq_acc, 2)},
            "decision_tree": {"trend": dt_pred, "accuracy": round(dt_acc, 2)}}



def get_analyst_recommendation_pred(stock_symbol):
    analyst_rcmd = pd.read_json('./data/analyst_data/' + stock_symbol + '_RCMD.json')
    analyst_rcmd_df = pd.DataFrame(analyst_rcmd)
    analyst_rcmd_df['dym'] = analyst_rcmd_df['date'].dt.to_period('M')
    rcmd_min_date = analyst_rcmd_df['date'].min()
    today = datetime.date.today()
    stock_prices_df = yf.download('AAPL', rcmd_min_date, today)
    stock_prices_df.drop(columns=['Open', 'High', 'Low', 'Volume', 'Close'], inplace=True)
    monthly_max_df = stock_prices_df['Adj Close'].resample('M').max().to_frame()
    monthly_max_df['dym'] = monthly_max_df.index.to_period('M')
    analyst_rcmd_df['dym'] = analyst_rcmd_df['date'].dt.to_period('M')
    merged_df = pd.merge(analyst_rcmd_df, monthly_max_df, how='left', left_on='dym', right_on='dym')
    merged_df = merged_df.sort_values('dym', ascending=True)
    merged_df = merged_df.reset_index(drop=True)

    merged_df['pct_change'] = (merged_df['Adj Close'].shift(-1) - merged_df['Adj Close']) / merged_df[
        'Adj Close'].shift(-1) * 100
    merged_df['trend'] = pd.cut(merged_df['pct_change'], bins=[-float('inf'), percentage_limit,
                                                               float('inf')], labels=['Hold', 'Buy'])
    ready_df = merged_df.drop(columns=['date', 'dym', 'symbol', 'pct_change', 'Adj Close'], inplace=False)
    ready_df = ready_df.dropna(subset=['trend'])
    label_encoder = LabelEncoder()
    ready_df['trend'] = label_encoder.fit_transform(ready_df['trend'])

    X = ready_df.drop(columns='trend', inplace=False)
    y = ready_df['trend']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    # Tried the time series split looking for better results

    # tscv = TimeSeriesSplit(n_splits=6)
    # for train_index, test_index in tscv.split(X):
    #     X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    #     y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', enable_categorical=True)
    model.fit(X_train, y_train)
    xg_y_pred = model.predict(X_test)
    xg_acc = accuracy_score(y_test, xg_y_pred)

    new_data = merged_df.tail(1).drop(columns=['date', 'dym', 'symbol', 'pct_change', 'trend', 'Adj Close'],
                                      inplace=False)
    xg_predicted_trend_encoded = model.predict(new_data)
    xg_pred = label_encoder.inverse_transform(xg_predicted_trend_encoded)[0]

    Dense = tf.keras.layers.Dense
    Input = tf.keras.layers.Input
    model = tf.keras.models.Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(2, activation='softmax')  # 2 classes: Buy, Hold
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=50, batch_size=10, validation_split=0.2)
    y_pred_prob = model.predict(X_test)
    seq_y_pred = np.argmax(y_pred_prob, axis=1)
    seq_acc = accuracy_score(y_test, seq_y_pred)
    seq_predicted_trend_encoded = model.predict(new_data)
    y_pred_new_data = np.argmax(seq_predicted_trend_encoded, axis=1)
    seq_pred = label_encoder.inverse_transform(y_pred_new_data)[0]

    model = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42)
    model.fit(X_train, y_train)

    dt_y_pred = model.predict(X_test)
    dt_acc = accuracy_score(y_test, dt_y_pred)
    dt_predicted_trend_encoded = model.predict(new_data)
    dt_pred = label_encoder.inverse_transform(dt_predicted_trend_encoded)[0]

    return {"xg_classifier": {"trend": xg_pred, "accuracy": round(xg_acc, 2)},
            "sequential": {"trend": seq_pred, "accuracy": round(seq_acc, 2)},
            "decision_tree": {"trend": dt_pred, "accuracy": round(dt_acc, 2)}}


def get_insider_pred(stock_symbol):
    insider_df = pd.read_csv('./data/insider_tran_data/' + stock_symbol + '.csv')
    insider_df = insider_df.drop('isDerivative', axis=1)
    insider_df['filingDate'] = pd.to_datetime(insider_df['filingDate'])
    insider_min_date = insider_df['filingDate'].min()
    today = datetime.date.today()

    stock_prices_df = yf.download(stock_symbol, insider_min_date, today)

    stock_prices_df.drop(columns=['Open', 'High', 'Low', 'Volume', 'Close'], inplace=True)
    monthly_max_df = stock_prices_df['Adj Close'].to_frame()
    monthly_max_df = monthly_max_df.reset_index()
    insider_df = insider_df[insider_df['transactionPrice'] != 0.000]
    insider_df = insider_df.reset_index(drop=True)
    merged_df = pd.merge(insider_df, monthly_max_df, how='right', left_on='filingDate', right_on='Date')
    merged_df = merged_df.sort_values('Date', ascending=False)
    merged_df['max_price_after_tran'] = (
        merged_df['Adj Close']
        .shift(-1)
        .rolling(window=30, min_periods=1)
        .max()
    )
    merged_df = merged_df.dropna(subset=['transactionPrice'])
    merged_df = merged_df.reset_index(drop=True)
    merged_df['pct_change'] = (merged_df['max_price_after_tran'] - merged_df['transactionPrice']) / merged_df[
        'max_price_after_tran'] * 100

    merged_df['trend'] = pd.cut(merged_df['pct_change'], bins=[-float('inf'), percentage_limit,
                                                               float('inf')], labels=['Hold', 'Buy'])
    # merged_df['trend'] = pd.cut(merged_df['pct_change'], bins=[-float('inf'), -percentage_limit, percentage
    # float('inf')], labels=['Sell', 'Hold', 'Buy'])

    ready_df = merged_df.drop(
        columns=['Date', 'id', 'symbol', 'pct_change', 'Adj Close', 'max_price_after_tran', 'currency', 'filingDate',
                 'transactionDate', 'source'], inplace=False)

    name_encoder = LabelEncoder()
    trend_encoder = LabelEncoder()
    onehot_encoder = OneHotEncoder(sparse_output=False)

    encoded_names = onehot_encoder.fit_transform(ready_df[['transactionCode']])
    encoded_names_df = pd.DataFrame(encoded_names, columns=onehot_encoder.get_feature_names_out(['transactionCode']))
    ready_df = pd.concat([ready_df, encoded_names_df], axis=1)
    ready_df.drop(columns=['transactionCode'], inplace=True)

    ready_df['trend'] = trend_encoder.fit_transform(ready_df['trend'])
    ready_df['name'] = name_encoder.fit_transform(ready_df['name'])

    last_row = ready_df.head(1).drop(columns=['trend'])

    ready_df = ready_df.iloc[1:]

    X = ready_df.drop(columns='trend', inplace=False)
    y = ready_df['trend']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    return {"xg_classifier": insider_xg_classifier_pred(X_test, X_train, last_row, trend_encoder, y_test, y_train),
            "sequential": insider_sequential_pred(X_test, X_train, last_row, trend_encoder, y_test, y_train),
            "decision_tree": insider_decision_tree_pred(X_test, X_train, last_row, trend_encoder, y_test, y_train)}


def insider_xg_classifier_pred(X_test, X_train, last_row, trend_encoder, y_test, y_train):
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', enable_categorical=True)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc_score = accuracy_score(y_test, y_pred)
    predicted_trend_encoded = model.predict(last_row)
    predicted_trend = trend_encoder.inverse_transform(predicted_trend_encoded)
    return {"trend": predicted_trend[0], "accuracy": round(acc_score, 2)}


def insider_sequential_pred(X_test, X_train, last_row, trend_encoder, y_test, y_train):
    Dense = tf.keras.layers.Dense
    Input = tf.keras.layers.Input
    model = tf.keras.models.Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(2, activation='softmax')
        # Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=50, batch_size=10, validation_split=0.2)
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    acc_score = accuracy_score(y_test, y_pred)
    predicted_trend_encoded = model.predict(last_row)
    last_row_pred = np.argmax(predicted_trend_encoded, axis=1)
    predicted_trend = trend_encoder.inverse_transform(last_row_pred)
    return {"trend": predicted_trend[0], "accuracy": round(acc_score, 2)}


def insider_decision_tree_pred(X_test, X_train, last_row, trend_encoder, y_test, y_train):
    model = RandomForestClassifier(n_estimators=100, max_depth=43, random_state=42)
    # model.fit(X_train_resampled, y_train_resampled)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # print(classification_report(y_test, y_pred, target_names=['Buy', 'Hold', 'Hold']))
    acc_score = accuracy_score(y_test, y_pred)
    predicted_trend_encoded = model.predict(last_row)
    predicted_trend = trend_encoder.inverse_transform(predicted_trend_encoded)
    return {"trend": predicted_trend[0], "accuracy": round(acc_score, 2)}
