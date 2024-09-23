import yfinance as yf
import numpy as np
import pandas as pd
import datetime
import missingno as msno
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, OneHotEncoder


def analyst_estimations(stock_symbol, percentage_limit):
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

    xg_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', enable_categorical=True)

    xg_model.fit(X_train, y_train)
    y_pred = xg_model.predict(X_test)

    xg_acc = accuracy_score(y_test, y_pred)
    xg_predicted_trend_encoded = xg_model.predict(new_data)
    xg_pred = label_encoder.inverse_transform(xg_predicted_trend_encoded)[0]

    Dense = tf.keras.layers.Dense
    Input = tf.keras.layers.Input
    seq_model = tf.keras.models.Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(2, activation='softmax')  # 2 classes: Buy, Hold
    ])
    seq_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    seq_model.fit(X_train, y_train, epochs=50, batch_size=10, validation_split=0.2)
    seq_y_pred_prob = seq_model.predict(X_test)
    seq_y_pred = np.argmax(seq_y_pred_prob, axis=1)

    seq_acc = accuracy_score(y_test, seq_y_pred)

    seq_predicted_trend_encoded = seq_model.predict(new_data)
    seq_y_pred_new_data = np.argmax(seq_predicted_trend_encoded, axis=1)
    seq_pred = label_encoder.inverse_transform(seq_y_pred_new_data)[0]

    # smote = SMOTE()
    # X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    rf_model = RandomForestClassifier(n_estimators=100, max_depth=43, random_state=42)
    # model.fit(X_train_resampled, y_train_resampled)
    rf_model.fit(X_train, y_train)

    rf_y_pred = rf_model.predict(X_test)

    # print(classification_report(y_test, y_pred, target_names=['Buy', 'Hold', 'Hold']))
    rf_acc = accuracy_score(y_test, rf_y_pred)

    rf_predicted_trend_encoded = rf_model.predict(new_data)
    rf_pred = label_encoder.inverse_transform(rf_predicted_trend_encoded)[0]
    return {"xg_classifier": {"trend": xg_pred, "accuracy": round(xg_acc, 2)},
            "sequential": {"trend": seq_pred, "accuracy": round(seq_acc, 2)},
            "random_forrest": {"trend": rf_pred, "accuracy": round(rf_acc, 2)}}