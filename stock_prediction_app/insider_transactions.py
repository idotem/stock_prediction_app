import yfinance as yf
import numpy as np
import pandas as pd
import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, OneHotEncoder


def insider_transactions(stock_symbol, percentage_limit):
    insider_df = pd.read_csv('./data/insider_tran_data/' + stock_symbol + '.csv')
    insider_df = insider_df.drop('isDerivative', axis=1)
    insider_df['filingDate'] = pd.to_datetime(insider_df['filingDate'])
    insider_min_date = insider_df['filingDate'].min()
    today = datetime.date.today()

    stock_prices_df = yf.download(stock_symbol, insider_min_date, today)
    stock_prices_df.columns = stock_prices_df.columns.droplevel(1)  # Drop the ticker symbol level

    stock_prices_df.drop(columns=['Open', 'High', 'Low', 'Volume'], inplace=True)
    monthly_max_df = stock_prices_df['Close'].resample('ME').max().to_frame()
    monthly_max_df = monthly_max_df.reset_index()
    insider_df = insider_df[insider_df['transactionPrice'] != 0.000]
    insider_df = insider_df.reset_index(drop=True)
    merged_df = pd.merge(insider_df, monthly_max_df, how='right', left_on='filingDate', right_on='Date')
    merged_df = merged_df.sort_values('Date', ascending=False)
    merged_df['max_price_after_tran'] = (
        merged_df['Close']
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
        columns=['Date', 'id', 'symbol', 'pct_change', 'Close', 'max_price_after_tran', 'currency', 'filingDate',
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

    new_data = ready_df.head(1).drop(columns=['trend'])

    ready_df = ready_df.iloc[1:]

    X = ready_df.drop(columns='trend', inplace=False)
    y = ready_df['trend']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    xg_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', enable_categorical=True)
    xg_model.fit(X_train, y_train)
    xg_y_pred = xg_model.predict(X_test)
    xg_y_pred_labels = trend_encoder.inverse_transform(xg_y_pred)
    xg_acc_score = accuracy_score(y_test, xg_y_pred)
    xg_predicted_trend_encoded = xg_model.predict(new_data)
    xg_predicted_trend = trend_encoder.inverse_transform(xg_predicted_trend_encoded)

    Dense = tf.keras.layers.Dense
    Input = tf.keras.layers.Input
    seq_model = tf.keras.models.Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(2, activation='softmax')
        # Dense(3, activation='softmax')
    ])
    seq_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    seq_model.fit(X_train, y_train, epochs=50, batch_size=10, validation_split=0.2)
    seq_y_pred_prob = seq_model.predict(X_test)
    seq_y_pred = np.argmax(seq_y_pred_prob, axis=1)
    seq_y_pred_labels = trend_encoder.inverse_transform(seq_y_pred)
    seq_acc_score = accuracy_score(y_test, seq_y_pred)
    seq_predicted_trend_encoded = seq_model.predict(new_data)
    seq_trend_pred_prob = np.argmax(seq_predicted_trend_encoded, axis=1)
    seq_predicted_trend_label = trend_encoder.inverse_transform(seq_trend_pred_prob)

    rf_model = RandomForestClassifier(n_estimators=100, max_depth=43, random_state=42)
    # rf_model.fit(X_train_resampled, y_train_resampled)
    rf_model.fit(X_train, y_train)

    rf_y_pred = rf_model.predict(X_test)

    rf_y_pred_labels = trend_encoder.inverse_transform(rf_y_pred)
    # print(classification_report(y_test, y_pred, target_names=['Buy', 'Hold', 'Hold']))
    rf_acc_score = accuracy_score(y_test, rf_y_pred)
    rf_predicted_trend_encoded = rf_model.predict(new_data)
    rf_predicted_trend = trend_encoder.inverse_transform(rf_predicted_trend_encoded)

    stacking_clf = StackingClassifier(estimators=[
        ('xgboost', xg_model),
        ('random-forest', rf_model)],
        final_estimator=LogisticRegression())

    # Fit the stacking classifier
    stacking_clf.fit(X_train, y_train)

    # Predict using the meta-classifier
    stacking_pred = stacking_clf.predict(X_test)
    stacking_acc = accuracy_score(y_test, stacking_pred)
    stacking_pred_label = trend_encoder.inverse_transform(stacking_pred)

    stacking_new_data_pred = stacking_clf.predict(new_data)
    stacking_trend_pred_label = trend_encoder.inverse_transform(stacking_new_data_pred)[0]

    xg_y_pred_prob = xg_model.predict_proba(X_test)
    rf_y_pred_prob = rf_model.predict_proba(X_test)
    avg_probs = (xg_y_pred_prob + rf_y_pred_prob + seq_y_pred_prob) / 3

    avg_y_pred = np.argmax(avg_probs, axis=1)
    avg_y_pred_label = trend_encoder.inverse_transform(avg_y_pred)
    avg_acc = accuracy_score(y_test, avg_y_pred)

    xg_predicted_trend_prob = xg_model.predict_proba(new_data)
    dt_predicted_trend_prob = rf_model.predict_proba(new_data)

    avg_trend_pred_probs = (xg_predicted_trend_prob + dt_predicted_trend_prob + seq_trend_pred_prob) / 3
    avg_final_trend_pred = np.argmax(avg_trend_pred_probs, axis=1)
    avg_final_trend_pred_label = trend_encoder.inverse_transform(avg_final_trend_pred)[0]

    return {"xg_classifier": {"trend": xg_predicted_trend[0], "pred": xg_y_pred_labels, "accuracy": round(xg_acc_score, 2), "pred_num": xg_y_pred, "pred_prob": xg_y_pred_prob},
            "sequential": {"trend": seq_predicted_trend_label[0], "pred": seq_y_pred_labels, "accuracy": round(seq_acc_score, 2), "pred_num": seq_y_pred, "pred_prob": seq_y_pred_prob},
            "random_forrest": {"trend": rf_predicted_trend[0], "pred": rf_y_pred_labels, "accuracy": round(rf_acc_score, 2), "pred_num": rf_y_pred, "pred_prob": rf_y_pred_prob},
            "stacking": {"trend": stacking_trend_pred_label, "pred": stacking_pred_label, "accuracy": round(stacking_acc, 2)},
            "average": {"trend": avg_final_trend_pred_label, "pred": avg_y_pred_label, "accuracy": round(avg_acc, 2)},
            "y_test": y_test}