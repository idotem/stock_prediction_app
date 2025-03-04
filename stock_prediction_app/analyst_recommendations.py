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
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, OneHotEncoder



def analyst_recommendations(stock_symbol, percentage_limit):
    analyst_rcmd = pd.read_json('./data/analyst_data/' + stock_symbol + '_RCMD.json')
    analyst_rcmd_df = pd.DataFrame(analyst_rcmd)
    analyst_rcmd_df['dym'] = analyst_rcmd_df['date'].dt.to_period('M')
    rcmd_min_date = analyst_rcmd_df['date'].min()
    today = datetime.date.today()
    stock_prices_df = yf.download('AAPL', rcmd_min_date, today)
    stock_prices_df.drop(columns=['Open', 'High', 'Low', 'Volume'], inplace=True)
    monthly_max_df = stock_prices_df['Close'].resample('M').max().to_frame()
    monthly_max_df['dym'] = monthly_max_df.index.to_period('M')
    analyst_rcmd_df['dym'] = analyst_rcmd_df['date'].dt.to_period('M')
    merged_df = pd.merge(analyst_rcmd_df, monthly_max_df, how='left', left_on='dym', right_on='dym')
    merged_df = merged_df.sort_values('dym', ascending=True)
    merged_df = merged_df.reset_index(drop=True)

    merged_df['pct_change'] = (merged_df['Close'].shift(-1) - merged_df['Close']) / merged_df[
        'Close'].shift(-1) * 100
    merged_df['trend'] = pd.cut(merged_df['pct_change'], bins=[-float('inf'), percentage_limit,
                                                               float('inf')], labels=['Hold', 'Buy'])
    ready_df = merged_df.drop(columns=['date', 'dym', 'symbol', 'pct_change', 'Close'], inplace=False)
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

    xg_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', enable_categorical=True)
    xg_model.fit(X_train, y_train)
    xg_y_pred = xg_model.predict(X_test)
    xg_acc = accuracy_score(y_test, xg_y_pred)
    xg_pred_labels = label_encoder.inverse_transform(xg_y_pred)

    new_data = merged_df.tail(1).drop(columns=['date', 'dym', 'symbol', 'pct_change', 'trend', 'Close'],
                                      inplace=False)
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
    seq_y_pred_labels = label_encoder.inverse_transform(seq_y_pred)

    seq_predicted_trend_prob = seq_model.predict(new_data)
    y_pred_new_data = np.argmax(seq_predicted_trend_prob, axis=1)
    seq_pred = label_encoder.inverse_transform(y_pred_new_data)[0]

    rf_model = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42)
    rf_model.fit(X_train, y_train)

    rf_y_pred = rf_model.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_y_pred)
    rf_y_pred_labels = label_encoder.inverse_transform(rf_y_pred)

    rf_predicted_trend_encoded = rf_model.predict(new_data)
    rf_pred = label_encoder.inverse_transform(rf_predicted_trend_encoded)[0]

    xg_y_pred_prob = xg_model.predict_proba(X_test)
    rf_y_pred_prob = rf_model.predict_proba(X_test)
    avg_probs = (xg_y_pred_prob + rf_y_pred_prob + seq_y_pred_prob) / 3

    avg_y_pred = np.argmax(avg_probs, axis=1)
    avg_acc = accuracy_score(y_test, avg_y_pred)
    avg_y_pred_label = label_encoder.inverse_transform(avg_y_pred)

    xg_predicted_trend_prob = xg_model.predict_proba(new_data)
    dt_predicted_trend_prob = rf_model.predict_proba(new_data)

    avg_trend_pred_probs = (xg_predicted_trend_prob + dt_predicted_trend_prob + seq_predicted_trend_prob) / 3
    final_trend_pred = np.argmax(avg_trend_pred_probs, axis=1)
    final_trend_pred_label = label_encoder.inverse_transform(final_trend_pred)[0]

    # Stack the models
    stacking_clf = StackingClassifier(estimators=[
        ('xgboost', xg_model),
        ('random-forest', rf_model)],
        final_estimator=LogisticRegression())

    # Fit the stacking classifier
    stacking_clf.fit(X_train, y_train)

    # Predict using the meta-classifier
    stacking_pred = stacking_clf.predict(X_test)
    stacking_acc = accuracy_score(y_test, stacking_pred)
    stacking_pred_label = label_encoder.inverse_transform(stacking_pred)

    stacking_new_data_pred = stacking_clf.predict(new_data)
    stacking_trend_pred_label = label_encoder.inverse_transform(stacking_new_data_pred)[0]

    return {"xg_classifier": {"trend": xg_pred, "pred": xg_pred_labels, "accuracy": round(xg_acc, 2), "pred_num": xg_y_pred, "pred_prob": xg_y_pred_prob},
            "sequential": {"trend": seq_pred, "pred": seq_y_pred_labels, "accuracy": round(seq_acc, 2), "pred_num": seq_y_pred, "pred_prob": seq_y_pred_prob},
            "random_forrest": {"trend": rf_pred, "pred": rf_y_pred_labels, "accuracy": round(rf_acc, 2), "pred_num": rf_y_pred, "pred_prob": rf_y_pred_prob},
            "stacking": {"trend": stacking_trend_pred_label, "pred": stacking_pred_label, "accuracy": round(stacking_acc, 2)},
            "average": {"trend": final_trend_pred_label, "pred": avg_y_pred_label, "accuracy": round(avg_acc, 2)},
            "y_test": y_test}