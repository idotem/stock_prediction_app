from django.shortcuts import render

from .analyst_estimations import analyst_estimations
from .analyst_recommendations import analyst_recommendations
from .graphrag_chat import answer_question
from .insider_transactions import insider_transactions
from sklearn.metrics import accuracy_score
from scipy import stats
import numpy as np


def default_graphrag(request):
    return render(request, "chat.html")

def ask_question(request):
    question = request.GET.get("question")
    if question:
        context_success = {"answer": answer_question(question)}
        return render(request, "answer.html", context_success)
    else:
        context_err = {"error": "You submitted an empty question. Submit a valid one!"}
        return render(request, "error.html", context_err)



# stock_symbols = ['AAPL', 'NVDA', 'SNOW', 'AMD', 'CAT', 'CHWY', 'ETSY', 'GOOGL', 'MSFT', 'NVDA', 'PAYC',
#                  'PAYO', 'PEP', 'PFE', 'PLTR', 'SHOP', 'SQ', 'TSLA', 'VRNT']
stock_symbols = ['AAPL', 'NVDA', 'WBA', 'NKE', 'SHOP']
percentage_limit = 5


def default_stocks(request):
    symbol = 'AAPL'
    return get_data_and_render_view(request, symbol)


def stocks(request, symbol):
    return get_data_and_render_view(request, symbol)


def get_data_and_render_view(request, symbol):
    insider_pred = insider_transactions(symbol, percentage_limit)
    analyst_rcmd = analyst_recommendations(symbol, percentage_limit)
    ins_xg_pred = insider_pred['xg_classifier']['pred_num']
    ins_rf_pred = insider_pred['random_forrest']['pred_num']
    ins_seq_pred = insider_pred['sequential']['pred_num']
    rcmd_xg_pred = insider_pred['xg_classifier']['pred_num']
    rcmd_rf_pred = insider_pred['random_forrest']['pred_num']
    rcmd_seq_pred = insider_pred['sequential']['pred_num']

    min_samples = min(ins_xg_pred.shape[0], ins_rf_pred.shape[0], ins_seq_pred.shape[0],
                      rcmd_xg_pred.shape[0], rcmd_rf_pred.shape[0], rcmd_seq_pred.shape[0])
    # rcmd_y_test = analyst_rcmd['y_test'][:min_samples]
    ins_y_test = insider_pred['y_test'][:min_samples]

    combined_predictions = np.column_stack(
        (ins_xg_pred[:min_samples], ins_rf_pred[:min_samples], ins_seq_pred[:min_samples],
         rcmd_xg_pred[:min_samples], rcmd_rf_pred[:min_samples], rcmd_seq_pred[:min_samples]))

    final_pred = stats.mode(combined_predictions, axis=1)[0].ravel()
    ins_final_acc = accuracy_score(ins_y_test, final_pred)
    # rcmd_final_acc = accuracy_score(rcmd_y_test, final_pred)
    context = {"stocks": stock_symbols,
               "insider": insider_pred,
               "analyst_rcmd": analyst_rcmd,
               # "analyst_est": get_analyst_estimations_pred(symbol),
               "selected_symbol": symbol,
               "final_pred": final_pred,
               "ins_final_acc": ins_final_acc,
               "rcmd_final_acc": ins_final_acc}
    return render(request, "stocks.html", context)

# def insider_xg_classifier_pred(X_test, X_train, last_row, trend_encoder, y_test, y_train):
#     scaler = MinMaxScaler()
#     X_train = scaler.fit_transform(X_train)
#     X_test = scaler.transform(X_test)
#     model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', enable_categorical=True)
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     y_pred_labels = trend_encoder.inverse_transform(y_pred)
#     acc_score = accuracy_score(y_test, y_pred)
#     predicted_trend_encoded = model.predict(last_row)
#     predicted_trend = trend_encoder.inverse_transform(predicted_trend_encoded)
#     return {"trend": predicted_trend[0], "pred": y_pred_labels, "accuracy": round(acc_score, 2)}
#

# def insider_sequential_pred(X_test, X_train, last_row, trend_encoder, y_test, y_train):
#     Dense = tf.keras.layers.Dense
#     Input = tf.keras.layers.Input
#     model = tf.keras.models.Sequential([
#         Input(shape=(X_train.shape[1],)),
#         Dense(64, activation='relu'),
#         Dense(32, activation='relu'),
#         Dense(2, activation='softmax')
#         # Dense(3, activation='softmax')
#     ])
#     model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#
#     model.fit(X_train, y_train, epochs=50, batch_size=10, validation_split=0.2)
#     y_pred_prob = model.predict(X_test)
#     y_pred = np.argmax(y_pred_prob, axis=1)
#     y_pred_labels = trend_encoder.inverse_transform(y_pred)
#     acc_score = accuracy_score(y_test, y_pred)
#     predicted_trend_encoded = model.predict(last_row)
#     last_row_pred = np.argmax(predicted_trend_encoded, axis=1)
#     predicted_trend = trend_encoder.inverse_transform(last_row_pred)
#     return {"trend": predicted_trend[0], "pred": y_pred_labels, "accuracy": round(acc_score, 2)}
#
#
# def insider_random_forrest_pred(X_test, X_train, last_row, trend_encoder, y_test, y_train):
#     model = RandomForestClassifier(n_estimators=100, max_depth=43, random_state=42)
#     # model.fit(X_train_resampled, y_train_resampled)
#     model.fit(X_train, y_train)
#
#     y_pred = model.predict(X_test)
#
#     y_pred_labels = trend_encoder.inverse_transform(y_pred)
#     # print(classification_report(y_test, y_pred, target_names=['Buy', 'Hold', 'Hold']))
#     acc_score = accuracy_score(y_test, y_pred)
#     predicted_trend_encoded = model.predict(last_row)
#     predicted_trend = trend_encoder.inverse_transform(predicted_trend_encoded)
#     return {"trend": predicted_trend[0], "pred": y_pred_labels, "accuracy": round(acc_score, 2)}
