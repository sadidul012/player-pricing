import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error


def split_price_history(x):
    x = x[1:-1].split(",")
    x = x if len(x) > 1 else []
    rows = []
    for i in range(0, len(x), 3):
        rows.append([(",".join(x[i:i+2])).strip()[1:-1], float(x[i+2])])
    return rows


def float_convert(x, index):
    try:
        x = x[1:-1].split("',")[index][1:].replace(",", "")
        return float(x)
    except Exception as e:
        return 0


def print_losses(y_hat, y_tru):
    print("rmse", np.sqrt(mean_squared_error(y_hat, y_tru)))
    print("mse", mean_squared_error(y_hat, y_tru))
    print("mape", mean_absolute_percentage_error(y_hat, y_tru))
    print("mae", mean_absolute_error(y_hat, y_tru))
