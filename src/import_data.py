import pandas as pd
import numpy as np

def load_data():
    train_csv_file = pd.read_csv("../datasets/国际象棋Checkmate预测/train.csv")
    X_train = np.array(train_csv_file.iloc[:, 0:6].values.astype(np.float32))
    Y_train = np.array(train_csv_file.iloc[:, 6].values.astype(np.int32))
    test_csv_file = pd.read_csv("../datasets/国际象棋Checkmate预测/test.csv")
    X_test = np.array(test_csv_file.iloc[:, 0:6].values.astype(np.float32))
    Y_test = np.array(test_csv_file.iloc[:, 6].values.astype(np.int32))
    return X_train, Y_train, X_test, Y_test
