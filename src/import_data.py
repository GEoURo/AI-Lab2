import pandas as pd
import numpy as np

def load_data_chess(filename=''):
    train_csv_file = pd.read_csv("../datasets/国际象棋Checkmate预测/" + "train" + filename + ".csv")
    X_train = np.array(train_csv_file.iloc[:, 0:6].values.astype(np.float32))
    Y_train = np.array(train_csv_file.iloc[:, 6].values.astype(np.int32))
    test_csv_file = pd.read_csv("../datasets/国际象棋Checkmate预测/" + "test" + filename + ".csv")
    X_test = np.array(test_csv_file.iloc[:, 0:6].values.astype(np.float32))
    Y_test = np.array(test_csv_file.iloc[:, 6].values.astype(np.int32))
    return X_train, Y_train, X_test, Y_test

def load_data_frog():
    train_csv_file = pd.read_csv("../datasets/青蛙聚类/Frogs_MFCCs.csv")
    X = np.array(train_csv_file.iloc[:, 0:22].values.astype(np.float32))
    Y = np.array(train_csv_file.iloc[:, 22:25])
    return X, Y
