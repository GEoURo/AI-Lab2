import pandas as pd
import numpy as np

label = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, 
"nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, 
"fifteen": 15, "sixteen": 16, "draw": 17}

csv_file = pd.read_csv("./datasets/国际象棋Checkmate预测/testset.csv")
WKing_col = csv_file.iloc[:, 0].values
WKing_row = csv_file.iloc[:, 1].values.astype(np.int8)
WRook_col = csv_file.iloc[:, 2].values
WRook_row = csv_file.iloc[:, 3].values.astype(np.int8)
BKing_col = csv_file.iloc[:, 4].values
BKing_row = csv_file.iloc[:, 5].values.astype(np.int8)
labels = csv_file.iloc[:, 6].values

for i in range(len(WKing_col)):
    WKing_col[i] = ord(WKing_col[i]) - ord("a") + 1
    WRook_col[i] = ord(WRook_col[i]) - ord("a") + 1
    BKing_col[i] = ord(BKing_col[i]) - ord("a") + 1
    labels[i] = label[labels[i]]

write_data = {"0": WKing_col, "1": WKing_row, "2": WRook_col, "3": WRook_row, "4": BKing_col, "5": BKing_row, "6": labels}
df = pd.DataFrame(write_data)
df.to_csv("./datasets/国际象棋Checkmate预测/test.csv", index=False)
print("Data processing complete!")