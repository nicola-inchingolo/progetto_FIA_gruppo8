
import pandas as pd
import numpy as np
from holdout_split import houldout_split
from MockKNN import MockKNN
from calculate_metrics import metrics
import matplotlib
matplotlib.use('TkAgg')  # oppure 'Qt5Agg', se hai PyQt installato
import matplotlib.pyplot as plt

def confusion_matrix_binary(y_test, y_pred, pos_label=4, neg_label=2):
    TP = np.sum((y_test == pos_label) & (y_pred == pos_label))
    TN = np.sum((y_test == neg_label) & (y_pred == neg_label))
    FP = np.sum((y_test == neg_label) & (y_pred == pos_label))
    FN = np.sum((y_test == pos_label) & (y_pred == neg_label))
    return np.array([[TN, FP],
                     [FN, TP]])


if __name__ == "__main__":

    # DataFrame di esempio
    df = pd.DataFrame({
        "ID": [1, 2, 3, 4, 5],
        "Sample code number": [1001, 1002, 1003, 1004, 1005],
        "Clump Thickness": [5, 3, 8, 1, 6],
        "Uniformity of Cell Size": [1, 4, 7, 2, 5],
        "Uniformity of Cell Shape": [2, 3, 8, 1, 6],
        "Marginal Adhesion": [1, 2, 9, 1, 4],
        "Single Epithelial Cell Size": [2, 3, 6, 1, 5],
        "Bare Nuclei": [1, 3, 7, 1, 5],
        "Bland Chromatin": [3, 4, 6, 2, 5],
        "Normal Nucleoli": [1, 2, 5, 1, 4],
        "Mitoses": [1, 2, 3, 1, 2],
        "Class": [2, 2, 4, 2, 4]  # 2 benigno, 4 maligno
    })

    print(df)

    dtr, dtst = houldout_split(df)
    print(f"train set: \n {dtr} \n")
    print(f"test set: \n {dtst} \n")

    x_train = dtr.drop(columns=["ID", "Sample code number", "Class"])
    y_train = dtr["Class"]

    x_test = dtst.drop(columns=["ID", "Sample code number", "Class"])
    y_test = dtst["Class"]

    print(x_train)
    print(y_train)
    print(x_test)
    print(y_test)

    #calculate prediction
    model = MockKNN(k=5)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    print(y_pred)

#making confusion matrix
    cm = confusion_matrix_binary(y_test, y_pred)
    print("Confusion Matrix:")
    print(type(cm))

    print(f"TP: {cm[0][0]}")
    print(f"FP: {cm[0][1]}")
    print(f"FN: {cm[1][0]}")
    print(f"TN: {cm[1][1]}")

    metrics_calculator = metrics(cm)
    accuracy = metrics_calculator.calculate_accuracy()
    error_rate = metrics_calculator.calculate_error_rate()
    sensitivity = metrics_calculator.calculate_sensitivity()
    specificity = metrics_calculator.calculate_specificity()
    g_mean = metrics_calculator.calculate_geometric_mean()
    precision = metrics_calculator.calculate_precision()
    f1_score = metrics_calculator.calculate_f1_score()
    auc = metrics_calculator.calculate_auc()



# Esempio confusion matrix
cm = np.array([[2, 1],
               [1, 1]])

fig, ax = plt.subplots()
im = ax.imshow(cm, cmap='Blues')

# Aggiungi annotazioni
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='black')

ax.set_xticks([0,1])
ax.set_yticks([0,1])
ax.set_xticklabels(["Pred 2","Pred 4"])
ax.set_yticklabels(["True 2","True 4"])
ax.set_title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()





