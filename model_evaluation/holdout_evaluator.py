import matplotlib
matplotlib.use("TkAgg")  # oppure "Qt5Agg"
import matplotlib.pyplot as plt
import random
import pandas as pd
import numpy as np
from MockKNN import MockKNN
from calculate_metrics import metrics
import matplotlib.pyplot as plt


def confusion_matrix_binary(y_test, y_pred, pos_label=4, neg_label=2, normalize=True):
    TP = np.sum((y_test == pos_label) & (y_pred == pos_label))
    TN = np.sum((y_test == neg_label) & (y_pred == neg_label))
    FP = np.sum((y_test == neg_label) & (y_pred == pos_label))
    FN = np.sum((y_test == pos_label) & (y_pred == neg_label))

    cm = np.array([[TN, FP],
                   [FN, TP]])

    if normalize:
        cm = cm.astype('float') / cm.sum()  # divide per il totale dei campioni

    return cm


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

    train_percentage = random.uniform(0.6, 0.9)

    split_index = int(len(df) * train_percentage)
    dtr = df.iloc[:split_index]
    dtst = df.iloc[split_index:]

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

    # calculate prediction
    model = MockKNN(k=5, seed=42, error_rate=0.3)
    model.fit(x_train, y_train)
    y_pred, y_score = model.predict(x_test, y_test=y_test)

    print(y_pred)

    # making confusion matrix
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
    # sensitivity = metrics_calculator.calculate_sensitivity()
    # specificity = metrics_calculator.calculate_specificity()
    g_mean = metrics_calculator.calculate_geometric_mean()
    precision = metrics_calculator.calculate_precision()
    f1_score = metrics_calculator.calculate_f1_score()

    # Calcolo ROC per più soglie
    thresholds = np.linspace(0, 1, 100)
    tpr_list = []
    fpr_list = []

    for thresh in thresholds:
        TP = np.sum((y_score >= thresh) & (y_test == 4))
        FP = np.sum((y_score >= thresh) & (y_test == 2))
        FN = np.sum((y_score < thresh) & (y_test == 4))
        TN = np.sum((y_score < thresh) & (y_test == 2))

        tpr = TP / (TP + FN) if (TP + FN) > 0 else 0
        fpr = FP / (FP + TN) if (FP + TN) > 0 else 0

        tpr_list.append(tpr)
        fpr_list.append(fpr)

    sorted_idx = np.argsort(fpr_list)
    fpr_list = np.array(fpr_list)[sorted_idx]
    tpr_list = np.array(tpr_list)[sorted_idx]
    auc = np.trapezoid(tpr_list, fpr_list)

    print(f"AUC: {auc}")

    # --- Traccia ROC ---
    plt.figure()
    plt.plot(fpr_list, tpr_list, marker='o', linestyle='-', color='violet')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.show()

    classes = ['2 (Benigno)', '4 (Maligno)']

    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix Normalizzata")
    plt.colorbar()

    # Tick sugli assi
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    # Scrive i valori nella matrice
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, f"{cm[i, j]:.2f}",
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.show()






