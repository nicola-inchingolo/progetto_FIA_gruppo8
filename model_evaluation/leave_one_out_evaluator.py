import matplotlib
matplotlib.use("TkAgg")  # oppure "Qt5Agg"
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from MockKNN import MockKNN
from calculate_metrics import metrics
from holdout_evaluator import confusion_matrix_binary

if __name__ == "__main__":

    num_samples = 200  # dataset più grande

    # DataFrame di esempio
    df = pd.DataFrame({
        "ID": range(1, num_samples + 1),
        "Sample code number": range(1001, 1001 + num_samples),
        "Clump Thickness": np.round(np.random.uniform(1, 10, num_samples), 1),
        "Uniformity of Cell Size": np.round(np.random.uniform(1, 10, num_samples), 1),
        "Uniformity of Cell Shape": np.round(np.random.uniform(1, 10, num_samples), 1),
        "Marginal Adhesion": np.round(np.random.uniform(1, 10, num_samples), 1),
        "Single Epithelial Cell Size": np.round(np.random.uniform(1, 10, num_samples), 1),
        "Bare Nuclei": np.round(np.random.uniform(1, 10, num_samples), 1),
        "Bland Chromatin": np.round(np.random.uniform(1, 10, num_samples), 1),
        "Normal Nucleoli": np.round(np.random.uniform(1, 10, num_samples), 1),
        "Mitoses": np.round(np.random.uniform(1, 5, num_samples), 1),
        # Classi distribuite in modo bilanciato, almeno 1 benigno per fold
        "Class": [2, 4] * (num_samples // 2)
    })

    print(df)

    rows, column = df.shape

    size = 0
    metrics_list = dict()
    tpr_array = np.array([])  # array vuoto
    fpr_array = np.array([])
    auc_list = np.array([])

    y_test_all = np.array([])
    y_score_all = np.array([])
    y_pred_all = np.array([])

    for i in range(rows):
        test_section = df.iloc[size:size + 1]
        train_section = pd.concat([df.iloc[:size], df.iloc[size + 1:]])
        print(f"test set: \n {test_section} \n")
        print(f"train set: \n {train_section} \n")

        x_train = train_section.drop(columns=["ID", "Sample code number", "Class"])
        y_train = train_section["Class"]

        x_test = test_section.drop(columns=["ID", "Sample code number", "Class"])
        y_test = test_section["Class"]

        model = MockKNN(k=5, seed=42, error_rate=0.3)
        model.fit(x_train, y_train)
        y_pred, y_score = model.predict(x_test)

        y_test_all = np.append(y_test_all, y_test)
        y_score_all = np.append(y_score_all, y_score)
        y_pred_all = np.append(y_pred_all, y_pred)

        cm = confusion_matrix_binary(y_test, y_pred, 4, 2, True)
        print("Confusion Matrix:")
        print(type(cm))

        print(f"TP: {cm[0][0]}")
        print(f"FP: {cm[0][1]}")
        print(f"FN: {cm[1][0]}")
        print(f"TN: {cm[1][1]}")

        metrics_calculator = metrics(cm)
        metrics_list[i] = metrics_calculator.calculate_all_the_above()
        print(f"\n metrcis list: \n {metrics_list} \n")

        size += 1

    tpr_list = []
    fpr_list = []

    thresholds = np.linspace(0, 1, 50)
    for thresh in thresholds:
        TP = np.sum((y_score_all >= thresh) & (y_test_all == 4))
        FP = np.sum((y_score_all >= thresh) & (y_test_all == 2))
        FN = np.sum((y_score_all < thresh) & (y_test_all == 4))
        TN = np.sum((y_score_all < thresh) & (y_test_all == 2))

        tpr = TP / (TP + FN) if (TP + FN) > 0 else 0
        fpr = FP / (FP + TN) if (FP + TN) > 0 else 0

        tpr_list.append(tpr)
        fpr_list.append(fpr)

        # Ordinamento per FPR crescente
    fpr_array = np.array(fpr_list)
    tpr_array = np.array(tpr_list)
    sorted_idx = np.argsort(fpr_array)
    fpr_array = fpr_array[sorted_idx]
    tpr_array = tpr_array[sorted_idx]
    auc = np.trapezoid(tpr_array, fpr_array)

    sum_metrics = {}
    for fold in metrics_list.values():
        for key, value in fold.items():
            if key not in sum_metrics:
                sum_metrics[key] = 0
            sum_metrics[key] += value

    num_folds = len(metrics_list)
    mean_metrics = {key: sum_metrics[key] / num_folds for key in sum_metrics}
    cm_all = confusion_matrix_binary(y_test_all, y_pred_all)
    mean_metrics["AUC"] = auc
    print("\nMedie delle metriche su tutti i fold:")
    print(mean_metrics)

    print(f"\nAUC (ROC): {auc}")

    # --- Traccia ROC ---
    plt.figure()
    plt.plot(fpr_array, tpr_array, marker='o', linestyle='-', color='blue')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.show()

    classes = ['2 (Benigno)', '4 (Maligno)']

    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Purples)
    plt.title("Confusion Matrix Normalizzata")
    plt.colorbar()

    # Tick sugli assi
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    # Scrive i valori nella matrice
    thresh = cm_all.max() / 2
    for i in range(cm_all.shape[0]):
        for j in range(cm_all.shape[1]):
            plt.text(j, i, f"{cm_all[i, j]:.2f}",
                     horizontalalignment="center",
                     color="white" if cm_all[i, j] > thresh else "black")

    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.show()