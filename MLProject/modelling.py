# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    zero_one_loss,
    classification_report,
    ConfusionMatrixDisplay,
)
import sys

if __name__ == "__main__":
    # Membaca train set dan test set untuk masing-masing fitur dan label
    X_train = pd.read_csv("weather_preprocessing/X_train.csv")
    X_test = pd.read_csv("weather_preprocessing/X_test.csv")
    y_train = pd.read_csv("weather_preprocessing/y_train.csv")
    y_test = pd.read_csv("weather_preprocessing/y_test.csv")

    # Membaca parameters
    n_estimators = int(sys.argv[1])
    max_depth = int(sys.argv[2])

    # Memulai running mlflow
    with mlflow.start_run():
        # Pelatihan model
        rf = RandomForestClassifier(
            max_depth=max_depth, n_estimators=n_estimators, random_state=1993
        )
        rf.fit(X_train, y_train)

        # Log parameters
        mlflow.log_params({"max_depth": 2, "n_estimators": 5, "random_state": 1993})

        # Menyimpan model dalam file lokal
        mlflow.sklearn.save_model(rf, "model_local_path")

        # Log model
        mlflow.sklearn.log_model(rf, "model")

        # Prediksi model
        y_pred = rf.predict(X_test)
        y_pred_prob = rf.predict_proba(X_test)[:, 1]

        # Evaluasi model
        accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
        precision = precision_score(y_true=y_test, y_pred=y_pred)
        recall = recall_score(y_true=y_test, y_pred=y_pred)
        f1score = f1_score(y_true=y_test, y_pred=y_pred)
        roc_auc = roc_auc_score(y_true=y_test, y_score=y_pred_prob)
        zero_one = zero_one_loss(y_true=y_test, y_pred=y_pred)

        # Manual log metrics yang ada juga dalam autolog dan yang tidak hanya tercover pada autolog
        mlflow.log_metrics(
            {
                "accuracy": accuracy,
                "precision_score": precision,
                "recall_score": recall,
                "f1_score": f1score,
                "roc_auc_score": roc_auc,
                "zero_one_loss": zero_one,
            }
        )

        # Manual log metrcis yang tidak hanya tercover pada autolog, lalu menyimpan file artefak MLflow di DagsHub
        classfication = classification_report(y_true=y_test, y_pred=y_pred)
        with open("classification_report.txt", "w") as file:
            file.write(classfication)
        mlflow.log_artifact("classification_report.txt")

        ConfusionMatrixDisplay.from_estimator(
            rf,
            X_test,
            y_pred,
            cmap=plt.cm.Blues,
            colorbar=False,
        )
        plt.savefig("confusion_matrix.jpg")
        mlflow.log_artifact("confusion_matrix.jpg")
