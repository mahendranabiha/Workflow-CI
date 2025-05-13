# Import Libraries
import mlflow
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    zero_one_loss,
    ConfusionMatrixDisplay,
    classification_report,
)

if __name__ == "__main__":
    # Membaca train set dan test set untuk masing-masing fitur dan label
    X_train = pd.read_csv("weather_preprocessing/X_train.csv")
    X_test = pd.read_csv("weather_preprocessing/X_test.csv")
    y_train = pd.read_csv("weather_preprocessing/y_train.csv")
    y_test = pd.read_csv("weather_preprocessing/y_test.csv")

    # Mendefinisikan dictionary dari hyperparameter yang dilakukan
    param_grid = {
        "n_estimators": [2, 4, 5],
        "max_depth": [10, 20, 50],
        "random_state": [1993],
    }

    # Memulai running mlflow
    with mlflow.start_run(run_name="model_tuning_latest"):
        # RandomForestClassifier object: rf
        rf = RandomForestClassifier()

        # Melakukan Grid Search Cross Validation pada train set
        grid_search = GridSearchCV(rf, param_grid, cv=5, n_jobs=1, verbose=2)

        grid_search.fit(X_train, y_train)

        # Log best model, lalu menyimpan file artefak MLflow di DagsHub
        mlflow.sklearn.log_model(grid_search, "model")

        best_params = grid_search.best_params_

        # Log parameters
        mlflow.log_params(best_params)

        # Prediksi model
        y_pred = grid_search.predict(X_test)
        y_pred_prob = grid_search.predict_proba(X_test)[:, 1]

        # Evaluasi model
        accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
        precision = precision_score(y_true=y_test, y_pred=y_pred)
        recall = recall_score(y_true=y_test, y_pred=y_pred)
        f1score = f1_score(y_true=y_test, y_pred=y_pred)
        roc_auc = roc_auc_score(y_true=y_test, y_score=y_pred_prob)
        zero_one = zero_one_loss(y_true=y_test, y_pred=y_pred)

        # Manual log metrics yang ada juga dalam autolog
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
            grid_search,
            X_test,
            y_pred,
            cmap=plt.cm.Blues,
            colorbar=False,
        )
        plt.savefig("confusion_matrix.jpg")
        mlflow.log_artifact("confusion_matrix.jpg")
