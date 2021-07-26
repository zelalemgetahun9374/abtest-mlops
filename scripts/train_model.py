import math
from config import Config
from csv_helper import CsvHelper
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

Config.MODELS_PATH.mkdir(parents=True, exist_ok=True)
helper = CsvHelper()

def eval_metrics(actual, pred):
  rmse = math.sqrt(mean_squared_error(actual, pred))
  mae = mean_absolute_error(actual, pred)
  r2 = r2_score(actual, pred)
  return rmse, mae, r2

X_train = helper.read_csv(str(Config.FEATURES_PATH / "train_features.csv"))
y_train = helper.read_csv(str(Config.FEATURES_PATH / "train_labels.csv"))


def train_model(_model, model_name):
    mlflow.set_experiment(model_name)

    kf = KFold(n_splits=5)
    solvers = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
    avg_score = 0
    avg_rmse = 0
    avg_mae = 0
    avg_r2 = 0
    best_model = None
    best_solver = solvers[0]

    if(model_name == 'XGBoost'):
        mlflow.xgboost.autolog()
    else:
        mlflow.sklearn.autolog()
    mlflow.log_param('Model', model_name)
    mlflow.log_param('Solvers', solvers)

    for solver in solvers:
        score_list = []
        rmse_list = []
        mae_list = []
        r2_list = []
        model = _model(solver)
        randomIter = kf.split(X_train)
        for i in range(5):
            train_index, val_index = next(randomIter)
            _X_train = X_train.iloc[train_index]
            _y_train = y_train.iloc[train_index]

            _X_val = X_train.iloc[val_index]
            _y_val = y_train.iloc[val_index]

            model.fit(_X_train, _y_train.to_numpy().ravel())
            y_pred = model.predict(_X_val)
            _score = accuracy_score(_y_val, y_pred)
            _rmse, _mae, _r2 = eval_metrics(_y_val, y_pred)
            score_list.append(_score)
            rmse_list.append(_rmse)
            mae_list.append(_mae)
            r2_list.append(_r2)

        avg_score_for_solver = sum(score_list) / len(score_list)
        if(avg_score_for_solver > avg_score):
            avg_score = avg_score_for_solver
            avg_rmse = sum(rmse_list) / len(rmse_list)
            avg_mae = sum(mae_list) / len(mae_list)
            avg_r2 = sum(r2_list) / len(r2_list)
            best_model = model
            best_solver = solver

    mlflow.log_param('Best Solver', best_solver)

    if(model_name == 'XGBoost'):
        mlflow.log_metric("Average Score", avg_score)
        mlflow.log_metric("RMSE", avg_rmse)
        mlflow.log_metric("MAE", avg_mae)
        mlflow.log_metric("R2", avg_r2)
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.xgboost.log_model(model, model_name, signature=signature)

    return best_model
