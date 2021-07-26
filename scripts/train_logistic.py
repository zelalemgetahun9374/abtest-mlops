import pickle
from config import Config
from train_model import train_model
from sklearn.linear_model import LogisticRegression

def model(solver):
    model = LogisticRegression(solver=solver, random_state=42)
    return model

final_model = train_model(model, "Logistic Regression")
pickle.dump(final_model, open(str(Config.MODELS_PATH / "logistic_model.pickle"), "wb"))