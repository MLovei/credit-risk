import pandas as pd
from pydantic import BaseModel
import pickle
from fastapi import FastAPI

class LoanApplication(BaseModel):
    EXT_SOURCE_3: float = None
    EXT_SOURCE_2: float = None
    EXT_SOURCE_1: float = None
    AMT_CREDIT: float
    AGE_YEARS: float
    CODE_GENDER: str
    NAME_EDUCATION_TYPE: str

class PredictionOut(BaseModel):
    default_proba: float

model_filename = 'xgboost_model.pkl'
with open(model_filename, 'rb') as f:
    model = pickle.load(f)
app = FastAPI()

@app.get("/")
def home():
    return {"message": "Loan Default Prediction App", "model_version": 0.1}

@app.post("/predict", response_model=PredictionOut)
def predict(payload: LoanApplication):
    cust_df = pd.DataFrame([payload.dict()])

    cust_df = cust_df[["EXT_SOURCE_3", "EXT_SOURCE_2", "EXT_SOURCE_1",
                       "AMT_CREDIT", "AGE_YEARS", "CODE_GENDER",
                       "NAME_EDUCATION_TYPE"]]

    cust_df = pd.get_dummies(cust_df, columns=["CODE_GENDER",
                                               "NAME_EDUCATION_TYPE"],
                             dummy_na=False)

    expected_columns = model.get_booster().feature_names
    cust_df = cust_df.reindex(columns=expected_columns, fill_value=0)

    preds = model.predict_proba(cust_df)[0, 1]
    result = {"default_proba": preds}
    return result
