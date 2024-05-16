from fastapi import FastAPI
import pandas as pd
import uvicorn
import joblib

app = FastAPI()

model = joblib.load("model.pkl")

@app.get("/predict")
def predict(
    Intake_Type: str,
    Intake_Condition: str,
    Sex_upon_Intake: str,
    Age_Upon_Intake_Days: int
):
    model_data_dict = {
            "Intake_Type": Intake_Type,
            "Intake_Condition": Intake_Condition,
            "Sex_upon_Intake": Sex_upon_Intake,
            "Age_Upon_Intake_Days": Age_Upon_Intake_Days
        }

    index = [0]  # You can create a more meaningful index here
    model_data_df = pd.DataFrame(model_data_dict, index=index)

    model_data_df.columns = model_data_df.columns.str.lower()

    result = model.predict(model_data_df).tolist()

    return {"result": result}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, debug=True)
