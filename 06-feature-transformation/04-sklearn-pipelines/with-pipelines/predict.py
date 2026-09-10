import pickle
import pandas as pd


# Load the complete fitted pipeline
with open("pipe.pkl", "rb") as file:
    pipe = pickle.load(file)


# ---------------------------------------------------------
# User input
#
# Features:
# Pclass, Sex, Age, SibSp, Parch, Fare, Embarked
# ---------------------------------------------------------
test_input = pd.DataFrame(
    {
        "Pclass": [2],
        "Sex": ["male"],
        "Age": [31.0],
        "SibSp": [0],
        "Parch": [0],
        "Fare": [10.5],
        "Embarked": ["S"],
    }
)


# ---------------------------------------------------------
# Make prediction
#
# The complete pipeline handles:
# - Missing-value imputation
# - One-hot encoding
# - Scaling
# - Feature selection
# - Decision tree prediction
# ---------------------------------------------------------
prediction = pipe.predict(test_input)

print("Prediction:", prediction[0])

if prediction[0] == 1:
    print("Passenger is predicted to survive.")
else:
    print("Passenger is predicted not to survive.")


# Optional: probability of survival
if hasattr(pipe, "predict_proba"):
    probability = pipe.predict_proba(test_input)[0, 1]
    print(f"Probability of survival: {probability:.2%}")