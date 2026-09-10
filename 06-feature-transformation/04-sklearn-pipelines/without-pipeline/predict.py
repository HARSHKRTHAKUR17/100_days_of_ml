import pickle
import numpy as np


# Load the fitted preprocessing objects and classifier
with open("models/ohe_sex.pkl", "rb") as file:
    ohe_sex = pickle.load(file)

with open("models/ohe_embarked.pkl", "rb") as file:
    ohe_embarked = pickle.load(file)

with open("models/clf.pkl", "rb") as file:
    clf = pickle.load(file)


# ---------------------------------------------------------
# Example user input
# Features:
# Pclass, Sex, Age, SibSp, Parch, Fare, Embarked
# ---------------------------------------------------------
test_input = np.array(
    [[2, "male", 31.0, 0, 0, 10.5, "S"]],
    dtype=object
)


# ---------------------------------------------------------
# Transform categorical features using the fitted encoders
# ---------------------------------------------------------
test_input_sex = ohe_sex.transform(
    test_input[:, 1].reshape(1, 1)
)

test_input_embarked = ohe_embarked.transform(
    test_input[:, 6].reshape(1, 1)
)


# Age is already numerical
test_input_age = test_input[:, 2].reshape(1, 1)


# ---------------------------------------------------------
# Combine features in the same order used during training
#
# Training order:
# Pclass, SibSp, Parch, Fare, Age,
# Sex_encoded, Embarked_encoded
# ---------------------------------------------------------
test_input_transformed = np.concatenate(
    (
        test_input[:, [0, 3, 4, 5]],
        test_input_age,
        test_input_sex,
        test_input_embarked,
    ),
    axis=1,
)


# ---------------------------------------------------------
# Make prediction
# ---------------------------------------------------------
prediction = clf.predict(test_input_transformed)

print("Prediction:", prediction[0])

if prediction[0] == 1:
    print("Passenger is predicted to survive.")
else:
    print("Passenger is predicted not to survive.")