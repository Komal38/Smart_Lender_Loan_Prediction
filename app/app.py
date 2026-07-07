import os
import pickle
import pandas as pd

from flask import Flask, render_template, request

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = pickle.load(open(os.path.join(BASE_DIR, "../models/rdf.pkl"), "rb"))
scaler = pickle.load(open(os.path.join(BASE_DIR, "../models/scaler.pkl"), "rb"))


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict", methods=["POST"])
def predict():

    gender = 1 if request.form["Gender"] == "Male" else 0
    married = 1 if request.form["Married"] == "Yes" else 0

    dep = request.form["Dependents"]
    dependents = 3 if dep == "3+" else int(dep)

    education = 0 if request.form["Education"] == "Graduate" else 1

    self_emp = 1 if request.form["Self_Employed"] == "Yes" else 0

    applicant_income = float(request.form["ApplicantIncome"])
    coapplicant_income = float(request.form["CoapplicantIncome"])
    loan_amount = float(request.form["LoanAmount"])
    loan_term = float(request.form["Loan_Amount_Term"])
    credit = int(request.form["Credit_History"])

    area = request.form["Property_Area"]

    if area == "Rural":
        area = 0
    elif area == "Semiurban":
        area = 1
    else:
        area = 2

    data = pd.DataFrame([[
        gender,
        married,
        dependents,
        education,
        self_emp,
        applicant_income,
        coapplicant_income,
        loan_amount,
        loan_term,
        credit,
        area
    ]], columns=[
        "Gender",
        "Married",
        "Dependents",
        "Education",
        "Self_Employed",
        "ApplicantIncome",
        "CoapplicantIncome",
        "LoanAmount",
        "Loan_Amount_Term",
        "Credit_History",
        "Property_Area"
    ])

    data = scaler.transform(data)

    prediction = model.predict(data)[0]

    if prediction == 1:
        result = "Loan Approved ✅"
    else:
        result = "Loan Rejected ❌"

    return render_template("output.html", prediction=result)


if __name__ == "__main__":
    app.run(debug=True)