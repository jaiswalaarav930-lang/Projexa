from flask import Flask, render_template, request
import pandas as pd
import joblib

# -------------------- APP SETUP --------------------
app = Flask(__name__)

# -------------------- LOAD DATA & MODEL --------------------
# Dataset
df = pd.read_csv("../dataset/student_study_data.csv")

# Trained ML model
model = joblib.load("../model/logistic_model.pkl")

# -------------------- RECOMMENDATION LOGIC --------------------
def generate_recommendation(math, physics, chemistry, attendance, study_hours):
    recommendations = []
    weak_subjects = []

    # Identify weak subjects
    if math < 50:
        weak_subjects.append("Math")
    if physics < 50:
        weak_subjects.append("Physics")
    if chemistry < 50:
        weak_subjects.append("Chemistry")

    # Attendance advice
    if attendance < 75:
        recommendations.append("Improve attendance to at least 75%")

    # Study hours advice
    if study_hours < 2:
        recommendations.append("Increase daily study hours")

    # Subject-based advice
    if weak_subjects:
        recommendations.append(
            "Focus more on weak subjects: " + ", ".join(weak_subjects)
        )

    # Default positive advice
    if not recommendations:
        recommendations.append("Maintain current study strategy and consistency")

    # Performance description
    if len(weak_subjects) == 0:
        performance = "Good academic standing"
    elif len(weak_subjects) == 1:
        performance = "Average performance, improvement needed in one subject"
    else:
        performance = "Low academic performance, multiple areas need improvement"

    return performance, weak_subjects, recommendations

# -------------------- ROUTE --------------------
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None

    if request.method == "POST":
        try:
            roll_no = int(request.form["roll_no"])

            # Fetch student data
            student = df[df["Student_ID"] == roll_no]

            if student.empty:
                error = "Roll number not found in records"
            else:
                student = student.iloc[0]

                # Prepare features for ML model
                features = [[
                    student["Math_Marks"],
                    student["Physics_Marks"],
                    student["Chemistry_Marks"],
                    student["Attendance_Percentage"],
                    student["Daily_Study_Hours"]
                ]]

                # ML Prediction
                prediction = model.predict(features)[0]

                if prediction == 1:
                    prediction_text = "Low Risk (Stable Performance)"
                else:
                    prediction_text = "High Risk (Needs Improvement)"

                # Generate recommendations
                performance, weak_subjects, recommendations = generate_recommendation(
                    student["Math_Marks"],
                    student["Physics_Marks"],
                    student["Chemistry_Marks"],
                    student["Attendance_Percentage"],
                    student["Daily_Study_Hours"]
                )

                # Final result dictionary (sent to HTML)
                result = {
                    "roll_no": roll_no,
                    "prediction": prediction_text,
                    "performance": performance,
                    "weak_subjects": weak_subjects,
                    "recommendations": recommendations,
                    "math": student["Math_Marks"],
                    "physics": student["Physics_Marks"],
                    "chemistry": student["Chemistry_Marks"],
                    "attendance": student["Attendance_Percentage"],
                    "study_hours": student["Daily_Study_Hours"]
                }

        except ValueError:
            error = "Invalid roll number input"

    return render_template("index.html", result=result, error=error)

# -------------------- RUN APP --------------------
if __name__ == "__main__":
    app.run(debug=True)
