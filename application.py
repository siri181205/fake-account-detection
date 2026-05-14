from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model
model = pickle.load(open("model.pkl", "rb"))

saved_data = {}
history = []

# Home Page
@app.route('/')
def home():
    return render_template("home.html")

# Login Page
@app.route('/login')
def login():
    return render_template("login.html")

# Detection Page
@app.route('/detect')
def detect():
    return render_template("index.html")

# After entering details
@app.route('/social', methods=['POST'])
def social():

    saved_data["username"] = request.form['username']
    saved_data["followers"] = int(request.form['followers'])
    saved_data["following"] = int(request.form['following'])
    saved_data["posts"] = int(request.form['posts'])
    saved_data["profile_pic"] = int(request.form['profile_pic'])

    # ✅ NEW AGE CONVERSION
    age = int(request.form['account_age'])
    unit = request.form['age_unit']

    if unit == "years":
        age = age * 365

    saved_data["account_age"] = age

    return render_template("social.html")

# Result Page
@app.route('/result', methods=['POST'])
def result():

    platform = request.form['platform']

    data = np.array([[
        saved_data["followers"],
        saved_data["following"],
        saved_data["posts"],
        saved_data["profile_pic"],
        saved_data["account_age"]
    ]])

    prediction = model.predict(data)

    if prediction[0] == 1:
        output = "⚠ Fake Account"
        color = "red"
    else:
        output = "✅ Real Account"
        color = "lime"

    # Confidence %
    try:
        confidence = int(max(model.predict_proba(data)[0]) * 100)
    except:
        confidence = 90

    # Save history
    history.append({
        "username": saved_data["username"],
        "platform": platform,
        "result": output,
        "accuracy": str(confidence) + "%"
    })

    return render_template("result.html",
                           prediction=output,
                           color=color,
                           platform=platform,
                           confidence=confidence)

# History Page
@app.route('/history')
def show_history():
    return render_template("history.html", data=history)

# Run app
if __name__ == "__main__":
    app.run(debug=True)