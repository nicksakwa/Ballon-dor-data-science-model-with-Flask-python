from flask import Flask, render_template
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# Load CSVs directly (make sure these files are in the same folder as app.py)
winners_df = pd.read_csv("ballon_dor_winners_2015-2024.csv")
shortlist_df = pd.read_csv("ballon_dor_2025_shortlist.csv")

# Normalize column names to avoid KeyError issues
winners_df.columns = winners_df.columns.str.strip()
shortlist_df.columns = shortlist_df.columns.str.strip()

# Define features we want to use (only those present in CSV)
possible_features = ["Major Club Trophies", "International Trophies", 
                     "Individual Awards", "Average Rating"]

features = [col for col in possible_features if col in winners_df.columns and col in shortlist_df.columns]

# Train the model
def train_model():
    if not features:
        raise ValueError("No valid features found in CSVs")

    X = winners_df[features]
    y = winners_df["Player"]

    # Encode target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y_encoded)

    return model, le

# Prediction function
def predict_ballon_dor():
    model, le = train_model()
    X_shortlist = shortlist_df[features]

    preds = model.predict(X_shortlist)
    shortlist_df["Prediction"] = le.inverse_transform(preds)

    # Get the most frequent predicted player
    winner = shortlist_df["Prediction"].mode()[0]
    return winner, shortlist_df


@app.route("/")
def index():
    try:
        prediction, shortlist = predict_ballon_dor()
        shortlist_html = shortlist.to_html(classes="table table-striped", index=False)
        return render_template("app.html", prediction=prediction, shortlist_table=shortlist_html)
    except Exception as e:
        return f"<h2>Error:</h2><pre>{e}</pre>"

if __name__ == "__main__":
    app.run(debug=True)
