from flask import Flask, render_template
import pandas as pd
import numpy as np
import traceback
import os

app = Flask(__name__)

UPLOADS = os.path.join(os.path.dirname(__file__), "uploads")
WINNERS_CSV = os.path.join(UPLOADS, "ballon_dor_winners_2015_2024.csv")
SHORTLIST_CSV = os.path.join(UPLOADS, "ballon_dor_2025_shortlist.csv")

def count_items(text, delimiter=','):
    if pd.isna(text):
        return 0
    s = str(text).strip()
    if s in ['', '—', '-', 'None', 'nan']:
        return 0
    # some award lists use ';' or ',' -- treat both as separators
    parts = [p.strip() for p in s.replace(';', ',').split(',') if p.strip()]
    return len(parts)

def find_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def load_and_prepare():
    # Load CSVs
    try:
        winners_df = pd.read_csv(WINNERS_CSV)
        shortlist_df = pd.read_csv(SHORTLIST_CSV)
    except Exception as e:
        raise FileNotFoundError(f"Unable to load CSV files from '{UPLOADS}': {e}")

    # Normalize column names
    winners_df.columns = winners_df.columns.astype(str).str.strip()
    shortlist_df.columns = shortlist_df.columns.astype(str).str.strip()

    # Candidate names
    avg_candidates = ['Avg. Rating', 'Average Rating', 'Avg Rating', 'Avg_Rating', 'Avg Rating (2024–25)']
    trophies_candidates = [
        'Major Club Trophies', 'Major Trophies (2024–25)', 'Major Trophies',
        'Major Club Trophies (2024–25)', 'Trophies'
    ]
    awards_candidates = [
        'Individual Awards', 'Individual Awards (2024–25)', 'Individual awards',
        'Individual_Awards', 'Awards'
    ]
    club_candidates = ['Club', 'Team', 'Club (league)']

    avg_col = find_column(winners_df, avg_candidates) or find_column(shortlist_df, avg_candidates)
    trophies_col = find_column(winners_df, trophies_candidates) or find_column(shortlist_df, trophies_candidates)
    awards_col = find_column(winners_df, awards_candidates) or find_column(shortlist_df, awards_candidates)
    club_col = find_column(shortlist_df, club_candidates) or find_column(winners_df, club_candidates)

    # Feature creation on shortlist (this is what we predict/rank)
    df = shortlist_df.copy()

    # Avg rating -> numeric
    if avg_col:
        df['Avg_Rating'] = pd.to_numeric(df[avg_col], errors='coerce')
    else:
        df['Avg_Rating'] = np.nan

    # Trophies / Awards counts
    if trophies_col:
        df['Num_Trophies'] = df[trophies_col].apply(lambda x: count_items(x, ','))
    else:
        df['Num_Trophies'] = 0

    if awards_col:
        df['Num_Awards'] = df[awards_col].apply(lambda x: count_items(x, ','))
    else:
        df['Num_Awards'] = 0

    # Attempt to extract league from Club column if present (used for optional filtering)
    if club_col and club_col in df.columns:
        league_match = df[club_col].astype(str).str.extract(r'\((.*?)\)')
        df['League'] = league_match.iloc[:, 0].astype(str).str.split(',').str[0].str.strip()
    else:
        df['League'] = np.nan

    detected = {
        'avg_col': avg_col,
        'trophies_col': trophies_col,
        'awards_col': awards_col,
        'club_col': club_col,
    }

    return winners_df, df, detected

def compute_scores(shortlist_df, detected, top5_only=False):
    # Define weights
    TROPHY_BONUS = 2.0
    AWARDS_BONUS = 1.5

    df = shortlist_df.copy()

    # Optional filter to top 5 leagues
    if top5_only and 'League' in df.columns:
        top_5_leagues = ['Premier League', 'La Liga', 'Serie A', 'Bundesliga', 'Ligue 1']
        df = df[df['League'].isin(top_5_leagues)].copy()

    # Coerce Avg_Rating to numeric; replace NaN with 0 for scoring (but keep original NaN for info)
    df['Avg_Rating_filled'] = df['Avg_Rating'].fillna(0)

    # Final score
    df['Final_Score'] = df['Avg_Rating_filled'] + (df['Num_Trophies'] * TROPHY_BONUS) + (df['Num_Awards'] * AWARDS_BONUS)

    # If all features are zero/NaN, raise informative error
    if df[['Avg_Rating', 'Num_Trophies', 'Num_Awards']].replace(0, np.nan).dropna(how='all').shape[0] == 0:
        raise ValueError(f"No valid features found in CSVs — detected columns: {detected}")

    df_sorted = df.sort_values(by='Final_Score', ascending=False).reset_index(drop=True)
    return df_sorted

@app.route("/")
def index():
    try:
        winners_df, shortlist_df, detected = load_and_prepare()
        df_pred = compute_scores(shortlist_df, detected, top5_only=False)

        if df_pred.shape[0] == 0:
            return render_template("error.html", error_message="No candidates available after filtering.")

        # Ensure required display columns exist
        display_cols = []
        for c in ['Player', 'Club']:
            if c in df_pred.columns:
                display_cols.append(c)
        # Add our computed columns
        display_cols += ['Num_Trophies', 'Num_Awards', 'Avg_Rating', 'Final_Score']
        # Keep only cols that exist
        display_cols = [c for c in display_cols if c in df_pred.columns]

        predictions_list = df_pred[display_cols].to_dict('records')
        predicted_winner = df_pred.iloc[0]['Player'] if 'Player' in df_pred.columns and not df_pred['Player'].isna().all() else "No clear winner"

        return render_template("app.html", predictions=predictions_list, predicted_winner=predicted_winner, detected=detected)
    except Exception as e:
        tb = traceback.format_exc()
        # Return an informative error page (templates/error.html should display error_message)
        return render_template("error.html", error_message=f"{e}\n\n{tb}")

if __name__ == "__main__":
    app.run(debug=True)