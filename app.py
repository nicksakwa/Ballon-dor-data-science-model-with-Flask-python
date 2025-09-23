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
    parts = [p.strip() for p in s.replace(';', ',').split(',') if p.strip()]
    return len(parts)

def find_column(df, candidates):
    if df is None:
        return None
    col_map = {str(c).lower().strip(): c for c in df.columns}
    for cand in candidates:
        key = str(cand).lower().strip()
        if key in col_map:
            return col_map[key]
    for cand in candidates:
        key = str(cand).lower().strip()
        for col in df.columns:
            if key in str(col).lower():
                return col
    return None

def load_and_prepare():
    winners_df = pd.read_csv(WINNERS_CSV, encoding="utf-8")
    shortlist_df = pd.read_csv(SHORTLIST_CSV, encoding="utf-8")
    winners_df.columns = [c.strip() for c in winners_df.columns]
    shortlist_df.columns = [c.strip() for c in shortlist_df.columns]
    avg_candidates = ['Avg. Rating', 'Average Rating', 'Rating']
    trophies_candidates = ['Major Club Trophies', 'Trophies', 'Club Honors']
    awards_candidates = ['Individual Awards', 'Awards']
    club_candidates = ['Club', 'Team', 'Club (league)']
    player_candidates = ['Player', 'Name', 'Full Name']
    detected = {
        'avg_col': find_column(shortlist_df, avg_candidates),
        'trophies_col': find_column(shortlist_df, trophies_candidates),
        'awards_col': find_column(shortlist_df, awards_candidates),
        'club_col': find_column(shortlist_df, club_candidates),
        'player_col': find_column(shortlist_df, player_candidates)
    }
    df = shortlist_df.copy()
    df['Player'] = df[detected['player_col']] if detected['player_col'] else df.iloc[:, 0]
    df['Avg_Rating'] = df[detected['avg_col']] if detected['avg_col'] else 0.0
    df['Num_Trophies'] = df[detected['trophies_col']].apply(count_items) if detected['trophies_col'] else 0
    df['Num_Awards'] = df[detected['awards_col']].apply(count_items) if detected['awards_col'] else 0
    if detected['club_col']:
        df['Club'] = df[detected['club_col']]
        df['League'] = df['Club'].apply(
            lambda x: str(x).split('(')[-1].replace(')', '').strip() if '(' in str(x) else ''
        )
    else:
        df['Club'] = ''
        df['League'] = ''
    return winners_df, df, detected

def compute_scores(shortlist_df, detected, top5_only=False):
    TROPHY_BONUS = 2.0
    AWARDS_BONUS = 1.5
    df = shortlist_df.copy()
    if top5_only:
        top5 = ["Premier League", "La Liga", "Serie A", "Bundesliga", "Ligue 1"]
        df = df[df['League'].isin(top5)]
    df['Final_Score'] = (
        df['Avg_Rating'].astype(float) +
        df['Num_Trophies'] * TROPHY_BONUS +
        df['Num_Awards'] * AWARDS_BONUS
    )
    df = df.sort_values('Final_Score', ascending=False).reset_index(drop=True)
    return df

@app.route("/")
def index():
    try:
        winners_df, shortlist_df, detected = load_and_prepare()
        df_pred = compute_scores(shortlist_df, detected, top5_only=False)
        if 'Player' not in df_pred.columns:
            df_pred['Player'] = df_pred.index.astype(str)
        if 'Club' not in df_pred.columns:
            df_pred['Club'] = ''
        df_pred['Avg_Rating'] = df_pred['Avg_Rating'].apply(
            lambda x: f"{float(x):.2f}" if pd.notnull(x) else ""
        )
        df_pred['Final_Score'] = df_pred['Final_Score'].apply(
            lambda x: f"{float(x):.2f}" if pd.notnull(x) else ""
        )
        predicted_winner = df_pred.iloc[0].to_dict() if not df_pred.empty else None
        predictions_list = df_pred.to_dict(orient="records")
        return render_template(
            "index.html",
            predictions=predictions_list,
            predicted_winner=predicted_winner,
            detected=detected
        )
    except Exception as e:
        return render_template("error.html", error=str(e), traceback=traceback.format_exc())

if __name__ == "__main__":
    app.run(debug=True)
