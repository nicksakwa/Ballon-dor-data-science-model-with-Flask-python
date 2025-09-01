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
    """
    Find a column name in df using a list of candidate names. Matching is case-insensitive.
    Returns the actual column name from df or None.
    """
    if df is None:
        return None
    # map lowercased column -> actual column
    col_map = {str(c).lower().strip(): c for c in df.columns}
    for cand in candidates:
        key = str(cand).lower().strip()
        if key in col_map:
            return col_map[key]
    # fallback: partial substring match
    for cand in candidates:
        key = str(cand).lower().strip()
        for col in df.columns:
            if key in str(col).lower():
                return col
    return None


def load_and_prepare():
    # Load CSVs
    try:
        winners_df = pd.read_csv(WINNERS_CSV, skipinitialspace=True, encoding='utf-8')
        shortlist_df = pd.read_csv(SHORTLIST_CSV, skipinitialspace=True, encoding='utf-8')
    except Exception as e:
        raise FileNotFoundError(f"Unable to load CSV files from '{UPLOADS}': {e}")

    # Normalize column names
    winners_df.columns = winners_df.columns.astype(str).str.strip()
    shortlist_df.columns = shortlist_df.columns.astype(str).str.strip()

    # Candidate header names (add more if your CSVs use different names)
    avg_candidates = ['Avg. Rating', 'Average Rating', 'Avg Rating', 'Avg_Rating', 'Avg Rating (2024–25)', 'Rating']
    trophies_candidates = [
        'Major Club Trophies', 'Major Trophies (2024–25)', 'Major Trophies',
        'Major Club Trophies (2024–25)', 'Trophies', 'Major trophies'
    ]
    awards_candidates = [
        'Individual Awards', 'Individual Awards (2024–25)', 'Individual awards',
        'Individual_Awards', 'Awards', 'Individual awards (2024–25)'
    ]
    club_candidates = ['Club', 'Team', 'Club (league)', 'Club (team)']
    player_candidates = ['Player', 'Name', 'Full name', 'Full Name', 'Player Name', 'player', 'name']

    # Find columns in each dataframe
    avg_col_short = find_column(shortlist_df, avg_candidates)
    avg_col_win = find_column(winners_df, avg_candidates)

    trophies_col_short = find_column(shortlist_df, trophies_candidates)
    trophies_col_win = find_column(winners_df, trophies_candidates)

    awards_col_short = find_column(shortlist_df, awards_candidates)
    awards_col_win = find_column(winners_df, awards_candidates)

    club_col_short = find_column(shortlist_df, club_candidates)
    club_col_win = find_column(winners_df, club_candidates)

    player_col_short = find_column(shortlist_df, player_candidates)
    player_col_win = find_column(winners_df, player_candidates)

    # Work on a shortlist copy (this is what we will score and display)
    df = shortlist_df.copy()

    # Map player column to 'Player' for template-consistency
    if player_col_short:
        if player_col_short != 'Player':
            df['Player'] = df[player_col_short]
    elif player_col_win:
        # winners has Player but shortlist doesn't — leave Player column NaN (can't map per-row)
        df['Player'] = np.nan
    else:
        df['Player'] = np.nan

    # Avg rating -> numeric (only if present on shortlist)
    if avg_col_short:
        df['Avg_Rating'] = pd.to_numeric(df[avg_col_short], errors='coerce')
        avg_source = 'shortlist'
        avg_col_used = avg_col_short
    else:
        df['Avg_Rating'] = np.nan
        avg_source = 'winners' if avg_col_win else None
        avg_col_used = avg_col_win

    # Trophies / Awards counts (only use shortlist columns when available)
    if trophies_col_short:
        df['Num_Trophies'] = df[trophies_col_short].apply(lambda x: count_items(x, ','))
        trophies_source = 'shortlist'
        trophies_col_used = trophies_col_short
    else:
        df['Num_Trophies'] = 0
        trophies_source = 'winners' if trophies_col_win else None
        trophies_col_used = trophies_col_win

    if awards_col_short:
        df['Num_Awards'] = df[awards_col_short].apply(lambda x: count_items(x, ','))
        awards_source = 'shortlist'
        awards_col_used = awards_col_short
    else:
        df['Num_Awards'] = 0
        awards_source = 'winners' if awards_col_win else None
        awards_col_used = awards_col_win

    # Club column mapping to 'Club' for template consistency
    if club_col_short:
        if club_col_short != 'Club':
            df['Club'] = df[club_col_short]
        club_source = 'shortlist'
        club_col_used = club_col_short
    elif club_col_win:
        df['Club'] = np.nan
        club_source = 'winners'
        club_col_used = club_col_win
    else:
        df['Club'] = np.nan
        club_source = None
        club_col_used = None

    # Try to extract League if club column contains "(League)"
    if club_col_short and club_col_short in df.columns:
        league_match = df[club_col_short].astype(str).str.extract(r'\((.*?)\)')
        df['League'] = league_match.iloc[:, 0].astype(str).str.split(',').str[0].str.strip()
    else:
        df['League'] = np.nan

    detected = {
        'avg': {'used_on': avg_source, 'column': avg_col_used},
        'trophies': {'used_on': trophies_source, 'column': trophies_col_used},
        'awards': {'used_on': awards_source, 'column': awards_col_used},
        'club': {'used_on': club_source, 'column': club_col_used},
        'player': {'shortlist_column': player_col_short, 'winners_column': player_col_win},
        'shortlist_columns': list(shortlist_df.columns),
        'winners_columns': list(winners_df.columns),
    }

    return winners_df, df, detected


def compute_scores(shortlist_df, detected, top5_only=False):
    TROPHY_BONUS = 2.0
    AWARDS_BONUS = 1.5

    df = shortlist_df.copy()

    if top5_only and 'League' in df.columns:
        top_5_leagues = ['Premier League', 'La Liga', 'Serie A', 'Bundesliga', 'Ligue 1']
        df = df[df['League'].isin(top_5_leagues)].copy()

    df['Avg_Rating_filled'] = df['Avg_Rating'].fillna(0)
    df['Final_Score'] = df['Avg_Rating_filled'] + (df['Num_Trophies'] * TROPHY_BONUS) + (df['Num_Awards'] * AWARDS_BONUS)

    # If there are no usable features on the shortlist rows, raise informative error
    usable_mask = df[['Avg_Rating', 'Num_Trophies', 'Num_Awards']].replace(0, np.nan).notna().any(axis=1)
    if usable_mask.sum() == 0:
        raise ValueError(f"No valid features found in shortlist CSV. Detected columns: {detected}")

    df_sorted = df.sort_values(by='Final_Score', ascending=False).reset_index(drop=True)
    return df_sorted


@app.route("/")
def index():
    try:
        winners_df, shortlist_df, detected = load_and_prepare()
        df_pred = compute_scores(shortlist_df, detected, top5_only=False)

        if df_pred.shape[0] == 0:
            return render_template("error.html", error_message="No candidates available after filtering.")

        # Ensure 'Player' and 'Club' exist in df_pred (map common alternatives)
        player_candidates = ['Player', 'Name', 'Full name', 'Full Name', 'Player Name', 'player', 'name']
        player_col = find_column(df_pred, player_candidates)
        if player_col and player_col != 'Player':
            df_pred['Player'] = df_pred[player_col]
        elif 'Player' not in df_pred.columns:
            df_pred['Player'] = np.nan

        club_candidates = ['Club', 'Team', 'Club (league)', 'club', 'team']
        club_col = find_column(df_pred, club_candidates)
        if club_col and club_col != 'Club':
            df_pred['Club'] = df_pred[club_col]
        elif 'Club' not in df_pred.columns:
            df_pred['Club'] = np.nan

        # Columns to display (only include existing ones)
        display_cols = [c for c in ['Player', 'Club', 'Num_Trophies', 'Num_Awards', 'Avg_Rating', 'Final_Score'] if c in df_pred.columns]

        predictions_list = df_pred[display_cols].to_dict('records')
        predicted_winner = df_pred.iloc[0]['Player'] if 'Player' in df_pred.columns and not df_pred['Player'].isna().all() else "No clear winner"

        # Render index.html (adjust template name if your project uses a different template)
        return render_template("index.html", predictions=predictions_list, predicted_winner=predicted_winner, detected=detected)
    except Exception as e:
        tb = traceback.format_exc()
        return render_template("error.html", error_message=f"{e}\n\n{tb}")


if __name__ == "__main__":
    app.run(debug=True)