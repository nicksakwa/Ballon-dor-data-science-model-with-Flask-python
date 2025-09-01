from flask import Flask, render_template
import pandas as pd
import numpy as np
import traceback

# Your provided function for predicting the Ballon d'Or winner
def predict_ballon_dor(winners_csv_path, shortlist_csv_path):
    """
    Predicts the Ballon d'Or winner based on a scoring model with data validation.
    
    Args:
        winners_csv_path (str): The file path to the CSV with past winners' data.
        shortlist_csv_path (str): The file path to the CSV with the 2025 shortlist data.

    Returns:
        pd.DataFrame: A DataFrame with the sorted predictions, including scores, or None on error.
    """
    try:
        # Load the data
        df_winners = pd.read_csv(winners_csv_path)
        df_shortlist = pd.read_csv(shortlist_csv_path)
    except FileNotFoundError as e:
        # This error is handled by the Flask route, which will render an error page
        raise FileNotFoundError(f"The required CSV file was not found: {e.filename}")
    except Exception as e:
        # Provide a detailed traceback for other loading errors
        return None, f"An error occurred while loading CSV files: {e}\n{traceback.format_exc()}"

    try:
        # Drop the "Not awarded" row from the winners' data
        df_winners = df_winners[df_winners['Year'] != 2020].reset_index(drop=True)

        # Function to count items in a string, handling missing values
        def count_items(text, delimiter):
            if pd.isna(text) or text == '—':
                return 0
            return len(str(text).split(delimiter))

        # Feature engineering for the winners' data
        df_winners['Num_Trophies'] = df_winners['Major Club Trophies'].apply(lambda x: count_items(x, ', '))
        df_winners['Num_Awards'] = df_winners['Individual Awards'].apply(lambda x: count_items(x, ', '))

        # Data Validation: Ensure 'Avg. Rating' is a numeric type
        df_shortlist['Avg. Rating'] = pd.to_numeric(df_shortlist['Avg. Rating'], errors='coerce')
        df_shortlist.dropna(subset=['Avg. Rating'], inplace=True)

        # Clean and filter the shortlist data
        league_match = df_shortlist['Club'].astype(str).str.extract(r'\((.*?)\)').iloc[:, 0]
        df_shortlist['League'] = league_match.str.split(',').str[0].str.strip()
        top_5_leagues = ['Premier League', 'La Liga', 'Serie A', 'Bundesliga', 'Ligue 1']
        df_shortlist_filtered = df_shortlist[df_shortlist['League'].isin(top_5_leagues)].copy()

        # Feature engineering for the filtered shortlist
        df_shortlist_filtered.loc[:, 'Num_Trophies'] = df_shortlist_filtered['Major Trophies (2024–25)'].apply(lambda x: count_items(x, ', '))
        df_shortlist_filtered.loc[:, 'Num_Awards'] = df_shortlist_filtered['Individual Awards (2024–25)'].apply(lambda x: count_items(x, '; '))

        # Define the scoring weights
        TROPHY_BONUS = 2.0
        AWARDS_BONUS = 1.5

        # Calculate the final score for each player
        df_shortlist_filtered.loc[:, 'Final_Score'] = (
            df_shortlist_filtered['Avg. Rating'] +
            (df_shortlist_filtered['Num_Trophies'] * TROPHY_BONUS) +
            (df_shortlist_filtered['Num_Awards'] * AWARDS_BONUS)
        )

        # Sort the players by their final score in descending order
        df_prediction = df_shortlist_filtered.sort_values(by='Final_Score', ascending=False).reset_index(drop=True)

        return df_prediction, None
    except Exception as e:
        # Catch any other unexpected errors and provide a detailed traceback
        return None, f"An unexpected error occurred during data processing: {e}\n{traceback.format_exc()}"

# Initialize the Flask application
app = Flask(__name__)

# Define the main route with error handling
@app.route('/')
def index():
    try:
        # Call your data science model to get the predictions
        predictions_df, error_message = predict_ballon_dor('ballon_dor_winners_2015_2024.csv', 'ballon_dor_2025_shortlist.csv')

        if error_message:
            return render_template('error.html', error_message=error_message)

        # Convert the DataFrame to a list of dictionaries for the template
        predictions_list = predictions_df[['Player', 'Club', 'Num_Trophies', 'Num_Awards', 'Avg. Rating', 'Final_Score']].to_dict('records')

        # Get the predicted winner
        predicted_winner = predictions_df.iloc[0]['Player']

        # Render the HTML template, passing the data to it
        return render_template('index.html', predictions=predictions_list, predicted_winner=predicted_winner)

    except FileNotFoundError as e:
        # Handle the error by rendering a custom error page
        return render_template('error.html', error_message=f"A CSV file was not found: {e}"), 404
    except Exception as e:
        # Catch any other unexpected errors and show a generic error page with a traceback
        error_info = f"An unexpected error occurred: {e}\n{traceback.format_exc()}"
        return render_template('error.html', error_message=error_info), 500

if __name__ == '__main__':
    # Run the Flask application in debug mode to see tracebacks
    app.run(debug=True)
