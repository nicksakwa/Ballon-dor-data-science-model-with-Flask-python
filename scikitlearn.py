from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def prepare_features(winners, shortlist):
    # Convert trophies/awards into counts
    def count_items(s):
        if pd.isna(s) or s=='—':
            return 0
        return len(s.split(','))
    
    for col in ['Major Club Trophies','Individual Awards']:
        winners[col+'_count'] = winners[col].apply(count_items)
        shortlist[col+'_count'] = shortlist[col].apply(count_items)
    
    # Encode leagues based on club
    def assign_league(club):
        club = str(club)
        if any(x in club for x in ['Barcelona','Real Madrid','Atletico']):
            return 'Spain'
        elif any(x in club for x in ['Liverpool','Manchester','Chelsea','Tottenham']):
            return 'England'
        elif any(x in club for x in ['Inter','Juventus','AC Milan','Napoli']):
            return 'Italy'
        elif any(x in club for x in ['Bayern','Dortmund']):
            return 'Germany'
        elif any(x in club for x in ['PSG','Marseille','Monaco']):
            return 'France'
        else:
            return 'Other'
    
    for df in [winners, shortlist]:
        df['League'] = df['Club'].apply(assign_league)
    
    # One-hot encode leagues
    encoder = OneHotEncoder()
    league_encoded_train = encoder.fit_transform(winners[['League']]).toarray()
    league_encoded_test = encoder.transform(shortlist[['League']]).toarray()
    
    X_train = pd.concat([
        winners[['Major Club Trophies_count','Individual Awards_count']],
        pd.DataFrame(league_encoded_train, columns=encoder.get_feature_names_out())
    ], axis=1)
    
    # All winners=1
    y_train = [1]*len(winners)
    
    # Use shortlist as negative examples (label=0)
    X_train_extra = pd.concat([
        shortlist[['Major Club Trophies_count','Individual Awards_count']],
        pd.DataFrame(league_encoded_test, columns=encoder.get_feature_names_out())
    ], axis=1)
    X_train = pd.concat([X_train, X_train_extra], ignore_index=True)
    y_train = y_train + [0]*len(shortlist)
    
    # Prepare test features
    X_test = pd.concat([
        shortlist[['Major Club Trophies_count','Individual Awards_count']],
        pd.DataFrame(league_encoded_test, columns=encoder.get_feature_names_out())
    ], axis=1)
    
    return X_train, y_train, X_test

@app.route('/', methods=['GET','POST'])
def index():
    if request.method == 'POST':
        winners_file = request.files['winners']
        shortlist_file = request.files['shortlist']
        
        winners_path = os.path.join(app.config['UPLOAD_FOLDER'], winners_file.filename)
        shortlist_path = os.path.join(app.config['UPLOAD_FOLDER'], shortlist_file.filename)
        
        winners_file.save(winners_path)
        shortlist_file.save(shortlist_path)
        
        # Load CSVs
        winners = pd.read_csv(winners_path)
        shortlist = pd.read_csv(shortlist_path)
        
        # Prepare features
        X_train, y_train, X_test = prepare_features(winners, shortlist)
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Predict probabilities
        shortlist['Ballon_dOr_Prob'] = model.predict_proba(X_test)[:,1]
        shortlist_sorted = shortlist.sort_values('Ballon_dOr_Prob', ascending=False)
        
        # Return top 10 predictions
        top10 = shortlist_sorted[['Player','Club','Ballon_dOr_Prob']].head(10).to_dict(orient='records')
        
        return render_template('index.html', top10=top10)
    
    return render_template('index.html', top10=None)

if __name__ == '__main__':
    app.run(debug=True)
