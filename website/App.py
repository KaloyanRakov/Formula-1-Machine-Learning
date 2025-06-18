import pandas as pd
import joblib
from flask import Flask, render_template, request
import numpy as np


app = Flask(__name__)

model = joblib.load('model/Individual_Predictions.pkl')
final_team_rankings_df = pd.read_csv("data/final_team_rankings.csv")
final_individual_rankings_df = pd.read_csv("data/final_individual_rankings.csv")


points_dict = {
    1: 25, 2: 18, 3: 15, 4: 12, 5: 10,
    6: 8, 7: 6, 8: 4, 9: 2, 10: 1
}

def get_points(pos):
    return points_dict.get(pos, 0)


team_logos = {
    "Red Bull Racing": "img/RedBullRacingLogo.png",
    "Mercedes": "img/MercedesLogo.png",
    "McLaren": "img/McLarenLogo.png",
}

driver_photos = {
    "Oscar Piastri": "img/oscar_piastri_headshot.png",
    "Lando Norris": "img/lando_norris_headshot.png",
    "Max Verstappen": "img/max_verstappen_headshot.png",
    "George Russell": "img/george_russell_headshot.png",
    "Andrea Kimi Antonelli": "img/Kimi_Antonelli.png",
    "Lewis Hamilton": "img/lewham01.png"

}

location_flags = {
    "Australia": "img/Flag_of_Australia.png",
    "China": "img/Flag_of_China.png",
    "Japan": "img/Japan_flag.png",
    "Bahrain": "img/Flag_of_Bahrain.png",
    "Saudi Arabia": "img/SaudiArabia_flag.png",
    "Miami": "img/USA_flag.png",
}

@app.route('/')
def home():
    top_3 = final_team_rankings_df.head(3)
    top_3["Logo"] = top_3["Team"].map(team_logos)
    locations = final_individual_rankings_df['Location'].unique()
    custom_top_3 = pd.concat([
        top_3.iloc[[1]],
        top_3.iloc[[0]],
        top_3.iloc[[2]]
    ])

    team_table = final_team_rankings_df.to_html(classes='data-table', index=False)
    return render_template('index.html', team_table=team_table, top_3_teams=custom_top_3, locations=locations, location_flags=location_flags)

@app.route('/race/<location_name>')
def race_details(location_name):
    race_data = final_individual_rankings_df[final_individual_rankings_df['Location'] == location_name]
    race_table_html = race_data.drop(columns=['Location']).to_html(classes='data-table', index=False)
    top_3_drivers = race_data.sort_values(by='Predicted').head(3).copy()
    custom_top_3_drivers = pd.concat([
        top_3_drivers.iloc[[1]],
        top_3_drivers.iloc[[0]],
        top_3_drivers.iloc[[2]]
    ])
    custom_top_3_drivers["Image"] = custom_top_3_drivers["Driver"].map(driver_photos)

    return render_template('race_details.html', location_name=location_name, race_table=race_table_html, top_3_drivers=top_3_drivers, custom_top_3_drivers=custom_top_3_drivers)


if __name__ == '__main__':
    app.run(debug=True)
