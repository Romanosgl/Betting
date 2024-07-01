import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import json
import os

config_path = 'config.json'

# Check if the file exists and is not empty
if not os.path.exists(config_path):
    raise FileNotFoundError(f"{config_path} does not exist")
if os.path.getsize(config_path) == 0:
    raise ValueError(f"{config_path} is empty")

# Load and parse the JSON file
try:
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
except json.JSONDecodeError as e:
    raise ValueError(f"Error parsing JSON: {e}")

project_dir = config.get('project_dir')
file_name = config.get('file_name')
if project_dir is None:
    raise ValueError("Project directory not set in config file")
if file_name is None:
    raise ValueError("File name not set in config file")

# Correctly join the directory and file name
file_path = os.path.join(project_dir, file_name)
if not os.path.exists(file_path):
    raise FileNotFoundError(f"The file {file_path} does not exist")

df = pd.read_excel(file_path)

#print(df.shape)
#print(df.dtypes)
#print(df.describe)
#print(df.columns)
#print(df.isnull().sum())
df.drop_duplicates(inplace=True)
#print(df.head())


data = {
    'DATE': [45462, 45462, 45463, 45463, 45464,45465,45465],
    'HOME TEAM':['GERMANY', 'SCOTLAND', 'SPAIN', 'DENMARK', 'NETHERLANDS','TURKEY', 'BELGIUM'],
    'AWAY TEAM': ['HUNGARY', 'SWITZERLAND', 'ITALY', 'ENGLAND', 'FRANCE','PORTUGAL', 'ROMANIA'],
    'HOME TEAM WINRATE': [1, 0, 1, 0, 1, 1, 0],
    'AWAY TEAM WINRATE': [0, 1, 0, 1, 0, 1, 1],
    'HOME TEAM FORM':["W","D","W","D","W","W","L"],
    'AWAY TEAM FORM' :["D","W","W","W","W","W","W"],                 
    'HEAD-TO-HEAD FORM': ["WLDDWW","DW","WWWD","LWD","LLW","LLLL","LW"],               
    'HOME GOAL DIFFERENCE': [4, -4, 3, 0, 1, 2, -1],             
    'AWAY GOAL DIFFERENCE': [-2, 2, 1, 1, 1, 1, 3],            
    'HOME TEAM FORM IN HOME MATCHES':  ["WWDW","DLD","WWW","DWW","WL","WW","LWD"],  
    'AWAY TEAM FORM IN AWAY MATCHES':  ["LWDDW","WWDLD","WDL","WD","DW","WW","WD"],  
    'HOME TEAM POINTS': [3, 0, 3, 1, 3, 3, 0],           
    'AWAY TEAM POINTS': [0, 3, 3, 3, 3, 6, 3],
    'HOME TEAM ODDS':[1.25, 4.60, 2.25, 5.50, 3.10, 5.75, 1.36],
    'DRAW ODDS': [6.25, 3.40, 3.10, 3.50, 3.10, 3.90, 5.00],
    'AWAY TEAM ODDS': [11.00, 1.85, 3.50, 1.70, 2.45, 1.62, 8.50],
    'RESULT': ['2-0', '1-1', '1-0', '1-1', '0-0', "0-3", "2-0"],
    'WINNER': ['GERMANY', 'DRAW', 'SPAIN', 'DRAW', 'DRAW', "PORTUGAL", "BELGIUM"]
}
#df = pd.DataFrame(data)

#pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)
#print(df)
def calculate_win_rate(form):
    wins = form.count('W')
    losses = form.count('L')
    total_matches = len(form)
    if total_matches == 0:
        return 0
    else:
        win_rate = (wins / total_matches) * 100
        return win_rate
    
df['HOME TEAM FORM'] = df['HOME TEAM FORM'].fillna('').astype(str)
df['AWAY TEAM FORM'] = df['AWAY TEAM FORM'].fillna('').astype(str)
df['HEAD-TO-HEAD FORM'] = df['HEAD-TO-HEAD FORM'].fillna('').astype(str)

home_head_to_head = []
away_head_to_head = []

for match_form in df['HEAD-TO-HEAD FORM']:
    home_form = {'W': 0, 'D': 0, 'L': 0}
    away_form = {'W': 0, 'D': 0, 'L': 0}
    
    # Count wins, draws, and losses for each team in head-to-head form
    for result in match_form:
        if result == 'W':
            home_form['W'] += 1
            away_form['L'] += 1
        elif result == 'D':
            home_form['D'] += 1
            away_form['D'] += 1
        elif result == 'L':
            home_form['L'] += 1
            away_form['W'] += 1
    
    # Append counts to respective lists
    home_head_to_head.append([home_form['W'], home_form['D'], home_form['L']])
    away_head_to_head.append([away_form['W'], away_form['D'], away_form['L']])

df['HOME HEAD-TO-HEAD FORM'] = home_head_to_head
df['AWAY HEAD-TO-HEAD FORM'] = away_head_to_head

df['HOME TEAM FORM WINRATE'] = df['HOME TEAM FORM'].apply(lambda form: calculate_win_rate(form))
df['AWAY TEAM FORM WINRATE'] = df['AWAY TEAM FORM'].apply(lambda form: calculate_win_rate(form))

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
print(df)

df[['HOME SCORE', 'AWAY SCORE']] = df['RESULT'].str.split('-', expand=True).astype(int)

def predict_winner(home_odds, draw_odds, away_odds, home_team_form_winrate, away_team_form_winrate,home_goal_difference,away_goal_difference):
    if home_odds <= draw_odds and home_odds <= away_odds and home_team_form_winrate > away_team_form_winrate:
        return 'HOME'
    elif draw_odds <= home_odds and draw_odds <= away_odds:
        return 'DRAW'
    elif away_odds < home_odds and away_odds < draw_odds and away_team_form_winrate > home_team_form_winrate:
        return 'AWAY'
    else:
        return 'DRAW' #if uncertain assume draw


df['PREDICTED WINNER'] = df.apply(lambda row: predict_winner(row['HOME TEAM ODDS'], row['DRAW ODDS'], row['AWAY TEAM ODDS'],
                                                             row['HOME TEAM FORM WINRATE'], row['AWAY TEAM FORM WINRATE'],row['HOME GOAL DIFFERENCE'],
                                                             row['AWAY GOAL DIFFERENCE']), axis=1)

df[['HOME SCORE', 'AWAY SCORE']] = df['RESULT'].str.split('-', expand=True).astype(int)

df['PREDICTED WINNER'] = df.apply(
    lambda row: predict_winner(
        row['HOME TEAM ODDS'], 
        row['DRAW ODDS'], 
        row['AWAY TEAM ODDS'],
        row['HOME TEAM FORM WINRATE'], 
        row['AWAY TEAM FORM WINRATE'],
        row['HOME GOAL DIFFERENCE'],
        row['AWAY GOAL DIFFERENCE']
    ), axis=1
)

def compare_winners(df):
    prediction = df['PREDICTED WINNER']
    actual_winners = df['WINNER'].tolist()
    
    results = [prediction[i] == actual_winners[i] for i in range(len(prediction))]
    
    for result in results:
        print(result)

def main():  
    compare_winners(df)
        
    df['CORRECT PREDICTIONS'] = df['WINNER'] == df['PREDICTED WINNER']
    total_predictions = df.shape[0]
    accuracy = df['CORRECT PREDICTIONS'].mean() * 100

    print(f"Accuracy of predictions: {accuracy:.2f}%")

    #plt.figure(figsize=(10, 6))
    #sns.countplot(data=df, x="CORRECT PREDICTIONS", hue="CORRECT PREDICTIONS", palette='coolwarm')
    #plt.legend([], [], frameon=False)
    #plt.title('Prediction Accuracy')
    #plt.xlabel('Correct Prediction')
    #plt.ylabel('Count')
    #plt.show()

if __name__ == "__main__":
    main()

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, StratifiedKFold


categorical_cols = ['HOME TEAM', 'AWAY TEAM']
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

# Fit and transform categorical columns
encoded_features = encoder.fit_transform(df[categorical_cols])
encoded_feature_names = encoder.get_feature_names_out(categorical_cols)

#Debugging
print(f"Shape of encoded features: {encoded_features.shape}")
print(f"Encoded feature names: {encoded_feature_names}")

encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names)

# Concatenate encoded features with original DataFrame
df_encoded = pd.concat([df, encoded_df], axis=1)

df_encoded[['HOME SCORE', 'AWAY SCORE']] = df_encoded['RESULT'].str.split('-', expand=True).astype(int)
df_encoded['DEBUG_WINNER'] = df_encoded.apply(
    lambda row: 'HOME' if row['WINNER'] == row['HOME TEAM'] else ('AWAY' if row['WINNER'] == row['AWAY TEAM'] else 'DRAW'), 
    axis=1
)
print(df_encoded[['WINNER', 'HOME TEAM', 'AWAY TEAM', 'DEBUG_WINNER']])

encoder = LabelEncoder()

# Encode categorical columns
df['HOME TEAM'] = encoder.fit_transform(df['HOME TEAM'])
df['AWAY TEAM'] = encoder.fit_transform(df['AWAY TEAM'])

df_encoded['HOME HEAD-TO-HEAD WINS'] = [x[0] for x in home_head_to_head]
df_encoded['HOME HEAD-TO-HEAD DRAWS'] = [x[1] for x in home_head_to_head]
df_encoded['HOME HEAD-TO-HEAD LOSSES'] = [x[2] for x in home_head_to_head]
df_encoded['AWAY HEAD-TO-HEAD WINS'] = [x[0] for x in away_head_to_head]
df_encoded['AWAY HEAD-TO-HEAD DRAWS'] = [x[1] for x in away_head_to_head]
df_encoded['AWAY HEAD-TO-HEAD LOSSES'] = [x[2] for x in away_head_to_head]

df_encoded['HOME TEAM FORM'] = encoder.fit_transform(df['HOME TEAM FORM'])
df_encoded['AWAY TEAM FORM'] = encoder.fit_transform(df['AWAY TEAM FORM'])


features = ['HOME TEAM ODDS', 'DRAW ODDS', 'AWAY TEAM ODDS',
            'HOME HEAD-TO-HEAD WINS', 'HOME HEAD-TO-HEAD DRAWS', 'HOME HEAD-TO-HEAD LOSSES',
            'AWAY HEAD-TO-HEAD WINS', 'AWAY HEAD-TO-HEAD DRAWS', 'AWAY HEAD-TO-HEAD LOSSES',
            'HOME GOAL DIFFERENCE','AWAY GOAL DIFFERENCE','HOME TEAM FORM','AWAY TEAM FORM'] + list(encoded_feature_names)
target = 'WINNER'

X = df_encoded[features]
y = df_encoded[target]

class_weights = {
    'HOME': 1.0,
    'AWAY': 1.0,
    'DRAW': 0.0  # Adjust weight for 'DRAW' class to penalize less
}

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42, class_weight=class_weights)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
accuracy_percentage = accuracy * 100
print(f"Accuracy of RandomForestClassifier: {accuracy_percentage:.2f}%")
print(predictions)
print("Distribution of classes in training set:")
print(y_train.value_counts())

print("Distribution of classes in test set:")
print(y_test.value_counts())

print(classification_report(y_test, predictions))

plt.figure(figsize=(10, 6))
sns.countplot(data=df_encoded, x="WINNER", hue="PREDICTED WINNER", palette='coolwarm')
plt.legend([], [], frameon=False)
plt.title('Prediction Accuracy')
plt.xlabel('Actual Winner')
plt.ylabel('Count')
plt.show()